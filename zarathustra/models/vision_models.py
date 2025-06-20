import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )


class ArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
        return output


class DetectionModule(nn.Module):
    '''
    DetectionModule(weights="DEFAULT", device=None, return_decoded=True)

    A custom real-time face detection model built on top of EfficientNet-B0 with a simple FPN-style neck and two heads:
    one for bounding box regression and another for objectness confidence.

    Args:
        weights (str): Path to weights file or "DEFAULT" to load from FaceDetectionWeights.pth.
        device (torch.device or str): The device to run the model on. Auto-detects CUDA if not specified.
        return_decoded (bool): If True, decodes bounding boxes into pixel coordinates with confidence scores.

    Usage:
        model = DetectionModule()
        output = model(image_tensor)
        output is a list of boxes with format [cx, cy, w, h, confidence] for each image in batch.
    '''
    def __init__(self, weights="DEFAULT", device=None, return_decoded=True):
        super().__init__()
        self.return_decoded = return_decoded
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = models.efficientnet_b0(weights="DEFAULT").features.to(self.device)
        self.neck = nn.Sequential(
            ConvBNAct(1280, 192, 3, padding=1),
            ConvBNAct(192, 128, 3, padding=1),
            ConvBNAct(128, 96, 3, padding=1)
        ).to(self.device)
        self.bbox_head = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 4, 1)
        ).to(self.device)
        self.obj_head = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1)
        ).to(self.device)
        if weights is not None:
            path = "FaceDetectionWeights.pth" if weights == "DEFAULT" else weights
            self.load_weights_safe(path)
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        bbox = self.bbox_head(x)
        obj = self.obj_head(x)
        x = torch.cat([bbox, obj], dim=1)
        if self.return_decoded:
            x = self.decode_predictions(x)
        return x

    def decode_predictions(self, preds, conf_thresh=0.05):
        B, _, H, W = preds.shape
        preds = preds.permute(0, 2, 3, 1).contiguous()
        stride_x = 256 / W
        stride_y = 256 / H
        all_boxes = []
        for b in range(B):
            pred = preds[b]
            conf = torch.sigmoid(pred[..., 4])
            conf_mask = conf > conf_thresh
            boxes = []
            ys, xs = conf_mask.nonzero(as_tuple=True)
            for y, x in zip(ys, xs):
                px, py, pw, ph = pred[y, x, :4]
                pconf = conf[y, x].item()
                cx = (x + px.item()) * stride_x
                cy = (y + py.item()) * stride_y
                w = pw.item() * stride_x
                h = ph.item() * stride_y
                boxes.append([cx, cy, w, h, pconf])
            all_boxes.append(boxes)
        return all_boxes

    def load_weights_safe(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name in own_state and param.size() == own_state[name].size():
                    own_state[name].copy_(param)
        except:
            pass


class RecognitionModule(nn.Module):
    '''
    RecognitionModule(emb_dim=384, num_classes=None, s=30.0, m=0.50, weights="DEFAULT", device="cpu")

    A custom face recognition model built using EfficientNet-B0 as a feature extractor, a 1x1 convolution neck,
    and an ArcFace classification head. If `num_classes` is None, it returns normalized embeddings.

    Args:
        emb_dim (int): Dimensionality of the output embedding.
        num_classes (int): Number of identity classes for classification. If None, the model outputs embeddings.
        s (float): ArcFace scale factor.
        m (float): ArcFace margin.
        weights (str): Path to weights file or "DEFAULT" to load from FaceRecognitionWeights.pth.
        device (torch.device or str): The device to run the model on.

    Usage:
        model = RecognitionModule(num_classes=1000)
        logits = model(image_tensor, labels)
        embeddings = model(image_tensor)
    '''
    def __init__(self, emb_dim=384, num_classes=None, s=30.0, m=0.50, weights="DEFAULT", device="cpu"):
        super().__init__()
        self.device = device
        self.backbone = models.efficientnet_b0(weights="DEFAULT").features.to(self.device)
        self.neck = ConvBNAct(1280, emb_dim, kernel_size=1).to(self.device)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(0.5),
        ).to(self.device)
        self.arcface = ArcFace(emb_dim, num_classes, s=s, m=m).to(self.device) if num_classes is not None else None
        if weights is not None:
            path = "FaceRecognitionWeights.pth" if weights == "DEFAULT" else weights
            self.load_weights_safe(path)
        self.to(self.device)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, dim=1)
        if labels is not None and self.arcface is not None:
            logits = self.arcface(x, labels)
            return logits
        return x

    def load_weights_safe(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name in own_state and param.size() == own_state[name].size():
                    own_state[name].copy_(param)
        except:
            pass




