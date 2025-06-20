

# DetectionModule - Real-Time Face Detection with EfficientNet

**DetectionModule** is a custom face detection model built using EfficientNet-B0 as a feature extractor, a lightweight FPN-style neck, and dual heads for bounding box regression and objectness confidence prediction.

---

## Table of Contents

- [Model Usage](#model-usage)
- [Parameters](#parameters)
- [Model Architecture](#model-architecture)
  - [DetectionModule](#detectionmodule-architecture)
  - [ConvBNAct](#convbnact-module-architecture)
- [Output Format](#output-format)

---

## Model Usage

### Instantiate the Model

```python
import zarathustra as Z
detector = Z.DetectionModule()

```

### Run Inference

```python
output = detector(image_tensor)

```

### Output

The output is a list of predicted bounding boxes per image in the format:

```python
[[cx, cy, w, h, confidence], ...]

```

Where:

-   `cx`, `cy` = center x/y of the box (in pixels)
    
-   `w`, `h` = width and height of the box
    
-   `confidence` = objectness score (0–1)
    

----------

## Parameters

- Parameter

- Type

- Default

- Description

`weights`

`str` or `None`

`"DEFAULT"`

Path to weights file. Use `"DEFAULT"` to load from `FaceDetectionWeights.pth`.

`device`

`str` or `torch.device`

`auto`

Device to run the model on. Automatically uses CUDA if available.

`return_decoded`

`bool`

`True`

If `True`, outputs bounding boxes in pixel space. Otherwise returns raw feature maps.

----------

## Model Architecture

### DetectionModule Architecture

```python
class DetectionModule(nn.Module):
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

```

----------

### ConvBNAct Module Architecture

```python
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

```

----------

## Output Format

If `return_decoded=True`, the model returns a list of bounding boxes per input image:

```python
[
  [[cx, cy, w, h, confidence], ...],  # image 1
  [[cx, cy, w, h, confidence], ...],  # image 2
  ...
]

```

-   Coordinates are in pixel space (assuming input image is 256x256).
    
-   Values are decoded using the model’s internal stride settings.
    

----------



```