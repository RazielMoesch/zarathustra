



# RecognitionModule - Face Recognition with ArcFace and EfficientNet

**RecognitionModule** is a modular face recognition model built with PyTorch. It uses EfficientNet-B0 as a feature extractor, a 1x1 convolutional neck, and an ArcFace head to produce discriminative face embeddings.

---

## Installation

Install via pip:

```bash
pip install zarathustra

```

----------

## Table of Contents

-   [Model Usage](https://chatgpt.com/?temporary-chat=true#model-usage)
    
-   [Parameters](https://chatgpt.com/?temporary-chat=true#parameters)
    
-   [Model Architecture](https://chatgpt.com/?temporary-chat=true#model-architecture)
    
    -   [RecognitionModule](https://chatgpt.com/?temporary-chat=true#recognitionmodule-architecture)
        
    -   [ConvBNAct](https://chatgpt.com/?temporary-chat=true#convbnact-module-architecture)
        
    -   [ArcFace](https://chatgpt.com/?temporary-chat=true#arcface-module-architecture)
        


----------

## Model Usage

### Instantiate the Model

```python
import zarathustra as Z
recognition = Z.RecognitionModule()

```

### Use for Embeddings

```python
embeddings = recognition(image_tensor)

```

### Use for Training (with labels)

```python
logits = recognition(image_tensor, labels)

```

----------

## Parameters

- Parameter

- Type

- Default

- Description


`emb_dim`

`int`

`384`

Dimensionality of the output face embedding. The output will be a 1xemb_dim Vector.

`num_classes`

`int` or `None`

`None`

Number of identity classes. Set this during training. If `None`, model returns embeddings.

`s`

`float`

`30.0`

ArcFace scale factor. Boosts softmax logits for better training convergence.

`m`

`float`

`0.5`

ArcFace angular margin. Encourages greater inter-class separation.

`weights`

`str` or `None`

`"DEFAULT"`

Path to a weights file. Use `"DEFAULT"` to load pretrained face recognition weights.

`device`

`str` or `torch.device`

`"cpu"`

Device to run the model on (`"cpu"` or `"cuda"`).

----------

## Model Architecture

### RecognitionModule Architecture

```python
class RecognitionModule(nn.Module):
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
            return self.arcface(x, labels)
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

### ArcFace Module Architecture

```python
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
        cosine = F.linear(embeddings, weights).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
        return output

```

----------

