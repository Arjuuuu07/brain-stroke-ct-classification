import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class ResNetClassifier(nn.Module):
    """ResNet backbone + global average pooling + one linear logit (used by scripts/train.py)."""

    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        base = getattr(tv_models, backbone)(weights="IMAGENET1K_V1" if pretrained else None)
        self.body = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                  base.layer1, base.layer2, base.layer3, base.layer4)
        self.fc = nn.Linear(base.fc.in_features, 1)

    def forward(self, x):
        return self.fc(self.body(x).mean((2, 3)))


# ----------------------------------------------------------------------
# Kept for the parked segmentation script (archive/segmentation_old.py).
# Not used by the classification pipeline.
# ----------------------------------------------------------------------
class Image3ChResNet(nn.Module):
    """Plain ResNet18 encoder, in_channels=3. Classification head only."""

    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def _encode(self, x):
        s0 = self.stem(x)
        s0p = self.maxpool(s0)
        s1 = self.layer1(s0p)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s1, s2, s3, s4

    def forward(self, x, seg=False):
        _, _, _, s4 = self._encode(x)
        pooled = self.avgpool(s4).flatten(1)
        class_logit = self.classifier(pooled)
        return class_logit, None


class Image3ChSegResNet(Image3ChResNet):
    """Same encoder as Image3ChResNet + U-Net decoder with fresh init."""

    def __init__(self):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.seg_head = nn.Conv2d(32, 1, 1)

    def forward(self, x, seg=True):
        s1, s2, s3, s4 = self._encode(x)
        cls_logit = self.classifier(self.avgpool(s4).flatten(1))
        if not seg:
            return cls_logit, None
        d = self.up4(s4); d = self.dec4(torch.cat([d, s3], 1))
        d = self.up3(d);  d = self.dec3(torch.cat([d, s2], 1))
        d = self.up2(d);  d = self.dec2(torch.cat([d, s1], 1))
        d = self.up1(d);  d = self.dec1(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        return cls_logit, self.seg_head(d)


def set_train_mode(model, keep_encoder_bn_eval=True):
    model.train()
    if keep_encoder_bn_eval:
        for mod in (model.stem, model.layer1, model.layer2, model.layer3, model.layer4):
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
