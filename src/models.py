import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,ResNet18_Weights,
    vgg19,
    mobilenet_v3_small,MobileNet_V3_Small_Weights,
    mobilenet_v3_large,MobileNet_V3_Large_Weights
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class SmallCnn(nn.Module):
    def __init__(self,num_classes=8,dropout=0.0,pooling="max"):
        super().__init__()
        if pooling == "max":
            pool1 = nn.MaxPool2d(2, 2)
            pool2 = nn.MaxPool2d(2, 2)
            pool3 = nn.MaxPool2d(2, 2)

        elif pooling == "avg":
            pool1 = nn.AvgPool2d(2, 2)
            pool2 = nn.AvgPool2d(2, 2)
            pool3 = nn.AvgPool2d(2, 2)

        else:
            raise ValueError( f"Unsupported pooling: {pooling}. Use 'max' or 'avg'.")

        self.conv = nn.Sequential(
        # Block 1
        nn.Conv2d(3,32,3,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),

        nn.Conv2d(32,32,3,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),

        pool1,

        # Block 2
        nn.Conv2d(32,64,3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64,64,3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        pool2,

        # Block 3
        nn.Conv2d(64,128,3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128,128,3,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        pool3,

        # Block 4
        nn.Conv2d(128,256,3,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
    )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256,num_classes),
        )
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# *************
# test model
# *************
model_smallcnn = SmallCnn(num_classes=8)
x_smallcnn = torch.randn(1,3,224,224)
y_smallcnn = model_smallcnn(x_smallcnn)
# print(y_smallcnn.shape)

total_params_smallcnn = sum(p.numel() for p in model_smallcnn.parameters())
trainable_params_smallcnn = sum(p.numel() for p in model_smallcnn.parameters() if p.requires_grad)
# print(f"Total parameters small cnn: {total_params_smallcnn:,}")        # 585,640
# print(f"Trainable parameters small cnn: {trainable_params_smallcnn:,}")        # 585,640


def resnet18_model(num_classes=8):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features,num_classes)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # print(f"Trainable params: {trainable_params}")
    # print(f"Total params: {total_params}")

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)

    model = model.to(device)

    return model


def mobilenet_v3_small_model(num_classes=8):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features,out_features=num_classes)
    model = model.to(device)
    return model


def build_mobilenet_v3_large(num_classes=8, pretrained=True):
    if pretrained:
        weights = MobileNet_V3_Large_Weights.DEFAULT
    else:
        weights = None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class DepthwiseCNN(nn.Module):
    def __init__(self, num_classes=8,dropout=0.0):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            DepthwiseSeparableConv(in_channels=3,out_channels=32),
            DepthwiseSeparableConv(in_channels=32,out_channels=32),
            nn.MaxPool2d(2,2),

            # Block 2
            DepthwiseSeparableConv(in_channels=32,out_channels=64),
            DepthwiseSeparableConv(in_channels=64,out_channels=64),
            nn.MaxPool2d(2,2),

            # Block 3
            DepthwiseSeparableConv(in_channels=64,out_channels=128),
            DepthwiseSeparableConv(in_channels=128,out_channels=128),
            nn.MaxPool2d(2,2),

            # Block 4
            DepthwiseSeparableConv(in_channels=128,out_channels=256),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256,num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x


# *************
# test model
# *************
model_depth = DepthwiseCNN(num_classes=8, dropout=0.0)
x_depth = torch.randn(4, 3, 224, 224)
y_depth = model_depth(x_depth)
# print("Input shape :", x_depth.shape)     #torch.Size([4, 3, 224, 224])
# print("Output shape:", y_depth.shape)     #torch.Size([4, 8])

total_params_depth = sum(p.numel() for p in model_depth.parameters())
trainable_params_depth = sum(p.numel() for p in model_depth.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params_depth:,}")        # 73,033
# print(f"Trainable parameters: {trainable_params_depth:,}")        # 73,033
