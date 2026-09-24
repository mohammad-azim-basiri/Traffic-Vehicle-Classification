import torch
import torch.nn as nn

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

# model = SmallCnn(num_classes=8)
# x = torch.randn(1,3,224,224)
# y = model(x)
# print(y.shape)
#
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(
#     p.numel() for p in model.parameters()
#     if p.requires_grad
# )
#
# print(f"Total parameters: {total_params:,}")        # 585,640
# print(f"Trainable parameters: {trainable_params:,}")        # 585,640