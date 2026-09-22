import torch
import torch.nn as nn

class SmallCnn(nn.Module):
    def __init__(self,num_classes=8,dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            # Block 2
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            # Block 3
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            # Block 4
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256,8),
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