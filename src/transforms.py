from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT

train_baseline_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)),
])

train_augmented_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)),
])

# train_augmented_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.GaussianBlur(kernel_size=3,sigma=(0.1, 1.5)),
#     transforms.ToTensor(),
#     transforms.RandomErasing(p=0.25,scale=(0.02, 0.1),ratio=(0.3, 3.3),value=0),
#     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
# ])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])