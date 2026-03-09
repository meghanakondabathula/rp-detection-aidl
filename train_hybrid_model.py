import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b4, vit_b_16
from tqdm import tqdm

# -------------------------------
# Settings
# -------------------------------
DATASET_DIR = "./final_dataset"   # Folder with subfolders Healthy/Retinitis_Pigmentosa
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "hybrid_model.pth"

classes = ["Healthy", "Retinitis_Pigmentosa"]

# -------------------------------
# Transformations (resize + normalize)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((380, 380)),  # EfficientNet-B4 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Dataset Loader
# -------------------------------
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# -------------------------------
# Hybrid Model Definition
# -------------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.effnet = efficientnet_b4(pretrained=True)
        self.effnet.classifier = nn.Identity()  # Remove classifier

        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()          # Remove classifier

        combined_dim = 1792 + 768
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # EfficientNet input: 380x380
        f1 = self.effnet(x)

        # ViT input: resize to 224x224 inside model
        vit_input = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        f2 = self.vit(vit_input)

        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)

# -------------------------------
# Initialize Model, Loss, Optimizer
# -------------------------------
model = HybridModel(num_classes=len(classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------------
# Training Loop
# -------------------------------
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    print(f"Validation Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Saved best model with val acc: {best_val_acc:.4f}")

# -------------------------------
# Testing
# -------------------------------
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / test_total
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"Hybrid model training complete! Saved as {MODEL_PATH}")
