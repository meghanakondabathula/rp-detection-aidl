import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b4, vit_b_16
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Settings
# -------------------------------
DATASET_DIR = "./final_dataset"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "hybrid_model.pth"
classes = ["Healthy", "Retinitis_Pigmentosa"]

# -------------------------------
# Transform (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Dataset Loader
# -------------------------------
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

# Split same way (70-15-15)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
_, _, test_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# Hybrid Model Definition
# -------------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.effnet = efficientnet_b4(weights="IMAGENET1K_V1")
        self.effnet.classifier = nn.Identity()

        self.vit = vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads = nn.Identity()

        combined_dim = 1792 + 768
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.effnet(x)
        f2 = self.vit(x)
        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)

# -------------------------------
# Load Trained Model
# -------------------------------
model = HybridModel(num_classes=len(classes)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# Evaluation
# -------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion Matrix & Report
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=classes))

# -------------------------------
# Plot Confusion Matrix
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
