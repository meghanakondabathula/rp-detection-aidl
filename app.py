import os
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from torchvision.models import efficientnet_b4, vit_b_16
import gdown
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1jCY4ZITeI33q2Hvblgpq03UbBgwaGya3"
MODEL_PATH = "hybrid_model.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

download_model()
# Model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = efficientnet_b4(weights=None)
        self.effnet.classifier = nn.Identity()
        self.vit = vit_b_16(weights=None)
        self.vit.heads = nn.Identity()
        self.classifier = nn.Linear(1792 + 768, 2)

    def forward(self, x):
        f1 = self.effnet(x)
        f2 = self.vit(x)
        return self.classifier(torch.cat((f1, f2), dim=1))

model = HybridModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

classes = ["Healthy", "Retinitis Pigmentosa"]

def predict(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return classes[pred.item()]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="RP Detection",
)

demo.launch(server_name="0.0.0.0", server_port=7860)