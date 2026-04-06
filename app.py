import os
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from torchvision.models import efficientnet_b4, vit_b_16
import gdown
import os
import gdown

def hello(name):
    return "Hello " + name

demo = gr.Interface(fn=hello, inputs="text", outputs="text")

demo.launch(server_name="0.0.0.0", server_port=7860)
