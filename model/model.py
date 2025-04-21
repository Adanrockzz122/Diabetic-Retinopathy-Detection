# model/model.py
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os

def get_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("model/retinopathy_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def generate_gradcam_with_confidence(model, image_tensor, image_path):
    target_layer = model.layer4[1].conv2
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence = probs.max().item()
    class_idx = torch.argmax(output).item()

    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].squeeze().detach().numpy()
    acts = activations[0].squeeze().detach().numpy()
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    orig = Image.open(image_path).resize((224, 224))
    orig = np.array(orig)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + orig

    gradcam_path = "static/gradcam.jpg"
    cv2.imwrite(gradcam_path, superimposed)

    hook_f.remove()
    hook_b.remove()

    return class_idx, confidence, gradcam_path
