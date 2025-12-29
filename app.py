import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import requests

# 1. Load Model (MobileNetV2 sebagai contoh)
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()

# 2. Download label kategori ImageNet
response = requests.get("https://git.io/JJknP")
labels = response.text.split("\n")

def predict(inp):
    # Pre-processing gambar
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(inp).unsqueeze(0)
    
    # Prediksi
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(input_tensor)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}    
    return confidences

# 3. Buat UI dengan Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Object Classifier",
    description="Unggah gambar untuk mengetahui objek apa itu."
)

demo.launch()
