import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import requests

# 1. Load Model langsung dari torchvision (Menghindari bug torch.hub)
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

# 2. Download label kategori ImageNet
try:
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    labels = response.text.split("\n")
except:
    # Fallback jika gagal download label
    labels = [f"Class {i}" for i in range(1000)]

def predict(inp):
    if inp is None:
        return None
        
    # Pre-processing gambar
    # Gradio mengirim gambar dalam format numpy array
    img = Image.fromarray(inp.astype('uint8'), 'RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    # Prediksi
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(input_tensor)[0], dim=0)
        # Ambil top 3 prediksi untuk ditampilkan di UI
        confidences = {labels[i]: float(prediction[i]) for i in range(len(labels)) if i < 1000}    
    
    return confidences

# 3. Buat UI dengan Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="AI Klasifikasi Gambar",
    description="Unggah gambar (misal: kucing, anjing, mobil) untuk melihat prediksi model MobileNetV2."
)

if __name__ == "__main__":
    demo.launch()
