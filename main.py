import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import urllib.request
import ssl

st.title("🖼️ Classification d'image avec IA (ResNet50)")

# Ignorer les erreurs de certificat SSL (temporairement)
ssl._create_default_https_context = ssl._create_unverified_context

# Chargement du modèle pré-entraîné
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

# Chargement des labels (ImageNet)
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_path = "imagenet_classes.txt"
    urllib.request.urlretrieve(url, labels_path)
    with open(labels_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

categories = load_labels()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Upload de l’image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image chargée', use_column_width=True)

    # Préparation de l'image
    input_tensor = transform(image).unsqueeze(0)

    # Prédiction
    with st.spinner("L’IA analyse l’image..."):
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("🧠 Résultats de la classification :")
    for i in range(top5_prob.size(0)):
        st.write(f"{categories[top5_catid[i]]} — {top5_prob[i].item() * 100:.2f}%")
