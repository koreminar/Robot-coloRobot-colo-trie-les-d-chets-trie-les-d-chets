import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import urllib.request

# 🧠 Mapping simple ImageNet → Catégorie déchet
recycling_map = {
    "plastic": ("Plastique", "Oui", "Poubelle Jaune"),
    "bottle": ("Verre", "Oui", "Poubelle Verte"),
    "can": ("Métal", "Oui", "Poubelle Jaune"),
    "paper": ("Papier", "Oui", "Poubelle Bleue"),
    "banana": ("Organique", "Non", "Compost / Ordures"),
    "food": ("Déchet Alimentaire", "Non", "Compost / Ordures"),
    "cup": ("Plastique", "Oui", "Poubelle Jaune"),
    "box": ("Carton", "Oui", "Poubelle Jaune"),
    "bag": ("Plastique", "Oui", "Poubelle Jaune"),
}

# Titre
st.title("♻️ Classificateur de déchets IA")

# Chargement modèle
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

# Labels ImageNet
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, "imagenet_classes.txt")
    with open("imagenet_classes.txt") as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels()

# Prétraitement image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Upload
uploaded_file = st.file_uploader("Charge une image de déchet", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléversée", use_column_width=True)

    # Prétraitement
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    st.subheader("🧠 Résultats de l’analyse IA :")
    found = False
    for i in range(top5_prob.size(0)):
        label = labels[top5_catid[i]]
        st.write(f"- {label} ({top5_prob[i].item()*100:.2f}%)")
        for keyword in recycling_map:
            if keyword in label.lower():
                type_dechet, recyclable, poubelle = recycling_map[keyword]
                st.success(f"✅ C'est un déchet de type **{type_dechet}**")
                st.info(f"♻️ Recyclable ? **{recyclable}**")
                st.warning(f"🗑️ À jeter dans : **{poubelle}**")
                found = True
                break
        if found:
            break
    if not found:
        st.error("⚠️ Ce type de déchet n’a pas pu être reconnu comme recyclable.")
