import streamlit as st
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Load pre-trained ResNet50 model for feature extraction
model = resnet50(weights='DEFAULT')
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    # Convert image to RGB to ensure it has 3 channels
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.numpy().flatten()

# Title of the app
st.title('Image Similarity Search App')

# Image upload for user
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load and display sample images
st.sidebar.title('Sample Images')
sample_images = {
    'Sample 1': 'sample1.png',
    'Sample 2': 'sample2.png',
    'Sample 3': 'sample3.png'
}

sample_image_paths = {}
for label, file_name in sample_images.items():
    img_path = f"sample_images/{file_name}"
    img = Image.open(img_path)
    st.sidebar.image(img, caption=label, use_column_width=True)
    sample_image_paths[label] = img

if uploaded_image is not None:
    # Extract features from the uploaded image
    uploaded_image = Image.open(uploaded_image)
    uploaded_features = extract_features(uploaded_image)
    
    similarities = {}
    for label, sample_image in sample_image_paths.items():
        sample_features = extract_features(sample_image)
        similarity = cosine_similarity([uploaded_features], [sample_features])[0][0]
        similarities[label] = similarity

    # Display similarity results
    st.write("Similarity Scores:")
    for label, score in similarities.items():
        st.write(f"{label}: {score:.4f}")
