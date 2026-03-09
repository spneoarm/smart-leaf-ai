# Smart Leaf AI 🌿

A computer vision app that detects plant diseases from leaf images using PyTorch and Streamlit.

## Features

- Upload leaf image
- Predict disease
- Show confidence score
- Interactive Streamlit UI

## Model

- ResNet18 (transfer learning)
- Trained with PyTorch

## Dataset

PlantVillage dataset (subset)

Classes:
- Potato Late Blight
- Potato Healthy
- Tomato Late Blight
- Tomato Healthy

## Run locally

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py