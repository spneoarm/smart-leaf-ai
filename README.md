# Plant Leaf Disease Multiclass Classification 🌿

A deployable Streamlit app that classifies potato and tomato leaf images using a PyTorch ResNet18 model.

## Features
- Upload a leaf image
- Predict plant disease class
- Show confidence score
- Display class probabilities
- Deployable on Streamlit Community Cloud

## Classes
- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Tomato Healthy

## Model
- ResNet18
- Transfer learning
- Trained with PyTorch on a PlantVillage subset

## Project Structure

```text
smart-leaf-ai/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
└── models/
    └── leaf_model_final.pth