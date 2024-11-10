import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('nutriguide_model.keras')

# Define class names
class_names = ["Aloo Paratha", "Dhokla", "Dosa", "Idli", "Malai Kofta", "Puran Poli", "Samosa"]

# Set up the Streamlit page
st.set_page_config(page_title="NutriGuide", page_icon="🍎", layout="centered")

# Custom styling for the app
st.markdown(
    """
    <style>
    .main-heading {
        font-size: 50px;
        color: #4CAF50;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
    }
    .sub-heading {
        font-size: 24px;
        color: #76c7c0;
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-button {
        background-color: #4CAF50; 
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
        border: none;
        margin-top: 20px;
    }
    .output-image {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main heading
st.markdown("<h1 class='main-heading'>NutriGuide</h1>", unsafe_allow_html=True)

# Sub-heading
st.markdown("<h3 class='sub-heading'>Food Image Recognition</h3>", unsafe_allow_html=True)

# Upload image section
uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])

# Placeholder for captured image button (functionality for real camera capture requires additional setup)
st.markdown("<div style='text-align: center;'>or</div>", unsafe_allow_html=True)
st.button("Capture Image")  # Placeholder; not functional in Streamlit

# Process and display the uploaded image
if uploaded_file:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img_resized = image.resize((224, 224))  # Resize to model's input size if required
    img_array = np.array(img_resized) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction using the model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_class]

    # Display the prediction result
    st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
