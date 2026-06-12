import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.markdown("""
    <style>
           .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }
    </style>
    """, unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
model_path = os.path.join(base_dir, "Model", "vgg16_covid_model.keras")

# Page config
st.set_page_config(page_title="COVID-19 X-Ray Detection",layout="centered")

# Load model
@st.cache_resource
def load_covid_model():
    return load_model(model_path)

model = load_covid_model()

# Class labels
class_names = [
                "Covid",
                "Normal",
                "Viral Pneumonia"
            ]

st.title("🩺 COVID-19 Chest X-Ray Classification")

st.write(
    """
    Upload a Chest X-Ray image and the model will classify it as:
    
    - Covid
    - Normal
    - Viral Pneumonia
    """
)

uploaded_file = st.file_uploader("Upload Chest X-Ray Image",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img,caption="Uploaded X-Ray",use_container_width=True)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    pred_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    if class_names[pred_class] == "Covid":
        st.error(f"⚠️ Prediction: {class_names[pred_class]}")

    elif class_names[pred_class] == "Normal":
        st.success(f"✅ Prediction: {class_names[pred_class]}")

    else:
        st.warning(f"🫁 Prediction: {class_names[pred_class]}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")

    st.write(
                {
                    class_names[i]:
                    float(prediction[0][i] * 100)
                    for i in range(3)
                }
            )
    
    prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability": prediction[0] * 100
        })

    st.bar_chart(prob_df.set_index("Class"))