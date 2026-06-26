import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Klasifikasi Bunga", page_icon="🌸", layout="centered")

st.title("🌸 Klasifikasi Citra Bunga")
st.write("Upload gambar bunga (Mawar, Matahari, Tulip) untuk prediksi.")

# --- LOAD MODEL ---
# Pastikan nama file di GitHub adalah model_klasifikasi_bunga_cnn.h5
MODEL_PATH = "model_klasifikasi_bunga_cnn.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File {MODEL_PATH} tidak ditemukan di GitHub!")
        return None
    try:
        # compile=False adalah kunci untuk menghindari error 'batch_shape'
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

model = load_my_model()

# --- INPUT GAMBAR ---
uploaded_file = st.file_uploader("Pilih gambar bunga...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Gambar yang diupload', width=300)
    
    # Preprocessing
    img_resized = image.resize((160, 160))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    prediction = model.predict(img_array)
    class_names = ["mawar", "matahari", "tulip"]
    result = class_names[np.argmax(prediction)]
    
    st.write(f"### 🎯 Hasil Prediksi: {result.capitalize()}")
