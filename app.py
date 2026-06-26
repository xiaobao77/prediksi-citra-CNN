import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

st.set_page_config(page_title="Klasifikasi Bunga", page_icon="🌸", layout="centered")

st.markdown("""
    <style>
        .judul { text-align: center; font-size: 2.2rem; font-weight: 700; color: #c0392b; }
        .subjudul { text-align: center; font-size: 1rem; color: #7f8c8d; margin-bottom: 1.5rem; }
        .kartu-hasil { background: #fdf6f0; border-left: 5px solid #e74c3c; border-radius: 8px; padding: 1rem 1.5rem; margin-top: 1rem; }
        .label-prediksi { font-size: 1.8rem; font-weight: 700; color: #2c3e50; }
        .confidence { font-size: 1rem; color: #7f8c8d; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="judul">🌸 Klasifikasi Citra Bunga</p>', unsafe_allow_html=True)
st.markdown('<p class="subjudul">Mawar · Matahari · Tulip — berbasis CNN</p>', unsafe_allow_html=True)
st.divider()

IMG_SIZE = 160
CLASS_NAMES = ["mawar", "matahari", "tulip"]
EMOJI = {"mawar": "🌹", "matahari": "🌻", "tulip": "🌷"}
DESKRIPSI = {
    "mawar": "Mawar dikenal dengan kelopak berlapis dan aroma harum.",
    "matahari": "Bunga matahari memiliki cakram cokelat besar dikelilingi kelopak kuning cerah.",
    "tulip": "Tulip berbentuk cangkir dengan warna beragam.",
}

# --- PERUBAHAN UTAMA DI SINI ---
MODEL_PATH = "model_klasifikasi_bunga_cnn.h5" 

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        # Menambahkan compile=False untuk menghindari error 'Unrecognized keyword arguments'
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

st.subheader("📤 Upload Gambar Bunga")
uploaded = st.file_uploader("Pilih file gambar (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img_pil, caption="Gambar yang diupload", use_container_width=True)
    with col2:
        with st.spinner("Memuat model..."):
            model = load_my_model()
            
        if model is None:
            st.warning(f"File {MODEL_PATH} tidak ditemukan atau rusak. Pastikan file .h5 ada di folder.")
        else:
            with st.spinner("Menganalisis gambar..."):
                img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0 # Normalisasi jika perlu
                img_array = np.expand_dims(img_array, axis=0)
                
                skor = model.predict(img_array, verbose=0)[0]
                idx = int(np.argmax(skor))
                label = CLASS_NAMES[idx]
                confidence = float(np.max(skor)) * 100
                
            st.markdown(f"""
            <div class="kartu-hasil">
                <div class="label-prediksi">{EMOJI[label]} {label.capitalize()}</div>
                <div class="confidence">Confidence: <b>{confidence:.2f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<br><small>💬 {DESKRIPSI[label]}</small>", unsafe_allow_html=True)
            st.markdown("#### 📊 Skor Semua Kelas")
            for nama, nilai in zip(CLASS_NAMES, skor):
                persen = float(nilai) * 100
                st.progress(min(int(persen)/100, 1.0), text=f"{EMOJI[nama]} {nama.capitalize()} — {persen:.2f}%")
