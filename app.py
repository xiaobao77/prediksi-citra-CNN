import streamlit as st
import numpy as np
from PIL import Image
os.environ["KERAS_BACKEND"] = "numpy"
import keras
import os

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Klasifikasi Bunga",
    page_icon="🌸",
    layout="centered"
)

# ==========================================================
# CSS KUSTOM
# ==========================================================
st.markdown("""
    <style>
        .judul {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            color: #c0392b;
            margin-bottom: 0;
        }
        .subjudul {
            text-align: center;
            font-size: 1rem;
            color: #7f8c8d;
            margin-bottom: 1.5rem;
        }
        .kartu-hasil {
            background: #fdf6f0;
            border-left: 5px solid #e74c3c;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-top: 1rem;
        }
        .label-prediksi {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
        }
        .confidence {
            font-size: 1rem;
            color: #7f8c8d;
        }
        .footer {
            text-align: center;
            font-size: 0.8rem;
            color: #bdc3c7;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# JUDUL
# ==========================================================
st.markdown('<p class="judul">🌸 Klasifikasi Citra Bunga</p>', unsafe_allow_html=True)
st.markdown('<p class="subjudul">Mawar · Matahari · Tulip — berbasis CNN</p>', unsafe_allow_html=True)
st.divider()

# ==========================================================
# LOAD MODEL
# ==========================================================
MODEL_PATH = "model_klasifikasi_bunga_cnn.keras"
IMG_SIZE   = 160
CLASS_NAMES = ["mawar", "matahari", "tulip"]
EMOJI       = {"mawar": "🌹", "matahari": "🌻", "tulip": "🌷"}
DESKRIPSI   = {
    "mawar":    "Mawar dikenal dengan kelopak berlapis dan aroma harum. Warna umumnya merah, pink, atau kuning.",
    "matahari": "Bunga matahari memiliki cakram cokelat besar dikelilingi kelopak kuning cerah.",
    "tulip":    "Tulip berbentuk cangkir dengan warna beragam: merah, ungu, oranye, kuning, hingga putih.",
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================================================
# UPLOAD GAMBAR
# ==========================================================
st.subheader("📤 Upload Gambar Bunga")
uploaded = st.file_uploader(
    "Pilih file gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload foto bunga mawar, matahari, atau tulip"
)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img_pil, caption="Gambar yang diupload", use_container_width=True)

    with col2:
        if model is None:
            st.error(
                "⚠️ Model belum tersedia.\n\n"
                "Jalankan dulu notebook Colab untuk melatih dan mengunduh "
                "`model_klasifikasi_bunga_cnn.keras`, lalu letakkan di folder yang sama dengan `app.py`."
            )
        else:
            with st.spinner("🔍 Menganalisis gambar..."):
                # Preprocessing
                img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
                img_array  = np.array(img_resized, dtype=np.float32)
                img_array  = np.expand_dims(img_array, axis=0)

                # Prediksi
                skor = model.predict(img_array, verbose=0)[0]
                idx  = int(np.argmax(skor))
                label       = CLASS_NAMES[idx]
                confidence  = float(np.max(skor)) * 100

            # Hasil utama
            st.markdown(f"""
                <div class="kartu-hasil">
                    <div class="label-prediksi">{EMOJI[label]} {label.capitalize()}</div>
                    <div class="confidence">Confidence: <b>{confidence:.2f}%</b></div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<br><small>💬 {DESKRIPSI[label]}</small>", unsafe_allow_html=True)

            # Skor semua kelas
            st.markdown("#### 📊 Skor Semua Kelas")
            for nama, nilai in zip(CLASS_NAMES, skor):
                persen = float(nilai) * 100
                st.progress(
                    int(persen),
                    text=f"{EMOJI[nama]} {nama.capitalize()} — {persen:.2f}%"
                )

# ==========================================================
# PANDUAN
# ==========================================================
with st.expander("ℹ️ Cara Menggunakan Aplikasi"):
    st.markdown("""
    1. **Latih model** terlebih dahulu menggunakan notebook Google Colab yang sudah disediakan.
    2. **Unduh file model** `model_klasifikasi_bunga_cnn.keras` dari Colab ke komputer kamu.
    3. **Letakkan file model** di folder yang sama dengan `app.py`.
    4. **Upload gambar** bunga (mawar, matahari, atau tulip) menggunakan tombol di atas.
    5. Hasil prediksi dan tingkat kepercayaan model akan ditampilkan secara otomatis.

    > **Tips:** Gunakan foto dengan pencahayaan baik, 1 bunga, dan latar tidak terlalu ramai
    > agar hasil prediksi lebih akurat.
    """)

with st.expander("🌸 Kelas Bunga yang Didukung"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🌹 Mawar\nKelopak berlapis, harum, warna merah/pink/kuning.")
    with c2:
        st.markdown("### 🌻 Matahari\nCakram cokelat besar, kelopak kuning, batang tinggi.")
    with c3:
        st.markdown("### 🌷 Tulip\nBentuk cangkir, warna beragam, daun panjang.")

# ==========================================================
# FOOTER
# ==========================================================
st.markdown('<p class="footer">Praktikum Big Data Analitik · Klasifikasi Citra Bunga CNN</p>', unsafe_allow_html=True)
