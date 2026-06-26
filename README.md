# 🌸 Aplikasi Klasifikasi Citra Bunga CNN

Aplikasi web untuk mengklasifikasikan gambar bunga **Mawar**, **Matahari**, dan **Tulip** menggunakan model CNN yang dilatih dengan TensorFlow/Keras.

---

## 📁 Struktur File

```
app_bunga/
├── app.py                              # Aplikasi Streamlit
├── requirements.txt                    # Library yang dibutuhkan
├── README.md                           # Dokumentasi ini
└── model_klasifikasi_bunga_cnn.keras   # Model CNN (diunduh dari Colab)
```

---

## 🚀 Cara Deploy ke Streamlit Cloud

### Langkah 1 — Latih model di Google Colab
1. Buka notebook `praktikum_klasifikasi_bunga_cnn.ipynb` di Google Colab
2. Jalankan semua cell dari atas sampai bawah
3. Setelah selesai, unduh file model:
   ```python
   from google.colab import files
   files.download("/content/model_klasifikasi_bunga_cnn.keras")
   ```

### Langkah 2 — Upload ke GitHub
1. Buat akun di [github.com](https://github.com) jika belum punya
2. Buat repository baru (klik **New repository**), beri nama misalnya `klasifikasi-bunga`
3. Upload semua file berikut ke repository tersebut:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `model_klasifikasi_bunga_cnn.keras`

### Langkah 3 — Deploy ke Streamlit Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub kamu
3. Klik **New app**
4. Pilih repository `klasifikasi-bunga`
5. Pada kolom **Main file path**, isi: `app.py`
6. Klik **Deploy** — tunggu beberapa menit
7. Aplikasi kamu akan online dengan URL seperti: `https://namakamu-klasifikasi-bunga.streamlit.app`

---

## 🌸 Kelas yang Didukung

| Kelas | Emoji | Ciri Khas |
|-------|-------|-----------|
| Mawar | 🌹 | Kelopak berlapis, warna merah/pink/kuning |
| Matahari | 🌻 | Cakram cokelat, kelopak kuning, batang tinggi |
| Tulip | 🌷 | Bentuk cangkir, warna beragam, daun panjang |

---

## 🛠️ Teknologi

- **Python** 3.10+
- **TensorFlow / Keras** — model CNN
- **Streamlit** — antarmuka web
- **Pillow** — pemrosesan gambar
