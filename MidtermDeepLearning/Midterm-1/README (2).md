
# UTS Deep Learning Project

| Nama | Program Studi | Kelas | NIM |
|------|--------------|--------|-----|
| **Darryl Satria Wibowo** | Teknik Komputer | TK-46-GAB | **1103223057** |

---


## 📌 Purpose of the Repository
Repository ini dibuat sebagai bagian dari tugas Ujian Tengah Semester (UTS) pada mata kuliah **Deep Learning**. Tujuannya adalah untuk mendokumentasikan eksperimen, preprocessing dataset, pelatihan model Neural Network, serta evaluasi performanya.

---

## 📘 Project Overview
Project ini berfokus pada proses pengembangan model Neural Network untuk klasifikasi berbasis dataset transaksi. Seluruh proses dilakukan menggunakan notebook interaktif sehingga dapat dipelajari dan direplikasi dengan mudah.

Alur utama project mencakup:

- Exploratory Data Analysis (EDA)
- Normalisasi atau preprocessing data
- Pembangunan model Neural Network
- Training dan evaluasi menggunakan metrik performa
- Analisis hasil dan interpretasi model

---

## 🤖 Model Used & Evaluation Metrics

Model utama yang digunakan dalam project ini adalah:

### 🧠 Neural Network Model (NN)
Model ini dilatih menggunakan dataset berukuran **68.378 sampel** dengan dua kelas target (binary classification).

Hasil evaluasi model berdasarkan classification report:

| Metric | Value |
|--------|-------|
| Accuracy | **0.98** |
| Precision | **0.98 (macro avg)** |
| Recall | **0.98 (macro avg)** |
| F1-Score | **0.98 (macro avg)** |
| Support | **68,378 samples** |

Per kelas performanya adalah:

| Class | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 0.0 | 0.97 | 0.99 | 0.98 | 34,189 |
| 1.0 | 0.99 | 0.97 | 0.98 | 34,189 |

Model menunjukkan performa yang **stabil dan seimbang antar kelas**, menunjukkan bahwa dataset tidak mengalami bias berlebih terhadap salah satu kelas.


---



