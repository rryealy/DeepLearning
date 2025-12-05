## Deep Learning: Prediksi Tahun (UTS Deep Learning)

| Nama | Program Studi | Kelas | NIM |
|------|--------------|--------|-----|
| **Darryl Satria Wibowo** | Teknik Komputer | TK-46-GAB | **1103223057** |
--
### Tujuan Repository
Repository ini berisi implementasi model Deep Learning (Multi-Layer Perceptron) menggunakan framework PyTorch untuk menyelesaikan tugas regresi. Tujuan utamanya adalah membangun, melatih, dan mengevaluasi model yang dapat memprediksi nilai target (tahun) dari sebuah dataset besar.

## Ikhtisar Proyek
Proyek ini berfokus pada pelatihan model deep learning untuk tugas regresi. Langkah-langkah utama yang dilakukan dalam notebook adalah: 
-  Pemuatan dan Pembersihan Data: Memuat dataset dari file CSV,    memeriksa bentuk aslinya, dan membersihkan data dari baris yang sepenuhnya kosong (NaN).
- Eksplorasi Data: Memisahkan fitur ($X$) dan target ($y$), menganalisis rentang dan distribusi target (Tahun), dan memvisualisasikan korelasi fitur serta keberadaan missing values.
- Pra-pemrosesan Data: Membagi data menjadi training (80%) dan testing (20%). Mengisi missing values pada fitur dengan median (Imputasi) dan menormalisasi fitur serta target menggunakan StandardScaler untuk mempersiapkan data bagi model deep learning.
- Desain dan Pelatihan Model: Membangun model SimpleMLP menggunakan PyTorch dengan beberapa layer tersembunyi, Batch Normalization, dan Dropout. Model dilatih menggunakan Loss Function MSELoss dan Optimizer Adam, serta memanfaatkan teknik Early Stopping dan Learning Rate Scheduler (ReduceLROnPlateau) untuk meningkatkan stabilitas dan efisiensi pelatihan.
- Evaluasi Model: Mengevaluasi model terbaik (checkpoint) pada test set menggunakan berbagai metrik regresi dalam skala nilai target aslinya.
- Visualisasi Hasil: Memvisualisasikan loss curve, hasil prediksi, dan distribusi residual untuk analisis lebih lanjut.

## Model dan Hasil metrik
### Arsitektur Model
Model yang digunakan adalah SimpleMLP dengan arsitektur sebagai berikut:
- Input Layer: Dimensi masukan sesuai jumlah fitur (90).
- Hidden Layers: Tiga layer tersembunyi dengan dimensi: [128, 64, 32].
  - Setiap layer tersembunyi diikuti oleh Batch Normalization, fungsi aktivasi ReLU, dan Dropout (tingkat 0.3) untuk regulasi.
  
- Output Layer: Satu layer linear dengan dimensi keluaran 1 (untuk regresi).
- Loss Function: Mean Squared Error (nn.MSELoss).
- Optimizer: Adam (lr = 0.001, weight_decay = 1e-5).

## Hasil Evaluasi (Skala Nilai Asli)
Model dihentikan pada Epoch 54 karena pemicu Early Stopping (kesabaran 10), dengan hasil terbaik dari validation set dan evaluasi akhir pada test set sebagai berikut:
### MSE : 71.377
### RMSE : 8.4485
### MAE : 5.8405
### R2 : 0.3935




