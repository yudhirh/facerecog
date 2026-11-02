# Face Similarity Checker

Aplikasi sederhana untuk mengecek similaritas dua wajah menggunakan:
- Face embedding model: ArcFace
- DeepFace similarity: `DeepFace.verify`

## Fitur
- Upload dua foto wajah
- Hitung cosine similarity dari embedding vector
- Tampilkan skor gabungan eksperimental dari cosine, Euclidean, korelasi, dan Manhattan
- Hitung similarity menggunakan model DeepFace
- Opsional tampilkan kotak wajah yang terdeteksi RetinaFace pada foto
- Tampilkan hasil detail seperti distance, threshold, dan verified status

### Membaca skor eksperimental

Panduan lengkap juga tersedia di aplikasi melalui expander **“Panduan lengkap
membaca skor eksperimental”**. Skor gabungan adalah indeks internal—bukan
persentase, probabilitas identitas, atau keputusan lolos/tidak lolos. Belum ada
threshold yang dikalibrasi, jadi jangan menafsirkan angka tertentu sebagai
batas kecocokan.

Skor menggabungkan empat metrik embedding ArcFace: cosine (45%), Euclidean
(25%), korelasi (15%), dan Manhattan (15%). Nilai lebih tinggi berarti skor
gabungan menurut rumus ini lebih tinggi, bukan peluang identitas lebih besar.
Cosine dan Euclidean berkaitan erat setelah normalisasi. Manhattan yang
membagi jarak dengan jumlah dimensi dapat terlihat tinggi secara bawaan pada
embedding berdimensi besar.

Konsistensi metrik dihitung sebagai `1 - simpangan baku` dari cosine, Euclidean,
dan korelasi yang dipetakan; Manhattan tidak ikut dihitung. Nilai ini hanya
heuristik keselarasan angka, bukan keyakinan atau bukti kebenaran.

Gunakan `Verified`, `Distance`, dan `Threshold` DeepFace/ArcFace untuk membaca
keputusan aplikasi saat ini. Skor eksperimental tidak memengaruhi keputusan
tersebut. Agar skor gabungan dapat dipakai untuk keputusan, rumus dan threshold
harus diuji serta dikalibrasi menggunakan pasangan foto berlabel yang representatif.

## Cara menjalankan

1. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
   Catatan: untuk kompatibilitas DeepFace + TensorFlow di Python 3.13, paket `tf-keras` sudah ditambahkan di `requirements.txt`.
2. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
3. Buka URL yang ditampilkan di terminal, biasanya:
   ```text
   http://localhost:8501
   ```

## Paket portable Windows

Untuk membuat folder portable yang bisa dipindahkan ke PC Windows x64 tanpa
Python terinstal dan tanpa mengunduh dependency atau model saat dijalankan:

1. Pastikan project sudah memiliki `.venv` dan model DeepFace sudah pernah
   diunduh (`arcface_weights.h5` dan `retinaface.h5`).
2. Jalankan `build_portable.ps1` di PowerShell.
3. Pindahkan seluruh folder `dist\FaceSimilarityPortable` ke PC tujuan.
4. Jalankan `Jalankan Face Similarity.bat` dari folder tersebut.

Build mengambil runtime Python dan dependency dari `.venv` project ini serta
menyalin model yang sudah ada di `%USERPROFILE%\.deepface\weights`. Karena
TensorFlow dan model wajah dibundel, ukuran folder hasil build dapat mencapai
beberapa gigabyte. Build memerlukan internet untuk mengunduh runtime Python;
penggunaan folder hasil build tidak memerlukan internet. Jangan memisahkan
file-file di dalam folder portable.

## Catatan
- Cara menjalankan dari source tetap memerlukan Python, instalasi dependency,
  dan internet untuk mengunduh model pada pemakaian pertama.
- Paket portable Windows hanya untuk Windows x64.
- Pastikan foto yang diupload jelas dan hanya menampilkan satu wajah pada tiap gambar.
