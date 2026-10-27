# SIGAP — Sistem Identifikasi Gambar untuk Analisis Pangan

SIGAP adalah aplikasi Streamlit lokal dengan **foto sebagai fitur utama** dan alat analisis stok lama sebagai fitur pendukung. Aplikasi tidak memerlukan OpenAI API, API key berbayar, maupun layanan inference eksternal.

## Tiga halaman

1. **Cek Foto** — unggah foto buah dan periksa indikasi visual Fresh/Rotten dengan model MobileNetV3-Small yang dilatih secara lokal.
2. **Analisis Stok** — gunakan model Decision Tree lama untuk satu produk. Analisis CSV banyak produk tersedia di bagian opsional, bukan alur utama.
3. **Model & Data** — lihat status kedua model, evaluasi model stok, dataset contoh, dan visualisasi Decision Tree.

## Dua model, dua tujuan

- `models/sigap_mobilenetv3_best.pt`: model visi komputer khusus buah. File ini dibuat setelah menjalankan pipeline training; tidak disertakan dalam ZIP karena model harus dihasilkan di mesin pengguna.
- `models/food_waste_model.joblib`: model Decision Tree lama untuk analisis risiko sisa stok. Model ini disertakan.
- `data/data_penjualan.csv`: dataset penjualan lama berisi 80 baris. Dataset ini disertakan sebagai sampel untuk halaman Analisis Stok.

Model foto mendeteksi pola visual pada jenis buah yang tercakup di data latihnya. Model stok mengklasifikasikan catatan inventaris, bukan kondisi visual buah. Hasil dari kedua model tidak boleh dianggap saling menggantikan.

## Menjalankan di Windows

Python 3.11 atau 3.12 direkomendasikan.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
streamlit run app.py
```

Aplikasi akan berjalan pada alamat lokal yang ditampilkan terminal, biasanya `http://localhost:8501`.

### Jika checkpoint foto sudah pernah kamu latih

Sebelum menyalin proyek ini ke folder baru, salin checkpoint dan metrik foto dari proyek SIGAP lamamu jika file-file tersebut tersedia:

```text
models/sigap_mobilenetv3_best.pt
models/metrics.json             (opsional)
models/training_history.csv     (opsional)
```

Letakkan file tersebut di folder `models/` proyek ini agar tidak perlu training ulang. ZIP ini mempertahankan model stok dan CSV lama; checkpoint foto yang tersimpan hanya di komputer pengguna tidak bisa disertakan otomatis ke ZIP.

### Jika model foto belum ada

Unduh data gambar dan latih model:

```powershell
python scripts/download_dataset.py
python scripts/train_model.py
```

Dataset gambar diunduh ke `data/raw/Fresh` dan `data/raw/Rotten`. Dataset mentah tidak disertakan dalam ZIP karena ukurannya besar. Pipeline membuat checkpoint di `models/sigap_mobilenetv3_best.pt`.

Dataset publik yang digunakan downloader: [Project-AgML/fresh_rotten_fruit_classification](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification), lisensi CC BY 4.0. Periksa kartu dataset dan pertahankan atribusi bila membagikan karya turunan.

## Analisis stok dan CSV opsional

Dataset lama `data/data_penjualan.csv` berisi kolom `stok`, `terjual`, `hari_ke`, `cuaca`, `hari_besar`, `sisa_persen`, dan `label`. Untuk input CSV, kolom minimal yang dibutuhkan:

```csv
product_name,stok,terjual,hari_ke,cuaca,hari_besar
Apel,100,65,15,0,0
Pisang,80,30,8,1,0
```

`product_name` opsional. Sistem menghitung sisa persediaan dari stok dan terjual. Kode `cuaca` menggunakan 0/1, tetapi arti kode pada sumber dataset lama tidak terdokumentasi.

Model stok memiliki 80 baris data, dan labelnya sangat erat terkait persentase sisa stok. Nilai evaluasi tinggi pada dataset kecil ini **bukan bukti akurasi di kondisi operasional nyata**. Gunakan sebagai demonstrasi pembelajaran, bukan satu-satunya dasar keputusan stok.

## Dependensi dan training

- `requirements.txt`: dependensi aplikasi.
- `requirements-train.txt`: dependensi aplikasi ditambah downloader dataset gambar.
- `src/local_vision.py` dan `src/model.py`: inferensi foto lokal dan arsitektur model gambar.
- `scripts/download_dataset.py` dan `scripts/train_model.py`: pipeline training foto.
- `src/predictor.py` dan `src/train_model.py`: prediksi inventaris lama serta pipeline training ulang model stok.
- `tests/`: tes untuk kedua alur model.

Untuk melatih ulang model inventaris dari CSV yang ada:

```powershell
python src/train_model.py
```

Untuk menjalankan seluruh tes:

```powershell
python -m pip install pytest
python -m pytest -q
```

## Batasan

- Model foto mengklasifikasikan gambar berdasarkan kelas data latih; tidak mencakup semua makanan.
- Skor model foto bukan probabilitas keamanan pangan yang terkalibrasi. Foto yang diprediksi Fresh tidak menjamin makanan aman.
- Model stok dan dataset lama hanya contoh kecil dan tidak memprediksi tanggal kedaluwarsa atau permintaan masa depan.
- Jangan mengunggah dataset gambar, virtual environment, atau file rahasia ke GitHub. Model inventaris kecil dan CSV contoh disertakan; file model foto yang dihasilkan dapat dikecualikan sesuai ukuran/lisensinya.
