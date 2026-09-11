# MLOps-PrediksiHargaPangan

Sistem prediksi dan peringatan dini disparitas harga pangan antar daerah di Indonesia.

Repositori ini merupakan proyek mata kuliah Machine Learning Operations (MLOps), Fakultas Ilmu Komputer, Universitas Brawijaya. Proyek dikembangkan secara bertahap melalui rangkaian Lembar Kerja (LK-01 sampai LK-14).

## Latar Belakang

Harga bahan pangan pokok berubah setiap hari dan berbeda-beda antar daerah. Sistem pemantauan yang ada saat ini bersifat pelaporan, bukan peramalan — memberi tahu harga hari ini, tetapi tidak memperkirakan harga minggu depan. Akibatnya intervensi pemerintah cenderung reaktif, dilakukan setelah harga terlanjur melonjak.

Proyek ini membangun sistem machine learning end-to-end yang memprediksi harga komoditas pangan per wilayah dan memberi peringatan dini ketika lonjakan harga diperkirakan terjadi.

## Sumber Data

| Sumber | Peran | Keterangan |
|---|---|---|
| SP2KP Kemendag | Sumber utama | Harga bahan pokok harian per provinsi |
| PIHPS Bank Indonesia | Sumber historis | Arsip harga untuk data latih awal |
| Open-Meteo | Sumber pendukung | Data cuaca sebagai variabel penjelas |

Data bersifat dinamis dan diperbarui setiap hari, sehingga menuntut mekanisme pengambilan berkala dan pelatihan ulang model secara berkelanjutan.

## Struktur Direktori

```
.
├── .devcontainer/    Konfigurasi lingkungan pengembangan Codespaces
├── config/           Berkas konfigurasi proyek
├── data/
│   ├── raw/          Data mentah hasil pengambilan (tidak di-commit)
│   └── processed/    Data hasil pengolahan (tidak di-commit)
├── docs/             Dokumentasi proyek
├── models/           Artefak model terlatih (tidak di-commit)
├── notebooks/        Notebook eksplorasi dan analisis
├── src/              Kode sumber
└── tests/            Pengujian
```

Isi folder `data/` dan `models/` sengaja diabaikan oleh Git melalui `.gitignore`. Git dirancang untuk mengelola kode, bukan berkas data berukuran besar. Pengelolaan versi data akan ditangani menggunakan DVC pada tahap LK-05. Berkas `.gitkeep` dipertahankan agar struktur folder tetap terbaca meskipun isinya kosong.

## Menjalankan Proyek

### Melalui GitHub Codespaces

1. Klik tombol **Code** pada halaman repositori
2. Pilih tab **Codespaces**, lalu **Create codespace on main**
3. Tunggu proses pembangunan selesai — dependencies terpasang otomatis melalui `devcontainer.json`

### Melalui lingkungan lokal

```bash
git clone https://github.com/sabrinaqalby/MLOps-PrediksiHargaPangan.git
cd MLOps-PrediksiHargaPangan
pip install -r requirements.txt
```

### Verifikasi lingkungan

```bash
python src/hello.py
```

Keluaran yang diharapkan: `Hello MLOps`

## Tumpukan Teknologi

GitHub dan GitHub Codespaces untuk version control dan lingkungan pengembangan. Python, Pandas, dan Scikit-learn untuk pengolahan data dan pemodelan. DVC untuk pemberian versi data, MLflow untuk pelacakan eksperimen, Docker dan GitHub Actions untuk kontainerisasi dan otomasi, serta Prometheus dan Grafana untuk pemantauan.

## Lisensi

MIT License — lihat berkas [LICENSE](LICENSE).
