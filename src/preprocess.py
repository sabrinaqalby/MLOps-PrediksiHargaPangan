import logging
from pathlib import Path
 
import pandas as pd
 
FOLDER_MENTAH = Path("data/raw")
FOLDER_BERSIH = Path("data/processed")
FILE_HASIL = FOLDER_BERSIH / "harga_bersih.csv"
 
KOLOM_KUNCI = ["tanggal", "wilayah", "komoditas"]
BATAS_INTERPOLASI = 3
BATAS_LONJAKAN = 50
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
 
 
def muat_data_mentah():
    """Menggabungkan seluruh file CSV di folder data mentah.
 
    Returns:
        pandas.DataFrame: Gabungan seluruh data mentah. DataFrame
        kosong jika tidak ada file yang ditemukan.
    """
    berkas = sorted(FOLDER_MENTAH.glob("sp2kp_*.csv"))
    if not berkas:
        logging.error("Tidak ada file CSV di %s", FOLDER_MENTAH)
        return pd.DataFrame()
 
    kumpulan = [pd.read_csv(item) for item in berkas]
    df = pd.concat(kumpulan, ignore_index=True)
    logging.info("Memuat %d file, total %d baris", len(berkas), len(df))
    return df
 
 
def samakan_tipe(df):
    """Menyeragamkan tipe data dan penulisan teks.
 
    Args:
        df (pandas.DataFrame): Data mentah gabungan.
 
    Returns:
        pandas.DataFrame: Data dengan tipe kolom yang seragam.
    """
    df = df.copy()
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")
 
    for kolom in ["wilayah", "komoditas", "satuan"]:
        df[kolom] = df[kolom].astype(str).str.strip()
 
    return df
 
 
def tandai_harga_kosong(df):
    """Mengubah harga nol menjadi nilai kosong.
 
    API SP2KP mengembalikan harga 0 pada hari tanpa survei pasar,
    misalnya akhir pekan dan hari libur. Nilai itu bukan harga
    sebenarnya, sehingga diperlakukan sebagai data kosong agar bisa
    diisi pada tahap berikutnya.
 
    Args:
        df (pandas.DataFrame): Data yang sudah diseragamkan tipenya.
 
    Returns:
        pandas.DataFrame: Data dengan harga nol menjadi NaN.
    """
    df = df.copy()
    jumlah = int((df["harga"] <= 0).sum())
    df.loc[df["harga"] <= 0, "harga"] = None
 
    if jumlah:
        logging.info("%d harga bernilai nol ditandai kosong", jumlah)
    return df
 
 
def buang_baris_tanpa_kunci(df):
    """Membuang baris yang kehilangan kolom kunci.
 
    Args:
        df (pandas.DataFrame): Data hasil penandaan harga kosong.
 
    Returns:
        pandas.DataFrame: Data yang kolom kuncinya lengkap.
    """
    jumlah_awal = len(df)
    df = df.dropna(subset=KOLOM_KUNCI)
 
    selisih = jumlah_awal - len(df)
    if selisih:
        logging.info("Membuang %d baris tanpa kolom kunci", selisih)
    return df
 
 
def buang_duplikat(df):
    """Menyisakan satu baris terbaru untuk tiap kombinasi kunci.
 
    Menjalankan ingestion beberapa kali pada tanggal yang sama
    menghasilkan baris berulang. Baris dengan waktu pengambilan
    paling akhir yang dipertahankan.
 
    Args:
        df (pandas.DataFrame): Data yang kolom kuncinya lengkap.
 
    Returns:
        pandas.DataFrame: Data tanpa duplikat.
    """
    jumlah_awal = len(df)
    df = df.sort_values("accessed_at")
    df = df.drop_duplicates(subset=KOLOM_KUNCI, keep="last")
 
    selisih = jumlah_awal - len(df)
    if selisih:
        logging.info("Menggabungkan %d baris duplikat", selisih)
    return df
 
 
def isi_harga_kosong(df):
    """Mengisi harga kosong dengan interpolasi antar tanggal.
 
    Pengisian dilakukan per kombinasi wilayah dan komoditas, dibatasi
    maksimal tiga hari berturut-turut. Baris hasil pengisian ditandai
    pada kolom hasil_interpolasi agar tetap dapat dibedakan dari
    harga hasil survei.
 
    Args:
        df (pandas.DataFrame): Data tanpa duplikat.
 
    Returns:
        pandas.DataFrame: Data dengan harga kosong yang sudah diisi.
    """
    df = df.sort_values(["wilayah", "komoditas", "tanggal"]).copy()
    df["hasil_interpolasi"] = df["harga"].isna()
 
    df["harga"] = df.groupby(["wilayah", "komoditas"])["harga"].transform(
        lambda seri: seri.interpolate(limit=BATAS_INTERPOLASI)
    )
 
    sisa = int(df["harga"].isna().sum())
    terisi = int(df["hasil_interpolasi"].sum()) - sisa
    logging.info("%d harga kosong diisi lewat interpolasi", terisi)
 
    if sisa:
        logging.warning("%d baris tetap kosong dan dibuang", sisa)
        df = df[df["harga"].notna()]
 
    df["harga"] = df["harga"].round().astype(int)
    return df
 
 
def tandai_lonjakan(df):
    """Menandai perubahan harga yang tidak wajar antar hari.
 
    Perubahan harga lebih dari 50 persen dibanding hari sebelumnya
    ditandai untuk diperiksa, bukan dihapus, karena lonjakan seperti
    ini bisa benar-benar terjadi pada komoditas seperti cabai.
 
    Args:
        df (pandas.DataFrame): Data yang harganya sudah lengkap.
 
    Returns:
        pandas.DataFrame: Data dengan kolom perubahan dan penanda.
    """
    df = df.sort_values(["wilayah", "komoditas", "tanggal"]).copy()
    kelompok = df.groupby(["wilayah", "komoditas"])["harga"]
 
    df["perubahan_persen"] = (kelompok.pct_change() * 100).round(2)
    df["lonjakan_ekstrem"] = df["perubahan_persen"].abs() > BATAS_LONJAKAN
 
    jumlah = int(df["lonjakan_ekstrem"].sum())
    if jumlah:
        logging.warning("%d baris ditandai lonjakan ekstrem", jumlah)
    return df
 
 
def simpan(df):
    """Menyimpan data bersih ke folder data/processed.
 
    Args:
        df (pandas.DataFrame): Data yang sudah dibersihkan.
    """
    FOLDER_BERSIH.mkdir(parents=True, exist_ok=True)
    kolom = KOLOM_KUNCI + [
        "satuan",
        "harga",
        "hasil_interpolasi",
        "perubahan_persen",
        "lonjakan_ekstrem",
        "sumber",
    ]
    hasil = df[kolom].sort_values(KOLOM_KUNCI)
    hasil.to_csv(FILE_HASIL, index=False)
    logging.info(
        "Data bersih disimpan: %s (%d baris)", FILE_HASIL, len(hasil)
    )
 
 
def main():
    """Menjalankan seluruh tahap prapemrosesan."""
    df = muat_data_mentah()
    if df.empty:
        return
 
    df = samakan_tipe(df)
    df = tandai_harga_kosong(df)
    df = buang_baris_tanpa_kunci(df)
    df = buang_duplikat(df)
    df = isi_harga_kosong(df)
    df = tandai_lonjakan(df)
    simpan(df)
 
if __name__ == "__main__":
    main()