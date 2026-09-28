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
    """Gabungkan semua CSV di data/raw/ jadi satu DataFrame."""

    berkas = sorted(FOLDER_MENTAH.glob("sp2kp_*.csv"))
    if not berkas:
        logging.error("Tidak ada file CSV di %s", FOLDER_MENTAH)
        return pd.DataFrame()

    kumpulan = [pd.read_csv(item) for item in berkas]
    df = pd.concat(kumpulan, ignore_index=True)
    logging.info("Memuat %d file, total %d baris", len(berkas), len(df))
    return df


def samakan_tipe(df):
    """Ubah tanggal jadi datetime, harga jadi angka, teks dirapikan. Perlu dilakukan karena hasil baca CSV semuanya jadi teks."""

    df = df.copy()
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

    for kolom in ["wilayah", "komoditas", "satuan"]:
        df[kolom] = df[kolom].astype(str).str.strip()

    return df


def tandai_harga_kosong(df):
    """Ubah harga 0 jadi kosong.
    API mengembalikan 0 kalau hari itu tidak ada survei pasar,
    biasanya akhir pekan. Itu bukan harga sebenarnya, jadi
    diperlakukan sebagai data kosong untuk diisi nanti.
    """

    df = df.copy()
    jumlah = int((df["harga"] <= 0).sum())
    df.loc[df["harga"] <= 0, "harga"] = None

    if jumlah:
        logging.info("%d harga bernilai nol ditandai kosong", jumlah)
    return df


def buang_baris_tanpa_kunci(df):
    """Buang baris yang tanggal, wilayah, atau komoditasnya kosong."""

    jumlah_awal = len(df)
    df = df.dropna(subset=KOLOM_KUNCI)

    selisih = jumlah_awal - len(df)
    if selisih:
        logging.info("Membuang %d baris tanpa kolom kunci", selisih)
    return df


def buang_duplikat(df):
    """Sisakan satu baris per tanggal-wilayah-komoditas.
    Kalau ingestion dijalankan dua kali di tanggal yang sama, akan
    ada baris kembar dan yang akan diambil ialah versi paling akhir.
    """

    jumlah_awal = len(df)
    df = df.sort_values("accessed_at")
    df = df.drop_duplicates(subset=KOLOM_KUNCI, keep="last")

    selisih = jumlah_awal - len(df)
    if selisih:
        logging.info("Menggabungkan %d baris duplikat", selisih)
    return df


def isi_harga_kosong(df):
    """Isi harga kosong dengan interpolasi antar tanggal.
    Dihitung per wilayah dan komoditas, maksimal tiga hari
    berturut-turut. Kolom hasil_interpolasi menandai mana harga
    isian dan mana harga asli dari survei.
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
    """Tandai perubahan harga di atas 50 persen.
    Cuma ditandai, tidak dihapus, karena harga cabai memang bisa
    melonjak sebesar itu dalam sehari.
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
    """Tulis hasil akhir ke data/processed/harga_bersih.csv."""
    
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
    """Jalankan semua tahap pembersihan secara berurutan."""

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