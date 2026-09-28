import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = (
    "https://api-sp2kp.kemendag.go.id/report/api/"
    "average-price/generate-perbandingan-harga"
)

# Komoditas yang dipantau beserta variant_id dari API SP2KP.
KOMODITAS = {
    52: "Beras Medium",
    25: "Telur Ayam Ras",
    2: "Cabai Merah Keriting",
    10: "Cabai Rawit Merah",
    13: "Bawang Merah",
}

# Wilayah yang dibandingkan: Kota Malang dan rata-rata nasional.
WILAYAH = {
    "Kota Malang": {"kode_provinsi": "35", "kode_kab_kota": "3573"},
    "Nasional": {},
}

FOLDER_OUTPUT = Path("data/raw")
JUMLAH_PERCOBAAN = 3
JEDA_PERCOBAAN = 5
JEDA_ANTAR_REQUEST = 1
BATAS_WAKTU = 30
WIB = timezone(timedelta(hours=7))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def ambil_data(tanggal, pembanding, nama_wilayah):
    """Memanggil API SP2KP untuk satu wilayah pada satu tanggal.

    Args:
        tanggal (str): Tanggal harga, format YYYY-MM-DD.
        pembanding (str): Tanggal pembanding, format YYYY-MM-DD.
        nama_wilayah (str): Kunci pada dict WILAYAH.

    Returns:
        tuple: (list data mentah, status code HTTP) atau (None, None)
        jika seluruh percobaan gagal.
    """
    payload = {
        "tanggal": tanggal,
        "tanggal_pembanding": pembanding,
        **WILAYAH[nama_wilayah],
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://sp2kp.kemendag.go.id/",
    }

    for percobaan in range(1, JUMLAH_PERCOBAAN + 1):
        try:
            respons = requests.post(
                API_URL,
                data=payload,
                headers=headers,
                timeout=BATAS_WAKTU,
            )
            respons.raise_for_status()
            isi = respons.json()

            if isi.get("status") != "success":
                raise ValueError(f"API menolak: {isi.get('message')}")

            logging.info(
                "Berhasil ambil %s tanggal %s (%d baris)",
                nama_wilayah,
                tanggal,
                len(isi.get("data", [])),
            )
            return isi.get("data", []), respons.status_code

        except (requests.RequestException, ValueError) as galat:
            logging.warning(
                "Percobaan %d/%d gagal untuk %s: %s",
                percobaan,
                JUMLAH_PERCOBAAN,
                nama_wilayah,
                galat,
            )
            if percobaan < JUMLAH_PERCOBAAN:
                time.sleep(JEDA_PERCOBAAN)

    logging.error("Gagal mengambil data %s tanggal %s", nama_wilayah, tanggal)
    return None, None


def susun_baris(data_mentah, nama_wilayah, tanggal, status_http):
    """Menyaring komoditas yang dipantau dan menyusun baris data.

    Args:
        data_mentah (list): Daftar komoditas dari respons API.
        nama_wilayah (str): Nama wilayah sumber data.
        tanggal (str): Tanggal harga.
        status_http (int): Status code HTTP saat pengambilan.

    Returns:
        list: Daftar dict siap dijadikan DataFrame.
    """
    waktu_ambil = datetime.now(WIB).isoformat(timespec="seconds")
    baris = []

    for item in data_mentah:
        variant_id = item.get("variant_id")
        if variant_id not in KOMODITAS:
            continue

        baris.append(
            {
                "tanggal": item.get("tanggal", tanggal),
                "wilayah": nama_wilayah,
                "komoditas": KOMODITAS[variant_id],
                "satuan": item.get("satuan_display"),
                "harga": item.get("harga"),
                "sumber": "sp2kp",
                "source_url": API_URL,
                "accessed_at": waktu_ambil,
                "http_status": status_http,
            }
        )

    return baris


def simpan(df, tanggal):
    """Menyimpan DataFrame ke CSV bertimestamp beserta metadatanya.

    Args:
        df (pandas.DataFrame): Data yang akan disimpan.
        tanggal (str): Tanggal harga, dipakai pada nama file.

    Returns:
        pathlib.Path: Lokasi file CSV yang dibuat.
    """
    FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    stempel = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    nama_file = f"sp2kp_{tanggal.replace('-', '')}_{stempel}.csv"
    lokasi = FOLDER_OUTPUT / nama_file

    df.to_csv(lokasi, index=False)

    metadata = {
        "source_url": API_URL,
        "accessed_at": datetime.now(WIB).isoformat(timespec="seconds"),
        "tanggal_data": tanggal,
        "record_count": len(df),
        "file": nama_file,
    }
    lokasi.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logging.info("Data disimpan: %s (%d baris)", lokasi, len(df))
    return lokasi


def baca_argumen():
    """Membaca argumen baris perintah.

    Returns:
        argparse.Namespace: Argumen tanggal dan tanggal pembanding.
    """
    hari_ini = datetime.now(WIB).date()
    parser = argparse.ArgumentParser(
        description="Mengambil data harga pangan dari API SP2KP."
    )
    parser.add_argument(
        "--tanggal",
        default=hari_ini.isoformat(),
        help="Tanggal harga (YYYY-MM-DD). Default: hari ini.",
    )
    parser.add_argument(
        "--pembanding",
        default=(hari_ini - timedelta(days=1)).isoformat(),
        help="Tanggal pembanding (YYYY-MM-DD). Default: kemarin.",
    )
    return parser.parse_args()


def main():
    """Menjalankan proses pengambilan data untuk seluruh wilayah."""
    argumen = baca_argumen()
    semua_baris = []

    for nama_wilayah in WILAYAH:
        data_mentah, status_http = ambil_data(
            argumen.tanggal, argumen.pembanding, nama_wilayah
        )
        if data_mentah:
            semua_baris.extend(
                susun_baris(
                    data_mentah, nama_wilayah, argumen.tanggal, status_http
                )
            )
        time.sleep(JEDA_ANTAR_REQUEST)

    if not semua_baris:
        logging.error("Tidak ada data yang berhasil diambil.")
        return

    simpan(pd.DataFrame(semua_baris), argumen.tanggal)

if __name__ == "__main__":
    main()