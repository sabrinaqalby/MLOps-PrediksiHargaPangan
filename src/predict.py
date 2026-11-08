import argparse
import logging
import os

import mlflow
import mlflow.pyfunc
import pandas as pd

from train import FITUR, HORIZON, bentuk_tabel_lebar, muat_data

URI_TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
NAMA_MODEL = "model-harga-pangan-malang"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def susun_uri(alias, versi):
    if versi:
        return "models:/{}/{}".format(NAMA_MODEL, versi)
    return "models:/{}@{}".format(NAMA_MODEL, alias)


def bentuk_fitur_inferensi(lebar):
    df = lebar.copy()
    kelompok = df.groupby("komoditas")["harga_malang"]

    df["lag_1"] = kelompok.shift(1)
    df["lag_3"] = kelompok.shift(3)
    df["lag_7"] = kelompok.shift(7)
    df["rata_7"] = kelompok.transform(lambda s: s.rolling(7).mean())
    df["simpangan_7"] = kelompok.transform(lambda s: s.rolling(7).std())

    selisih = df["harga_malang"] - df["harga_nasional"]
    df["selisih_nasional"] = selisih / df["harga_nasional"] * 100
    df["hari_dalam_minggu"] = df["tanggal"].dt.dayofweek
    df["kode_komoditas"] = df["komoditas"].astype("category").cat.codes

    siap = df.dropna(subset=FITUR).sort_values("tanggal")
    return siap.groupby("komoditas").tail(1).reset_index(drop=True)


def susun_hasil(terkini, prediksi):
    hasil = pd.DataFrame(
        {
            "komoditas": terkini["komoditas"],
            "tanggal_acuan": terkini["tanggal"].dt.date,
            "harga_terakhir": terkini["harga_malang"].astype(int),
            "tanggal_prediksi": (
                terkini["tanggal"] + pd.Timedelta(days=HORIZON)
            ).dt.date,
            "harga_prediksi": [round(nilai) for nilai in prediksi],
        }
    )

    selisih = hasil["harga_prediksi"] - hasil["harga_terakhir"]
    hasil["perubahan_persen"] = (
        selisih / hasil["harga_terakhir"] * 100
    ).round(2)
    return hasil


def baca_argumen():
    parser = argparse.ArgumentParser(
        description="Memuat model dari MLflow Registry dan memprediksi harga."
    )
    parser.add_argument(
        "--alias",
        default="production",
        help="Alias model di registry. Default: production.",
    )
    parser.add_argument(
        "--versi",
        type=int,
        help="Muat nomor versi tertentu, mengabaikan alias.",
    )
    parser.add_argument(
        "--komoditas",
        help="Tampilkan satu komoditas saja.",
    )
    return parser.parse_args()


def main():
    argumen = baca_argumen()
    mlflow.set_tracking_uri(URI_TRACKING)

    uri = susun_uri(argumen.alias, argumen.versi)
    logging.info("Memuat model dari %s", uri)

    try:
        model = mlflow.pyfunc.load_model(uri)
    except mlflow.exceptions.MlflowException as galat:
        logging.error("Gagal memuat model: %s", galat)
        return

    logging.info("Model berhasil dimuat")

    df = muat_data()
    if df.empty:
        return

    terkini = bentuk_fitur_inferensi(bentuk_tabel_lebar(df))
    if argumen.komoditas:
        terkini = terkini[terkini["komoditas"] == argumen.komoditas]

    if terkini.empty:
        logging.error("Tidak ada data yang bisa diprediksi")
        return

    prediksi = model.predict(terkini[FITUR].astype(float))
    hasil = susun_hasil(terkini, prediksi)

    print("\nPrediksi harga {} hari ke depan (Kota Malang)".format(HORIZON))
    print(hasil.to_string(index=False))
    print()


if __name__ == "__main__":
    main()