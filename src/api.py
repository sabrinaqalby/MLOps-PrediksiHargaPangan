import logging
import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from predict import bentuk_fitur_inferensi
from train import FITUR, HORIZON, bentuk_tabel_lebar, muat_data

URI_TRACKING = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
)
URI_MODEL = os.environ.get(
    "MODEL_URI", "models:/model-harga-pangan-malang@production"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

mlflow.set_tracking_uri(URI_TRACKING)

app = FastAPI(
    title="API Prediksi Harga Pangan Kota Malang",
    description=(
        "Memprediksi harga komoditas pangan tujuh hari ke depan "
        "memakai model yang diambil dari MLflow Model Registry."
    ),
    version="1.0",
)

_model = None
_keterangan = {}


class BarisFitur(BaseModel):

    lag_1: float
    lag_3: float
    lag_7: float
    rata_7: float
    simpangan_7: float
    harga_nasional: float
    selisih_nasional: float
    hari_dalam_minggu: int
    kode_komoditas: int


class Permintaan(BaseModel):

    data: list[BarisFitur]


def pecah_uri_model(uri):
    sisa = uri.replace("models:/", "", 1)

    if "@" in sisa:
        nama, alias = sisa.split("@", 1)
        return nama, alias, None

    if "/" in sisa:
        nama, versi = sisa.rsplit("/", 1)
        return nama, None, versi

    return sisa, None, None


def ambil_keterangan_model():
    nama, alias, versi = pecah_uri_model(URI_MODEL)
    klien = MlflowClient()

    if alias:
        objek = klien.get_model_version_by_alias(nama, alias)
    else:
        objek = klien.get_model_version(nama, versi)

    return {
        "nama": nama,
        "versi": int(objek.version),
        "alias": list(objek.aliases),
        "tahap": objek.tags.get("tahap", "None"),
        "run_id": objek.run_id,
        "uri": URI_MODEL,
    }


def muat_model():
    global _model, _keterangan

    if _model is not None:
        return _model, _keterangan

    logging.info("Memuat model %s dari %s", URI_MODEL, URI_TRACKING)
    try:
        _keterangan = ambil_keterangan_model()
        _model = mlflow.pyfunc.load_model(URI_MODEL)
    except Exception as galat:  # noqa: BLE001
        logging.error("Gagal memuat model: %s", galat)
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum tersedia di registry. Daftarkan model "
                "lebih dulu lalu beri alias sesuai MODEL_URI."
            ),
        ) from galat

    logging.info("Model versi %s siap melayani", _keterangan["versi"])
    return _model, _keterangan


def susun_tabel_hasil(terkini, prediksi):
    hasil = []

    for (_, baris), nilai in zip(terkini.iterrows(), prediksi):
        terakhir = int(baris["harga_malang"])
        ramalan = int(round(nilai))
        hasil.append(
            {
                "komoditas": baris["komoditas"],
                "tanggal_acuan": str(baris["tanggal"].date()),
                "harga_terakhir": terakhir,
                "tanggal_prediksi": str(
                    (baris["tanggal"] + pd.Timedelta(days=HORIZON)).date()
                ),
                "harga_prediksi": ramalan,
                "perubahan_persen": round(
                    (ramalan - terakhir) / terakhir * 100, 2
                ),
            }
        )

    return hasil


@app.get("/")
def beranda():
    return {
        "layanan": "API Prediksi Harga Pangan Kota Malang",
        "tracking_server": URI_TRACKING,
        "model": URI_MODEL,
        "endpoint": {
            "GET /health": "status layanan",
            "GET /model": "keterangan versi model yang dipakai",
            "GET /prediksi": "prediksi dari data terbaru tiap komoditas",
            "POST /prediksi": "prediksi dari fitur yang dikirim sendiri",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracking_server": URI_TRACKING,
        "model_uri": URI_MODEL,
        "model_dimuat": _model is not None,
    }


@app.get("/model")
def keterangan_model():
    _, keterangan = muat_model()
    return keterangan


@app.get("/prediksi")
def prediksi_terbaru(komoditas: str | None = None):
    model, keterangan = muat_model()

    df = muat_data()
    if df.empty:
        raise HTTPException(
            status_code=503, detail="Data harga belum tersedia."
        )

    terkini = bentuk_fitur_inferensi(bentuk_tabel_lebar(df))
    if komoditas:
        terkini = terkini[terkini["komoditas"] == komoditas]

    if terkini.empty:
        raise HTTPException(
            status_code=404, detail="Komoditas tidak ditemukan pada data."
        )

    keluaran = model.predict(terkini[FITUR].astype(float))

    return {
        "model_versi": keterangan["versi"],
        "horizon_hari": HORIZON,
        "jumlah": len(terkini),
        "hasil": susun_tabel_hasil(terkini, keluaran),
    }


@app.post("/prediksi")
def prediksi_dari_masukan(permintaan: Permintaan):
    model, keterangan = muat_model()

    if not permintaan.data:
        raise HTTPException(status_code=400, detail="Data tidak boleh kosong.")

    masukan = pd.DataFrame(
        [baris.model_dump() for baris in permintaan.data]
    )[FITUR].astype(float)

    keluaran = model.predict(masukan)

    return {
        "model_versi": keterangan["versi"],
        "horizon_hari": HORIZON,
        "harga_prediksi": [int(round(nilai)) for nilai in keluaran],
    }