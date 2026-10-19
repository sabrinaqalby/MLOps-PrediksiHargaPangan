import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE_DATA = Path("data/processed/harga_bersih.csv")
FILE_VERSI_DATA = Path("data/raw.dvc")
NAMA_EKSPERIMEN = "prediksi-harga-pangan-malang"
NAMA_MODEL_REGISTRY = "model-harga-pangan-malang"

HORIZON = 7
RASIO_UJI = 0.2
SEED = 42

FITUR = [
    "lag_1",
    "lag_3",
    "lag_7",
    "rata_7",
    "simpangan_7",
    "harga_nasional",
    "selisih_nasional",
    "hari_dalam_minggu",
    "kode_komoditas",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def baca_versi_data():
    """Ambil hash versi dataset dari berkas penunjuk DVC."""

    if not FILE_VERSI_DATA.exists():
        return "tidak-diketahui"

    for baris in FILE_VERSI_DATA.read_text().splitlines():
        if "md5:" in baris:
            return baris.split("md5:")[1].strip()
    return "tidak-diketahui"


def muat_data():
    """Baca hasil prapemrosesan dari data/processed."""

    if not FILE_DATA.exists():
        logging.error(
            "Berkas %s tidak ditemukan. Jalankan preprocess.py", FILE_DATA
        )
        return pd.DataFrame()

    df = pd.read_csv(FILE_DATA)
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    logging.info("Memuat %d baris dari %s", len(df), FILE_DATA)
    return df


def bentuk_tabel_lebar(df):
    """Ubah data menjadi satu baris per tanggal dan komoditas."""
    
    lebar = df.pivot_table(
        index=["tanggal", "komoditas"],
        columns="wilayah",
        values="harga",
    ).reset_index()
    lebar.columns.name = None

    lebar = lebar.rename(
        columns={"Kota Malang": "harga_malang", "Nasional": "harga_nasional"}
    )
    lebar = lebar.dropna(subset=["harga_malang", "harga_nasional"])
    return lebar.sort_values(["komoditas", "tanggal"]).reset_index(drop=True)


def bentuk_fitur(lebar):
    """Susun fitur dan target untuk pelatihan."""

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

    df["target"] = df.groupby("komoditas")["harga_malang"].shift(-HORIZON)

    siap = df.dropna(subset=FITUR + ["target"]).reset_index(drop=True)
    logging.info(
        "Terbentuk %d baris siap latih dari %d baris", len(siap), len(df)
    )
    return siap


def bagi_data(df):
    """Pisahkan data latih dan data uji berdasarkan urutan waktu."""

    df = df.sort_values("tanggal").reset_index(drop=True)
    batas = int(len(df) * (1 - RASIO_UJI))

    latih, uji = df.iloc[:batas], df.iloc[batas:]
    logging.info(
        "Data latih %d baris (s.d. %s), data uji %d baris (sejak %s)",
        len(latih),
        latih["tanggal"].max().date(),
        len(uji),
        uji["tanggal"].min().date(),
    )
    return (
        latih[FITUR],
        uji[FITUR],
        latih["target"],
        uji["target"],
        uji,
    )


def hitung_metrik(y_benar, y_prediksi):
    """Hitung metrik evaluasi regresi."""

    y_benar = np.asarray(y_benar, dtype=float)
    y_prediksi = np.asarray(y_prediksi, dtype=float)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_benar, y_prediksi))),
        "mae": float(mean_absolute_error(y_benar, y_prediksi)),
        "mape": float(np.mean(np.abs((y_benar - y_prediksi) / y_benar)) * 100),
        "r2": float(r2_score(y_benar, y_prediksi)),
    }


def simpan_model(model, contoh_input, daftarkan):
    """Catat artefak model ke MLflow."""

    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=contoh_input,
        registered_model_name=NAMA_MODEL_REGISTRY if daftarkan else None,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    if daftarkan:
        logging.info("Model didaftarkan sebagai '%s'", NAMA_MODEL_REGISTRY)


def baca_argumen():
    """Baca parameter eksperimen dari baris perintah."""

    parser = argparse.ArgumentParser(
        description="Melatih model prediksi harga pangan dan catat ke MLflow."
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument(
        "--nama-run",
        default=None,
        help="Nama run di MLflow. Format: [model]_[versi]_[tanggal].",
    )
    parser.add_argument(
        "--daftarkan",
        action="store_true",
        help="Daftarkan model ke MLflow Model Registry.",
    )
    return parser.parse_args()


def main():
    """Jalankan satu eksperimen pelatihan dan catat hasilnya ke MLflow."""

    argumen = baca_argumen()

    df = muat_data()
    if df.empty:
        return

    siap = bentuk_fitur(bentuk_tabel_lebar(df))
    if len(siap) < 30:
        logging.error(
            "Data terlalu sedikit (%d baris) untuk dilatih", len(siap)
        )
        return

    X_latih, X_uji, y_latih, y_uji, data_uji = bagi_data(siap)

    versi_data = baca_versi_data()
    nama_run = argumen.nama_run or "rf_{}_{}".format(
        argumen.n_estimators, pd.Timestamp.today().strftime("%Y%m%d")
    )

    mlflow.set_experiment(NAMA_EKSPERIMEN)

    with mlflow.start_run(
        run_name=nama_run,
        tags={
            "task": "regression",
            "dataset_version": versi_data,
            "wilayah": "Kota Malang",
        },
    ):
        model = RandomForestRegressor(
            n_estimators=argumen.n_estimators,
            max_depth=argumen.max_depth,
            min_samples_leaf=argumen.min_samples_leaf,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_latih, y_latih)
        prediksi = model.predict(X_uji)

        metrik = hitung_metrik(y_uji, prediksi)

        # Pembanding naif: harga tujuh hari ke depan dianggap sama dengan
        # harga hari ini. Model baru berguna bila mengungguli angka ini.
        metrik_naif = hitung_metrik(y_uji, data_uji["harga_malang"])
        metrik["rmse_naif"] = metrik_naif["rmse"]
        metrik["perbaikan_vs_naif_persen"] = (
            (metrik_naif["rmse"] - metrik["rmse"]) / metrik_naif["rmse"] * 100
        )

        mlflow.log_param("algoritma", "RandomForestRegressor")
        mlflow.log_param("n_estimators", argumen.n_estimators)
        mlflow.log_param("max_depth", argumen.max_depth)
        mlflow.log_param("min_samples_leaf", argumen.min_samples_leaf)
        mlflow.log_param("horizon_hari", HORIZON)
        mlflow.log_param("jumlah_fitur", len(FITUR))
        mlflow.log_param("baris_latih", len(X_latih))
        mlflow.log_param("baris_uji", len(X_uji))
        mlflow.log_param("versi_data_dvc", versi_data)

        for nama, nilai in metrik.items():
            mlflow.log_metric(nama, nilai)

        simpan_model(model, X_uji.head().astype(float), argumen.daftarkan)

        logging.info(
            "RMSE %.1f | MAE %.1f | MAPE %.2f%% | R2 %.3f"
            " | unggul %.1f%% dari naif",
            metrik["rmse"],
            metrik["mae"],
            metrik["mape"],
            metrik["r2"],
            metrik["perbaikan_vs_naif_persen"],
        )

if __name__ == "__main__":
    main()