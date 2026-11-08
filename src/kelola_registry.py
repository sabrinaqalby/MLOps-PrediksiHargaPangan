import argparse
import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

URI_TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
NAMA_MODEL = "model-harga-pangan-malang"

TAHAP_ALIAS = {"Staging": "staging", "Production": "production"}

DESKRIPSI_MODEL = (
    "Prediksi harga lima komoditas pangan di Kota Malang untuk tujuh hari "
    "ke depan. Masukan model berupa riwayat harga lokal, harga nasional, "
    "selisih keduanya, dan penanda hari. Data berasal dari API SP2KP "
    "Kemendag dan versinya dilacak dengan DVC. "
    "Alias 'production' menandai versi yang dipakai layanan inferensi, "
    "alias 'staging' menandai versi yang masih diuji."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def buat_klien():
    mlflow.set_tracking_uri(URI_TRACKING)
    return MlflowClient()


def ambil_versi(klien, nomor):
    try:
        versi = klien.get_model_version(NAMA_MODEL, str(nomor))
    except mlflow.exceptions.MlflowException as galat:
        logging.error("Versi %s tidak ditemukan: %s", nomor, galat)
        return None, None

    return versi, klien.get_run(versi.run_id)


def susun_deskripsi(versi, run):
    par = run.data.params
    met = run.data.metrics

    return "\n".join(
        [
            "Versi {} dari run '{}'.".format(
                versi.version, run.info.run_name
            ),
            "",
            "Algoritma      : {}".format(par.get("algoritma", "-")),
            "Hyperparameter : n_estimators={}, max_depth={}, "
            "min_samples_leaf={}".format(
                par.get("n_estimators", "-"),
                par.get("max_depth", "-"),
                par.get("min_samples_leaf", "-"),
            ),
            "Horizon        : {} hari".format(par.get("horizon_hari", "-")),
            "Fitur          : {} kolom".format(par.get("jumlah_fitur", "-")),
            "Data latih     : {} baris".format(par.get("baris_latih", "-")),
            "Data uji       : {} baris".format(par.get("baris_uji", "-")),
            "Versi data DVC : {}".format(par.get("versi_data_dvc", "-")),
            "",
            "RMSE {:.1f} | MAE {:.1f} | MAPE {:.2f}% | R2 {:.3f}".format(
                met.get("rmse", 0),
                met.get("mae", 0),
                met.get("mape", 0),
                met.get("r2", 0),
            ),
            "Pembanding naif: RMSE {:.1f} (selisih {:+.1f}%)".format(
                met.get("rmse_naif", 0),
                met.get("perbaikan_vs_naif_persen", 0),
            ),
        ]
    )


def isi_metadata(klien, versi, run):
    klien.update_registered_model(NAMA_MODEL, description=DESKRIPSI_MODEL)
    klien.update_model_version(
        NAMA_MODEL, versi.version, description=susun_deskripsi(versi, run)
    )

    par = run.data.params
    met = run.data.metrics
    tag = {
        "algoritma": par.get("algoritma", "-"),
        "n_estimators": par.get("n_estimators", "-"),
        "max_depth": str(par.get("max_depth", "-")),
        "min_samples_leaf": par.get("min_samples_leaf", "-"),
        "horizon_hari": par.get("horizon_hari", "-"),
        "versi_data_dvc": par.get("versi_data_dvc", "-"),
        "rmse": "{:.1f}".format(met.get("rmse", 0)),
        "r2": "{:.3f}".format(met.get("r2", 0)),
        "wilayah": "Kota Malang",
        "sumber_data": "SP2KP Kemendag",
    }

    for kunci, nilai in tag.items():
        klien.set_model_version_tag(NAMA_MODEL, versi.version, kunci, nilai)

    logging.info("Metadata versi %s diperbarui", versi.version)


def coba_tahap_lama(klien, nomor, tahap):
    if not hasattr(klien, "transition_model_version_stage"):
        logging.info("MLflow versi ini memakai alias, bukan stage")
        return

    try:
        klien.transition_model_version_stage(NAMA_MODEL, str(nomor), tahap)
        logging.info("Stage lama diisi: %s", tahap)
    except Exception as galat:  # noqa: BLE001
        logging.info("Stage lama tidak dipakai (%s)", type(galat).__name__)


def atur_tahap(klien, nomor, tahap):
    alias = TAHAP_ALIAS[tahap]

    klien.set_model_version_tag(NAMA_MODEL, str(nomor), "tahap", tahap)
    klien.set_registered_model_alias(NAMA_MODEL, alias, str(nomor))
    logging.info("Versi %s diberi alias '%s'", nomor, alias)

    if tahap == "Production":
        try:
            klien.delete_registered_model_alias(NAMA_MODEL, "staging")
            logging.info("Alias 'staging' dilepas")
        except mlflow.exceptions.MlflowException:
            pass

    coba_tahap_lama(klien, nomor, tahap)


def tampilkan_registry(klien):
    logging.info("Tracking URI: %s", mlflow.get_tracking_uri())

    daftar = klien.search_model_versions("name='{}'".format(NAMA_MODEL))
    if not daftar:
        logging.warning("Belum ada versi untuk model '%s'", NAMA_MODEL)
        return

    print("\nModel: {}".format(NAMA_MODEL))
    print("-" * 72)
    for ringkas in sorted(daftar, key=lambda v: int(v.version)):
        versi = klien.get_model_version(NAMA_MODEL, ringkas.version)
        run = klien.get_run(versi.run_id)
        print(
            "Versi {:<3} | tahap {:<12} | alias {:<28} | RMSE {}".format(
                versi.version,
                versi.tags.get("tahap", "None"),
                ", ".join(versi.aliases) or "-",
                versi.tags.get(
                    "rmse",
                    "{:.1f}".format(run.data.metrics.get("rmse", 0)),
                ),
            )
        )
    print("-" * 72)


def baca_argumen():
    parser = argparse.ArgumentParser(
        description="Mengelola metadata dan tahap model di MLflow Registry."
    )
    parser.add_argument(
        "--lihat",
        action="store_true",
        help="Tampilkan seluruh versi model beserta tahapnya.",
    )
    parser.add_argument(
        "--versi",
        type=int,
        help="Nomor versi model yang akan diproses.",
    )
    parser.add_argument(
        "--tahap",
        choices=sorted(TAHAP_ALIAS),
        help="Tahap tujuan versi model.",
    )
    return parser.parse_args()


def main():
    argumen = baca_argumen()
    klien = buat_klien()

    if argumen.lihat or argumen.versi is None:
        tampilkan_registry(klien)
        return

    versi, run = ambil_versi(klien, argumen.versi)
    if versi is None:
        return

    isi_metadata(klien, versi, run)

    if argumen.tahap:
        atur_tahap(klien, argumen.versi, argumen.tahap)

    tampilkan_registry(klien)


if __name__ == "__main__":
    main()