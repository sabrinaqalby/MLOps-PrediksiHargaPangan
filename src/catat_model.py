import argparse
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import mlflow.artifacts
import yaml
from mlflow.tracking import MlflowClient

from train import baca_versi_data

URI_TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
NAMA_MODEL = "model-harga-pangan-malang"

FOLDER_MODEL = Path("models/model_produksi")
FILE_METADATA = Path("models/model_produksi.yaml")
FILE_DATA_OLAHAN = Path("data/processed/harga_bersih.csv")
FILE_VERSI_DATA = Path("data/raw.dvc")

WIB = timezone(timedelta(hours=7))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def ambil_versi_aktif(klien, alias):
    try:
        return klien.get_model_version_by_alias(NAMA_MODEL, alias)
    except mlflow.exceptions.MlflowException:
        logging.error(
            "Alias '%s' belum dipasang. Jalankan kelola_registry.py dulu",
            alias,
        )
        return None


def unduh_artefak(uri):
    try:
        with tempfile.TemporaryDirectory() as sementara:
            asal = mlflow.artifacts.download_artifacts(
                artifact_uri=uri, dst_path=sementara
            )

            if FOLDER_MODEL.exists():
                shutil.rmtree(FOLDER_MODEL)
            FOLDER_MODEL.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(asal, FOLDER_MODEL)
    except (mlflow.exceptions.MlflowException, OSError) as galat:
        logging.error("Gagal mengunduh artefak: %s", galat)
        return None

    logging.info("Artefak model disalin ke %s", FOLDER_MODEL)
    return FOLDER_MODEL


def susun_metadata(versi, run, alias):
    return {
        "model": {
            "nama_registry": NAMA_MODEL,
            "versi": int(versi.version),
            "alias": alias,
            "tahap": versi.tags.get("tahap", "None"),
            "uri": "models:/{}@{}".format(NAMA_MODEL, alias),
            "run_id": run.info.run_id,
            "nama_run": run.info.run_name,
            "folder_artefak": str(FOLDER_MODEL),
        },
        "data": {
            "berkas_penunjuk": str(FILE_VERSI_DATA),
            "md5": baca_versi_data(),
            "berkas_olahan": str(FILE_DATA_OLAHAN),
            "sumber": "API SP2KP Kemendag",
        },
        "parameter": dict(sorted(run.data.params.items())),
        "metrik": {
            nama: round(float(nilai), 4)
            for nama, nilai in sorted(run.data.metrics.items())
        },
        "dicatat_pada": datetime.now(WIB).isoformat(timespec="seconds"),
    }


def simpan_metadata(metadata):
    FILE_METADATA.parent.mkdir(parents=True, exist_ok=True)
    FILE_METADATA.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logging.info("Metadata model ditulis ke %s", FILE_METADATA)


def baca_argumen():
    parser = argparse.ArgumentParser(
        description="Mencatat metadata model aktif untuk dilacak DVC."
    )
    parser.add_argument(
        "--alias",
        default="production",
        help="Alias model di registry. Default: production.",
    )
    return parser.parse_args()


def main():
    argumen = baca_argumen()
    mlflow.set_tracking_uri(URI_TRACKING)
    klien = MlflowClient()

    versi = ambil_versi_aktif(klien, argumen.alias)
    if versi is None:
        return

    run = klien.get_run(versi.run_id)
    uri = "models:/{}@{}".format(NAMA_MODEL, argumen.alias)

    if unduh_artefak(uri) is None:
        return

    simpan_metadata(susun_metadata(versi, run, argumen.alias))

    print("\nModel aktif : {} versi {}".format(NAMA_MODEL, versi.version))
    print("Versi data  : {}".format(baca_versi_data()))
    print("\nLanjutkan dengan:")
    print("    dvc add models/model_produksi")
    print("    dvc push\n")


if __name__ == "__main__":
    main()