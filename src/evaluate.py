import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import yaml

from kelola_registry import (
    NAMA_MODEL,
    ambil_versi,
    atur_tahap,
    buat_klien,
    isi_metadata,
)

NAMA_EKSPERIMEN = "prediksi-harga-pangan-malang"
FILE_AMBANG = Path("config/ambang_model.yaml")
TAHAP_TUJUAN = "Staging"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def muat_ambang():
    if not FILE_AMBANG.exists():
        logging.error("Berkas %s tidak ditemukan", FILE_AMBANG)
        raise SystemExit(1)

    return yaml.safe_load(FILE_AMBANG.read_text(encoding="utf-8"))


def ambil_run_terakhir(klien):
    eksperimen = klien.get_experiment_by_name(NAMA_EKSPERIMEN)
    if eksperimen is None:
        logging.error("Eksperimen '%s' belum ada", NAMA_EKSPERIMEN)
        return None

    hasil = klien.search_runs(
        [eksperimen.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not hasil:
        logging.error("Belum ada run pada eksperimen '%s'", NAMA_EKSPERIMEN)
        return None

    return hasil[0]


def periksa_gerbang(metrik, ambang):
    return [
        (
            "MAPE",
            metrik.get("mape", float("inf")),
            "<= {:.2f}%".format(ambang["mape_maksimum"]),
            metrik.get("mape", float("inf")) <= ambang["mape_maksimum"],
        ),
        (
            "R2",
            metrik.get("r2", float("-inf")),
            ">= {:.3f}".format(ambang["r2_minimum"]),
            metrik.get("r2", float("-inf")) >= ambang["r2_minimum"],
        ),
        (
            "RMSE",
            metrik.get("rmse", float("inf")),
            "<= {:.1f}".format(ambang["rmse_maksimum"]),
            metrik.get("rmse", float("inf")) <= ambang["rmse_maksimum"],
        ),
    ]


def periksa_regresi(metrik, acuan):
    batas = acuan["rmse"] * (1 + acuan["toleransi_penurunan_persen"] / 100)
    nilai = metrik.get("rmse", float("inf"))

    return (
        "RMSE vs model produksi",
        nilai,
        "<= {:.1f}".format(batas),
        nilai <= batas,
    )


def periksa_target_produksi(metrik, target):
    mape = metrik.get("mape", float("inf"))
    unggul = metrik.get("perbaikan_vs_naif_persen", float("-inf"))
    batas_unggul = target["keunggulan_vs_naif_minimum_persen"]

    return [
        (
            "MAPE H+7",
            mape,
            "< {:.1f}%".format(target["mape_maksimum"]),
            mape < target["mape_maksimum"],
        ),
        (
            "Keunggulan vs naif",
            unggul,
            ">= {:.1f}%".format(batas_unggul),
            unggul >= batas_unggul,
        ),
    ]


def tulis_ringkasan(baris):
    tujuan = os.environ.get("GITHUB_STEP_SUMMARY")
    if not tujuan:
        return

    with open(tujuan, "a", encoding="utf-8") as berkas:
        berkas.write("\n".join(baris) + "\n")


def susun_tabel(judul, pemeriksaan, kata_lolos, kata_gagal):
    baris = [
        "### {}".format(judul),
        "",
        "| Kriteria | Nilai | Syarat | Hasil |",
        "| --- | --- | --- | --- |",
    ]

    for nama, nilai, syarat, status in pemeriksaan:
        baris.append(
            "| {} | {:.3f} | {} | {} |".format(
                nama, nilai, syarat, kata_lolos if status else kata_gagal
            )
        )

    baris.append("")
    return baris


def susun_ringkasan(run, gerbang, target, lolos):
    baris = [
        "## Validasi Model Otomatis",
        "",
        "Run dievaluasi: `{}`".format(run.info.run_name),
        "",
    ]

    baris += susun_tabel(
        "Syarat masuk Staging", gerbang, "LOLOS", "GAGAL"
    )
    baris += susun_tabel(
        "Target produksi (LK-01)", target, "TERCAPAI", "BELUM"
    )

    baris += [
        "**Kesimpulan:** {}".format(
            "model didaftarkan ke " + TAHAP_TUJUAN
            if lolos
            else "model ditolak, tidak didaftarkan"
        ),
    ]

    if lolos and not all(status for _, _, _, status in target):
        baris.append("")
        baris.append(
            "Target produksi LK-01 belum terpenuhi, sehingga model "
            "berhenti di Staging dan tidak dinaikkan ke Production."
        )

    return baris


def daftarkan_ke_staging(klien, run):
    versi_baru = mlflow.register_model(
        "runs:/{}/model".format(run.info.run_id), NAMA_MODEL
    )
    nomor = versi_baru.version

    versi, run_asal = ambil_versi(klien, nomor)
    if versi is not None:
        isi_metadata(klien, versi, run_asal)

    atur_tahap(klien, nomor, TAHAP_TUJUAN)
    logging.info(
        "Model terdaftar sebagai versi %s dengan tahap %s",
        nomor,
        TAHAP_TUJUAN,
    )
    return nomor


def baca_argumen():
    parser = argparse.ArgumentParser(
        description="Memvalidasi model terbaru dan mempromosikannya."
    )
    parser.add_argument(
        "--tanpa-promosi",
        action="store_true",
        help="Hanya memeriksa ambang batas tanpa mendaftarkan model.",
    )
    return parser.parse_args()


def main():
    argumen = baca_argumen()
    konfigurasi = muat_ambang()
    klien = buat_klien()

    run = ambil_run_terakhir(klien)
    if run is None:
        sys.exit(1)

    metrik = run.data.metrics
    logging.info("Mengevaluasi run '%s'", run.info.run_name)

    gerbang = periksa_gerbang(metrik, konfigurasi["gerbang_staging"])
    gerbang.append(periksa_regresi(metrik, konfigurasi["model_acuan"]))
    target = periksa_target_produksi(
        metrik, konfigurasi["target_produksi"]
    )

    logging.info("--- Syarat masuk Staging ---")
    for nama, nilai, syarat, status in gerbang:
        logging.info(
            "%-24s %10.3f  syarat %-12s %s",
            nama,
            nilai,
            syarat,
            "LOLOS" if status else "GAGAL",
        )

    logging.info("--- Target produksi LK-01 ---")
    for nama, nilai, syarat, status in target:
        logging.info(
            "%-24s %10.3f  target %-12s %s",
            nama,
            nilai,
            syarat,
            "TERCAPAI" if status else "BELUM",
        )

    lolos = all(status for _, _, _, status in gerbang)
    siap_produksi = all(status for _, _, _, status in target)
    tulis_ringkasan(susun_ringkasan(run, gerbang, target, lolos))

    if not lolos:
        logging.error(
            "Model tidak memenuhi syarat Staging, promosi dibatalkan"
        )
        sys.exit(1)

    if siap_produksi:
        logging.info(
            "Seluruh target LK-01 tercapai, model layak diajukan "
            "naik ke Production"
        )
    else:
        logging.warning(
            "Target LK-01 belum tercapai, model berhenti di %s",
            TAHAP_TUJUAN,
        )

    if argumen.tanpa_promosi:
        logging.info("Seluruh syarat lolos, promosi dilewati atas permintaan")
        return

    daftarkan_ke_staging(klien, run)


if __name__ == "__main__":
    main()