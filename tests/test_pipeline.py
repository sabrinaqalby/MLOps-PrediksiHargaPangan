import numpy as np
import pandas as pd
import pytest

from train import (
    FITUR,
    HORIZON,
    bagi_data,
    bentuk_fitur,
    bentuk_tabel_lebar,
    hitung_metrik,
)

JUMLAH_HARI = 40


@pytest.fixture
def data_bersih():
    # Sediakan data bersih dua komoditas pada dua wilayah.
    tanggal = pd.date_range("2026-09-01", periods=JUMLAH_HARI, freq="D")
    acuan = {"Beras Medium": 14000, "Cabai Rawit Merah": 55000}
    baris = []

    for komoditas, dasar in acuan.items():
        for wilayah, pengali in [("Kota Malang", 1.0), ("Nasional", 0.95)]:
            for urutan, hari in enumerate(tanggal):
                baris.append(
                    {
                        "tanggal": hari,
                        "wilayah": wilayah,
                        "komoditas": komoditas,
                        "harga": int(dasar * pengali) + urutan * 10,
                    }
                )

    return pd.DataFrame(baris)


@pytest.fixture
def data_siap(data_bersih):
    # Sediakan data yang fitur dan targetnya sudah lengkap
    return bentuk_fitur(bentuk_tabel_lebar(data_bersih))


def test_tabel_lebar_memisahkan_harga_per_wilayah(data_bersih):
    # Harga Malang dan harga nasional menjadi dua kolom terpisah.
    lebar = bentuk_tabel_lebar(data_bersih)

    assert "harga_malang" in lebar.columns
    assert "harga_nasional" in lebar.columns
    assert len(lebar) == JUMLAH_HARI * 2


def test_tabel_lebar_membuang_tanggal_yang_tidak_lengkap(data_bersih):
    kurang = data_bersih[
        ~(
            (data_bersih["wilayah"] == "Nasional")
            & (data_bersih["komoditas"] == "Beras Medium")
            & (data_bersih["tanggal"] == pd.Timestamp("2026-09-10"))
        )
    ]

    lebar = bentuk_tabel_lebar(kurang)
    tersisa = lebar[lebar["tanggal"] == pd.Timestamp("2026-09-10")]

    assert len(tersisa) == 1
    assert tersisa.iloc[0]["komoditas"] == "Cabai Rawit Merah"


def test_bentuk_fitur_menghasilkan_seluruh_kolom(data_siap):
    for kolom in FITUR + ["target"]:
        assert kolom in data_siap.columns

    assert not data_siap[FITUR + ["target"]].isna().any().any()


def test_target_adalah_harga_tujuh_hari_kemudian(data_siap, data_bersih):
    # Target harus berisi harga Malang pada H+7, bukan harga hari itu.
    malang = data_bersih[data_bersih["wilayah"] == "Kota Malang"]
    contoh = data_siap.iloc[0]

    seharusnya = malang[
        (malang["komoditas"] == contoh["komoditas"])
        & (malang["tanggal"] == contoh["tanggal"]
           + pd.Timedelta(days=HORIZON))
    ]["harga"].iloc[0]

    assert contoh["target"] == seharusnya


def test_bagi_data_tidak_mencampur_urutan_waktu(data_siap):
    X_latih, X_uji, _, _, data_uji = bagi_data(data_siap)

    urut = data_siap.sort_values("tanggal").reset_index(drop=True)
    tanggal_latih_terakhir = urut.iloc[len(X_latih) - 1]["tanggal"]

    assert len(X_latih) + len(X_uji) == len(data_siap)
    assert data_uji["tanggal"].min() >= tanggal_latih_terakhir


def test_bagi_data_memakai_kolom_fitur_yang_sama(data_siap):
    X_latih, X_uji, _, _, _ = bagi_data(data_siap)

    assert list(X_latih.columns) == FITUR
    assert list(X_uji.columns) == FITUR


def test_metrik_prediksi_sempurna_bernilai_nol():
    benar = np.array([10000.0, 20000.0, 30000.0])

    metrik = hitung_metrik(benar, benar)

    assert metrik["rmse"] == pytest.approx(0.0)
    assert metrik["mae"] == pytest.approx(0.0)
    assert metrik["mape"] == pytest.approx(0.0)
    assert metrik["r2"] == pytest.approx(1.0)


def test_metrik_menghitung_selisih_dengan_benar():
    benar = np.array([100.0, 200.0])
    prediksi = np.array([110.0, 180.0])

    metrik = hitung_metrik(benar, prediksi)

    assert metrik["mae"] == pytest.approx(15.0)
    assert metrik["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))