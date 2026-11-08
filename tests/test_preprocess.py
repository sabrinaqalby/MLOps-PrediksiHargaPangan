import pandas as pd
import pytest

from preprocess import (
    BATAS_LONJAKAN,
    buang_baris_tanpa_kunci,
    buang_duplikat,
    isi_harga_kosong,
    samakan_tipe,
    tandai_harga_kosong,
    tandai_lonjakan,
)


def baris(tanggal, harga, wilayah="Kota Malang", komoditas="Beras Medium",
          accessed_at="2026-10-19T08:00:00"):
    return {
        "tanggal": tanggal,
        "wilayah": wilayah,
        "komoditas": komoditas,
        "satuan": "Rp/kg",
        "harga": harga,
        "sumber": "sp2kp",
        "accessed_at": accessed_at,
    }


@pytest.fixture
def data_mentah():
    # Sediakan data mentah kecil yang sudah diseragamkan tipenya.
    return samakan_tipe(
        pd.DataFrame(
            [
                baris("2026-10-01", 14000),
                baris("2026-10-02", 14200),
                baris("2026-10-03", 14400),
            ]
        )
    )


def test_samakan_tipe_mengubah_tanggal_dan_harga():
    # Kolom tanggal jadi tipe waktu dan harga jadi tipe angka.
    df = samakan_tipe(
        pd.DataFrame([baris("2026-10-01", "14000"), baris("2026-10-02", 0)])
    )

    assert pd.api.types.is_datetime64_any_dtype(df["tanggal"])
    assert pd.api.types.is_numeric_dtype(df["harga"])


def test_samakan_tipe_membuang_spasi_berlebih():
    # Spasi di awal dan akhir nama wilayah ikut dibersihkan.
    df = samakan_tipe(
        pd.DataFrame([baris("2026-10-01", 14000, wilayah="  Nasional  ")])
    )

    assert df.loc[0, "wilayah"] == "Nasional"


def test_tandai_harga_kosong_mengubah_nol_jadi_kosong():
    # Harga nol dari hari tanpa survei diperlakukan sebagai data kosong.
    df = samakan_tipe(
        pd.DataFrame(
            [
                baris("2026-10-01", 14000),
                baris("2026-10-02", 0),
                baris("2026-10-03", -50),
            ]
        )
    )

    hasil = tandai_harga_kosong(df)

    assert hasil["harga"].isna().sum() == 2
    assert hasil.loc[0, "harga"] == 14000


def test_buang_baris_tanpa_kunci_membuang_tanggal_kosong():
    # Baris tanpa tanggal tidak dapat dipakai dan harus dibuang.
    df = samakan_tipe(
        pd.DataFrame([baris("2026-10-01", 14000), baris(None, 14100)])
    )

    hasil = buang_baris_tanpa_kunci(df)

    assert len(hasil) == 1


def test_buang_duplikat_menyisakan_pengambilan_terbaru():
    # Ingestion berulang pada tanggal sama disisakan yang paling akhir.
    df = samakan_tipe(
        pd.DataFrame(
            [
                baris("2026-10-01", 14000, accessed_at="2026-10-01T08:00:00"),
                baris("2026-10-01", 14500, accessed_at="2026-10-01T17:00:00"),
            ]
        )
    )

    hasil = buang_duplikat(df)

    assert len(hasil) == 1
    assert hasil.iloc[0]["harga"] == 14500


def test_isi_harga_kosong_menandai_hasil_interpolasi():
    # Harga kosong di tengah diisi dan baris hasilnya diberi penanda.
    df = tandai_harga_kosong(
        samakan_tipe(
            pd.DataFrame(
                [
                    baris("2026-10-01", 14000),
                    baris("2026-10-02", 0),
                    baris("2026-10-03", 14400),
                ]
            )
        )
    )

    hasil = isi_harga_kosong(df).sort_values("tanggal")

    assert hasil["harga"].isna().sum() == 0
    assert hasil.iloc[1]["harga"] == 14200
    assert bool(hasil.iloc[1]["hasil_interpolasi"]) is True
    assert bool(hasil.iloc[0]["hasil_interpolasi"]) is False


def test_isi_harga_kosong_menghasilkan_harga_bulat(data_mentah):
    hasil = isi_harga_kosong(data_mentah)

    assert pd.api.types.is_integer_dtype(hasil["harga"])


def test_tandai_lonjakan_menandai_perubahan_ekstrem():
    # Perubahan harga di atas ambang ditandai, bukan dihapus.
    df = isi_harga_kosong(
        tandai_harga_kosong(
            samakan_tipe(
                pd.DataFrame(
                    [
                        baris("2026-10-01", 20000),
                        baris("2026-10-02", 21000),
                        baris("2026-10-03", 60000),
                    ]
                )
            )
        )
    )

    hasil = tandai_lonjakan(df).sort_values("tanggal")

    assert len(hasil) == 3
    assert bool(hasil.iloc[2]["lonjakan_ekstrem"]) is True
    assert bool(hasil.iloc[1]["lonjakan_ekstrem"]) is False
    assert hasil.iloc[2]["perubahan_persen"] > BATAS_LONJAKAN


def test_tandai_lonjakan_tidak_mencampur_antar_komoditas():
    # Perubahan harga dihitung per komoditas, bukan antar komoditas.
    df = isi_harga_kosong(
        tandai_harga_kosong(
            samakan_tipe(
                pd.DataFrame(
                    [
                        baris("2026-10-01", 14000, komoditas="Beras Medium"),
                        baris("2026-10-01", 60000, komoditas="Cabai Rawit"),
                    ]
                )
            )
        )
    )

    hasil = tandai_lonjakan(df)

    assert hasil["lonjakan_ekstrem"].sum() == 0