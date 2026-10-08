# Perbaikan evaluasi Agri-AI EWS (untuk naskah tesis)

## Cara menerapkan

Salin seluruh isi folder ini ke repositori `ews_prophet_lstm` (menimpa berkas lama), atau dari folder repositori:

```bash
git apply perbaikan.patch
```

## Cara mengisi angka Bab IV (wajib dengan internet)

```bash
pip install -r requirements.txt pytest
python scripts/thesis_outputs.py --lintas keduanya --uji-unit
```

- Durasi: sekitar 5–10 menit untuk kombinasi utama, ditambah sekitar 1–3 jam untuk validasi lintas 43 deret (CPU).
- Kovariat iklim diunduh sekali dari Open-Meteo dan NOAA, lalu disimpan di `data/cache/`. Jika unduhan gagal, skrip **berhenti** dan tidak memakai data sintetis.
- Semua keluaran ada di `thesis_outputs/`. Berkas `ringkasan_untuk_naskah.md` memetakan setiap angka ke tabel di naskah:

| Berkas | Naskah |
|---|---|
| `4_2_metrik_model.csv` | Tabel 4.2 (baris model AI), paragraf ±10% |
| `gambar_4_1_prediksi_vs_aktual.png` | Gambar 4.1 |
| `4_2_mape_per_horizon.csv` | Tabel 4.3 |
| `4_2_bobot_ensemble.json` | Bobot Smart Ensemble di 4.2 dan 4.9 |
| `4_2_cakupan_interval.csv` | Tabel 4.4 |
| `4_3_ringkasan_lintas.csv` | Tabel 4.5 |
| `4_4_skor_ews_per_titik_asal.csv` | Tabel 4.6 (kolom prediksi, skor EWS, level) |
| `4_6_korelasi_iklim.csv` | Tabel 4.7 |
| `4_8_pengujian_unit.csv` | Tabel 4.8 |
| `konfigurasi.json` | Versi pustaka, hash data, konfigurasi (untuk lampiran) |

Opsi `--tanpa-kovariat` dan `--cepat` hanya untuk uji alur; hasilnya diberi peringatan dan tidak boleh dipakai di naskah.

## Masalah yang diperbaiki

1. **TFT dievaluasi dengan data uji ikut dilatih dan prediksi datar.**
   - `prepare_dataset` memotong data latih 30 hari sebelum akhir data, sehingga hampir seluruh periode uji ikut dilatih.
   - Model hanya memprediksi 30 hari terakhir, lalu sisanya diisi dengan mengulang nilai terakhir (`np.pad(..., 'edge')`).
   - Sekarang: `build_training_dataset()` memotong tepat di akhir data latih, dan prediksi dibuat per jendela 30 hari.
2. **Metrik TFT di Dashboard di-hardcode** (`MAPE 8.5, RMSE 120, MAE 95`). Sekarang metrik diambil dari protokol evaluasi.
3. **BiLSTM praktis tidak terlatih.**
   - Pelatihan full-batch membuat "10 epoch" hanya berarti 10 langkah gradien.
   - Sekarang: mini-batch 64, maksimum 50 epoch, early stopping (patience 5), seed tetap.
   - MC Dropout divektorkan (50 sampel sekaligus).
4. **Smart Ensemble tidak sejajar dan bobotnya tidak sesuai naskah.**
   - Prediksi dipotong dari awal, tetapi nilai aktual diambil dari akhir (selisih ±6 hari).
   - Bobot yang dipakai adalah bawaan 40/30/30; 45/10/45 hanyalah aturan untuk prakiraan lebih dari 30 hari.
   - Sekarang: `fit_weights_grid()` mencari bobot (kelipatan 5%) pada periode validasi, dan `combine_series()` menggabungkan prediksi per tanggal dengan fallback (koefisien variasi > 15%).
5. **Protokol uji tidak setara.** SMA-30 dan BiLSTM diuji 1 langkah ke depan dengan harga aktual, sedangkan Prophet dan ARIMA memprediksi seluruh periode uji sekaligus. Sekarang semua model memakai protokol rolling-origin yang sama (`models/evaluation_protocol.py`).
6. **Kovariat iklim tidak pernah dipakai saat evaluasi**, dan `weather_client` diam-diam memakai cuaca sintetis bila API gagal.
   - Sekarang Prophet memakai regresor iklim dan TFT memakai ENSO (diketahui) serta cuaca (teramati), keduanya ex-ante.
   - Mode `strict=True` menolak data sintetis.
7. **Prakiraan Dashboard keliru.**
   - TFT "memprakirakan" 30 hari terakhir data lama, bukan masa depan. Sekarang memakai `forecast_future()`.
   - BiLSTM selalu memprediksi 1 hari ke depan berapa pun tanggal targetnya. Sekarang rekursif sampai tanggal target, dengan interval dari MC Dropout (bukan ±5% tetap).
8. **Backtester EWS memakai harga terbaru sebagai "harga saat ini".** Sekarang mesin EWS dibangun dari data sampai tanggal pengecekan.
9. **Prophet dalam evaluasi tidak memakai hari libur nasional**, padahal naskah menyebutkannya. Sekarang memakai `add_country_holidays('ID')`.
10. **Directional Accuracy** kini dihitung relatif terhadap harga di titik asal, dan kategori MAPE mengikuti Lewis (1982).

## Berkas yang berubah

- Baru:
  - `models/evaluation_protocol.py`
  - `scripts/thesis_outputs.py`
  - `PERUBAHAN.md`
- Diubah:
  - `models/evaluation.py`, `models/lstm_forecast.py`, `models/tft_forecast.py`, `models/ensemble.py`
  - `data/weather_client.py`, `engine/backtester.py`
  - `pages/1_🏠_Dashboard.py`, `pages/3_🔬_Model_Laboratory.py`, `pages/5_ℹ️_About.py`

## Catatan untuk Streamlit Cloud

Dashboard sekarang menghitung metrik dengan protokol yang sama dengan Model Laboratory, dan hasilnya di-cache per kombinasi. Pemuatan pertama satu kombinasi bisa memakan beberapa menit karena model penyusun dilatih dua kali (validasi dan uji). Untuk demo, turunkan epoch di sidebar.
