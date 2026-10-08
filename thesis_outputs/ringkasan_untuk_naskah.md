# Ringkasan keluaran untuk naskah

Dibuat: 08-10-2026 09:37  
Data: `food_prices_real.csv` (SHA-256 `671ec3a012d7629a…`)  

## Subbab 4.1 — Tabel 4.1 Ringkasan pemeriksaan data

| Butir | Nilai |
|---|---|
| Jumlah catatan | 647709 |
| Tanggal awal | 2021-01-04 |
| Tanggal akhir | 2026-05-07 |
| Jumlah komoditas | 10 |
| Jumlah provinsi | 34 |
| Catatan duplikat (tanggal-provinsi-komoditas) | 0 |
| Harga kosong | 0 |
| Harga nol/negatif | 0 |
| Jumlah deret (komoditas × provinsi) | 334 |
| Total hari kalender seluruh deret | 651039 |
| Hari tanpa catatan | 3330 |
| Diisi forward-fill (≤ 7 hari) | 3330 |
| Diinterpolasi linear (celah > 7 hari) | 0 |
| Celah terpanjang (hari) | 3 |

## Subbab 4.2 — protokol

Data latih 1560 hari (s.d. 2025-04-12), data uji 390 hari (2025-04-13 s.d. 2026-05-07), 13 titik asal, horizon 30 hari. Validasi bobot ensemble: 312 hari mulai 2024-06-05. Kovariat iklim: ya.


### Tabel 4.2 Perbandingan kinerja

| Model | RMSE | MAE | MAPE (%) | SMAPE (%) | R² | Directional Accuracy (%) | Kategori MAPE |
|---|---|---|---|---|---|---|---|
| Naive Seasonal | 12.273 | 9.829 | 23,26 | 26,64 | -5,693 | 42,4 | Cukup |
| SMA-30 | 5.974 | 4.965 | 12,22 | 11,90 | -0,586 | 43,7 | Baik |
| ARIMA(5,1,0) | 5.918 | 3.844 | 9,20 | 8,64 | -0,556 | 57,6 | Sangat akurat |
| Prophet | 7.685 | 6.135 | 14,65 | 14,31 | -1,624 | 62,5 | Baik |
| BiLSTM | 4.351 | 3.090 | 7,28 | 7,22 | 0,159 | 72,4 | Sangat akurat |
| TFT | 11.037 | 8.573 | 20,17 | 23,41 | -4,413 | 64,3 | Cukup |
| Smart Ensemble | 4.343 | 3.087 | 7,28 | 7,22 | 0,162 | 76,2 | Sangat akurat |

### Persentase hari dengan galat dalam toleransi (untuk paragraf ±X%)

| Model | Dalam ±5% (%) | Dalam ±10% (%) |
|---|---|---|
| Naive Seasonal | 9,74 | 22,31 |
| SMA-30 | 26,41 | 46,15 |
| ARIMA(5,1,0) | 51,54 | 72,56 |
| Prophet | 25,64 | 36,92 |
| BiLSTM | 52,56 | 72,05 |
| TFT | 13,59 | 28,97 |
| Smart Ensemble | 53,59 | 71,28 |

### Tabel 4.3 MAPE menurut horizon

| Model | 1–7 hari | 8–14 hari | 15–30 hari |
|---|---|---|---|
| Naive Seasonal | 22,99 | 25,33 | 22,46 |
| SMA-30 | 9,02 | 11,17 | 14,07 |
| ARIMA(5,1,0) | 2,54 | 8,10 | 12,59 |
| Prophet | 11,91 | 13,23 | 16,47 |
| BiLSTM | 2,48 | 6,37 | 9,78 |
| TFT | 20,77 | 19,68 | 20,11 |
| Smart Ensemble | 2,62 | 6,29 | 9,75 |

### Bobot Smart Ensemble (hasil grid search pada validasi)

Prophet 0%, BiLSTM 95%, TFT 5%; MAPE validasi gabungan 12,26%; fallback aktif pada 133 hari (34,1% hari uji); model validasi terbaik: BiLSTM.

### Tabel 4.4 Cakupan interval ketidakpastian

| Model | Interval | Cakupan aktual (%) | Lebar rata-rata (% harga) |
|---|---|---|---|
| Prophet | Interval 90% Prophet | 74,62 | 42,46 |
| BiLSTM | Interval 90% MC Dropout (50 sampel) | 51,28 | 10,93 |
| TFT | Interval kuantil 2–98% TFT | 12,82 | 8,64 |

## Subbab 4.4 — Tabel 4.6 Skor EWS per titik asal

| Tanggal data terakhir | Tanggal target | Harga terakhir | Prediksi ensemble | Harga aktual target | Perubahan prediksi (%) | Perubahan aktual (%) | Skor EWS | Level EWS | Skor risiko pasokan | Kenaikan aktual ≥ 10% |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-04-12 | 2025-05-12 | 48.200 | 45.897 | 40.250 | -4,8 | -16,5 | 16,3 | Normal | 23,5 | 0 |
| 2025-05-12 | 2025-06-11 | 40.250 | 39.025 | 38.550 | -3,0 | -4,2 | 12,2 | Normal | 24,5 | 0 |
| 2025-06-11 | 2025-07-11 | 38.550 | 38.025 | 41.500 | -1,4 | 7,7 | 12,4 | Normal | 44,5 | 0 |
| 2025-07-11 | 2025-08-10 | 41.500 | 41.757 | 53.733 | 0,6 | 29,5 | 26,6 | Watch | 29,6 | 1 |
| 2025-08-10 | 2025-09-09 | 53.733 | 45.412 | 36.800 | -15,5 | -31,5 | 51,5 | Alert | 19,7 | 0 |
| 2025-09-09 | 2025-10-09 | 36.800 | 35.127 | 36.500 | -4,5 | -0,8 | 29,3 | Watch | 37,8 | 0 |
| 2025-10-09 | 2025-11-08 | 36.500 | 37.051 | 41.233 | 1,5 | 13,0 | 14,4 | Normal | 38,1 | 1 |
| 2025-11-08 | 2025-12-08 | 41.233 | 40.365 | 51.450 | -2,1 | 24,8 | 14,8 | Normal | 27,5 | 1 |
| 2025-12-08 | 2026-01-07 | 51.450 | 46.830 | 38.600 | -9,0 | -25,0 | 33,7 | Watch | 53,6 | 0 |
| 2026-01-07 | 2026-02-06 | 38.600 | 36.245 | 38.550 | -6,1 | -0,1 | 25,7 | Watch | 27,2 | 0 |
| 2026-02-06 | 2026-03-08 | 38.550 | 38.961 | 40.200 | 1,1 | 4,3 | 12,7 | Normal | 35,6 | 0 |
| 2026-03-08 | 2026-04-07 | 40.200 | 38.618 | 41.150 | -3,9 | 2,4 | 15,1 | Normal | 24,5 | 0 |
| 2026-04-07 | 2026-05-07 | 41.150 | 40.057 | 43.650 | -2,7 | 6,1 | 15,5 | Normal | 28,6 | 0 |

## Subbab 4.5 — nilai korelasi tiap fitur (Gambar 4.2)

| Fitur | Kelompok | |r| |
|---|---|---|
| Lag 1 hari | Autolag harga | 0,997 |
| Lag 7 hari | Autolag harga | 0,947 |
| Lag 14 hari | Autolag harga | 0,848 |
| Lag 30 hari | Autolag harga | 0,617 |
| Korelasi dengan Telur Ayam | Lintas komoditas | 0,475 |
| Korelasi dengan Daging Sapi | Lintas komoditas | 0,441 |
| Korelasi dengan Gula Pasir | Lintas komoditas | 0,415 |
| Korelasi dengan Beras | Lintas komoditas | 0,355 |
| Korelasi dengan Cabai Merah | Lintas komoditas | 0,341 |
| Korelasi dengan Minyak Goreng | Lintas komoditas | 0,308 |
| Korelasi dengan Daging Ayam | Lintas komoditas | 0,239 |
| Korelasi dengan Cabai Rawit | Lintas komoditas | 0,218 |
| Korelasi dengan Bawang Putih | Lintas komoditas | 0,189 |
| Bulan | Temporal | 0,158 |
| Hari dalam seminggu | Temporal | 0,006 |

## Subbab 4.6 — Tabel 4.7 Korelasi kovariat iklim dan harga

| Target | Kovariat | Lag (hari) | Pearson r | p (Pearson) | Spearman ρ | p (Spearman) | n |
|---|---|---|---|---|---|---|---|
| Harga | Curah hujan (rata-rata 30 hari) | 0 | 0,140 | 0,0000 | 0,245 | 0,0000 | 1.950 |
| Harga | Curah hujan (rata-rata 30 hari) | 7 | 0,180 | 0,0000 | 0,276 | 0,0000 | 1.943 |
| Harga | Curah hujan (rata-rata 30 hari) | 14 | 0,221 | 0,0000 | 0,309 | 0,0000 | 1.936 |
| Harga | Curah hujan (rata-rata 30 hari) | 30 | 0,303 | 0,0000 | 0,367 | 0,0000 | 1.920 |
| Harga | Suhu (rata-rata 30 hari) | 0 | -0,044 | 0,0540 | -0,042 | 0,0611 | 1.950 |
| Harga | Suhu (rata-rata 30 hari) | 7 | -0,055 | 0,0158 | -0,059 | 0,0089 | 1.943 |
| Harga | Suhu (rata-rata 30 hari) | 14 | -0,066 | 0,0035 | -0,079 | 0,0005 | 1.936 |
| Harga | Suhu (rata-rata 30 hari) | 30 | -0,062 | 0,0070 | -0,099 | 0,0000 | 1.920 |
| Harga | Indeks ENSO (ONI) | 0 | -0,085 | 0,0002 | 0,097 | 0,0000 | 1.950 |
| Harga | Indeks ENSO (ONI) | 7 | -0,083 | 0,0003 | 0,085 | 0,0002 | 1.943 |
| Harga | Indeks ENSO (ONI) | 14 | -0,080 | 0,0004 | 0,070 | 0,0021 | 1.936 |
| Harga | Indeks ENSO (ONI) | 30 | -0,086 | 0,0002 | 0,024 | 0,2913 | 1.920 |
| Perubahan harga 30 hari (%) | Curah hujan (rata-rata 30 hari) | 0 | 0,196 | 0,0000 | 0,198 | 0,0000 | 1.920 |
| Perubahan harga 30 hari (%) | Curah hujan (rata-rata 30 hari) | 7 | 0,187 | 0,0000 | 0,182 | 0,0000 | 1.920 |
| Perubahan harga 30 hari (%) | Curah hujan (rata-rata 30 hari) | 14 | 0,185 | 0,0000 | 0,179 | 0,0000 | 1.920 |
| Perubahan harga 30 hari (%) | Curah hujan (rata-rata 30 hari) | 30 | 0,175 | 0,0000 | 0,139 | 0,0000 | 1.920 |
| Perubahan harga 30 hari (%) | Suhu (rata-rata 30 hari) | 0 | 0,043 | 0,0626 | 0,005 | 0,8183 | 1.920 |
| Perubahan harga 30 hari (%) | Suhu (rata-rata 30 hari) | 7 | 0,034 | 0,1391 | -0,014 | 0,5294 | 1.920 |
| Perubahan harga 30 hari (%) | Suhu (rata-rata 30 hari) | 14 | 0,016 | 0,4909 | -0,033 | 0,1424 | 1.920 |
| Perubahan harga 30 hari (%) | Suhu (rata-rata 30 hari) | 30 | 0,016 | 0,4954 | -0,005 | 0,8107 | 1.920 |
| Perubahan harga 30 hari (%) | Indeks ENSO (ONI) | 0 | 0,022 | 0,3426 | -0,059 | 0,0095 | 1.920 |
| Perubahan harga 30 hari (%) | Indeks ENSO (ONI) | 7 | 0,024 | 0,3012 | -0,070 | 0,0022 | 1.920 |
| Perubahan harga 30 hari (%) | Indeks ENSO (ONI) | 14 | 0,026 | 0,2636 | -0,083 | 0,0003 | 1.920 |
| Perubahan harga 30 hari (%) | Indeks ENSO (ONI) | 30 | 0,021 | 0,3613 | -0,108 | 0,0000 | 1.920 |

## Subbab 4.3 — Tabel 4.5 validasi lintas (43 kombinasi)

| Model | MAPE rata-rata (%) | MAPE median (%) | MAPE min (%) | MAPE maks (%) | Jumlah kombinasi terbaik |
|---|---|---|---|---|---|
| Naive Seasonal | 19,05 | 19,70 | 2,02 | 47,86 | 0 |
| SMA-30 | 11,25 | 11,32 | 0,58 | 44,92 | 0 |
| ARIMA(5,1,0) | 7,77 | 7,44 | 0,38 | 32,15 | 23 |
| Prophet | 12,94 | 12,92 | 0,84 | 38,75 | 0 |
| BiLSTM | 8,77 | 7,28 | 0,61 | 36,94 | 19 |
| Smart Ensemble | 8,75 | 7,68 | 0,61 | 36,94 | 1 |

## Subbab 4.8 — Tabel 4.8 Hasil pengujian unit

| Modul uji | Lulus | Gagal | Dilewati |
|---|---|---|---|
| test_api | 16 | 0 | 0 |
| test_backtester | 4 | 0 | 0 |
| test_database | 15 | 0 | 0 |
| test_ensemble | 9 | 0 | 0 |
| test_evaluation | 20 | 0 | 0 |
| test_ews_engine | 5 | 0 | 0 |
| test_ews_engine_v2 | 7 | 0 | 0 |
| test_lstm_forecast | 6 | 0 | 0 |
| test_pihps_harmonizer | 2 | 0 | 0 |
| test_pihps_scraper | 8 | 0 | 0 |
| test_price_narrative | 13 | 0 | 0 |
| test_prophet_forecast | 4 | 0 | 0 |
| test_scheduler | 4 | 0 | 0 |
| test_supply_risk | 8 | 0 | 0 |
| test_tft_forecast | 4 | 0 | 1 |
| test_theme | 7 | 0 | 0 |
| test_weather_client | 16 | 0 | 0 |
