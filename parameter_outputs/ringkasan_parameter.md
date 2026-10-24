# Ringkasan analisis sensitivitas parameter

Bawang Merah, Jawa Timur; uji 390 hari, 13 titik asal; kovariat iklim: ya.

| Model | Parameter | Nilai | MAPE (%) | Δ MAPE vs bawaan | RMSE | DA (%) | Waktu (detik) |
|---|---|---|---|---|---|---|---|
| Naive Seasonal | Periode musiman (hari) | 7 | 9,58 | -13,68 | 5.525 | 28,4 | 0,0 |
| Naive Seasonal | Periode musiman (hari) | 30 | 13,59 | -9,67 | 7.153 | 37,2 | 0,0 |
| Naive Seasonal | Periode musiman (hari) | 365 (bawaan) | 23,26 | 0,00 | 12.273 | 42,4 | 0,0 |
| SMA | Jendela rata-rata (hari) | 7 | 9,42 | -2,79 | 5.476 | 35,9 | 0,0 |
| SMA | Jendela rata-rata (hari) | 14 | 10,27 | -1,95 | 5.584 | 35,9 | 0,0 |
| SMA | Jendela rata-rata (hari) | 30 (bawaan) | 12,22 | 0,00 | 5.974 | 43,7 | 0,0 |
| SMA | Jendela rata-rata (hari) | 60 | 11,92 | -0,29 | 6.410 | 53,2 | 0,0 |
| ARIMA | Orde (p, d, q) | (1, 1, 0) | 9,13 | -0,07 | 5.582 | 40,1 | 2,6 |
| ARIMA | Orde (p, d, q) | (2, 1, 0) | 9,16 | -0,04 | 5.616 | 55,3 | 1,0 |
| ARIMA | Orde (p, d, q) | (5, 1, 0) (bawaan) | 9,20 | 0,00 | 5.918 | 57,6 | 3,0 |
| ARIMA | Orde (p, d, q) | (1, 1, 1) | 9,23 | 0,03 | 5.942 | 64,1 | 3,9 |
| ARIMA | Orde (p, d, q) | (2, 1, 2) | 9,15 | -0,05 | 5.991 | 64,1 | 9,3 |
| Prophet | changepoint_prior_scale | 0,01 | 14,04 | -0,61 | 7.615 | 55,8 | 24,5 |
| Prophet | changepoint_prior_scale | 0,05 (bawaan) | 14,65 | 0,00 | 7.685 | 62,5 | 20,3 |
| Prophet | changepoint_prior_scale | 0,1 | 14,15 | -0,50 | 7.152 | 52,5 | 26,8 |
| Prophet | changepoint_prior_scale | 0,5 | 13,05 | -1,60 | 7.603 | 43,4 | 96,0 |
| Prophet | Efek hari libur nasional | ID (bawaan) | 14,65 | 0,00 | 7.685 | 62,5 | 20,3 |
| Prophet | Efek hari libur nasional | tidak dipakai | 14,59 | -0,07 | 7.591 | 59,7 | 22,8 |
| Prophet | Kovariat iklim | ya (bawaan) | 14,65 | 0,00 | 7.685 | 62,5 | 20,3 |
| Prophet | Kovariat iklim | tidak | 14,61 | -0,04 | 7.603 | 58,9 | 29,3 |
| BiLSTM | Panjang sekuens (hari) | 14 | 10,38 | 3,10 | 6.742 | 51,4 | 86,9 |
| BiLSTM | Panjang sekuens (hari) | 30 (bawaan) | 7,28 | 0,00 | 4.351 | 72,4 | 142,8 |
| BiLSTM | Panjang sekuens (hari) | 60 | 13,53 | 6,25 | 9.109 | 49,1 | 349,6 |
| BiLSTM | Ukuran tersembunyi | 32 | 7,47 | 0,19 | 4.581 | 68,7 | 52,3 |
| BiLSTM | Ukuran tersembunyi | 64 | 8,31 | 1,03 | 5.265 | 65,6 | 134,0 |
| BiLSTM | Ukuran tersembunyi | 128 (bawaan) | 7,28 | 0,00 | 4.351 | 72,4 | 142,8 |
| BiLSTM | Dropout | 0,1 | 6,25 | -1,03 | 3.879 | 81,7 | 367,7 |
| BiLSTM | Dropout | 0,2 (bawaan) | 7,28 | 0,00 | 4.351 | 72,4 | 142,8 |
| BiLSTM | Dropout | 0,3 | 8,61 | 1,33 | 5.129 | 61,5 | 414,9 |
| TFT | Panjang encoder (hari) | 30 | 10,58 | -9,59 | 7.204 | 65,9 | 277,9 |
| TFT | Panjang encoder (hari) | 60 | 18,89 | -1,27 | 10.325 | 53,7 | 178,1 |
| TFT | Panjang encoder (hari) | 90 (bawaan) | 20,17 | 0,00 | 11.037 | 64,3 | 237,4 |
| TFT | Ukuran tersembunyi | 16 | 18,14 | -2,03 | 10.069 | 61,2 | 105,8 |
| TFT | Ukuran tersembunyi | 32 (bawaan) | 20,17 | 0,00 | 11.037 | 64,3 | 237,4 |
| TFT | Ukuran tersembunyi | 64 | 9,88 | -10,28 | 6.525 | 56,1 | 299,4 |
| TFT | Jumlah epoch | 5 | 19,18 | -0,98 | 11.460 | 49,4 | 70,8 |
| TFT | Jumlah epoch | 15 (bawaan) | 20,17 | 0,00 | 11.037 | 64,3 | 237,4 |
| TFT | Jumlah epoch | 30 | 18,74 | -1,43 | 10.116 | 62,0 | 1.001,9 |
| TFT | Kovariat iklim | ya (bawaan) | 20,17 | 0,00 | 11.037 | 64,3 | 237,4 |
| TFT | Kovariat iklim | tidak | 16,69 | -3,48 | 9.484 | 45,7 | 163,4 |
| Smart Ensemble | Langkah grid bobot | 0,05 (bawaan) | 7,28 | 0,00 | 4.343 | 76,2 | 1.284,9 |
| Smart Ensemble | Langkah grid bobot | 0,1 | 7,30 | 0,03 | 4.350 | 76,7 | 0,2 |
| Smart Ensemble | Langkah grid bobot | 0,25 | 7,28 | 0,00 | 4.351 | 72,4 | 0,1 |
| Smart Ensemble | Ambang fallback (KV) | tidak dipakai | 7,03 | -0,25 | 4.252 | 72,9 | 0,8 |
| Smart Ensemble | Ambang fallback (KV) | 0,1 | 7,20 | -0,08 | 4.320 | 76,5 | 0,7 |
| Smart Ensemble | Ambang fallback (KV) | 0,15 (bawaan) | 7,28 | 0,00 | 4.343 | 76,2 | 0,6 |
| Smart Ensemble | Ambang fallback (KV) | 0,2 | 7,18 | -0,10 | 4.342 | 74,7 | 0,6 |
