# DESIGN.md: Agri-AI EWS

Arah ini dipilih pemilik proyek ("hijau tani, tenang", tema awal terang) dan ditulis oleh agen. Silakan revisi; `theme.py` dan `.streamlit/config.toml` mengikuti file ini.

Dial: ENERGY 1 / RHYTHM 2 / MOTION 1

## Identitas
- Alat kerja untuk analis dan pembuat kebijakan pangan, bukan halaman pemasaran. Data dan status peringatan adalah pusat perhatian.
- Nada tenang dan faktual. Bahasa UI: Indonesia. Istilah teknis model (MAPE, RMSE, BiLSTM, TFT) tetap seperti aslinya.

## Palet
| Token | Terang | Gelap | Alasan |
|---|---|---|---|
| Latar | `#F7F6F2` | `#121613` | Netral hangat seperti kertas laporan, tidak putih klinis |
| Latar sekunder | `#ECEAE3` | `#1C221E` | Sidebar, input, panel |
| Teks | `#1C2421` | `#E6EBE5` | Kontras tinggi untuk tabel angka |
| Utama (hijau daun) | `#2E6B3F` | `#7FBF8E` | Satu aksen: pertanian, dipakai untuk elemen aktif saja |

Warna status adalah sistem terpisah, berupa chip padat dengan teks yang lolos AA di kedua tema:
| Status | Latar | Teks |
|---|---|---|
| Bahaya / harga naik | `#B42318` | putih |
| Waspada | `#B54708` | putih |
| Perhatian | `#F5C451` | `#1C2421` |
| Normal / harga turun | `#1F6E35` | putih |
| Netral / stabil | `#475467` | putih |

Garis grafik (lolos 3:1 di kedua tema): aktual `#3E8E55`, prakiraan `#A86A10`, target `#C2412D`, pembanding `#4E7BA8`, referensi `#7A847E`.

## Tipografi
IBM Plex Sans: humanis tetapi tegas, angkanya jelas untuk harga dan metrik, dan mendukung angka tabular.

## Motif identitas
Chip status padat (warna + teks, tidak pernah warna saja) dipakai sama di kartu EWS, arah prediksi, faktor penyebab, dan chatbot.

## Aturan
- Tanpa gradien, glow, blur, atau emoji dekoratif. Bayangan tidak dipakai; batas tipis memisahkan panel.
- Tema mengikuti sistem pengguna; diganti lewat menu ⋮ > Settings > Theme.
