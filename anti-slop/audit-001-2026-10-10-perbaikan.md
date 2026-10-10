# Laporan perbaikan audit 001

- Tanggal: 2026-10-10
- Disetujui: semua nomor (1-27)
- Arah desain dari pemilik: "hijau tani, tenang", tema awal terang. Ditulis ke `DESIGN.md`, dial ENERGY 1 / RHYTHM 2 / MOTION 1.

## Perubahan arsitektur

Toggle tema kustom berbasis `session_state` dan CSS override sebanyak 450 baris diganti dengan tema bawaan Streamlit (`[theme.light]` dan `[theme.dark]` di `.streamlit/config.toml`). Akibatnya tabel, input, chat input, ring fokus, dan grafik Plotly ikut berganti tema. `theme.py` sekarang hanya memuat token warna (status dan grafik, lolos AA di kedua tema), komponen kecil (`status_chip`, `ews_card`, `render_sidebar_brand`, `render_footer`), dan CSS untuk komponen itu saja.

Halaman diganti nama menjadi bahasa Indonesia tanpa emoji. URL ikut berubah: `/Dashboard`, `/Analisis_Regional`, `/Laboratorium_Model`, `/Laporan`, `/Tentang`, `/Chatbot`.

## Status per nomor

| No | Status | Yang dilakukan |
|---|---|---|
| 1 | Selesai | Kartu EWS memakai warna solid per level, teks lolos AA (5.4:1 sampai 9.7:1); label 0.65rem kapital diganti 0.9rem. Nama level disamakan dengan pesan engine (Bahaya, Waspada, Perhatian, Normal); skor disembunyikan saat prakiraan gagal |
| 2 | Selesai | Gradien tombol dihapus; tombol memakai gaya bawaan dengan warna tema |
| 3 | Selesai | Label arah dan dampak memakai chip padat (`STATUS`); sel terbaik di tabel Lab memakai latar dan teks dengan kontras 6.3:1 |
| 4 | Selesai | Opacity footer dihapus; warna teks tema penuh |
| 5 | Selesai | Override `box-shadow`/`border` dihapus; ditambah outline `:focus-visible` 2px `#3E8E55` (3.7:1 terang, 4.5:1 gelap) |
| 6 | Selesai | Header tidak lagi disembunyikan; tombol buka sidebar terlihat di 375px |
| 7 | Selesai | Em dash dihapus dari semua teks UI, termasuk 2 kalimat di `engine/price_narrative.py`; label tren `supply_risk` dipetakan ke teks Indonesia di Dashboard |
| 8 | Selesai | "Uptime 99.9%" dan klaim BMKG dihapus; diganti alasan yang bisa diverifikasi tentang Open-Meteo. Sitasi Prophet (Taylor & Letham, 2018) dan TFT (Lim et al., 2021) ditambahkan |
| 9 | Selesai | Delta palsu di beranda dihapus; delta Dashboard diberi konteks ("dalam 7 hari", "dari harga terakhir") |
| 10 | Selesai | Tema bawaan Streamlit terang/gelap; terverifikasi di browser untuk kedua mode |
| 11 | Selesai | Logo Flaticon diganti wordmark teks "Agri-AI EWS" (placeholder sampai ada logo resmi) |
| 12 | Selesai | Badge "Online" dihapus |
| 13 | Selesai | Gradien, glow, dan latar gradien dihapus; palet hijau tani di `DESIGN.md` |
| 14 | Selesai di halaman | Emoji dihapus dari judul, tab, tombol, label, sidebar, dan nama file. Favicon 🌾 dipertahankan |
| 15 | Selesai | Semua bayangan dihapus; panel dipisahkan oleh batas tipis |
| 16 | Selesai | Kartu fitur beranda diganti `st.page_link` yang benar-benar menuju halaman |
| 17 | Selesai | Tracking lebar dan huruf kapital dihapus; font IBM Plex Sans (alasan di `DESIGN.md`), angka tabular pada metrik |
| 18 | Selesai | Traceback mentah tidak lagi tampil; pesan error menyebut penyebab dan langkah berikutnya. `showErrorDetails = false` |
| 19 | Selesai, dengan catatan | Tema mengikuti sistem pengguna: terang di sistem terang. Streamlit tidak bisa memaksa tema terang sekaligus menyediakan tema gelap kustom |
| 20 | Selesai | `borderColor` widget 3.5:1 (terang) dan 3.6:1 (gelap) |
| 21 | Selesai di halaman | Semua label UI berbahasa Indonesia; buzzword dihapus |
| 22 | Selesai | Palet GitHub/Tailwind diganti palet sendiri |
| 23 | Selesai | Satu set token status dan grafik di `theme.py` |
| 24 | Selesai | Footer `flex-wrap`; terverifikasi tidak meluap di 375px |
| 25 | Selesai | Angka komoditas dan provinsi di beranda dibaca dari database |
| 26 | Selesai | Halaman Tentang punya sidebar yang sama; selektor CSS yang tidak pernah cocok dihapus |
| 27 | Selesai | CSS chat ganda dihapus |

## Temuan baru saat verifikasi

- Caption bawaan Streamlit memakai opacity 0.6: hanya 4.0:1 di sidebar terang. Dinaikkan ke 0.8 (7.3:1).
- Tabel markdown di halaman Tentang melebar sampai 409px di layar 375px dan terpotong. Sekarang menggulir di tempat.
- Nilai dan delta metrik terpotong ("IDR 2...") di kolom sempit. Sekarang membungkus ke baris berikut.
- Kartu EWS memakai label "Siaga", sementara pesan engine menulis "WASPADA". Label sudah disamakan.
- "Skor risiko pasokan 0/100" tampil saat prakiraan gagal, padahal skor ini tidak bergantung pada prakiraan. Sekarang selalu dihitung.

## Lanjutan (permintaan kedua: "betulkan semuanya")

- **Emoji di keluaran engine dihapus.** Sumbernya `engine/chatbot_engine.py` (74 baris), `engine/ews_engine_v2.py` (17), dan `engine/price_narrative.py` (10). Pesan EWS tidak lagi diawali "🔸 WASPADA:", karena level sudah tampil di kartu. Narasi faktor memakai daftar biasa, tidak lagi titik warna. Panah ↑↓→ di tren narasi dihapus karena tanda +/- sudah menyampaikan arah.
- **Label tren pasokan** (`engine/supply_risk.py`) diganti dari "↑ Upward / ↓ Downward / → Stable / — Insufficient data" menjadi "Naik / Turun / Stabil / Data belum cukup". Label ini juga keluar lewat API `/api/data/supply-risk` dan PDF; `tests/test_supply_risk.py` sudah diperbarui.
- **Label "app" di navigasi** diganti "Beranda". `app.py` sekarang menjadi router `st.navigation`: menjalankan database dan scheduler, lalu halaman yang dipilih. Isi beranda dipindah ke `pages/0_Beranda.py`. URL halaman lain tetap sama. Akibatnya scheduler sinkronisasi PIHPS sekarang berjalan apa pun halaman yang pertama dibuka, tidak hanya dari beranda.
- **Tetap tidak diubah:** `calculate_warning_level` di `ews_engine_v2.py` (antarmuka v1, berbahasa Inggris, tidak tampil di UI, diuji oleh `tests/test_ews_engine*.py`) hanya dibersihkan emojinya. Tombol buka sidebar 28px adalah komponen bawaan Streamlit.

## Perbaikan error (permintaan ketiga)

**Environment Python 3.12 (Intel Mac, x86_64)**
- PyTorch 2.2.2 adalah versi terakhir untuk Intel Mac dan dikompilasi untuk NumPy 1.x. Karena itu NumPy diturunkan dari 2.5.1 ke **1.26.4**; ini juga menghilangkan peringatan SciPy (yang mensyaratkan NumPy < 2.3). Sebelum memasang, sudah dicek: tidak ada paket terpasang yang mensyaratkan NumPy 2 (kecuali pandas, dan itu hanya untuk Python 3.14).
- Paket dari `requirements.txt` yang belum terpasang kini dipasang: `fpdf2`, `openpyxl`, `fastapi`, `uvicorn`, `httpx`, `pytorch-forecasting` 1.8.0, `pytorch-lightning` 2.6.6, dengan torch, numpy, pandas, scipy, dan scikit-learn dikunci di versi sekarang. `pip check`: tidak ada konflik. `pytest` juga dipasang.

**Bug kode yang muncul setelah dependensi lengkap**
- `report/pdf_generator.py`: `FPDFException: Not enough horizontal space`. Di fpdf2 versi baru, `multi_cell` membiarkan kursor di sisi kanan, sehingga rekomendasi kedua tidak punya ruang. Kursor sekarang dikembalikan ke margin kiri.
- PDF: pesan EWS memakai `cell` sehingga teks panjang meluap ke luar halaman; sekarang `multi_cell`.
- PDF: "Perubahan Harga" selalu +0.00% karena EWS tidak pernah mengirim `pct_change`; sekarang dihitung dari harga.
- PDF: "Target Prediksi" berisi tanggal hari ini, padahal prakiraannya 30 hari ke depan. `pages/4_Laporan.py` sekarang memakai tanggal akhir prakiraan untuk EWS, narasi, dan PDF.
- PDF: interpretasi MAPE selalu "Baik" untuk semua nilai di atas 10%; sekarang mengikuti kategori Lewis (1982).
- PDF (antislop): kotak status putih-di-atas-kuning diganti warna status yang lolos AA; sampul gelap diganti terang dengan aksen hijau; label dan nama bulan berbahasa Indonesia; footer abu-abu 3:1 diganti 5:1.

## Bukti klik per elemen (browser, Streamlit 1.50)

| Elemen | Hasil |
|---|---|
| Beranda: 5 `page_link` | href ke halaman yang ada; "Laboratorium Model" diklik dan membuka `/Laboratorium_Model` |
| Beranda: metrik | Tanpa delta; nilai dari database (683,057 baris, 34 provinsi, 10 komoditas) |
| Analisis Regional | 4 grafik tampil, tanpa error, terang dan gelap |
| Tentang, mode gelap | Latar `#121613`, teks terang, tabel status memakai chip |
| Tentang, 375px | `scrollWidth` 375, tidak ada elemen terpotong; tombol buka sidebar terlihat |
| Laporan: Tab ke "Buat laporan PDF" | Ring fokus terlihat; Enter menjalankan tombol dan menampilkan pesan error yang jelas (`fpdf2` belum terpasang di Python ini) |
| Laporan: "Buat file Excel" | Pesan error yang jelas (`openpyxl` belum terpasang di Python ini) |
| Chatbot: tombol cepat "Tren cabai" | Pesan pengguna dan jawaban tampil |
| Dashboard, model default (ensemble) | Gagal karena environment (`RuntimeError: Numpy is not available` dari PyTorch); pesan error, kartu "Belum ada" tanpa skor, dan empty state grafik/metrik tampil |
| Dashboard, pilih "Prophet saja" | Kartu "Waspada 55/100", 4 metrik tidak terpotong, chip arah dan faktor tampil, kategori MAPE tampil |
| Tes (setelah perbaikan environment) | `pytest`: 149 lulus, 1 dilewati (tes yang hanya berjalan bila TFT *tidak* terpasang) |
| API: `/health`, `/api/data/supply-risk` | 200; `trend_direction: "Stabil"` |
| Laporan: "Buat laporan PDF" | "Laporan PDF siap diunduh", tombol unduh muncul, target 08-11-2026 |
| Laporan: "Buat file Excel" | "File Excel siap diunduh (65 baris)" |
| Dashboard, model default ensemble (setelah perbaikan environment) | Selesai dalam sekitar 15 menit pada pemuatan pertama; bobot Prophet 20%, BiLSTM 80%, TFT 0%; kartu "Normal 23/100"; 3 grafik; MAPE 11.81% (Baik); 0 error, 0 emoji |
| Navigasi | Label: Beranda, Dashboard, Analisis Regional, Laboratorium Model, Laporan, Tentang, Chatbot |
| Chatbot: "Bagaimana tren harga cabai merah?" | Jawaban tampil, 0 emoji |
| Dashboard, "Prophet saja" (setelah perubahan engine) | Kartu "Waspada 52/100", pesan tanpa awalan, caption "Tren harga 7 hari: stabil", rekomendasi tanpa emoji; 0 emoji, 0 em dash, 0 panah di seluruh halaman termasuk tab tersembunyi |

## Pembersihan akhir (requirements dan peringatan)

- `requirements.txt`: `numpy<2` khusus Intel Mac (`platform_machine == "x86_64"`), `streamlit>=1.56.0`, `httpx2` menggantikan `httpx` untuk TestClient Starlette 1.x, ditambah `pytest`.
- `apscheduler` dan `statsmodels` ternyata belum terpasang. Akibatnya selama ini sinkronisasi harian PIHPS diam-diam dilewati, dan ARIMA pembanding tidak tersedia. Keduanya sudah dipasang.
- `pytest.ini` membatasi koleksi ke `tests/`, sehingga skrip pelatihan `test_tft.py` di root tidak ikut berjalan.
- `api/schemas.py`: `Field(example=...)` diganti `examples=[...]` (Pydantic v2).
- `use_container_width=True` (22 tempat) diganti `width="stretch"`.
- Hasil: `pytest` 149 lulus, 1 dilewati, **0 peringatan**; AppTest semua halaman (lewat `app.py`), termasuk Dashboard ensemble: 0 exception, 0 error, 0 deprecation; `pip check` bersih; `pip install -r requirements.txt` tidak memasang apa pun lagi.
