# Petunjuk revisi (masukan Bu Eneng, 21 Oktober 2026)

## 1. Pasang kode baru
Salin isi folder `kode/` ke repositori (menimpa berkas lama), atau dari folder repositori jalankan:
    git apply revisi.patch

Berkas kode:
- BARU: `models/forecast_ahead.py`, `scripts/prakiraan_ke_depan.py`, `scripts/analisis_parameter.py`, `scripts/analisis_slr.py`
- DIUBAH: `pages/1_Dashboard.py` (tab "Prakiraan periodik") dan `models/evaluation_protocol.py`
  (perbaikan Naive Seasonal untuk periode < 30 hari; hasil Bab IV yang memakai periode 365 hari tidak berubah)

## 2. Prakiraan ke depan (revisi 2) → Subbab 4.9
    python scripts/prakiraan_ke_depan.py
- Memakai data terbaru di database sistem dan bobot ensemble dari `thesis_outputs/4_2_bobot_ensemble.json`.
- Butuh internet untuk kovariat iklim.
- Hasil di `prakiraan_outputs/`: isi Tabel 4.9–4.10 dari `ringkasan_prakiraan.md` dan sisipkan `gambar_prakiraan_ke_depan.png` sebagai Gambar 4.4.
- Deploy kode ke Streamlit, buka Dashboard, lalu tab **Prakiraan periodik**. Ambil tangkapan layarnya untuk Gambar 4.5.

## 3. Analisis sensitivitas parameter (revisi 3) → Subbab 4.10 dan 4.11.1
    python scripts/analisis_parameter.py
- Waktu jalan sekitar 45–60 menit.
- Jalankan di komputer yang sama dengan evaluasi Bab IV, supaya baris bawaan sama dengan Tabel 4.2.
- Hasil di `parameter_outputs/`: isi Tabel 4.11 dari `ringkasan_parameter.md`, sisipkan `gambar_sensitivitas_parameter.png` sebagai Gambar 4.6, lalu lengkapi kalimat [ISI ...] di 4.11.1.

## 4. SLR berbasis jaringan (revisi 1) → Subbab 2.3
Sudah terisi lengkap. Untuk menjalankan ulang (misalnya setelah daftar pustaka berubah):
    python scripts/analisis_slr.py --naskah Fahmi_Prasanda_Seminar_Hasil_revisi_dosen.docx --output slr_outputs

## 5. Yang perlu dicek
- Tiga rujukan baru: Clauset dkk. (2004), Donthu dkk. (2021), van Eck & Waltman (2010).
- Buka dan ekspor ke PDF dengan Microsoft Word, karena LibreOffice salah menomori tabel dan gambar per bab.
