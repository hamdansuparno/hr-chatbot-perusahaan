# 🏢 HR Chatbot & Sistem Manajemen Peraturan Perusahaan (Web & Mobile)

Aplikasi web cerdas berbasis **Flask** dan **Natural Language Processing (NLP)** yang dirancang untuk menjawab pertanyaan karyawan seputar peraturan, kebijakan, hak, dan kewajiban di perusahaan secara interaktif, baik melalui **Laptop/Komputer** maupun **HP / Smartphone**.

---

## 📱 Akses Melalui HP / Smartphone

Aplikasi ini dapat langsung diakses dari perangkat HP (Android / iOS) yang terhubung ke jaringan Wi-Fi yang sama:

1. **Alamat URL Langsung di HP:**
   ```
   http://192.168.110.62:5000
   ```
2. **Scan QR Code:**
   - Di tampilan komputer, klik tombol **`📱 Buka di HP`** pada bilah atas (*header*).
   - Arahkan kamera smartphone ke kode QR yang muncul di layar.
3. **Jadikan Aplikasi Mandiri (Add to Home screen):**
   - Di peramban Chrome (Android) atau Safari (iPhone), buka menu opsi dan pilih **"Tambahkan ke Layar Utama" (Add to Home screen)**.
   - Aplikasi akan tampil dengan ikon sendiri di layar HP dan dapat dibuka layar penuh (*full screen*) seperti aplikasi bawaan.

---

## 🚀 Fitur Utama & Penyempurnaan

1. **Tampilan Responsif Mobile & Desktop:**
   - **Mode HP:** Tampilan percakapan penuh yang nyaman dengan sentuhan jempol, keyboard-friendly (*cegah auto-zoom*), dan panel admin yang dapat dibuka/tutup menggunakan menu laci (*Offcanvas Drawer*).
   - **Mode Desktop:** Menyediakan 3 pilihan mode tampilan (*Split View, Chat Penuh, Admin Penuh*).
2. **NLP Bahasa Indonesia (Sastrawi + TF-IDF + N-Gram):**
   - Dilengkapi *stemming* otomatis dan filter *stopwords* untuk mengenali ragam pertanyaan karyawan secara akurat.
3. **Database Aturan Permanen:**
   - Aturan disimpan di `corpus_perusahaan.txt` dan dapat diperbarui *real-time* melalui panel Admin.
4. **Intent & Voice Assistant:**
   - Menjawab sapaan dan ucapan terima kasih dengan ramah serta dilengkapi fitur pembacaan suara (*Text-to-Speech*).

---

## 📁 Struktur Berkas

```
d:\DATA MAINING\PERUSAHAAN\
├── app.py                     # Backend server Flask & NLP Engine
├── corpus_perusahaan.txt      # Berkas korpus aturan perusahaan
├── requirements.txt           # Daftar dependensi Python
├── run.bat                    # Script batch 1-klik untuk menjalankan di Windows
├── static/
│   └── manifest.json          # Konfigurasi PWA untuk instalasi di HP
├── templates/
│   └── index.html             # Antarmuka responsif (Mobile Drawer + Desktop)
└── .venv/                     # Virtual environment Python lokal
```

---

## ⚡ Cara Menjalankan Aplikasi

Cukup klik dua kali (double-click) pada:
```
run.bat
```
Lalu scan QR Code atau akses:
- **Di Laptop/PC:** `http://127.0.0.1:5000`
- **Di HP/Mobile:** `http://192.168.110.62:5000`
