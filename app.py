import os
import re
import sys
import string
import socket
import logging
from flask import Flask, render_template, request, jsonify
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Konfigurasi Logging & Flask ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = Flask(__name__)

def get_local_ip() -> str:
    """Deteksi alamat IP LAN komputer agar bisa diakses dari HP / smartphone dalam satu Wi-Fi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

# --- Sastrawi Stemmer & Stopwords Setup ---
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    
    stemmer = StemmerFactory().create_stemmer()
    stopwords_set = set(StopWordRemoverFactory().get_stop_words())
    # Stopwords tambahan khusus percakapan/tanya jawab umum agar kata bantu tidak mendominasi TF-IDF
    custom_sw = {
        'hari', 'ini', 'itu', 'berapa', 'apa', 'apakah', 'siapakah', 'kenapa', 
        'mengapa', 'bagaimana', 'bisa', 'tolong', 'mohon', 'ada', 'dan', 'atau', 
        'pada', 'di', 'ke', 'dari', 'kah', 'lah'
    }
    stopwords_set.update(custom_sw)
except Exception as e:
    stemmer = None
    stopwords_set = set()
    logging.warning(f"Sastrawi tidak aktif ({e}). Menggunakan pemrosesan teks standar.")

# --- Pastikan NLTK Resources Siap ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "corpus_perusahaan.txt")

DEFAULT_CORPUS = """Jam kerja operasional kantor perusahaan dimulai pukul 08.00 WIB dan berakhir pukul 17.00 WIB setiap hari kerja Senin sampai Jumat.
Waktu istirahat kerja karyawan adalah pukul 12.00 WIB sampai 13.00 WIB pada hari Senin hingga Kamis, serta pukul 11.30 WIB sampai 13.00 WIB pada hari Jumat untuk sholat Jumat.
Karyawan wajib melakukan presensi kehadiran (clock-in dan clock-out) menggunakan mesin sidik jari atau aplikasi mobile absensi saat datang dan pulang kerja.
Keterlambatan masuk kerja lebih dari 15 menit tanpa pemberitahuan resmi akan dikenakan sanksi pemotongan tunjangan kehadiran sebesar 5% per kejadian.
Ketentuan pakaian seragam kerja hari Senin sampai Kamis adalah kemeja rapi, celana/rok bahan formal atau seragam kantor resmi.
Ketentuan pakaian kerja hari Jumat adalah baju batik nusantara atau pakaian smart casual yang sopan, rapi, dan bersepatu tertutup.
Hak cuti tahunan diberikan sebanyak 12 hari kerja per tahun setelah karyawan bekerja minimal selama 1 tahun berturut-turut.
Pengajuan permohonan cuti tahunan wajib diajukan minimal 3 hari sebelum tanggal cuti melalui sistem aplikasi HRIS perusahaan.
Izin tidak masuk kerja karena sakit wajib disertai dengan surat keterangan dokter resmi dan diserahkan kepada HRD maksimal 2 hari kerja setelah masuk kembali.
Karyawan perempuan berhak mengambil cuti melahirkan selama 3 bulan kalender dengan tetap memperoleh hak upah penuh.
Karyawan berhak atas izin cuti khusus berbayar, yaitu cuti menikah selama 3 hari kerja dan cuti duka cita keluarga inti selama 2 hari kerja.
Pekerjaan lembur (overtime) hanya sah dan dihitung jika ada formulir Surat Perintah Lembur (SPL) resmi yang disetujui atasan langsung sebelum lembur dilaksanakan.
Jadwal pembayaran gaji dan transfer slip gaji karyawan (gajian) dilakukan secara rutin setiap tanggal 27 setiap bulannya via transfer bank.
Tunjangan Hari Raya (THR) keagamaan wajib dibayarkan oleh perusahaan paling lambat 7 hari kalender (H-7) sebelum hari raya keagamaan karyawan.
Perusahaan memberikan bonus tahunan pada setiap akhir tahun kepada karyawan yang mencapai dan melampaui target Key Performance Indicator (KPI).
Semua karyawan berhak didaftarkan ke dalam jaminan sosial ketenagakerjaan dan kesehatan melalui BPJS Ketenagakerjaan dan BPJS Kesehatan.
Dilarang keras merokok, vape, atau rokok elektrik di dalam seluruh ruangan gedung kantor, merokok hanya diperbolehkan di area luar khusus merokok (smoking area).
Seluruh karyawan wajib menjaga kerahasiaan data perusahaan (NDA), dilarang membocorkan rahasia dagang, strategi bisnis, atau data pelanggan kepada pihak luar.
Prosedur pengunduran diri (resign kerja) wajib menyampaikan surat resign tertulis minimal 30 hari sebelumnya (one month notice) kepada atasan dan HRD.
Evaluasi penilaian kinerja karyawan (KPI / Key Performance Indicator) dilaksanakan secara berkala oleh manajemen setiap 6 bulan sekali.
Fasilitas penggantian biaya kacamata (klaim optik) dapat diajukan oleh karyawan tetap setahun sekali dengan plafon maksimal sesuai golongan jabatan."""

# Variabel State Global
current_corpus = ""
vectorizer = None
tfidf_matrix = None
sentences = []

def clean_and_stem(text: str) -> str:
    """Bersihkan teks, hilangkan tanda baca, filter stopwords, dan lakukan stemming Sastrawi."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [w for w in text.split() if w and w not in stopwords_set]
    
    if not words:
        words = text.split()

    if stemmer:
        try:
            return " ".join([stemmer.stem(w) for w in words])
        except Exception:
            return " ".join(words)
    return " ".join(words)

def load_or_init_corpus() -> str:
    """Muat korpus dari file lokal; jika belum ada, buat dengan korpus default."""
    if os.path.exists(CORPUS_FILE):
        try:
            with open(CORPUS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception as e:
            logging.error(f"Gagal membaca {CORPUS_FILE}: {e}")
    
    try:
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CORPUS)
    except Exception as e:
        logging.error(f"Gagal menulis {CORPUS_FILE}: {e}")
    return DEFAULT_CORPUS

def train_model():
    """Latih ulang model TF-IDF berdasarkan isi korpus saat ini."""
    global vectorizer, tfidf_matrix, sentences, current_corpus
    
    raw_lines = [line.strip() for line in current_corpus.split('\n') if line.strip()]
    if not raw_lines:
        sentences = [DEFAULT_CORPUS.split('\n')[0]]
    else:
        sentences = raw_lines

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(),
        preprocessor=clean_and_stem,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)
    logging.info(f"Model berhasil dilatih dengan {len(sentences)} kalimat aturan perusahaan.")

def check_intent_response(user_input: str):
    """Mendeteksi sapaan, ucapan terima kasih, dan identitas bot sebelum kalkulasi TF-IDF."""
    clean_input = user_input.lower().strip()
    clean_no_punct = clean_input.translate(str.maketrans('', '', string.punctuation))

    # 1. Sapaan (Greetings)
    greetings = ['halo', 'hai', 'hei', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'assalamualaikum', 'assalamu alaikum', 'assalamu\'alaikum', 'hi', 'hello']
    for g in greetings:
        if re.search(r'\b' + re.escape(g) + r'\b', clean_no_punct):
            return "Halo! Selamat datang di Portal Tanya Jawab HR Perusahaan. Ada aturan kantor, jadwal gajian, atau hak karyawan yang ingin Anda ketahui?"

    # 2. Ucapan Terima Kasih (Gratitude)
    thanks = ['terima kasih', 'makasih', 'terimakasih', 'thanks', 'thank you', 'tengkyu', 'matur nuwun']
    for t in thanks:
        if re.search(r'\b' + re.escape(t) + r'\b', clean_no_punct):
            return "Sama-sama! Senang bisa membantu Anda. Jika masih ada yang ingin ditanyakan terkait peraturan kerja, silakan ketik langsung."

    # 3. Identitas Bot
    bot_identity = ['kamu siapa', 'siapa kamu', 'siapa anda', 'apa ini', 'bantuan', 'apa yang bisa kamu lakukan', 'fungsi bot']
    for bi in bot_identity:
        if bi in clean_no_punct:
            return "Saya adalah <strong>HR Virtual Assistant</strong> yang siap membantu menjawab pertanyaan seputar kebijakan perusahaan, hak cuti, jam kerja, seragam, jadwal gaji, tunjangan, dan prosedur kantor."

    # 4. Pamitan (Goodbyes)
    goodbyes = ['dadah', 'selamat tinggal', 'bye', 'sampai jumpa', 'good bye']
    for gb in goodbyes:
        if re.search(r'\b' + re.escape(gb) + r'\b', clean_no_punct):
            return "Sampai jumpa! Selamat melanjutkan pekerjaan dan tetap semangat!"

    return None

# --- Inisialisasi Model Awal ---
current_corpus = load_or_init_corpus()
train_model()

# --- Endpoint Routing ---

@app.route('/')
def home():
    lan_ip = get_local_ip()
    port = int(os.environ.get("PORT", 5000))
    return render_template(
        'index.html',
        corpus=current_corpus,
        total_rules=len(sentences),
        lan_ip=lan_ip,
        port=port
    )

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    data = request.get_json(silent=True) or {}
    user_input = data.get('msg', '').strip()

    if not user_input:
        return jsonify({
            'response': 'Mohon masukkan pertanyaan Anda.',
            'confidence': 0.0,
            'matched_rule': None
        })

    # 1. Cek Sapaan & Intent
    intent_reply = check_intent_response(user_input)
    if intent_reply:
        return jsonify({
            'response': intent_reply,
            'confidence': 1.0,
            'matched_rule': 'Intent Handler'
        })

    # 2. Perhitungan Kemiripan Kosinus (TF-IDF Cosine Similarity)
    try:
        user_vec = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, tfidf_matrix)[0]
        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        # Ambang Batas Kemiripan Optimal
        SIMILARITY_THRESHOLD = 0.18

        if best_score < SIMILARITY_THRESHOLD:
            response = (
                "Maaf, saya belum menemukan aturan perusahaan yang sesuai dengan pertanyaan tersebut.<br><br>"
                "💡 <strong>Saran Topik Populer:</strong><br>"
                "• <em>Jam kerja & waktu istirahat</em><br>"
                "• <em>Jadwal gajian & THR</em><br>"
                "• <em>Aturan pakaian kerja / batik</em><br>"
                "• <em>Cuti tahunan & izin sakit</em><br>"
                "• <em>Sanksi keterlambatan & lembur (SPL)</em><br>"
                "• <em>Prosedur resign (one month notice)</em>"
            )
            return jsonify({
                'response': response,
                'confidence': round(best_score, 2),
                'matched_rule': None
            })

        matched_rule = sentences[best_idx]
        return jsonify({
            'response': f"📌 <strong>Ketentuan Resmi Perusahaan:</strong><br>{matched_rule}",
            'confidence': round(best_score, 2),
            'matched_rule': matched_rule
        })

    except Exception as e:
        logging.error(f"Error processing response: {e}")
        return jsonify({
            'response': 'Terjadi kesalahan sistem dalam memproses jawaban. Silakan coba beberapa saat lagi.',
            'confidence': 0.0,
            'matched_rule': None
        }), 500

@app.route('/update_model', methods=['POST'])
def update_model_route():
    global current_corpus
    data = request.get_json(silent=True) or {}
    new_corpus = data.get('corpus', '').strip()

    if not new_corpus:
        return jsonify({'success': False, 'error': 'Database aturan tidak boleh kosong.'}), 400

    try:
        current_corpus = new_corpus
        with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
            f.write(current_corpus)
        train_model()
        return jsonify({
            'success': True,
            'message': 'Database peraturan berhasil disimpan dan model telah diperbarui!',
            'total_rules': len(sentences)
        })
    except Exception as e:
        logging.error(f"Gagal memperbarui model: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reset_corpus', methods=['POST'])
def reset_corpus_route():
    global current_corpus
    try:
        current_corpus = DEFAULT_CORPUS
        with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_CORPUS)
        train_model()
        return jsonify({
            'success': True,
            'corpus': DEFAULT_CORPUS,
            'total_rules': len(sentences)
        })
    except Exception as e:
        logging.error(f"Gagal reset corpus: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    lan_ip = get_local_ip()
    port = int(os.environ.get("PORT", 5000))
    return jsonify({
        'total_rules': len(sentences),
        'total_vocabulary': len(vectorizer.vocabulary_) if vectorizer else 0,
        'lan_ip': lan_ip,
        'port': port,
        'mobile_url': f"http://{lan_ip}:{port}",
        'status': 'online'
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    lan_ip = get_local_ip()
    print(f"\n========================================================")
    print(f"🚀 HR Chatbot & Portal Peraturan Perusahaan Aktif!")
    print(f"💻 Akses di Laptop/PC : http://127.0.0.1:{port}")
    print(f"📱 Akses di HP/Mobile : http://{lan_ip}:{port}")
    print(f"========================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
