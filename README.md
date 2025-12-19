# PDF Text Editor & AI Tools

O aplicație Django completă pentru manipularea fișierelor PDF, incluzând editare text, **AI Rephrasing**, și utilitare avansate (Split, Merge, Compress, etc.).

## 🌟 Funcționalități Principale

### 1. Editare Text & AI Rephrase
- **Find & Replace**: Căutare și înlocuire text în tot documentul.
- **AI Rephrase (Ollama)**: Selectează o zonă cu mouse-ul și cere AI-ului să reformuleze textul.
  - **Safe Mode**: Înlocuiește textul doar în cutia selectată (micșorează fontul dacă e nevoie).
  - **Flow Mode**: Reformulează și **rearanjează** textul în pagină (mută conținutul de dedesubt, extinde pagina dacă e necesar).
  - **Păstrare Stil**: Încearcă să mențină fontul și alinierea originală (Left/Right/Justify/Center).

### 2. Utilitare PDF
- **Split PDF**: Împarte un PDF în mai multe fișiere (după pagini sau intervale).
- **Merge PDF**: Unește mai multe PDF-uri într-unul singur.
- **Rotate Pages**: Rotește paginile (90, 180, 270 grade).
- **Watermark**: Adaugă watermark text sau imagine (cu transparență).
- **Page Numbers**: Adaugă numerotare pagini (poziționare customizabilă).
- **Compress PDF**: Reduce dimensiunea fișierului (optimizare imagini).

### 3. OCR & Conversie
- **OCR to Text**: Extrage text din PDF-uri scanate (folosind Tesseract).
- **Preview**: Vizualizare PDF în browser înainte și după modificare.

## 📋 Cerințe

- Python 3.8+
- **Ollama** (pentru AI Rephrase) - trebuie să ruleze local sau pe un server accesibil.
- **Tesseract OCR** (pentru funcția OCR).

## 🚀 Instalare și Pornire

### 1. Clonează proiectul
```bash
git clone https://github.com/Alexandru2984/pdf_Editor_v2.git
cd pdf_Editor_v2
```

### 2. Setup Mediu Virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# sau venv\Scripts\activate pe Windows
pip install -r requirements.txt
```

### 3. Configurare .env
Creează un fișier `.env` în rădăcina proiectului:
```env
DEBUG=True
SECRET_KEY='cheia-ta-secreta'
ALLOWED_HOSTS=localhost,127.0.0.1
# OLLAMA_HOST=http://localhost:11434 (opțional)
```

### 4. Pornire Server
```bash
python manage.py migrate
python manage.py runserver
```
Accesează: **http://localhost:8000/**

## 🔧 Detalii Tehnice

- **Backend**: Django 4.2
- **PDF Engine**: PyMuPDF (fitz) - manipulare directă a stream-urilor PDF.
- **AI**: Ollama (Llama 3, Mistral, etc.) via API.
- **Frontend**: HTML5, CSS3, JavaScript (PDF.js pentru selecție vizuală).

## ⚠️ Limitări Cunoscute (Flow Mode)

- **Bullet Points**: Elementele grafice (buline, linii) nu sunt mutate automat în Flow Mode momentan.
- **Layout Complex**: Tabelele sau layout-urile multi-coloană complexe pot suferi modificări nedorite la reflow.

## 🧹 Cleanup

Fișierele temporare sunt șterse automat printr-o comandă de management:
```bash
python manage.py cleanup_old_pdfs --hours 24
```

---
**Made with ❤️ using Django + PyMuPDF + Ollama**
