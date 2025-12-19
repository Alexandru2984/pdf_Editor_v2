# PDF Text Editor

O aplicație Django care permite editarea textului în fișiere PDF, **păstrând layout-ul original** al documentului (paginare, poziții, fonturi, imagini).

## ⚠️ Ce Face Această Aplicație

- ✅ Găsește și înlocuiește text în PDF-uri existente
- ✅ Păstrează formatul original (layout, fonturi, imagini, paginare)
- ✅ Suportă căutare case-sensitive și page range selection
- ✅ Vizualizare rezultate cu număr de înlocuiri
- ✅ Download PDF modificat

## ❌ Ce NU Face

- Nu poate edita PDF-uri scanate fără layer de text OCR (doar imagini)
- Nu garantează 100% păstrarea layout-ului în cazuri extreme (fonturi foarte speciale, text rotit complex)

## 📋 Cerințe

- Python 3.8+
- pip și virtualenv

## 🚀 Instalare și Pornire

### 1. Clonează/Descarcă proiectul

```bash
cd pdf_Editor
```

### 2. Creează virtual environment și instalează dependințele

```bash
# Creează virtual environment
python3 -m venv venv

# Activează virtual environment
source venv/bin/activate  # Linux/Mac
# SAU
venv\Scripts\activate  # Windows

# Instalează dependințele
pip install -r requirements.txt
```

### 3. Rulează migrațiile

```bash
python manage.py migrate
```

### 4. Pornește serverul de development

```bash
python manage.py runserver
```

### 5. Accesează aplicația

Deschide browser-ul la: **http://localhost:8000/**

## 📖 Cum se Folosește - Exemplu Workflow

### Pas 1: Upload PDF
1. Accesează pagina principală
2. Click pe "Alege fișier PDF"
3. Selectează un PDF (ex: `document.pdf`)
4. Click "Încarcă și continuă"

### Pas 2: Găsește și Înlocuiește
1. Introdu textul de căutat (ex: `"test"`)
2. Introdu textul nou (ex: `"exemplu"`)
3. (Opțional) Bifează/Debifează "Case sensitive"
4. (Opțional) Specifică interval de pagini (ex: `"1-3,5"` sau lasă gol pentru toate)
5. Click "Aplică modificările"

### Pas 3: Descarcă Rezultatul
1. Vezi câte înlocuiri s-au făcut
2. Verifică warnings (dacă există)
3. Click "Descarcă PDF modificat"
4. Salvează fișierul modificat

## 🔧 Tehnologie Folosită

- **Backend**: Django 4.2
- **PDF Library**: PyMuPDF (fitz) 1.23+
- **Approach**: Modificare directă a content stream-urilor PDF, **NU** regenerare PDF de la zero

## ⚠️ Limitări și Cazuri Speciale

### 1. PDF-uri Scanate (Doar Imagini)
**Problemă**: Dacă PDF-ul conține doar imagini scanate (fără text selectabil), nu se poate modifica textul.

**Detectare**: Aplicația detectează automat și afișează un warning.

**Soluție**: Folosește un tool OCR pentru a adăuga layer de text sau editează manual în Adobe Acrobat.

### 2. Text Fragmentat
**Problemă**: În unele PDF-uri, textul este stocat character-by-character (ex: "t"+"e"+"s"+"t" în loc de "test").

**Comportament**: PyMuPDF reunește automat fragmentele la căutare, dar înlocuirea poate avea rezultate variabile.

**Recomandare**: Testează pe un PDF de probă mai întâi.

### 3. Fonturi Embedded Speciale
**Problemă**: Dacă PDF-ul folosește fonturi embedded cu subset limitat de caractere, textul nou poate să nu se afișeze corect.

**Comportament**: Aplicația încearcă să păstreze fontul original, dar face fallback la Helvetica dacă nu reușește.

**Warnings**: Vei vedea un warning în pagina de rezultat dacă apar probleme.

### 4. Text Rotit sau cu Transformări Complexe
**Problemă**: Text cu rotații complexe sau transformări matriciale avansate.

**Comportament**: Înlocuirea funcționează pentru text normal, dar poate avea probleme cu text foarte rotit/distorsionat.

## 🧹 Gestionarea Fișierelor Temporare

### Cleanup Automat (Recomandat)

Aplicația stochează fișierele în `media/uploads/` și `media/processed/`. Pentru a șterge fișierele vechi automat:

```bash
# Șterge fișiere mai vechi de 24h (default)
python manage.py cleanup_old_pdfs

# Șterge fișiere mai vechi de 6 ore
python manage.py cleanup_old_pdfs --hours 6
```

### Cron Job (Producție)

Pentru a rula cleanup automat în producție, adaugă în crontab:

```bash
# Rulează cleanup la fiecare 6 ore
0 */6 * * * cd /path/to/pdf_Editor && ./venv/bin/python manage.py cleanup_old_pdfs
```

### Configurare

Poți ajusta timpul de cleanup în `pdf_project/settings.py`:

```python
PDF_CLEANUP_HOURS = 24  # Schimbă cu valoarea dorită
```

## 🧪 Teste

Aplicația include teste pentru:
- Procesare PDF (find & replace, page range parsing, text detection)
- Views (upload, edit, result, download)
- Workflow complet end-to-end

### Rulare Teste

```bash
# Rulează toate testele
python manage.py test pdfeditor

# Rulează cu verbose output
python manage.py test pdfeditor -v 2

# Rulează un test specific
python manage.py test pdfeditor.tests.PDFProcessorTests.test_find_and_replace_basic
```

## 📁 Structura Proiectului

```
pdf_Editor/
├── venv/                          # Virtual environment
├── pdf_project/                   # Django project
│   ├── settings.py                # Configurări (MEDIA_ROOT, etc.)
│   └── urls.py                    # URL routing principal
├── pdfeditor/                     # Django app
│   ├── views.py                   # Views pentru upload/edit/result/download
│   ├── forms.py                   # FindReplaceForm
│   ├── pdf_processor.py           # Core logic PyMuPDF (TEXT REPLACEMENT)
│   ├── urls.py                    # URL routing app
│   ├── templates/pdfeditor/       # HTML templates
│   ├── tests.py                   # Unit tests
│   └── management/commands/
│       └── cleanup_old_pdfs.py    # Cleanup command
├── media/
│   ├── uploads/                   # PDF-uri urcate
│   └── processed/                 # PDF-uri modificate (temporar)
├── static/css/
│   └── style.css                  # Styling simplu
├── requirements.txt               # Django + PyMuPDF
├── README.md                      # Acest fișier
└── manage.py
```

## 🔍 Cum Funcționează Tehnic

### PyMuPDF Approach

1. **Deschide PDF-ul original** cu `fitz.open()`
2. **Caută textul** cu `page.search_for(search_text)`
3. **Pentru fiecare match găsit:**
   - Extrage informații despre font (nume, dimensiune, culoare)
   - Adaugă redaction annotation (șterge textul vechi cu alb)
   - Aplică redaction: `page.apply_redactions()`
   - Inserează textul nou în aceeași poziție cu același font
4. **Salvează PDF-ul modificat** optimizat

### De Ce NU Regenerăm PDF-ul?

❌ **Abordare greșită**: Extrage tot textul → Pune-l într-un template nou → Generează PDF nou
- Pierderi: pozițiile exacte, fonturile originale, imaginile, layout-ul complex

✅ **Abordare corectă**: Modifică direct content stream-ul PDF-ului
- Păstrează: totul intact, doar textul specificat este înlocuit

## 💡 Tips & Best Practices

1. **Testează pe o copie mai întâi** - Nu edita direct PDF-ul important
2. **Verifică rezultatul vizual** - Deschide PDF-ul modificat și verifică layout-ul
3. **Folosește preview înainte de aplicare** - Asigură-te că search text-ul e corect
4. **Page range** - Dacă știi exact paginile, specifică-le pentru performanță mai bună
5. **Case sensitive** - Activează dacă vrei exactitate maximă

## 🐛 Troubleshooting

### "PDF-ul nu conține text selectabil"
- PDF-ul este un scan - folosește OCR sau editează manual
- Verifică dacă poți selecta text cu mouse-ul în Adobe Reader

### "Nu s-a putut insera textul pe pagina X"
- Font incompatibil - PDF-ul folosește un font special care nu acceptă caracterul nou
- Solution: Încearcă text diferit sau editează manual pagina respectivă

### Fișierele procesate ocupă prea mult spațiu
- Rulează: `python manage.py cleanup_old_pdfs`
- Configurează un cron job pentru cleanup automat

## 📝 Licență

Acest proiect este open-source și disponibil pentru uz personal și educational.

## 🙏 Credite

- **PyMuPDF**: https://pymupdf.readthedocs.io/
- **Django**: https://www.djangoproject.com/

---

**Made with ❤️ using Django + PyMuPDF**
