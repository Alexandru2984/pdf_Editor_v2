"""Public, bilingual content for crawlable PDF tool landing pages.

These pages explain the real operation and its limitations before a visitor
uploads anything.  Product views remain separate and session-scoped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalizedText:
    en: str
    ro: str

    def get(self, language: str) -> str:
        return self.ro if language == "ro" else self.en


@dataclass(frozen=True, slots=True)
class LocalizedFaq:
    question: LocalizedText
    answer: LocalizedText


@dataclass(frozen=True, slots=True)
class ToolLanding:
    key: str
    slug: str
    operation_url_name: str
    icon: str
    name: LocalizedText
    category: LocalizedText
    short_description: LocalizedText
    seo_title: LocalizedText
    seo_description: LocalizedText
    intro: LocalizedText
    highlights: tuple[LocalizedText, ...]
    steps: tuple[LocalizedText, ...]
    note: LocalizedText
    faqs: tuple[LocalizedFaq, ...]
    multiple_upload: bool = False

    @property
    def url_name(self) -> str:
        return f"tool_{self.key}"

    def localized_url_name(self, language: str) -> str:
        """Return the canonical route name for one supported locale."""

        if language not in {"en", "ro"}:
            raise ValueError(f"Unsupported marketing language: {language}")
        return f"{self.url_name}_{language}"

    def localized(self, language: str) -> dict[str, object]:
        return {
            "key": self.key,
            "slug": self.slug,
            "url_name": self.url_name,
            "operation_url_name": self.operation_url_name,
            "icon": self.icon,
            "name": self.name.get(language),
            "category": self.category.get(language),
            "short_description": self.short_description.get(language),
            "seo_title": self.seo_title.get(language),
            "seo_description": self.seo_description.get(language),
            "intro": self.intro.get(language),
            "highlights": [item.get(language) for item in self.highlights],
            "steps": [item.get(language) for item in self.steps],
            "note": self.note.get(language),
            "faqs": [
                {
                    "question": faq.question.get(language),
                    "answer": faq.answer.get(language),
                }
                for faq in self.faqs
            ],
            "multiple_upload": self.multiple_upload,
        }


@dataclass(frozen=True, slots=True)
class DirectToolLink:
    """A crawlable product page that is already usable without a PDF upload."""

    url_name: str
    icon: str
    name: LocalizedText
    category: LocalizedText
    short_description: LocalizedText

    def localized_url_name(self, language: str) -> str:
        if language not in {"en", "ro"}:
            raise ValueError(f"Unsupported marketing language: {language}")
        return f"{self.url_name}_{language}"

    def localized(self, language: str) -> dict[str, object]:
        return {
            "url_name": self.url_name,
            "icon": self.icon,
            "name": self.name.get(language),
            "category": self.category.get(language),
            "short_description": self.short_description.get(language),
        }


def text(en: str, ro: str) -> LocalizedText:
    return LocalizedText(en=en, ro=ro)


def faq(question_en: str, question_ro: str, answer_en: str, answer_ro: str) -> LocalizedFaq:
    return LocalizedFaq(
        question=text(question_en, question_ro),
        answer=text(answer_en, answer_ro),
    )


TOOL_LANDINGS: tuple[ToolLanding, ...] = (
    ToolLanding(
        key="merge_pdf",
        slug="merge-pdf",
        operation_url_name="merge",
        icon="🔗",
        name=text("Merge PDF", "Unește PDF-uri"),
        category=text("Organize", "Organizare"),
        short_description=text(
            "Combine several PDF files into one document in the order you choose.",
            "Combină mai multe fișiere PDF într-un singur document, în ordinea aleasă.",
        ),
        seo_title=text(
            "Merge PDF Files Online | PDF Platform",
            "Unește PDF-uri online | PDF Platform",
        ),
        seo_description=text(
            "Merge two or more PDF files online, arrange them in the right order and download one combined document. No account required.",
            "Unește online două sau mai multe PDF-uri, aranjează-le în ordinea dorită și descarcă un singur document. Fără cont obligatoriu.",
        ),
        intro=text(
            "Put contracts, reports, scans or chapters into one clean PDF. Upload all source files together, then select and arrange their order before creating the result.",
            "Adună contracte, rapoarte, scanări sau capitole într-un singur PDF. Încarcă toate fișierele, apoi selectează și stabilește ordinea lor înainte de generare.",
        ),
        highlights=(
            text(
                "Upload multiple PDFs in one selection.",
                "Încarci mai multe PDF-uri dintr-o singură selecție.",
            ),
            text(
                "Choose the exact merge order before processing.",
                "Alegi ordinea exactă înainte de procesare.",
            ),
            text(
                "Download one combined PDF when the job is complete.",
                "Descarci un singur PDF combinat la final.",
            ),
        ),
        steps=(
            text("Choose at least two PDF files below.", "Alege cel puțin două fișiere PDF mai jos."),
            text(
                "Select and arrange the files on the merge screen.",
                "Selectează și ordonează fișierele în ecranul de unire.",
            ),
            text(
                "Merge, review and download the new document.", "Unește, verifică și descarcă documentul nou."
            ),
        ),
        note=text(
            "Merging a digitally signed PDF changes the document and invalidates its existing signature. Review interactive forms and bookmarks in the output as well.",
            "Unirea unui PDF semnat digital modifică documentul și invalidează semnătura existentă. Verifică și formularele interactive sau semnele de carte din rezultat.",
        ),
        faqs=(
            faq(
                "How many PDFs do I need?",
                "De câte PDF-uri am nevoie?",
                "At least two. You can upload several files at once, subject to the per-file upload limit.",
                "Cel puțin două. Poți încărca mai multe simultan, în limita stabilită pentru fiecare fișier.",
            ),
            faq(
                "Can I change their order?",
                "Pot schimba ordinea lor?",
                "Yes. The merge screen lets you arrange the selected documents before creating the output.",
                "Da. Ecranul de unire îți permite să aranjezi documentele selectate înainte de generare.",
            ),
        ),
        multiple_upload=True,
    ),
    ToolLanding(
        key="split_pdf",
        slug="split-pdf",
        operation_url_name="split",
        icon="✂️",
        name=text("Split PDF", "Separă PDF"),
        category=text("Organize", "Organizare"),
        short_description=text(
            "Extract page ranges into one or several smaller PDF files.",
            "Extrage intervale de pagini în unul sau mai multe PDF-uri mai mici.",
        ),
        seo_title=text(
            "Split PDF by Page Range Online | PDF Platform", "Separă PDF pe pagini online | PDF Platform"
        ),
        seo_description=text(
            "Split a PDF into separate files using page ranges such as 1-5, 8 or 10-14. Create exactly the parts you need online.",
            "Separă un PDF în fișiere distincte folosind intervale precum 1-5, 8 sau 10-14. Generează online exact părțile de care ai nevoie.",
        ),
        intro=text(
            "Create a chapter, extract a signed appendix or divide a large scan without rebuilding the document manually. Every range becomes a separate PDF.",
            "Creează un capitol, extrage o anexă sau împarte o scanare mare fără să reconstruiești manual documentul. Fiecare interval devine un PDF separat.",
        ),
        highlights=(
            text(
                "Supports single pages and inclusive page ranges.",
                "Acceptă pagini individuale și intervale inclusive.",
            ),
            text(
                "Creates several outputs in one operation.",
                "Creează mai multe rezultate într-o singură operațiune.",
            ),
            text(
                "Validates ranges against the real page count.",
                "Validează intervalele față de numărul real de pagini.",
            ),
        ),
        steps=(
            text("Upload the PDF you want to divide.", "Încarcă PDF-ul pe care vrei să îl împarți."),
            text(
                "Enter ranges such as 1-3, 7-10 or individual pages.",
                "Introdu intervale precum 1-3, 7-10 sau pagini individuale.",
            ),
            text(
                "Run the split and download the generated parts.",
                "Pornește separarea și descarcă părțile generate.",
            ),
        ),
        note=text(
            "Page numbers are one-based and both ends of a range are included. A range outside the document is rejected instead of producing a partial result.",
            "Numerotarea începe de la 1, iar ambele capete ale intervalului sunt incluse. Un interval din afara documentului este respins, nu procesat parțial.",
        ),
        faqs=(
            faq(
                "Can I extract only one page?",
                "Pot extrage o singură pagină?",
                "Yes. Enter a single page number, such as 6, and it becomes its own PDF.",
                "Da. Introdu un singur număr, de exemplu 6, iar pagina va deveni un PDF separat.",
            ),
            faq(
                "What does 1-3, 8-9 create?",
                "Ce creează 1-3, 8-9?",
                "It creates two files: one containing pages 1 through 3 and another containing pages 8 and 9.",
                "Creează două fișiere: unul cu paginile 1-3 și altul cu paginile 8 și 9.",
            ),
        ),
    ),
    ToolLanding(
        key="compress_pdf",
        slug="compress-pdf",
        operation_url_name="compress",
        icon="📦",
        name=text("Compress PDF", "Comprimă PDF"),
        category=text("Optimize", "Optimizare"),
        short_description=text(
            "Reduce PDF size with a choice between stronger compression and better image quality.",
            "Redu dimensiunea PDF-ului alegând între compresie mai puternică și calitate mai bună a imaginilor.",
        ),
        seo_title=text("Compress PDF Online | PDF Platform", "Comprimă PDF online | PDF Platform"),
        seo_description=text(
            "Compress PDF files online with maximum, balanced or quality-focused settings. Compare the original and resulting file size before download.",
            "Comprimă PDF-uri online cu setări maximă, echilibrată sau orientată spre calitate. Compară dimensiunea inițială cu rezultatul înainte de descărcare.",
        ),
        intro=text(
            "Make image-heavy reports and scans easier to email or archive. The tool cleans PDF streams and recompresses embedded images using the quality profile you select.",
            "Fă rapoartele și scanările bogate în imagini mai ușor de trimis sau arhivat. Instrumentul curăță fluxurile PDF și recomprimă imaginile conform profilului ales.",
        ),
        highlights=(
            text("Three practical compression profiles.", "Trei profiluri practice de compresie."),
            text(
                "Shows original size, result size and saved percentage.",
                "Afișează dimensiunea inițială, rezultatul și procentul economisit.",
            ),
            text(
                "Keeps text as text instead of rasterizing every page.",
                "Păstrează textul ca text, fără rasterizarea tuturor paginilor.",
            ),
        ),
        steps=(
            text(
                "Upload the PDF and open the compression tool.",
                "Încarcă PDF-ul și deschide instrumentul de compresie.",
            ),
            text(
                "Choose maximum, balanced or minimal compression.",
                "Alege compresie maximă, echilibrată sau minimă.",
            ),
            text("Compare the result and download it.", "Compară rezultatul și descarcă-l."),
        ),
        note=text(
            "Savings depend on the source. A PDF that already contains optimized images may shrink very little, and stronger settings can visibly affect photographs or scans.",
            "Economia depinde de sursă. Un PDF deja optimizat poate scădea foarte puțin, iar setările puternice pot afecta vizibil fotografiile sau scanările.",
        ),
        faqs=(
            faq(
                "Which profile should I use?",
                "Ce profil ar trebui să aleg?",
                "Balanced is the safest starting point. Use maximum for small email attachments and minimal when visual fidelity matters most.",
                "Profilul echilibrat este punctul bun de pornire. Alege maxim pentru atașamente mici și minim când calitatea vizuală contează cel mai mult.",
            ),
            faq(
                "Will every PDF become smaller?",
                "Orice PDF va deveni mai mic?",
                "No. Already optimized or mostly text-only documents can remain similar in size, and the result page reports the actual change.",
                "Nu. Documentele deja optimizate sau formate aproape doar din text pot rămâne similare; pagina de rezultat arată schimbarea reală.",
            ),
        ),
    ),
    ToolLanding(
        key="pdf_to_word",
        slug="pdf-to-word",
        operation_url_name="convert",
        icon="📝",
        name=text("PDF to Word", "PDF în Word"),
        category=text("Convert", "Conversie"),
        short_description=text(
            "Turn a text-based PDF into an editable DOCX document.",
            "Transformă un PDF cu text într-un document DOCX editabil.",
        ),
        seo_title=text(
            "Convert PDF to Word Online | PDF Platform", "Convertește PDF în Word online | PDF Platform"
        ),
        seo_description=text(
            "Convert a text-based PDF to an editable DOCX file for Microsoft Word, LibreOffice or Google Docs. OCR scanned PDFs first for better results.",
            "Convertește un PDF cu text într-un DOCX editabil în Microsoft Word, LibreOffice sau Google Docs. Aplică mai întâi OCR documentelor scanate.",
        ),
        intro=text(
            "Recover editable paragraphs and layout from reports, letters or forms. The converter works best with normal text pages and creates a standard .docx download.",
            "Recuperează paragrafe și structură editabilă din rapoarte, scrisori sau formulare. Conversia funcționează cel mai bine pe pagini cu text normal și produce un fișier .docx standard.",
        ),
        highlights=(
            text("Produces an editable DOCX file.", "Produce un fișier DOCX editabil."),
            text(
                "Works with Word, LibreOffice and Google Docs.",
                "Funcționează cu Word, LibreOffice și Google Docs.",
            ),
            text(
                "Runs longer conversions as background jobs with progress.",
                "Rulează conversiile mai lungi în fundal, cu progres afișat.",
            ),
        ),
        steps=(
            text(
                "Upload a PDF that contains selectable text.", "Încarcă un PDF care conține text selectabil."
            ),
            text("Start the PDF-to-Word conversion.", "Pornește conversia PDF în Word."),
            text(
                "Download the DOCX and review complex formatting.",
                "Descarcă DOCX-ul și verifică formatarea complexă.",
            ),
        ),
        note=text(
            "Multi-column layouts, equations and complex tables may need manual cleanup. An image-only scan should be made searchable with OCR before conversion.",
            "Structurile pe coloane, ecuațiile și tabelele complexe pot necesita corecții manuale. O scanare fără text trebuie trecută mai întâi prin OCR.",
        ),
        faqs=(
            faq(
                "Why is my Word file almost empty?",
                "De ce fișierul Word este aproape gol?",
                "The source is probably an image-only scan. Run the OCR tool first, then convert the searchable result.",
                "Sursa este probabil o scanare fără strat de text. Rulează mai întâi OCR, apoi convertește rezultatul căutabil.",
            ),
            faq(
                "Will the layout be identical?",
                "Structura va fi identică?",
                "Simple documents usually convert well, but PDF is a fixed-layout format and complex pages cannot always map perfectly to Word.",
                "Documentele simple se convertesc de obicei bine, dar PDF este un format cu structură fixă, iar paginile complexe nu pot fi transpuse întotdeauna perfect în Word.",
            ),
        ),
    ),
    ToolLanding(
        key="pdf_to_images",
        slug="pdf-to-images",
        operation_url_name="to_images",
        icon="🖼️",
        name=text("PDF to Images", "PDF în imagini"),
        category=text("Convert", "Conversie"),
        short_description=text(
            "Render every PDF page as PNG or JPG and download the set as a ZIP.",
            "Transformă fiecare pagină PDF în PNG sau JPG și descarcă setul într-o arhivă ZIP.",
        ),
        seo_title=text(
            "Convert PDF to JPG or PNG Online | PDF Platform",
            "Convertește PDF în JPG sau PNG online | PDF Platform",
        ),
        seo_description=text(
            "Convert every PDF page to PNG or JPG at 72, 150 or 300 DPI. Download multi-page results together in one ZIP archive.",
            "Convertește fiecare pagină PDF în PNG sau JPG la 72, 150 sau 300 DPI. Descarcă rezultatele cu mai multe pagini într-o singură arhivă ZIP.",
        ),
        intro=text(
            "Create image previews, publish individual pages or extract scans in a widely supported format. You control both output type and render resolution.",
            "Creează previzualizări, publică pagini individuale sau extrage scanări într-un format ușor de folosit. Controlezi formatul și rezoluția rezultatului.",
        ),
        highlights=(
            text(
                "PNG for sharp graphics or JPG for smaller photos and scans.",
                "PNG pentru grafică precisă sau JPG pentru fotografii și scanări mai mici.",
            ),
            text("72, 150 and 300 DPI presets.", "Profiluri de 72, 150 și 300 DPI."),
            text(
                "Predictable page-numbered filenames inside one ZIP.",
                "Nume de fișiere numerotate predictibil într-un singur ZIP.",
            ),
        ),
        steps=(
            text("Upload the source PDF.", "Încarcă PDF-ul sursă."),
            text("Choose PNG or JPG and the required DPI.", "Alege PNG sau JPG și rezoluția DPI necesară."),
            text("Export and download the ZIP archive.", "Exportă și descarcă arhiva ZIP."),
        ),
        note=text(
            "Image export rasterizes each page, so selectable text and interactive elements do not exist in the image files. Higher DPI also increases processing time and archive size.",
            "Exportul rasterizează fiecare pagină, astfel că imaginile nu mai conțin text selectabil sau elemente interactive. Un DPI mai mare crește timpul și dimensiunea arhivei.",
        ),
        faqs=(
            faq(
                "Should I choose PNG or JPG?",
                "Să aleg PNG sau JPG?",
                "Use PNG for diagrams, screenshots and text-heavy pages. Use JPG for photographs or scans when smaller files matter more.",
                "Folosește PNG pentru diagrame, capturi și pagini cu text. Alege JPG pentru fotografii sau scanări când contează dimensiunea mai mică.",
            ),
            faq(
                "Which DPI is right?",
                "Ce rezoluție DPI este potrivită?",
                "72 DPI suits quick screen previews, 150 is a balanced default and 300 is intended for high-quality print output.",
                "72 DPI este potrivit pentru previzualizări, 150 este varianta echilibrată, iar 300 este destinat tipăririi de calitate.",
            ),
        ),
    ),
    ToolLanding(
        key="protect_pdf",
        slug="protect-pdf",
        operation_url_name="protect",
        icon="🔒",
        name=text("Protect PDF", "Protejează PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Encrypt a PDF with an opening password using AES-256.",
            "Criptează un PDF cu o parolă de deschidere folosind AES-256.",
        ),
        seo_title=text(
            "Password Protect PDF with AES-256 | PDF Platform",
            "Protejează PDF cu parolă AES-256 | PDF Platform",
        ),
        seo_description=text(
            "Add an opening password to a PDF with AES-256 encryption. Create and download a protected copy online without changing the source file.",
            "Adaugă o parolă de deschidere unui PDF prin criptare AES-256. Creează și descarcă online o copie protejată, fără modificarea sursei.",
        ),
        intro=text(
            "Require recipients to enter a password before they can open the document. The original upload remains separate and the result is a newly encrypted PDF copy.",
            "Solicită destinatarilor o parolă înainte de deschiderea documentului. Fișierul original rămâne separat, iar rezultatul este o copie PDF nou criptată.",
        ),
        highlights=(
            text("AES-256 document encryption.", "Criptare AES-256 a documentului."),
            text(
                "Password confirmation prevents typing mistakes.",
                "Confirmarea parolei previne greșelile de tastare.",
            ),
            text("Creates a separate protected output.", "Creează un rezultat protejat separat."),
        ),
        steps=(
            text("Upload an unencrypted PDF.", "Încarcă un PDF necriptat."),
            text(
                "Enter and confirm a strong opening password.",
                "Introdu și confirmă o parolă puternică de deschidere.",
            ),
            text("Create the protected copy and download it.", "Creează copia protejată și descarc-o."),
        ),
        note=text(
            "The platform cannot recover a forgotten document password. Share it through a different channel from the PDF and keep your own secure copy.",
            "Platforma nu poate recupera o parolă uitată. Trimite parola prin alt canal decât PDF-ul și păstrează propria copie în siguranță.",
        ),
        faqs=(
            faq(
                "Does this modify my original PDF?",
                "Îmi modifică PDF-ul original?",
                "No. The operation creates a separate encrypted output that you download.",
                "Nu. Operațiunea creează un rezultat criptat separat, pe care îl descarci.",
            ),
            faq(
                "Can you recover my password later?",
                "Puteți recupera parola mai târziu?",
                "No. Choose a strong password you can retain safely; it is required to open or later unlock the file.",
                "Nu. Alege o parolă puternică pe care o poți păstra în siguranță; va fi necesară la deschidere sau deblocare.",
            ),
        ),
    ),
    ToolLanding(
        key="unlock_pdf",
        slug="unlock-pdf",
        operation_url_name="unprotect",
        icon="🔓",
        name=text("Unlock PDF", "Deblochează PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Remove password protection from a PDF when you know the password.",
            "Elimină protecția unui PDF atunci când cunoști parola.",
        ),
        seo_title=text(
            "Unlock PDF and Remove Password Online | PDF Platform",
            "Elimină parola unui PDF online | PDF Platform",
        ),
        seo_description=text(
            "Remove password protection from a PDF you are authorized to open. Enter the existing password and download a sanitized, unencrypted copy.",
            "Elimină protecția unui PDF pe care ești autorizat să îl deschizi. Introdu parola existentă și descarcă o copie sanitizată, necriptată.",
        ),
        intro=text(
            "Create an unencrypted copy for a workflow that cannot handle password-protected files. You must supply the correct user or owner password.",
            "Creează o copie necriptată pentru un flux care nu acceptă fișiere protejate. Trebuie să furnizezi parola corectă de utilizator sau proprietar.",
        ),
        highlights=(
            text(
                "Accepts the current user or owner password.",
                "Acceptă parola curentă de utilizator sau proprietar.",
            ),
            text(
                "Sanitizes the decrypted result before release.",
                "Sanitizează rezultatul decriptat înainte de livrare.",
            ),
            text("Produces a separate unprotected copy.", "Produce o copie neprotejată separată."),
        ),
        steps=(
            text("Upload the encrypted PDF.", "Încarcă PDF-ul criptat."),
            text(
                "Enter its current password on the unlock screen.",
                "Introdu parola actuală în ecranul de deblocare.",
            ),
            text("Download the sanitized, unprotected copy.", "Descarcă copia sanitizată și neprotejată."),
        ),
        note=text(
            "Use this only for a document you own or are authorized to modify. The tool does not crack, guess or bypass an unknown password.",
            "Folosește instrumentul doar pentru un document pe care îl deții sau ești autorizat să îl modifici. Nu sparge, nu ghicește și nu ocolește o parolă necunoscută.",
        ),
        faqs=(
            faq(
                "Can this open a PDF if I forgot the password?",
                "Poate deschide PDF-ul dacă am uitat parola?",
                "No. A valid existing password is required; the tool does not perform password recovery or cracking.",
                "Nu. Este necesară parola existentă validă; instrumentul nu recuperează și nu sparge parole.",
            ),
            faq(
                "Is the downloaded copy still encrypted?",
                "Copia descărcată mai este criptată?",
                "No. After successful authentication and sanitization, the output is saved without PDF encryption.",
                "Nu. După autentificare și sanitizare, rezultatul este salvat fără criptare PDF.",
            ),
        ),
    ),
    ToolLanding(
        key="ocr_pdf",
        slug="ocr-pdf",
        operation_url_name="searchable",
        icon="🔎",
        name=text("OCR PDF", "OCR pentru PDF"),
        category=text("Extract", "Extragere"),
        short_description=text(
            "Add a searchable text layer to image-only English or Romanian scans.",
            "Adaugă un strat de text căutabil scanărilor în engleză sau română.",
        ),
        seo_title=text(
            "OCR PDF Online — Make Scans Searchable | PDF Platform",
            "OCR PDF online — transformă scanări în text | PDF Platform",
        ),
        seo_description=text(
            "Run OCR on scanned PDF pages in English, Romanian or both. Add a searchable, selectable text layer while preserving the page appearance.",
            "Rulează OCR pe pagini PDF scanate în română, engleză sau ambele. Adaugă un strat de text căutabil și selectabil, păstrând aspectul paginii.",
        ),
        intro=text(
            "Turn scanned contracts, receipts and archives into documents you can search and copy from. Pages that already contain text are copied without OCR.",
            "Transformă contracte, chitanțe și arhive scanate în documente în care poți căuta și copia text. Paginile care conțin deja text sunt copiate fără OCR.",
        ),
        highlights=(
            text(
                "English, Romanian or combined recognition.", "Recunoaștere în română, engleză sau combinată."
            ),
            text(
                "Processes only pages that lack a text layer.", "Procesează doar paginile fără strat de text."
            ),
            text(
                "Adjustable rendering resolution from 72 to 600 DPI.",
                "Rezoluție ajustabilă între 72 și 600 DPI.",
            ),
        ),
        steps=(
            text("Upload the scanned PDF.", "Încarcă PDF-ul scanat."),
            text(
                "Choose the document language and rendering DPI.",
                "Alege limba documentului și rezoluția DPI.",
            ),
            text(
                "Start OCR, follow job progress and download the searchable PDF.",
                "Pornește OCR, urmărește progresul și descarcă PDF-ul căutabil.",
            ),
        ),
        note=text(
            "OCR is probabilistic. Small fonts, handwriting, skewed pages and low-contrast scans can produce mistakes, so verify critical names, dates and amounts.",
            "OCR nu este infailibil. Fonturile mici, scrisul de mână, paginile înclinate și contrastul slab pot produce erori; verifică nume, date și sume importante.",
        ),
        faqs=(
            faq(
                "Does OCR change how the scan looks?",
                "OCR schimbă aspectul scanării?",
                "The goal is to preserve the visible page while adding an invisible text layer for search and selection.",
                "Scopul este păstrarea aspectului vizibil, adăugând un strat invizibil pentru căutare și selectare.",
            ),
            faq(
                "What DPI should I choose?",
                "Ce DPI ar trebui să aleg?",
                "200 DPI is a good default. Increase it for very small print, understanding that processing time and memory use also rise.",
                "200 DPI este valoarea recomandată. Mărește-o pentru text foarte mic, ținând cont că timpul și memoria necesare cresc.",
            ),
        ),
    ),
    ToolLanding(
        key="redact_pdf",
        slug="redact-pdf",
        operation_url_name="redact",
        icon="⬛",
        name=text("Redact PDF", "Cenzurează PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Permanently remove matching text instead of merely covering it.",
            "Elimină permanent textul găsit, nu doar îl acoperă vizual.",
        ),
        seo_title=text(
            "Permanently Redact Text in PDF Online | PDF Platform",
            "Elimină permanent text din PDF online | PDF Platform",
        ),
        seo_description=text(
            "Find and permanently redact names, IDs or other text in a PDF. Remove underlying content and replace matches with black redaction boxes.",
            "Găsește și elimină permanent nume, identificatori sau alt text dintr-un PDF. Șterge conținutul și înlocuiește potrivirile cu zone negre.",
        ),
        intro=text(
            "Prepare a document for disclosure by removing repeated names, identifiers or confidential terms. Matches are removed from the page content, not hidden behind a cosmetic rectangle.",
            "Pregătește un document pentru distribuire eliminând nume, identificatori sau termeni confidențiali. Potrivirile sunt șterse din conținut, nu ascunse cosmetic.",
        ),
        highlights=(
            text(
                "Case-insensitive matching for one or more terms.",
                "Căutare fără diferențiere între litere mari și mici, pentru unul sau mai mulți termeni.",
            ),
            text("Optional page-range restriction.", "Restricționare opțională la anumite pagini."),
            text(
                "Applies true PDF redactions to the underlying content.",
                "Aplică redactări PDF reale asupra conținutului.",
            ),
        ),
        steps=(
            text("Upload a PDF with selectable text.", "Încarcă un PDF cu text selectabil."),
            text(
                "Enter one sensitive term per line and optionally a page range.",
                "Introdu câte un termen sensibil pe linie și, opțional, un interval de pagini.",
            ),
            text(
                "Run redaction, inspect every page and download the result.",
                "Rulează redactarea, verifică fiecare pagină și descarcă rezultatul.",
            ),
        ),
        note=text(
            "Always review the output before sharing it. Image-only text needs OCR first, spelling variants need separate terms, and redaction cannot be undone in the generated copy.",
            "Verifică întotdeauna rezultatul înainte de distribuire. Textul din imagini necesită mai întâi OCR, variantele de scriere trebuie introduse separat, iar redactarea nu poate fi anulată în copia generată.",
        ),
        faqs=(
            faq(
                "Is the text really removed?",
                "Textul este eliminat cu adevărat?",
                "For matched selectable text, the operation applies PDF redactions and removes the covered content rather than only drawing over it.",
                "Pentru textul selectabil găsit, operațiunea aplică redactări PDF și șterge conținutul acoperit, nu doar desenează peste el.",
            ),
            faq(
                "Will it find text inside a scan?",
                "Găsește textul dintr-o scanare?",
                "Not until the scan has a text layer. Run OCR first, then redact the resulting searchable PDF and verify it manually.",
                "Nu înainte ca scanarea să aibă un strat de text. Rulează OCR, apoi redactează PDF-ul căutabil și verifică-l manual.",
            ),
        ),
    ),
    ToolLanding(
        key="watermark_pdf",
        slug="watermark-pdf",
        operation_url_name="watermark",
        icon="💧",
        name=text("Watermark PDF", "Adaugă filigran PDF"),
        category=text("Edit", "Editare"),
        short_description=text(
            "Place a text or image watermark on every page with custom position and opacity.",
            "Aplică un filigran text sau imagine pe fiecare pagină, cu poziție și opacitate configurabile.",
        ),
        seo_title=text(
            "Add Text or Image Watermark to PDF | PDF Platform",
            "Adaugă filigran text sau imagine în PDF | PDF Platform",
        ),
        seo_description=text(
            "Add a text or image watermark to every PDF page. Choose position, opacity, rotation and text size before downloading a new copy.",
            "Adaugă un filigran text sau imagine pe fiecare pagină PDF. Alege poziția, opacitatea, rotația și dimensiunea textului înainte de descărcare.",
        ),
        intro=text(
            "Mark drafts, confidential documents, review copies or branded material consistently across all pages. Use text for a quick label or upload a logo image.",
            "Marchează drafturi, documente confidențiale, copii pentru revizie sau materiale de brand pe toate paginile. Folosește text sau încarcă o imagine cu logo-ul.",
        ),
        highlights=(
            text("Text and image watermark modes.", "Moduri de filigran text și imagine."),
            text(
                "Nine page positions plus opacity and rotation controls.",
                "Nouă poziții pe pagină, plus control pentru opacitate și rotație.",
            ),
            text("Applies the chosen watermark to every page.", "Aplică filigranul ales pe fiecare pagină."),
        ),
        steps=(
            text("Upload the PDF to mark.", "Încarcă PDF-ul pe care vrei să îl marchezi."),
            text(
                "Choose text or image and configure its appearance.",
                "Alege text sau imagine și configurează aspectul.",
            ),
            text(
                "Generate, preview and download the watermarked copy.",
                "Generează, previzualizează și descarcă versiunea cu filigran.",
            ),
        ),
        note=text(
            "A watermark communicates status or ownership but is not access control. Use PDF encryption as well when the document itself must require a password.",
            "Filigranul indică statutul sau proprietatea, dar nu controlează accesul. Folosește și criptare PDF dacă documentul trebuie protejat cu parolă.",
        ),
        faqs=(
            faq(
                "Can I use my logo?",
                "Pot folosi logo-ul meu?",
                "Yes. Choose image mode and upload a supported image; the tool sizes, positions, rotates and applies opacity to it.",
                "Da. Alege modul imagine și încarcă un format acceptat; instrumentul îl dimensionează, poziționează, rotește și îi aplică opacitatea.",
            ),
            faq(
                "Does the watermark protect against copying?",
                "Filigranul împiedică copierea?",
                "No. It is a visible mark, not a DRM mechanism. Pair it with password protection and appropriate sharing controls when needed.",
                "Nu. Este un marcaj vizibil, nu un mecanism DRM. Combină-l cu protecție prin parolă și controlul distribuirii atunci când este necesar.",
            ),
        ),
    ),
    ToolLanding(
        key="rotate_pdf",
        slug="rotate-pdf",
        operation_url_name="rotate",
        icon="🔄",
        name=text("Rotate PDF", "Rotește PDF"),
        category=text("Organize", "Organizare"),
        short_description=text(
            "Rotate all pages or a selected range by 90, 180 or 270 degrees.",
            "Rotește toate paginile sau un interval cu 90, 180 ori 270 de grade.",
        ),
        seo_title=text("Rotate PDF Pages Online | PDF Platform", "Rotește pagini PDF online | PDF Platform"),
        seo_description=text(
            "Rotate every PDF page or only selected ranges by 90, 180 or 270 degrees. Fix sideways scans and download a corrected copy online.",
            "Rotește toate paginile PDF sau doar intervalele alese cu 90, 180 ori 270 de grade. Corectează scanările și descarcă o copie nouă.",
        ),
        intro=text(
            "Correct sideways scans, upside-down pages or mixed-orientation documents without re-scanning them. Apply one rotation to the whole file or target specific pages.",
            "Corectează scanări întoarse, pagini răsturnate sau documente cu orientări mixte fără să le rescanezi. Aplică rotația întregului fișier sau doar anumitor pagini.",
        ),
        highlights=(
            text("90°, 180° and 270° rotation choices.", "Opțiuni de rotație la 90°, 180° și 270°."),
            text("Optional page ranges such as 1-3,5.", "Intervale opționale precum 1-3,5."),
            text(
                "Creates a corrected copy without rasterizing pages.",
                "Creează o copie corectată fără rasterizarea paginilor.",
            ),
        ),
        steps=(
            text(
                "Upload the PDF with incorrectly oriented pages.",
                "Încarcă PDF-ul cu pagini orientate greșit.",
            ),
            text(
                "Choose the angle and, if needed, enter page ranges.",
                "Alege unghiul și, dacă este necesar, introdu intervalele.",
            ),
            text("Rotate and download the corrected file.", "Rotește și descarcă fișierul corectat."),
        ),
        note=text(
            "Rotation changes page orientation metadata; it does not deskew a scan whose content is tilted by a few degrees inside the page image.",
            "Rotația schimbă orientarea paginii; nu îndreaptă o scanare al cărei conținut este înclinat cu doar câteva grade în interiorul imaginii.",
        ),
        faqs=(
            faq(
                "Can I rotate only one page?",
                "Pot roti o singură pagină?",
                "Yes. Enter that page number in the optional range field; leave it empty to rotate the whole document.",
                "Da. Introdu numărul paginii în câmpul opțional; lasă-l gol pentru a roti întregul document.",
            ),
            faq(
                "Does rotation reduce quality?",
                "Rotația reduce calitatea?",
                "The pages are not intentionally rendered into images, so rotation does not apply lossy image compression by itself.",
                "Paginile nu sunt transformate intenționat în imagini, deci rotația nu aplică în sine compresie cu pierderi.",
            ),
        ),
    ),
    ToolLanding(
        key="reorder_pdf",
        slug="reorder-pdf-pages",
        operation_url_name="reorder",
        icon="📑",
        name=text("Reorder PDF Pages", "Reordonează pagini PDF"),
        category=text("Organize", "Organizare"),
        short_description=text(
            "Drag pages into a new order and omit the pages you want to remove.",
            "Trage paginile într-o ordine nouă și elimină paginile de care nu ai nevoie.",
        ),
        seo_title=text(
            "Reorder and Delete PDF Pages Online | PDF Platform",
            "Reordonează și șterge pagini PDF online | PDF Platform",
        ),
        seo_description=text(
            "Reorder PDF pages visually, remove unwanted pages and download a clean copy. Drag page thumbnails into the exact sequence you need.",
            "Reordonează vizual paginile PDF, elimină paginile nedorite și descarcă o copie curată. Trage miniaturile în secvența dorită.",
        ),
        intro=text(
            "Fix scanning order, move an appendix or remove accidental blank pages using page thumbnails instead of typing a complicated sequence.",
            "Corectează ordinea scanării, mută o anexă sau elimină pagini goale folosind miniaturi, fără să tastezi o secvență complicată.",
        ),
        highlights=(
            text("Visual thumbnail-based page ordering.", "Ordonare vizuală pe baza miniaturilor."),
            text(
                "Remove pages by leaving them out of the final sequence.",
                "Elimină pagini prin excluderea lor din secvența finală.",
            ),
            text(
                "Validates that at least one unique page remains.",
                "Validează că rămâne cel puțin o pagină unică.",
            ),
        ),
        steps=(
            text("Upload the PDF you want to reorganize.", "Încarcă PDF-ul pe care vrei să îl reorganizezi."),
            text(
                "Drag thumbnails into order and remove unwanted pages.",
                "Trage miniaturile în ordine și elimină paginile nedorite.",
            ),
            text("Create and download the reorganized PDF.", "Creează și descarcă PDF-ul reorganizat."),
        ),
        note=text(
            "Removing a page from the generated copy is permanent. Keep the source until you have reviewed the final order and confirmed that no required page is missing.",
            "Eliminarea unei pagini din copia generată este permanentă. Păstrează sursa până verifici ordinea finală și confirmi că nu lipsește nicio pagină necesară.",
        ),
        faqs=(
            faq(
                "Can I delete blank pages?",
                "Pot șterge paginile goale?",
                "Yes. Remove those thumbnails from the final sequence, while keeping at least one page in the document.",
                "Da. Elimină miniaturile respective din secvența finală, păstrând cel puțin o pagină în document.",
            ),
            faq(
                "Can the same page appear twice?",
                "Aceeași pagină poate apărea de două ori?",
                "No. This operation reorders or removes existing pages; duplicate page numbers are rejected.",
                "Nu. Operațiunea reordonează sau elimină paginile existente; numerele duplicate sunt respinse.",
            ),
        ),
    ),
    ToolLanding(
        key="sign_pdf",
        slug="digitally-sign-pdf",
        operation_url_name="sign",
        icon="✍️",
        name=text("Digitally Sign PDF", "Semnează digital PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Apply a cryptographic PDF signature with your own PKCS#12 certificate.",
            "Aplică o semnătură PDF criptografică folosind propriul certificat PKCS#12.",
        ),
        seo_title=text(
            "Digitally Sign PDF with a P12 Certificate | PDF Platform",
            "Semnează digital PDF cu certificat P12 | PDF Platform",
        ),
        seo_description=text(
            "Digitally sign a PDF with your own P12 or PFX certificate. Choose the visible signature position and optionally add a trusted timestamp.",
            "Semnează digital un PDF cu propriul certificat P12 sau PFX. Alege poziția semnăturii vizibile și adaugă opțional o marcă temporală.",
        ),
        intro=text(
            "Create a real PKCS#7 document signature, not just a picture of a handwritten name. Supply your certificate archive, choose where the visible widget appears and download the signed PDF.",
            "Creează o semnătură PKCS#7 reală, nu doar imaginea unui nume scris de mână. Furnizează arhiva certificatului, alege poziția vizibilă și descarcă PDF-ul semnat.",
        ),
        highlights=(
            text(
                "Accepts PKCS#12 .p12 and .pfx certificate archives.",
                "Acceptă arhive de certificat PKCS#12 .p12 și .pfx.",
            ),
            text("Configurable visible page and position.", "Pagină și poziție vizibilă configurabile."),
            text(
                "Optional RFC 3161 timestamp and validation information.",
                "Marcă temporală RFC 3161 și informații de validare opționale.",
            ),
        ),
        steps=(
            text("Upload the PDF that must be signed.", "Încarcă PDF-ul care trebuie semnat."),
            text(
                "Provide your P12/PFX certificate, its password and signature details.",
                "Furnizează certificatul P12/PFX, parola și detaliile semnăturii.",
            ),
            text(
                "Sign, download and verify the result with the verification tool.",
                "Semnează, descarcă și verifică rezultatul cu instrumentul de validare.",
            ),
        ),
        note=text(
            "A test certificate proves the workflow but not a legally trusted identity. For production documents, use a certificate issued and governed for your jurisdiction and purpose.",
            "Un certificat de test demonstrează fluxul, dar nu confirmă juridic identitatea. Pentru documente oficiale, folosește un certificat emis și reglementat pentru jurisdicția și scopul tău.",
        ),
        faqs=(
            faq(
                "Is this the same as pasting a signature image?",
                "Este același lucru cu lipirea unei imagini a semnăturii?",
                "No. The tool creates a cryptographic PDF signature linked to the private key in your certificate archive.",
                "Nu. Instrumentul creează o semnătură PDF criptografică legată de cheia privată din arhiva certificatului.",
            ),
            faq(
                "How can I check the result?",
                "Cum pot verifica rezultatul?",
                "Use the PDF signature verification page to inspect integrity, signer information, certificate trust and timestamps.",
                "Folosește pagina de verificare a semnăturilor PDF pentru integritate, informații despre semnatar, încrederea certificatului și mărci temporale.",
            ),
        ),
    ),
    ToolLanding(
        key="crop_pdf",
        slug="crop-pdf",
        operation_url_name="crop",
        icon="✂️",
        name=text("Crop PDF Margins", "Decupează margini PDF"),
        category=text("Organize", "Organizare"),
        short_description=text(
            "Trim page margins in millimeters or points across all pages or a range.",
            "Ajustează marginile paginilor în milimetri sau puncte pentru tot documentul sau un interval.",
        ),
        seo_title=text(
            "Crop PDF Pages and Trim Margins Online | PDF Platform",
            "Decupează pagini PDF și margini online | PDF Platform",
        ),
        seo_description=text(
            "Crop PDF pages and trim unwanted margins in millimeters or points. Apply consistent margins across every page or a selected range online.",
            "Decupează paginile PDF și elimină marginile nedorite în milimetri sau puncte. Aplică margini uniforme tuturor paginilor sau unui interval.",
        ),
        intro=text(
            "Remove wide white borders, scanner artifacts or printer crop marks by adjusting top, right, bottom and left margins in millimeters or points.",
            "Elimină bordurile albe late, urmele de scanare sau marcajele de tipar ajustând marginile sus, dreapta, jos și stânga în milimetri sau puncte.",
        ),
        highlights=(
            text(
                "Supports millimeters (mm) and PDF points (pt).", "Suportă milimetri (mm) și puncte PDF (pt)."
            ),
            text(
                "Apply to all pages or target specific page ranges.",
                "Aplică tuturor paginilor sau doar unor intervale specifice.",
            ),
            text(
                "Updates the PDF CropBox without rasterizing text.",
                "Actualizează CropBox-ul PDF fără rasterizarea textului.",
            ),
        ),
        steps=(
            text("Upload the PDF you want to crop.", "Încarcă PDF-ul pe care vrei să îl decupezi."),
            text(
                "Enter top, right, bottom and left margin insets.",
                "Introdu valorile de decupare pentru sus, dreapta, jos și stânga.",
            ),
            text(
                "Apply the crop and download the trimmed PDF.", "Aplică decuparea și descarcă PDF-ul ajustat."
            ),
        ),
        note=text(
            "Cropping adjusts the visible page box (CropBox) rather than permanently erasing outside objects. Use redaction if hidden margin content is confidential.",
            "Decuparea ajustează zona vizibilă a paginii (CropBox), fără a șterge fizic obiectele exterioare. Folosește redactarea dacă marginile conțin date confidențiale.",
        ),
        faqs=(
            faq(
                "Does cropping change image or text quality?",
                "Decuparea afectează calitatea textului sau a imaginilor?",
                "No. Cropping modifies the page boundary coordinates without re-encoding text or embedded images.",
                "Nu. Decuparea modifică doar coordonatele marginilor paginii, fără a recoda textul sau imaginile.",
            ),
            faq(
                "Can I crop only odd or specific pages?",
                "Pot decupa doar anumite pagini?",
                "Yes. Specify a page range such as 1-3,5,8-10 or leave the field blank to crop every page.",
                "Da. Specifică un interval precum 1-3,5,8-10 sau lasă câmpul gol pentru a decupa toate paginile.",
            ),
        ),
    ),
    ToolLanding(
        key="flatten_pdf",
        slug="flatten-pdf",
        operation_url_name="flatten",
        icon="🧱",
        name=text("Flatten PDF", "Aplatizează PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Bake interactive form fields and annotations into static page content.",
            "Transformă câmpurile de formular și adnotările interactive în conținut static.",
        ),
        seo_title=text(
            "Flatten PDF Forms and Annotations Online | PDF Platform",
            "Aplatizează formulare și adnotări PDF | PDF Platform",
        ),
        seo_description=text(
            "Flatten interactive PDF form fields and annotations into static page content so values cannot be edited accidentally by recipients.",
            "Aplatizează câmpurile interactive și adnotările PDF în conținut static pe pagină, astfel încât valorile să nu fie modificate accidental.",
        ),
        intro=text(
            "Lock in completed form values, stamps and comments before sending a PDF to clients or printers so interactive widgets no longer float above the page.",
            "Fixează valorile completate în formulare, ștampilele și comentariile înainte de trimitere sau tipărire, eliminând câmpurile interactive.",
        ),
        highlights=(
            text(
                "Merges AcroForm fields and annotations into the page stream.",
                "Contopește câmpurile AcroForm și adnotările în fluxul paginii.",
            ),
            text(
                "Improves compatibility with printers and archival viewers.",
                "Îmbunătățește compatibilitatea la tipărire și arhivare.",
            ),
            text(
                "Generates a clean flattened copy while keeping your original.",
                "Generează o copie aplatizată curată, păstrând originalul intact.",
            ),
        ),
        steps=(
            text("Upload the PDF with forms or annotations.", "Încarcă PDF-ul cu formulare sau adnotări."),
            text("Confirm the flatten operation.", "Confirmă operațiunea de aplatizare."),
            text("Download the static flattened PDF.", "Descarcă PDF-ul static aplatizat."),
        ),
        note=text(
            "Once flattened, form fields can no longer be edited as interactive inputs. Keep the unflattened file if you need to update values later.",
            "După aplatizare, câmpurile de formular nu mai pot fi editate interactiv. Păstrează fișierul original dacă vei avea nevoie de modificări ulterioare.",
        ),
        faqs=(
            faq(
                "What does flattening a PDF do?",
                "Ce face aplatizarea unui PDF?",
                "It converts interactive form inputs, checkboxes and annotations into regular static graphics and text on the page.",
                "Convertește câmpurile interactive, căsuțele de bifat și adnotările în grafică și text static obișnuit pe pagină.",
            ),
            faq(
                "Is flattening the same as password protection?",
                "Aplatizarea este același lucru cu protecția prin parolă?",
                "No. Flattening removes interactive form widgets, whereas password protection encrypts the file and restricts permissions.",
                "Nu. Aplatizarea elimină elementele interactive de formular, în timp ce protecția cu parolă criptează fișierul.",
            ),
        ),
    ),
    ToolLanding(
        key="page_numbers_pdf",
        slug="add-page-numbers-pdf",
        operation_url_name="page_numbers",
        icon="🔢",
        name=text("Add Page Numbers", "Numerotează pagini PDF"),
        category=text("Edit", "Editare"),
        short_description=text(
            "Insert customizable page numbers in the header or footer of your PDF.",
            "Inserează numere de pagină personalizabile în antetul sau subsolul PDF-ului.",
        ),
        seo_title=text(
            "Add Page Numbers to PDF Online | PDF Platform",
            "Adaugă numere de pagină în PDF online | PDF Platform",
        ),
        seo_description=text(
            "Insert customizable page numbers into any PDF document. Choose the numbering format, starting page, font size and header or footer position.",
            "Inserează numere de pagină personalizabile în orice document PDF. Alege formatul numerotării, pagina de start, dimensiunea și poziția.",
        ),
        intro=text(
            "Prepare reports, legal bundles and academic submissions with clear, consistent page numbering in the header or footer.",
            "Pregătește rapoarte, dosare juridice și lucrări academice cu o numerotare clară și uniformă în antet sau subsol.",
        ),
        highlights=(
            text(
                "Multiple formats including Page X of Y and simple numbers.",
                "Formate multiple, inclusiv „Pagina X din Y” sau numere simple.",
            ),
            text(
                "Six header and footer positions with adjustable margins.",
                "Șase poziții în antet și subsol, cu margini ajustabile.",
            ),
            text(
                "Custom starting number and font size controls.",
                "Control pentru numărul de start și dimensiunea fontului.",
            ),
        ),
        steps=(
            text(
                "Upload the PDF document to number.", "Încarcă documentul PDF pe care vrei să îl numerotezi."
            ),
            text(
                "Choose position, numbering format, font size and start number.",
                "Alege poziția, formatul numerotării, dimensiunea fontului și numărul de start.",
            ),
            text("Generate and download the numbered PDF.", "Generează și descarcă PDF-ul numerotat."),
        ),
        note=text(
            "If your document already has printed page numbers, adding new ones will overlay them unless you choose a different corner or crop the old footer first.",
            "Dacă documentul are deja numere tipărite, noile numere se vor suprapune dacă nu alegi alt colț sau nu decupezi mai întâi vechiul subsol.",
        ),
        faqs=(
            faq(
                "Can I start numbering from a number other than 1?",
                "Pot începe numerotarea de la alt număr decât 1?",
                "Yes. Set the starting number to match continuous multi-volume pagination.",
                "Da. Setează numărul de start dorit pentru a continua numerotarea între mai multe volume.",
            ),
            faq(
                "Where can page numbers be placed?",
                "Unde pot fi plasate numerele de pagină?",
                "You can place them at the top-left, top-center, top-right, bottom-left, bottom-center or bottom-right of every page.",
                "Le poți plasa sus-stânga, sus-centru, sus-dreapta, jos-stânga, jos-centru sau jos-dreapta pe fiecare pagină.",
            ),
        ),
    ),
    ToolLanding(
        key="metadata_pdf",
        slug="edit-pdf-metadata",
        operation_url_name="metadata",
        icon="🏷️",
        name=text("Edit PDF Metadata", "Editează metadate PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Inspect, update or completely strip title, author, subject and keyword metadata.",
            "Verifică, actualizează sau șterge complet titlul, autorul, subiectul și cuvintele-cheie.",
        ),
        seo_title=text(
            "View, Edit or Strip PDF Metadata Online | PDF Platform",
            "Editează sau șterge metadate PDF online | PDF Platform",
        ),
        seo_description=text(
            "Inspect, update or completely strip PDF metadata fields including title, author, subject and keywords before sharing your document online.",
            "Verifică, actualizează sau șterge complet metadatele PDF precum titlul, autorul, subiectul și cuvintele-cheie înainte de a distribui fișierul.",
        ),
        intro=text(
            "Clean up hidden document properties before publishing a file, or set accurate title, author and keyword fields for indexing and archives.",
            "Curăță proprietățile ascunse ale documentului înainte de publicare sau setează corect titlul, autorul și cuvintele-cheie pentru arhivare.",
        ),
        highlights=(
            text(
                "Edit Title, Author, Subject, Keywords, Creator and Producer.",
                "Editează Titlul, Autorul, Subiectul, Cuvintele-cheie, Creatorul și Producătorul.",
            ),
            text(
                "One-click option to strip all metadata for privacy.",
                "Opțiune rapidă pentru ștergerea completă a metadatelor.",
            ),
            text(
                "Preserves page layout and visual appearance.",
                "Păstrează intacte aspectul vizual și structura paginilor.",
            ),
        ),
        steps=(
            text("Upload the PDF to inspect.", "Încarcă PDF-ul pe care vrei să îl verifici."),
            text(
                "Edit the metadata fields or select strip all metadata.",
                "Editează câmpurile de metadate sau alege ștergerea lor completă.",
            ),
            text("Save and download the updated PDF.", "Salvează și descarcă PDF-ul actualizat."),
        ),
        note=text(
            "Stripping document metadata removes standard info dictionary and XMP properties, but does not remove author names written visibly inside page text.",
            "Ștergerea metadatelor elimină proprietățile din dicționarul info și XMP, dar nu șterge numele de autori scrise vizibil în textul paginii.",
        ),
        faqs=(
            faq(
                "Why should I strip PDF metadata?",
                "De ce ar trebui să șterg metadatele PDF?",
                "PDFs often store the creator username, software version and internal document title, which you may not want to disclose publicly.",
                "Fișierele PDF păstrează adesea numele utilizatorului, aplicația folosită și titluri interne pe care nu dorești să le expui public.",
            ),
            faq(
                "Does editing metadata alter the pages?",
                "Editarea metadatelor modifică paginile?",
                "No. Only the document information properties are updated; page text and images remain untouched.",
                "Nu. Doar proprietățile informative ale documentului sunt actualizate; textul și imaginile rămân neschimbate.",
            ),
        ),
    ),
    ToolLanding(
        key="pdfa_pdf",
        slug="pdf-to-pdfa",
        operation_url_name="pdfa",
        icon="🏛️",
        name=text("Convert to PDF/A", "Convertește în PDF/A"),
        category=text("Convert", "Conversie"),
        short_description=text(
            "Convert standard PDFs into archival PDF/A format for long-term preservation.",
            "Convertește fișiere PDF standard în formatul de arhivare PDF/A pe termen lung.",
        ),
        seo_title=text(
            "Convert PDF to Archival PDF/A Online | PDF Platform",
            "Convertește PDF în format PDF/A online | PDF Platform",
        ),
        seo_description=text(
            "Convert standard PDF files into ISO-standardized archival PDF/A documents with embedded color profiles and metadata for long-term storage.",
            "Convertește fișiere PDF standard în documente PDF/A de arhivare conform ISO, cu profiluri de culoare și metadate încorporate pe termen lung.",
        ),
        intro=text(
            "Prepare documents for public institutions, court filings and long-term digital archives that require self-contained PDF/A compliance.",
            "Pregătește documente pentru instituții publice, depuneri oficiale și arhive digitale pe termen lung care solicită conformitate PDF/A.",
        ),
        highlights=(
            text(
                "Supports PDF/A-1b, PDF/A-2b and PDF/A-3b conformance levels.",
                "Suportă nivelurile de conformitate PDF/A-1b, PDF/A-2b și PDF/A-3b.",
            ),
            text(
                "Embeds output intent color profiles and XMP metadata.",
                "Încorporează profiluri de culoare și metadate XMP standardizate.",
            ),
            text(
                "Removes forbidden dynamic content for archival stability.",
                "Elimină elementele dinamice interzise pentru stabilitate în arhivă.",
            ),
        ),
        steps=(
            text("Upload the PDF you need to archive.", "Încarcă PDF-ul pe care trebuie să îl arhivezi."),
            text(
                "Select the target PDF/A conformance level.",
                "Selectează nivelul de conformitate PDF/A dorit.",
            ),
            text(
                "Convert and download the archival PDF/A file.",
                "Convertește și descarcă fișierul PDF/A de arhivă.",
            ),
        ),
        note=text(
            "If the source PDF uses unembedded fonts missing on the server, verify the converted document with your target archive validator before submission.",
            "Dacă PDF-ul sursă folosește fonturi neîncorporate, verifică documentul convertit cu validatorul instituției înainte de depunerea oficială.",
        ),
        faqs=(
            faq(
                "Which PDF/A version should I choose?",
                "Ce versiune PDF/A ar trebui să aleg?",
                "PDF/A-2b is recommended for most modern archives as it supports JPEG2000 compression and transparency.",
                "PDF/A-2b este recomandat pentru majoritatea arhivelor moderne, deoarece suportă transparență și compresie eficientă.",
            ),
            faq(
                "Can an encrypted PDF be converted to PDF/A?",
                "Poate fi convertit în PDF/A un PDF criptat?",
                "PDF/A forbids password encryption. Unlock the PDF first before converting it to PDF/A.",
                "Standardul PDF/A interzice criptarea cu parolă. Deblochează mai întâi PDF-ul înainte de conversia în PDF/A.",
            ),
        ),
    ),
    ToolLanding(
        key="compare_pdf",
        slug="compare-pdf",
        operation_url_name="compare",
        icon="⚖️",
        name=text("Compare PDFs", "Compară două PDF-uri"),
        category=text("Edit", "Editare"),
        short_description=text(
            "Spot text changes and visual differences between two versions of a PDF.",
            "Identifică modificările de text și diferențele vizuale dintre două versiuni PDF.",
        ),
        seo_title=text(
            "Compare Two PDF Files Side by Side | PDF Platform",
            "Compară două fișiere PDF vizual și text | PDF Platform",
        ),
        seo_description=text(
            "Compare two PDF versions to spot text changes and visual page differences. Generate a unified diff summary and highlighted comparison PDF.",
            "Compară două versiuni PDF pentru a identifica modificările de text și diferențele vizuale. Generează un raport diff și un PDF evidențiat.",
        ),
        intro=text(
            "Check contract revisions, proofread updated layouts or verify that only approved clauses changed between two PDF drafts.",
            "Verifică reviziile unui contract, compară machete actualizate sau confirmă că doar clauzele aprobate s-au schimbat între două variante PDF.",
        ),
        highlights=(
            text(
                "Extracts a unified text diff across pages.",
                "Extrage diferențele de text (diff) pe parcursul paginilor.",
            ),
            text(
                "Generates a visual comparison PDF highlighting changed regions.",
                "Generează un PDF comparativ care evidențiază vizual zonele modificate.",
            ),
            text(
                "Upload two files at once or compare against an existing upload.",
                "Încarcă două fișiere simultan sau compară cu un document deja încărcat.",
            ),
        ),
        steps=(
            text(
                "Upload the base PDF and the revised PDF.", "Încarcă PDF-ul de bază și versiunea revizuită."
            ),
            text("Run the comparison engine.", "Rulează motorul de comparare."),
            text(
                "Review the text diff and download the visual comparison PDF.",
                "Analizează diferențele de text și descarcă PDF-ul comparativ.",
            ),
        ),
        note=text(
            "Scanned PDFs without a text layer will only show visual pixel differences unless you run OCR on both files first.",
            "PDF-urile scanate fără strat de text vor afișa doar diferențe vizuale la nivel de pixeli dacă nu rulezi mai întâi OCR pe ambele fișiere.",
        ),
        faqs=(
            faq(
                "How are visual differences shown?",
                "Cum sunt afișate diferențele vizuale?",
                "Pages are rendered and compared so regions that changed between the two versions are highlighted in the output PDF.",
                "Paginile sunt randate și comparate, iar zonele care diferă între cele două versiuni sunt evidențiate în PDF-ul rezultat.",
            ),
            faq(
                "Can I compare PDFs with different page counts?",
                "Pot compara PDF-uri cu număr diferit de pagini?",
                "Yes. Shared pages are compared pair by pair and extra pages are noted in the comparison output.",
                "Da. Paginile comune sunt comparate pereche cu pereche, iar paginile suplimentare sunt semnalate în rezultat.",
            ),
        ),
        multiple_upload=True,
    ),
    ToolLanding(
        key="outline_pdf",
        slug="pdf-bookmarks-outline",
        operation_url_name="outline",
        icon="🔖",
        name=text("PDF Bookmarks and Outline", "Semne de carte și cuprins PDF"),
        category=text("Organize", "Organizare"),
        short_description=text(
            "View, add or edit hierarchical bookmarks and table-of-contents entries.",
            "Vizualizează, adaugă sau editează semne de carte ierarhice și intrări de cuprins.",
        ),
        seo_title=text(
            "Edit PDF Bookmarks and Table of Contents | PDF Platform",
            "Editează semne de carte și cuprins PDF | PDF Platform",
        ),
        seo_description=text(
            "View, add or replace PDF bookmarks and hierarchical outline entries so readers can jump directly to chapters and sections in your document.",
            "Vizualizează, adaugă sau înlocuiește semnele de carte și cuprinsul ierarhic PDF, astfel încât cititorii să sară rapid la capitole și secțiuni.",
        ),
        intro=text(
            "Make long manuals, e-books and legal filings easy to navigate by defining a clean sidebar bookmark tree pointing to exact page numbers.",
            "Fă manualele lungi, cărțile electronice și dosarele ușor de parcurs definind un arbore de semne de carte către paginile exacte.",
        ),
        highlights=(
            text(
                "Pre-fills existing bookmarks so you can edit them quickly.",
                "Preîncarcă semnele de carte existente pentru editare rapidă.",
            ),
            text(
                "Supports nested hierarchy levels (chapters, sections, subsections).",
                "Suportă niveluri ierarhice imbricate (capitole, secțiuni, subsecțiuni).",
            ),
            text(
                "Simple line-by-line format: level | title | page.",
                "Format simplu linie cu linie: nivel | titlu | pagină.",
            ),
        ),
        steps=(
            text("Upload the PDF document.", "Încarcă documentul PDF."),
            text(
                "Edit or enter bookmark lines with level, title and target page.",
                "Editează sau introdu liniile de cuprins cu nivel, titlu și pagină.",
            ),
            text(
                "Apply the outline and download the updated PDF.",
                "Aplică cuprinsul și descarcă PDF-ul actualizat.",
            ),
        ),
        note=text(
            "PDF bookmarks update the viewer navigation sidebar; they do not insert a newly typeset table-of-contents page inside the document body.",
            "Semnele de carte actualizează panoul lateral de navigare al cititorului PDF; nu inserează o pagină nouă de cuprins în interiorul documentului.",
        ),
        faqs=(
            faq(
                "How do I create sub-bookmarks?",
                "Cum creez sub-capitole în cuprins?",
                "Use level 1 for top-level chapters, level 2 for sections under the preceding level 1, and so on.",
                "Folosește nivelul 1 pentru capitole principale, nivelul 2 pentru secțiuni sub capitolul precedent ș.a.m.d.",
            ),
            faq(
                "Can I remove all existing bookmarks?",
                "Pot șterge toate semnele de carte existente?",
                "Yes. Clear the outline editor and submit to strip the bookmark tree from the generated copy.",
                "Da. Golește editorul de cuprins și salvează pentru a elimina arborele de semne de carte din copia generată.",
            ),
        ),
    ),
    ToolLanding(
        key="form_fill_pdf",
        slug="fill-pdf-forms",
        operation_url_name="form_fill",
        icon="📋",
        name=text("Fill PDF Forms", "Completează formulare PDF"),
        category=text("Edit", "Editare"),
        short_description=text(
            "Complete interactive AcroForm fields online and optionally flatten the result.",
            "Completează online câmpurile AcroForm interactive și aplatizează opțional rezultatul.",
        ),
        seo_title=text(
            "Fill Interactive PDF Forms Online | PDF Platform",
            "Completează formulare PDF interactive | PDF Platform",
        ),
        seo_description=text(
            "Detect and fill interactive AcroForm fields in your PDF online, including text inputs, checkboxes and dropdowns, with optional flattening.",
            "Detectează și completează online câmpurile AcroForm dintr-un PDF, inclusiv casete de text, bife și liste, cu opțiune de aplatizare finală.",
        ),
        intro=text(
            "Automatically list every interactive field in an official application, tax form or contract, type your values in a clean web form and export the completed PDF.",
            "Listează automat toate câmpurile interactive dintr-o cerere, formular fiscal sau contract, completează valorile și descarcă PDF-ul gata completat.",
        ),
        highlights=(
            text(
                "Detects text fields, checkboxes, radio groups and dropdowns.",
                "Detectează câmpuri de text, căsuțe de bifat, butoane radio și liste.",
            ),
            text(
                "Groups fields by page for fast, structured completion.",
                "Grupează câmpurile pe pagini pentru o completare rapidă și clară.",
            ),
            text(
                "Optional one-step flatten so completed values cannot be altered.",
                "Opțiune de aplatizare directă pentru blocarea valorilor completate.",
            ),
        ),
        steps=(
            text(
                "Upload a PDF with interactive form fields.",
                "Încarcă un PDF cu câmpuri de formular interactive.",
            ),
            text(
                "Fill in the detected fields and choose whether to flatten.",
                "Completează câmpurile detectate și alege dacă dorești aplatizarea.",
            ),
            text("Generate and download the filled PDF.", "Generează și descarcă PDF-ul completat."),
        ),
        note=text(
            "If your PDF is a flat scan without interactive AcroForm widgets, use the visual PDF Editor to place text directly over the page.",
            "Dacă PDF-ul este o scanare simplă fără câmpuri AcroForm interactive, folosește Editorul PDF vizual pentru a plasa text direct pe pagină.",
        ),
        faqs=(
            faq(
                "Why does the tool say no form fields were found?",
                "De ce instrumentul spune că nu s-au găsit câmpuri de formular?",
                "Many PDFs look like forms visually but lack interactive AcroForm widgets. Open those files in Edit PDF to add text boxes anywhere.",
                "Multe PDF-uri arată ca formulare, dar nu conțin câmpuri AcroForm interactive. Deschide-le în Editează PDF pentru a adăuga text oriunde.",
            ),
            faq(
                "Should I flatten the form after filling it?",
                "Ar trebui să aplatizez formularul după completare?",
                "Flatten it if you are submitting a final read-only copy; leave it unflattened if the recipient needs to edit the fields further.",
                "Aplatizează-l dacă trimiți varianta finală; lasă-l neaplatizat dacă destinatarul trebuie să mai modifice câmpurile.",
            ),
        ),
    ),
    ToolLanding(
        key="summarize_pdf",
        slug="ai-pdf-summarizer",
        operation_url_name="summarize",
        icon="🧠",
        name=text("AI PDF Summarizer", "Rezumat PDF cu AI"),
        category=text("Convert", "Conversie"),
        short_description=text(
            "Generate concise executive summaries and key takeaways from any PDF in your language.",
            "Generează rezumate executive concise și idei principale din orice PDF, în limba dorită.",
        ),
        seo_title=text(
            "AI PDF Summarizer and Key Takeaways | PDF Platform",
            "Rezumat PDF cu AI și idei principale | PDF Platform",
        ),
        seo_description=text(
            "Generate concise AI-powered summaries and key bullet points from any PDF document in your preferred language, then copy or download the text.",
            "Generează rezumate concise și idei principale cu AI din orice document PDF în limba dorită, apoi copiază sau descarcă textul rezultat.",
        ),
        intro=text(
            "Turn lengthy reports, research papers and contracts into structured summaries with bulleted key points in seconds.",
            "Transformă în câteva secunde rapoarte lungi, articole de cercetare și contracte în rezumate structurate cu ideile esențiale.",
        ),
        highlights=(
            text(
                "Supports output in English, Romanian, German, French, Spanish and more.",
                "Suportă rezumate în română, engleză, germană, franceză, spaniolă și alte limbi.",
            ),
            text(
                "Automatic fallback across local and cloud AI models.",
                "Comutare automată între modelele AI disponibile.",
            ),
            text(
                "One-click copy to clipboard or .txt download.",
                "Copiere rapidă în clipboard sau descărcare ca fișier .txt.",
            ),
        ),
        steps=(
            text(
                "Upload the PDF document you want to summarize.",
                "Încarcă documentul PDF pe care vrei să îl rezumi.",
            ),
            text(
                "Select your preferred output language and AI model.",
                "Selectează limba dorită pentru rezumat și modelul AI.",
            ),
            text("Generate, copy or download the summary.", "Generează, copiază sau descarcă rezumatul."),
        ),
        note=text(
            "AI summaries can omit nuances or misinterpret dense tables. Always verify critical clauses, figures and obligations in the original PDF.",
            "Rezumatele generate cu AI pot omite nuanțe sau pot interpreta greșit tabele complexe. Verifică mereu cifrele și clauzele critice în documentul original.",
        ),
        faqs=(
            faq(
                "Can I summarize a PDF in another language?",
                "Pot rezuma un PDF în altă limbă?",
                "Yes. Choose your target output language (such as Romanian or English) and the AI will summarize and translate the key points.",
                "Da. Alege limba dorită (de exemplu română sau engleză), iar AI-ul va rezuma și traduce ideile principale.",
            ),
            faq(
                "What if my PDF is a scanned image?",
                "Ce fac dacă PDF-ul meu este o scanare?",
                "Run Searchable PDF (OCR) first so the document has selectable text before generating an AI summary.",
                "Rulează mai întâi funcția PDF căutabil (OCR) pentru a extrage textul înainte de generarea rezumatului cu AI.",
            ),
        ),
    ),
    ToolLanding(
        key="qr_code_pdf",
        slug="add-qr-code-to-pdf",
        operation_url_name="qr_code",
        icon="📱",
        name=text("Add QR Code to PDF", "Adaugă cod QR pe PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Stamp a scannable QR code onto one or all pages of a PDF for verification links or invoices.",
            "Aplică un cod QR scanabil pe una sau pe toate paginile unui PDF pentru linkuri de verificare sau facturi.",
        ),
        seo_title=text(
            "Add QR Code Stamp to PDF Pages Online | PDF Platform",
            "Adaugă cod QR pe paginile PDF online | PDF Platform",
        ),
        seo_description=text(
            "Generate and stamp a high-resolution scannable QR code onto any PDF page with custom position, size and caption.",
            "Generează și aplică un cod QR scanabil de înaltă rezoluție pe orice pagină PDF, cu poziție, dimensiune și etichetă configurabile.",
        ),
        intro=text(
            "Embed a crisp verification URL, payment link or document reference as a scannable QR code on the first page, last page or every page of your PDF.",
            "Integrează un link de verificare, link de plată sau referință de document sub formă de cod QR scanabil pe prima pagină, ultima pagină sau pe toate paginile PDF-ului.",
        ),
        highlights=(
            text(
                "High-contrast QR generation with medium error correction.",
                "Generare cod QR cu contrast ridicat și corecție de erori medie.",
            ),
            text(
                "Target the first page, last page, all pages or a custom page range.",
                "Aplicare pe prima pagină, ultima pagină, toate paginile sau un interval ales.",
            ),
            text(
                "Optional text caption printed neatly below the QR code.",
                "Etichetă opțională de text afișată sub codul QR.",
            ),
        ),
        steps=(
            text(
                "Upload the PDF document that needs a QR stamp.",
                "Încarcă documentul PDF pe care vrei să aplici codul QR.",
            ),
            text(
                "Enter the target URL or text, choose corner position, size and pages.",
                "Introdu URL-ul sau textul dorit, alege colțul paginii, dimensiunea și paginile.",
            ),
            text(
                "Apply the QR stamp and download the updated PDF.",
                "Aplică ștampila QR și descarcă PDF-ul actualizat.",
            ),
        ),
        note=text(
            "Test the stamped QR code with your phone camera before printing or distributing large batches.",
            "Testează codul QR aplicat cu camera telefonului înainte de tipărire sau distribuire.",
        ),
        faqs=(
            faq(
                "Can I place the QR code only on the first or last page?",
                "Pot plasa codul QR doar pe prima sau pe ultima pagină?",
                "Yes. Choose First page only, Last page only, All pages or a custom page range such as 1-3, 5.",
                "Da. Poți alege doar prima pagină, doar ultima pagină, toate paginile sau un interval personalizat precum 1-3, 5.",
            ),
            faq(
                "Does the QR code work when printed in black and white?",
                "Funcționează codul QR atunci când documentul este tipărit alb-negru?",
                "Yes. The QR stamp is rendered in pure black on white with a clean quiet zone so laser and inkjet printers reproduce it reliably.",
                "Da. Codul QR este generat negru pe alb cu margine de siguranță clară, astfel încât imprimantele îl reproduc fără probleme.",
            ),
        ),
    ),
)

TOOL_LANDINGS_BY_SLUG = {tool.slug: tool for tool in TOOL_LANDINGS}
TOOL_LANDINGS_BY_URL_NAME = {tool.url_name: tool for tool in TOOL_LANDINGS}
DIRECT_TOOL_LINKS: tuple[DirectToolLink, ...] = (
    DirectToolLink(
        url_name="images_to_pdf",
        icon="🧩",
        name=text("Images to PDF", "Imagini în PDF"),
        category=text("Convert", "Conversie"),
        short_description=text(
            "Combine JPG, PNG, TIFF, WebP, BMP or GIF images into one ordered PDF.",
            "Combină imagini JPG, PNG, TIFF, WebP, BMP sau GIF într-un PDF ordonat.",
        ),
    ),
    DirectToolLink(
        url_name="verify_signature",
        icon="🔍",
        name=text("Verify PDF Signatures", "Verifică semnături PDF"),
        category=text("Security", "Securitate"),
        short_description=text(
            "Inspect integrity, signer identity, certificate trust and trusted timestamps.",
            "Verifică integritatea, semnatarul, încrederea certificatului și mărcile temporale.",
        ),
    ),
)
UPLOAD_DESTINATION_NAMES = frozenset(tool.operation_url_name for tool in TOOL_LANDINGS)


def is_upload_destination_allowed(url_name: str) -> bool:
    """Return whether an upload may continue to this fixed internal view."""

    return url_name in UPLOAD_DESTINATION_NAMES
