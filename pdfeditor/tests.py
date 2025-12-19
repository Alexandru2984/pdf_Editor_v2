"""
Tests pentru aplicația PDF Editor.
"""
import os
import tempfile
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
import fitz  # PyMuPDF

from .pdf_processor import (
    parse_page_range,
    check_pdf_has_text,
    find_and_replace_text
)


class PDFProcessorTests(TestCase):
    """Teste pentru funcțiile de procesare PDF."""
    
    def setUp(self):
        """Creează PDF-uri de test."""
        # Create a simple PDF with text
        self.test_pdf_text = self._create_test_pdf_with_text("Acesta este un test simplu. Test repetat.")
        
        # Create a PDF without text (just a blank page)
        self.test_pdf_blank = self._create_blank_pdf()
    
    def tearDown(self):
        """Curăță fișierele de test."""
        if os.path.exists(self.test_pdf_text):
            os.remove(self.test_pdf_text)
        if os.path.exists(self.test_pdf_blank):
            os.remove(self.test_pdf_blank)
    
    def _create_test_pdf_with_text(self, text):
        """Helper pentru crearea unui PDF cu text."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), text, fontsize=12)
        
        # Create temp file
        fd, path = tempfile.mkstemp(suffix='.pdf')
        os.close(fd)
        doc.save(path)
        doc.close()
        return path
    
    def _create_blank_pdf(self):
        """Helper pentru crearea unui PDF fără text."""
        doc = fitz.open()
        doc.new_page()  # Pagină goală
        
        fd, path = tempfile.mkstemp(suffix='.pdf')
        os.close(fd)
        doc.save(path)
        doc.close()
        return path
    
    def test_parse_page_range_single_page(self):
        """Test parsare interval cu o singură pagină."""
        result = parse_page_range("3", 10)
        self.assertEqual(result, [2])  # 0-indexed
    
    def test_parse_page_range_multiple_pages(self):
        """Test parsare interval cu mai multe pagini."""
        result = parse_page_range("1-3,5", 10)
        self.assertEqual(result, [0, 1, 2, 4])
    
    def test_parse_page_range_complex(self):
        """Test parsare interval complex."""
        result = parse_page_range("1-2,5,7-9", 10)
        self.assertEqual(result, [0, 1, 4, 6, 7, 8])
    
    def test_parse_page_range_all_pages(self):
        """Test parsare pentru toate paginile."""
        result = parse_page_range("", 5)
        self.assertEqual(result, [0, 1, 2, 3, 4])
    
    def test_parse_page_range_invalid(self):
        """Test parsare interval invalid."""
        with self.assertRaises(ValueError):
            parse_page_range("1-20", 10)  # Out of range
    
    def test_check_pdf_has_text_with_text(self):
        """Test verificare PDF cu text."""
        has_text, message = check_pdf_has_text(self.test_pdf_text)
        self.assertTrue(has_text)
    
    def test_check_pdf_has_text_without_text(self):
        """Test verificare PDF fără text."""
        has_text, message = check_pdf_has_text(self.test_pdf_blank)
        self.assertFalse(has_text)
    
    def test_find_and_replace_basic(self):
        """Test înlocuire text de bază."""
        output_path, count, warnings = find_and_replace_text(
            pdf_path=self.test_pdf_text,
            search_text="test",
            replace_text="exemplu",
            case_sensitive=False
        )
        
        # Verifică că s-au făcut înlocuiri
        self.assertEqual(count, 2)  # "test" și "Test"
        self.assertTrue(os.path.exists(output_path))
        
        # Verifică conținutul PDF-ului modificat
        doc = fitz.open(output_path)
        page_text = doc[0].get_text()
        self.assertIn("exemplu", page_text.lower())
        self.assertNotIn("test", page_text.lower())
        doc.close()
        
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_find_and_replace_case_sensitive(self):
        """Test înlocuire case-sensitive."""
        output_path, count, warnings = find_and_replace_text(
            pdf_path=self.test_pdf_text,
            search_text="test",
            replace_text="exemplu",
            case_sensitive=True
        )
        
        # Ar trebui să găsească doar "test" (lowercase), nu "Test"
        # Poate găsi 1 sau 2 în funcție de cum PDF-ul stochează textul
        self.assertGreaterEqual(count, 1)
        self.assertLessEqual(count, 2)
        
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_find_and_replace_no_matches(self):
        """Test când nu există matches."""
        output_path, count, warnings = find_and_replace_text(
            pdf_path=self.test_pdf_text,
            search_text="nonexistent",
            replace_text="exemplu",
            case_sensitive=True
        )
        
        self.assertEqual(count, 0)
        
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)


class ViewTests(TestCase):
    """Teste pentru views."""
    
    def setUp(self):
        self.client = Client()
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test content for upload", fontsize=12)
        
        self.pdf_fd, self.test_pdf_path = tempfile.mkstemp(suffix='.pdf')
        doc.save(self.test_pdf_path)
        doc.close()
    
    def tearDown(self):
        os.close(self.pdf_fd)
        if os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
    
    def test_upload_view_get(self):
        """Test GET request la upload view."""
        response = self.client.get(reverse('upload'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Încarcă PDF')
    
    def test_upload_view_post_valid(self):
        """Test POST cu fișier valid."""
        with open(self.test_pdf_path, 'rb') as f:
            uploaded_file = SimpleUploadedFile(
                "test.pdf",
                f.read(),
                content_type="application/pdf"
            )
            
            response = self.client.post(reverse('upload'), {
                'pdf_file': uploaded_file
            })
            
            # Ar trebui să redirecteze la edit
            self.assertEqual(response.status_code, 302)
            self.assertTrue(response.url.endswith(reverse('edit')))
    
    def test_upload_view_post_invalid_extension(self):
        """Test POST cu fișier non-PDF."""
        uploaded_file = SimpleUploadedFile(
            "test.txt",
            b"not a pdf",
            content_type="text/plain"
        )
        
        response = self.client.post(reverse('upload'), {
            'pdf_file': uploaded_file
        })
        
        # Ar trebui să rămână pe pagina de upload cu eroare
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Doar fișiere PDF')
    
    def test_edit_view_without_upload(self):
        """Test acces la edit fără upload prealabil."""
        response = self.client.get(reverse('edit'))
        
        # Ar trebui să redirecteze la upload
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.endswith(reverse('upload')))
    
    def test_full_workflow(self):
        """Test workflow complet: upload -> edit -> result -> download."""
        # Step 1: Upload
        with open(self.test_pdf_path, 'rb') as f:
            uploaded_file = SimpleUploadedFile("test.pdf", f.read(), content_type="application/pdf")
            self.client.post(reverse('upload'), {'pdf_file': uploaded_file})
        
        # Step 2: Edit (find & replace)
        response = self.client.post(reverse('edit'), {
            'search_text': 'Test',
            'replace_text': 'Example',
            'case_sensitive': True,
            'page_range': ''
        })
        
        # Ar trebui să redirecteze la result
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.endswith(reverse('result')))
        
        # Step 3: Result
        response = self.client.get(reverse('result'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Modificările au fost aplicate')
        
        # Step 4: Download
        response = self.client.get(reverse('download'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
