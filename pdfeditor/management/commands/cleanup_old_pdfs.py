"""
Management command pentru ștergerea fișierelor PDF procesate vechi.
"""
import os
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Șterge fișierele PDF procesate mai vechi de PDF_CLEANUP_HOURS ore'

    def add_arguments(self, parser):
        parser.add_argument(
            '--hours',
            type=int,
            default=None,
            help='Numărul de ore după care să șteargă fișierele (default: din settings.PDF_CLEANUP_HOURS)'
        )

    def handle(self, *args, **options):
        cleanup_hours = options.get('hours') or getattr(settings, 'PDF_CLEANUP_HOURS', 24)
        cleanup_threshold = datetime.now() - timedelta(hours=cleanup_hours)
        
        # Cleanup both uploads and processed directories
        directories = [
            os.path.join(settings.MEDIA_ROOT, 'uploads'),
            os.path.join(settings.MEDIA_ROOT, 'processed')
        ]
        
        total_deleted = 0
        total_size = 0
        
        for directory in directories:
            if not os.path.exists(directory):
                continue
            
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                
                # Skip if not a file
                if not os.path.isfile(filepath):
                    continue
                
                # Check modification time
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_mtime < cleanup_threshold:
                    file_size = os.path.getsize(filepath)
                    try:
                        os.remove(filepath)
                        total_deleted += 1
                        total_size += file_size
                        self.stdout.write(
                            self.style.SUCCESS(f'Șters: {filename} ({file_size / 1024:.2f} KB)')
                        )
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f'Eroare la ștergerea {filename}: {str(e)}')
                        )
        
        if total_deleted > 0:
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nTotal: {total_deleted} fișiere șterse ({total_size / 1024 / 1024:.2f} MB)'
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'Nu s-au găsit fișiere mai vechi de {cleanup_hours} ore.')
            )
