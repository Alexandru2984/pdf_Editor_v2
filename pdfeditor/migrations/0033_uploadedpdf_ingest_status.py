from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("pdfeditor", "0032_webhookdelivery"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedpdf",
            name="ingest_status",
            field=models.CharField(
                choices=[
                    ("sanitized", "Sanitized"),
                    ("encrypted", "Encrypted — unprotect only"),
                ],
                db_index=True,
                default="sanitized",
                max_length=16,
            ),
        ),
    ]
