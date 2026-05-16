from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("pdfeditor", "0024_alter_processedpdf_kind"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="celery_task_id",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
    ]
