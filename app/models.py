from django.db import models
from RAG.read_chunk import read_docx, chunk_text
from RAG.embed_text import embed_text
import logging

logger = logging.getLogger(__name__)


class Document(models.Model):
    pdf_file = models.FileField(upload_to="docx/")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"docx File uploaded on {self.created_at}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.pdf_file:
            text = read_docx(self.pdf_file.path)
            chunks = chunk_text(text)
            embeds = embed_text(chunks)

            for chunk, embed in zip(chunks, embeds):
                TextChunk.objects.create(document=self, chunk=chunk, embed=embed)
        else:
            logger.error("No docx file to process.")



class TextChunk(models.Model):
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="textchunks"
    )
    chunk = models.TextField()
    embed = models.JSONField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Topic(models.Model):
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Conversation(models.Model):
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, null = True)
    question = models.CharField(max_length=100)
    answer = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)



