

from collections.abc import Iterable
from django.db import models
from .utils.chunk_pdf import extract_text_from_pdf, chunk_text
from .utils.embed_text import embed_text
import logging

logger = logging.getLogger(__name__)



class Prompt(models.Model):
    input_text = models.TextField()
    length_of_line = models.IntegerField(null=True)
    generated_prompt = models.TextField(null=True)

    def __str__(self):
        return f"Prompt: {self.input_text[:50]}..."  # Display first 50 characters
    

class Document(models.Model):
    pdf_file = models.FileField(upload_to='pdfs/')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"PDF File uploaded on {self.created_at}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.pdf_file:  
            text = extract_text_from_pdf(self.pdf_file.path)
            chunks = chunk_text(text)
            embeds = embed_text(chunks)

            for chunk, embed in zip(chunks, embeds):
                TextChunk.objects.create(document=self, chunk=chunk, embed=embed)
        else:
            logger.error("No PDF file to process.")

class TextChunk(models.Model):

    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    chunk = models.TextField()
    embed = models.JSONField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

