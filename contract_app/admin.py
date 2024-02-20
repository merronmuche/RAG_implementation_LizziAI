from django.contrib import admin


from django.contrib import admin
from .models import Document, TextChunk

class DocumentAdmin(admin.ModelAdmin):
    list_display = ('pdf_file', 'created_at', 'updated_at')
    search_fields = ('pdf_file',)
    readonly_fields = ('created_at', 'updated_at')

class TextChunkAdmin(admin.ModelAdmin):
    list_display = ('chunk', 'embed', 'updated_at')
  
admin.site.register(Document, DocumentAdmin)
admin.site.register(TextChunk, TextChunkAdmin)

