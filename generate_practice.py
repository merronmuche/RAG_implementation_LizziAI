import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_chatbot.settings')
django.setup()

import sys
sys.path.append('..')
from app.views import generate_response_with_gpt_turbo
from RAG.embed_text import embed_text
from RAG.similarity import cosine_similarity
from app.models import Document, TextChunk
from asgiref.sync import sync_to_async


async def get_reponse(user_question, selected_document_name):

    selected_documents = await sync_to_async(Document.objects.filter)(pdf_file__icontains=selected_document_name)
    selected_document = await sync_to_async(selected_documents.first)()


    # Embed the user's input
    embeded_question = embed_text([user_question])[0]

    best_text_chunks = []
    chunks = await sync_to_async(list)(TextChunk.objects.filter(document=selected_document))

    for text_chunk in chunks:
        similarity = cosine_similarity(embeded_question, text_chunk.embed)
        
        if len(best_text_chunks) < 3:
            best_text_chunks.append((similarity, text_chunk.chunk))
        else:
            min_similarity_index = min(range(3), key=lambda i: best_text_chunks[i][0])
            if similarity > best_text_chunks[min_similarity_index][0]:
                best_text_chunks[min_similarity_index] = (similarity, text_chunk.chunk)

    # Use a different variable name for the final list
    final_best_text_chunks = []
    for _, chunk in best_text_chunks:
        final_best_text_chunks.append(chunk)

    total_text = ''.join(final_best_text_chunks)

    response = None  # Add a default value
    if best_text_chunks:
        response = generate_response_with_gpt_turbo(user_question, total_text)

    return response.content  




