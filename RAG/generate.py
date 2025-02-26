
from django.shortcuts import HttpResponse, render

from embed_text import embed_text
from similarity import cosine_similarity
from dotenv import load_dotenv
from app.models import Document, TextChunk
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


load_dotenv()

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

def generate_response_with_gpt_turbo(user_question, relevant_text_chunk):
    # Combine the user's question and relevant text chunk into a prompt
    prompt = f"AI Language Model Assistant, based on the provided information:\n"\
            f"User Question: '{user_question}'\n"\
            f"Relevant Text Chunk: '{relevant_text_chunk}'\n"\
            f"Please provide a concise and fact-based answer to the user's question:\n"\
            f"Answer: "
    
    response = chat.invoke([
            HumanMessage(
                content=prompt
            ),
            
        ])

    return response

def generate_response(request):

    if request.method == 'POST':
        user_question = request.POST.get('input_text')
        selected_document_name = request.POST.get('document', '')
        selected_document = Document.objects.filter(pdf_file__icontains=selected_document_name)[0]

        # Embed the user's input
        embeded_question = embed_text([user_question])[0]

        best_text_chunks = []
        # Compare with embeddings in TextChunk objects
        chunks = TextChunk.objects.filter(document=selected_document)

        for text_chunk in chunks:
            similarity = cosine_similarity(embeded_question, text_chunk.embed)
            
            # Add the current chunk to the list if there are less than 3 chunks
            if len(best_text_chunks) < 3:
                best_text_chunks.append((similarity, text_chunk.chunk))
            else:
                # Check if the current chunk is more similar than the least similar chunk in the list
                min_similarity_index = min(range(3), key=lambda i: best_text_chunks[i][0])
                if similarity > best_text_chunks[min_similarity_index][0]:
                    # Replace the least similar chunk with the current chunk
                    best_text_chunks[min_similarity_index] = (similarity, text_chunk.chunk)

        # Extract only the chunks from the tuple
        best_text_chunks = [chunk for _, chunk in best_text_chunks]
        total_text = ''.join(best_text_chunks)
        if best_text_chunks:
            # Use the user's question and the relevant text chunk directly
            response = generate_response_with_gpt_turbo(user_question, total_text)

            return render(request, 'app/prompt_result.html', context={'generated_response': response.content})
            # return render(request, 'app/generate_response.html', {'generated_response': response})

        else:
            return HttpResponse("No similar documents found.")
    
    elif request.method == 'GET':

        documents = Document.objects.all()
        return render(request, 'app/generate_response.html', context={'documents': documents})
