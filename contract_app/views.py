
# from django.shortcuts import HttpResponse, render

# from contract_app.utils.prompt_gen import get_prompt
# from .utils.embed_text import embed_text
# from .utils.similarity import cosine_similarity

# from .models import Prompt, TextChunk



# def generate_prompt_from_vector(text_chunk, user_question):

#     prompt = text_chunk + " \n based on the above data, give an answer to \
#             the following question. restrict yourself to the above data only. \
#             if you can't get an answer based on the data, you can feel free to \
#                 say i don't know. here is the question. \n" + user_question +\
#                  "DONOT add anything from yourself."
#     return prompt



# def generate_prompt(request):
#     if request.method == 'POST':
#         input_text = request.POST.get('input_text')

#         # Embed the user's input
#         embeded_question = embed_text([input_text])[0]

#         highest_similarity = -1
#         best_text_chunk = None

#         # Compare with embeddings in TextChunk objects
#         for text_chunk in TextChunk.objects.all():
#             similarity = cosine_similarity(embeded_question, text_chunk.embed)

#             if similarity > highest_similarity:
#                 highest_similarity = similarity
#                 best_text_chunk = text_chunk.chunk

#         if best_text_chunk is not None:
#             generated_prompt = generate_prompt_from_vector(best_text_chunk, input_text)
#             # return HttpResponse(generated_prompt)
#             return render(request, 'contract_app/prompt_result.html', {'generated_prompt': generated_prompt})
#         else:
#             return HttpResponse("No similar documents found.")
#     else:
#         return render(request, 'contract_app/generate_prompt.html')
    
       

from django.shortcuts import HttpResponse, render

from .utils.embed_text import embed_text
from .utils.similarity import cosine_similarity
from dotenv import load_dotenv
from .models import TextChunk
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


load_dotenv()

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)



def generate_response_with_gpt_turbo(user_question, relevant_text_chunk):
    # Combine the user's question and relevant text chunk into a prompt
    prompt = f"based on the thi relevant_text_chunk, give an answer to \
              the users question. restrict yourself to the given data only. \
              if you can't get an answer based on the data, you can feel free to \
              say i don't know.\n + {user_question}\n{relevant_text_chunk}\n"

    # Implement the GPT-3.5 Turbo API call to generate a response
    # Replace the following line with your actual API call
    response = chat.invoke([
            HumanMessage(
                content=prompt
            ),
            
        ])
    

    # Assuming response is the generated answer
    return response

def generate_response(request):
    if request.method == 'POST':
        user_question = request.POST.get('input_text')

        # Embed the user's input
        embeded_question = embed_text([user_question])[0]

        highest_similarity = -1
        best_text_chunk = None

        # Compare with embeddings in TextChunk objects
        for text_chunk in TextChunk.objects.all():
            similarity = cosine_similarity(embeded_question, text_chunk.embed)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_text_chunk = text_chunk.chunk

        if best_text_chunk is not None:
            # Use the user's question and the relevant text chunk directly
            response = generate_response_with_gpt_turbo(user_question, best_text_chunk)

            return render(request, 'contract_app/prompt_result.html', {'generated_response': response})
            # return render(request, 'contract_app/generate_response.html', {'generated_response': response})

        else:
            return HttpResponse("No similar documents found.")
    elif request.method == 'GET':
        return render(request, 'contract_app/generate_response.html')
