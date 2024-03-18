from django.shortcuts import HttpResponse, render
from RAG.embed_text import embed_text
from RAG.similarity import cosine_similarity
from dotenv import load_dotenv
from .models import Conversation, Document, TextChunk
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from django.contrib.auth.decorators import login_required

load_dotenv()

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)


def generate_response_with_gpt_turbo(user_question, relevant_text_chunk, history=""):
    prompt = (
        f"AI Language Model Assistant, based on the provided information:\n"
        f"User Question: '+ \n {history} \n {user_question}'\n"
        f"Relevant Text Chunk: '{relevant_text_chunk}'\n"
        f"Please provide a short, concise and fact-based answer to the user's question:\n"
        f"Answer: "
    )

    # Implement the GPT-3.5 Turbo API call to generate a response
    response = chat.invoke(
        [
            HumanMessage(content=prompt),
        ]
    )
    return response


@login_required
def generate_response(request):
    documents = Document.objects.all()

    if request.method == "POST":
        user_question = request.POST.get("input_text")

        # Retriever
        """
        The retriever takes in a user question and returns chunks.
        """
        selected_document_name = request.POST.get("document", "")
        selected_document = Document.objects.filter(
            pdf_file__icontains=selected_document_name
        )[0]
        converstations = Conversation.objects.all()

        history = {}
        for conv in converstations:
            x = {"Question": conv.question, "Answer": conv.answer}
            history.update(x)

        embeded_question = embed_text([user_question])[0]

        best_text_chunks = []
        chunks = TextChunk.objects.filter(document=selected_document)

        for text_chunk in chunks:
            similarity = cosine_similarity(embeded_question, text_chunk.embed)

            if len(best_text_chunks) < 3:
                best_text_chunks.append((similarity, text_chunk.chunk))
            else:
                min_similarity_index = min(
                    range(3), key=lambda i: best_text_chunks[i][0]
                )
                if similarity > best_text_chunks[min_similarity_index][0]:
                    best_text_chunks[min_similarity_index] = (
                        similarity,
                        text_chunk.chunk,
                    )

        best_text_chunks = [chunk for _, chunk in best_text_chunks]
        total_text = "".join(best_text_chunks)
        ############################################# end of retriever

        # Generator
        response = generate_response_with_gpt_turbo(user_question, total_text, history)

        # save the conversation
        Conversation.objects.create(question=user_question, answer=response.content)
        ###################### end of generator
        return render(
            request,
            "contract_app/generate_response.html",
            context={
                "generated_response": response.content,
                "user_question": user_question,
                "documents": documents,
            },
        )

    elif request.method == "GET":
        return render(
            request,
            "contract_app/generate_response.html",
            context={"documents": documents},
        )
