
from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    content = []

    current_question = None
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text.endswith('?'):
            # This is a question
            current_question = {'question': text, 'answer': None}
        elif current_question is not None:
            # This is an answer
            current_question['answer'] = text
            content.append(current_question)
            current_question = None

    return content

if __name__ == "__main__":
    file_path = "/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Raptor Q&A2.docx"
    content = read_docx(file_path)
    print(content)
