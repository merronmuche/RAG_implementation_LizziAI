



from RAG_evaluation.create_and_evaluate_ragas_dataset import create_ragas_dataset, evaluate_ragas_dataset

from RAG_evaluation.read_qa import read_docx

file_path = "/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Raptor Q&A2.docx"
content = read_docx(file_path)

data = create_ragas_dataset(content)

eval_result = evaluate_ragas_dataset(data)
print(eval_result)
