



from RAG_evaluation.create_and_evaluate_ragas_dataset import create_ragas_dataset, evaluate_ragas_dataset

from RAG_evaluation.read_qa import read_docx

file_path = "/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Raptor Q&A2.docx"
content = read_docx(file_path)

basic_qa_ragas_dataset = create_ragas_dataset(content)
print(basic_qa_ragas_dataset)
print(type(basic_qa_ragas_dataset))
eval_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)
print(eval_result)
