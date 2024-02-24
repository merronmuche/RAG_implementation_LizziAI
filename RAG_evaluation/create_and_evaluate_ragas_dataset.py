


from tqdm import tqdm
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)
import pandas as pd
from datasets import Dataset

from RAG_evaluation.generate_resp import get_reponse


async def create_ragas_dataset(eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    # answer, context = get_reponse(row["question"], selected_document_name='pdfs/Raptor_Contract.docx_odQK02a.pdf')
    answer, context = await get_reponse(row["question"], selected_document_name='pdfs/Raptor_Contract.docx_odQK02a.pdf')
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer,
         "contexts" : [context[0]],
         "ground_truths" : [row["answer"]] # This is the ground truth
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result


if __name__=="__main__":
  
    from read_qa import read_docx

    file_path = "/home/meron/Documents/work/tenacademy/week11/RAG_implementation_LizziAI/datalizzy/Raptor Q&A2.docx"
    content = read_docx(file_path)
    
    data = create_ragas_dataset(content)

    print(data)

