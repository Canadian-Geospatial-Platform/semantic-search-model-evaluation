import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from finetune import preprocess_records_into_text,preprocess_records_into_text_fr
from inference import model_fn,predict_fn
from tqdm import tqdm

def embed_text(model, tokenizer, texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        data = {'inputs': batch_texts}
        batch_embeddings = predict_fn(data, (model, tokenizer))
        embeddings.append(torch.tensor(batch_embeddings))

    # Stack all batch tensors vertically to create a single tensor
    return torch.cat(embeddings, dim=0)


def semantic_search(model, query_embedding, record_embeddings):
    hits = util.semantic_search(query_embedding, record_embeddings, top_k=5)
    return [hit['corpus_id'] for hit in hits[0]]
    
def keyword_search(query, records):
    query = query.lower().split()
    return [i for i, text in enumerate(records) if any(word in text.lower() for word in query)][:5]



# Load the fine-tuned model
model_path = 'models/paraphrase-multilingual-MiniLM-L12-v2-finetune-huggingface'
# model_path = 'sentence-transformers/all-MiniLM-L6-v2'
model,tokenizer = model_fn(model_dir=model_path) 
# model,tokenizer = model_fn(model_path) 

# Load the datasets
labels_df = pd.read_csv('updated_labels.csv', encoding='latin-1')

# labels_df = pd.read_csv('updated_labels_en_and_fr.csv', encoding='latin-1')
records_df = pd.read_parquet('records.parquet')
labels_df = labels_df.dropna(subset=['doc_id'])


# Create embeddings for all records
records_df['text'] = preprocess_records_into_text(records_df) # Using your preprocess function

# records_df['text'] = preprocess_records_into_text_fr(records_df) # Using your preprocess function
record_texts = records_df['text'].tolist()
# assert False
record_embeddings = embed_text(model,tokenizer, record_texts)

# Evaluate
num_correct = 0
keyword_num_correct = 0
predicted_row = []
mrr_total=0
for _, row in labels_df.iterrows():
    query_embedding = embed_text(model,tokenizer, [row['query']])
    top_doc_ids = semantic_search(model, query_embedding, record_embeddings)
    predicted_row.append((row['doc_id'],top_doc_ids))
    # # Find the labeled doc_id in records_df
    labeled_doc_id = row['doc_id']  # Assuming 'result' contains the correct doc_id
    if any(records_df.iloc[doc_id]['features_properties_id'] == labeled_doc_id for doc_id in top_doc_ids):
        num_correct += 1
        mrr_total += 1/(1+[records_df.iloc[doc_id]['features_properties_id'] for doc_id in top_doc_ids].index(labeled_doc_id))
        
    top_keyword_doc_ids = keyword_search(row['query-fr'], record_texts)
    keyword_correct = any(records_df.iloc[doc_id]['features_properties_id'] == labeled_doc_id for doc_id in top_keyword_doc_ids)
    keyword_num_correct += keyword_correct
    
keyword_accuracy = keyword_num_correct / len(labels_df)

dfp = pd.DataFrame.from_records(predicted_row,columns=['doc_id','top_ids'])
d = model_path.split('/')[-1]
print(d)
dfp.to_csv(f'finetuned_predicted_result_{d}.csv')
accuracy = num_correct / len(labels_df)
print(f"Accuracy: {accuracy}")
print(f"MRR is {mrr_total / len(labels_df)}")
print(f"Keyword Search Accuracy: {keyword_accuracy}")
