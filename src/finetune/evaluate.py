import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from finetune import preprocess_records_into_text
def embed_text(model, texts):
    with torch.no_grad():
        return model.encode(texts, convert_to_tensor=True)

def semantic_search(model, query_embedding, record_embeddings):
    hits = util.semantic_search(query_embedding, record_embeddings, top_k=5)
    return [hit['corpus_id'] for hit in hits[0]]
    
def keyword_search(query, records):
    query = query.lower().split()
    return [i for i, text in enumerate(records) if any(word in text.lower() for word in query)][:5]

# Load the fine-tuned model
model_path = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_path)

# Load the datasets
labels_df = pd.read_csv('updated_labels.csv')
records_df = pd.read_parquet('records.parquet')
labels_df = labels_df.dropna(subset=['doc_id'])


# Create embeddings for all records
records_df['text'] = preprocess_records_into_text(records_df) # Using your preprocess function
record_texts = records_df['text'].tolist()
record_embeddings = embed_text(model, record_texts)

# Evaluate
num_correct = 0
keyword_num_correct = 0

for _, row in labels_df.iterrows():
    query_embedding = embed_text(model, [row['query']])
    top_doc_ids = semantic_search(model, query_embedding, record_embeddings)
    
    # # Find the labeled doc_id in records_df
    labeled_doc_id = row['doc_id']  # Assuming 'result' contains the correct doc_id
    if any(records_df.iloc[doc_id]['features_properties_id'] == labeled_doc_id for doc_id in top_doc_ids):
        num_correct += 1
        
    top_keyword_doc_ids = keyword_search(row['query'], record_texts)
    keyword_correct = any(records_df.iloc[doc_id]['features_properties_id'] == labeled_doc_id for doc_id in top_keyword_doc_ids)
    keyword_num_correct += keyword_correct
    
keyword_accuracy = keyword_num_correct / len(labels_df)


accuracy = num_correct / len(labels_df)
print(f"Accuracy: {accuracy}")
print(f"Keyword Search Accuracy: {keyword_accuracy}")
