from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
tqdm.pandas()

MODEL_NAME = 'doc2query/msmarco-14langs-mt5-base-v1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

PREFIXES = ["What is a ", "What is the ", "What is ", "What are ", "What are the "]


def _clean_option(option):
    """Clean generated option by removing common prefixes."""
    text = option.strip()
    for prefix in PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    return text


def _generate_queries_batch(texts, max_length=64):
    """
    Generate queries for a batch of texts using beam search.
    
    Args:
        texts: List of input texts
        max_length: Maximum length of generated queries
    
    Returns:
        List of generated and cleaned query strings
    """
    # Tokenize batch with padding
    encoded = TOKENIZER(
        list(texts),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = MODEL.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            top_k=10,
            num_return_sequences=1,
        )
    
    # Decode and clean (move outputs back to CPU for decoding)
    queries = []
    for idx, output in enumerate(outputs):
        decoded = TOKENIZER.decode(output.cpu(), skip_special_tokens=True)
        cleaned = _clean_option(decoded)
        queries.append(cleaned)
        
    return queries


def create_queries(df, text_col, new_col, batch_size=32, **generate_kwargs):
    """
    Map source text column to generated queries and add them as a new column.
    Uses efficient batch processing for inference.
    
    Args:
        df: Input DataFrame
        text_col: Column name containing source text
        new_col: Column name for generated queries
        batch_size: Batch size for inference (default: 32)
        **generate_kwargs: Generation parameters (max_length)
    
    Returns:
        DataFrame with new_col containing generated query strings
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")
    
    df = df.copy()
    texts = df[text_col].tolist()
    
    # Process texts in batches
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating queries"):
        batch_texts = texts[i:i + batch_size]
        batch_queries = _generate_queries_batch(batch_texts, **generate_kwargs)
        results.extend(batch_queries)
    
    df[new_col] = results
    return df




