from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = 'doc2query/msmarco-14langs-mt5-base-v1'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def _generate_queries(text, max_length=64, top_p=0.95, top_k=10):
    input_ids = TOKENIZER.encode(str(text), return_tensors='pt')
    with torch.no_grad():
        sampling_outputs = MODEL.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=2,
        )
    
    options = [
        TOKENIZER.decode(output, skip_special_tokens=True)
        for output in sampling_outputs
    ]

    prefixes = ["What is a ", "What is the ", "What is ", "What are ", "What are the "]

    def _clean_option(option):
        text = option.strip()
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        return text

    cleaned_options = [_clean_option(option) for option in options]

    # return longer query for more specificity
    return cleaned_options[1] if len(cleaned_options[1]) > len(cleaned_options[0]) else cleaned_options[0]


def create_queries(df, text_col, new_col, **generate_kwargs):
    """Map the source text column to generated query lists and add them as a new column."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")
    
    df = df.copy()
    df[new_col] = df[text_col].map(
        lambda text: _generate_queries(text, **generate_kwargs)
    )
    return df




