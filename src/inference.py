import logging

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check for CUDA device and use it if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def model_fn(model_dir):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir,add_pooling_layer=False)

    # Move model to the appropriate device (GPU or CPU)
    model.to(device)

    return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    # Destruct model and tokenizer
    model, tokenizer = model_and_tokenizer

    # Tokenize sentences
    sentences = data.pop("inputs", data)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Move tensors to the same device as model
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Return dictionary, which will be JSON serializable
    return sentence_embeddings.tolist()
