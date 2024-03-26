import pandas as pd
from nltk.tokenize import sent_tokenize
from finetune import preprocess_records_into_text
def row_to_sentence_pairs(df, text_column):
    pairs = []
    for text in df[text_column]:
        sentences = sent_tokenize(text)
        # Create pairs of sentences from the text
        for i in range(len(sentences) - 1):
            pairs.append({"set": [sentences[i], sentences[i + 1]]})
    return pairs

def split_text_middle(text):
    sentences = sent_tokenize(text)
    middle_index = len(sentences) // 2
    part1 = " ".join(sentences[:middle_index])
    part2 = " ".join(sentences[middle_index:])
    return part1, part2

def dataframe_to_sentence_pairs(df, text_column):
    pairs = []
    for text in df[text_column]:
        part1, part2 = split_text_middle(text)
        pairs.append({"set": [part1, part2]})
    return pairs

# Example usage

df = pd.read_parquet('records.parquet')
df['text'] = preprocess_records_into_text(df)

sentence_pairs = dataframe_to_sentence_pairs(df, 'text')

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Convert the sentence pairs to InputExample format
examples = [InputExample(texts=pair["set"]) for pair in sentence_pairs]

# DataLoader
train_dataloader = DataLoader(examples, batch_size=16, shuffle=True)

# MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Training
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
