import pandas as pd
from nltk.tokenize import sent_tokenize
from finetune import preprocess_records_into_text
import nltk

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import logging
import sys

nltk.download('punkt')

# Configure logging to file and console
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training_log_MPR.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

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

def callback(score, epoch, steps):
    logger.info(f"Epoch: {epoch}, Steps: {steps}, Loss: {score}")
    
def main(path_to_training_data, model_save_directory, num_train_epochs):
    df = pd.read_parquet(path_to_training_data)
    df['text'] = preprocess_records_into_text(df)
    
    sentence_pairs = dataframe_to_sentence_pairs(df, 'text')
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    examples = [InputExample(texts=pair["set"]) for pair in sentence_pairs[:128]]
    
    # DataLoader
    train_dataloader = DataLoader(examples, batch_size=32, shuffle=True)
    
    # MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              epochs=num_train_epochs, 
              warmup_steps=100, 
              output_path=model_save_directory, 
              callback=callback)

# Example usage
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script_name.py [path_to_training_data] [model_save_directory] [num_train_epochs]")
        sys.exit(1)
    
    path_to_training_data = sys.argv[1]
    model_save_directory = sys.argv[2]
    num_train_epochs = int(sys.argv[3])

    main(path_to_training_data, model_save_directory, num_train_epochs)