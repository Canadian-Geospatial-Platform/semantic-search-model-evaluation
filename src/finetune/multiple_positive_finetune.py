import pandas as pd
from nltk.tokenize import sent_tokenize
from finetune import preprocess_records_into_text
import nltk

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import logging
import sys
import os 

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

def dataframe_to_sentence_pairs(texts):
    pairs = []
    for text in texts:
        part1, part2 = split_text_middle(text)
        pairs.append({"set": [part1, part2]})
    return pairs

def callback(score, epoch, steps):
    logger.info(f"Epoch: {epoch}, Steps: {steps}, Loss: {score}")
    
def main(path_to_training_data, model_save_directory, num_train_epochs,model_name,add_french=False):
    logger.info(f"Starting fine-tuning for {model_name}")
    # Check if the model save directory exists, create if not
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
        logger.info(f"Created model save directory: {model_save_directory}")
        
    df = pd.read_parquet(path_to_training_data)
    texts = preprocess_records_into_text(df,add_french).to_list()
    
    sentence_pairs = dataframe_to_sentence_pairs(texts)
    
    model = SentenceTransformer(model_name)
    
    examples = [InputExample(texts=pair["set"]) for pair in sentence_pairs[:]]
    
    # DataLoader
    train_dataloader = DataLoader(examples, batch_size=32, shuffle=True)
    
    # MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Define output path for model saving
    model_output_path = os.path.join(model_save_directory, model_name.split('/')[-1])
    
    try:
      model.fit(train_objectives=[(train_dataloader, train_loss)], 
                epochs=num_train_epochs, 
                warmup_steps=100, 
                output_path=model_output_path, 
                callback=callback)
      logger.info(f"Finished fine-tuning for {model_name}")
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")

# Example usage
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script_name.py [path_to_training_data] [model_save_directory] [num_train_epochs]")
        sys.exit(1)
    
    path_to_training_data = sys.argv[1]
    model_save_directory = sys.argv[2]
    num_train_epochs = int(sys.argv[3])
    models = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]

    for model in models:
        if True or model=='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':
            main(path_to_training_data, model_save_directory, num_train_epochs,model,add_french=True)
        else:
            main(path_to_training_data, model_save_directory, num_train_epochs,model,add_french=False)
            
        