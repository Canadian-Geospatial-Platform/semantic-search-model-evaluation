
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import boto3
import os
import logging
from torch.utils.data import Dataset

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __getitem__(self, idx):
        # Tokenize the text at the given index
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze() for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)


def fine_tune_model(model_name, data, save_directory):
    logger.info(f"Starting fine-tuning for {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Tokenize the data
    dataset = TextDataset(tokenizer, data['text'].tolist())

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=save_directory,
        num_train_epochs=3,  # example value, adjust as needed
        per_device_train_batch_size=16,  # example value, adjust as needed
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save the model to the specified directory
    model.save_pretrained(save_directory)
    logger.info(f"Finished fine-tuning for {model_name}")

def upload_to_s3(bucket_name, model_name, directory):
    s3 = boto3.client('s3')
    for filename in os.listdir(directory):
        s3.upload_file(os.path.join(directory, filename), bucket_name, f"{model_name}/{filename}")
        logger.info(f"Uploaded {filename} to S3 bucket {bucket_name}/{model_name}/{filename}")

def main():
    # Load your data
    df = pd.read_csv('records_text.csv')  

    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
    bucket_name = 'semanticsearch-nlp-finetune'  

    for model in models:
        save_directory = f"models/{model}-finetune"
        os.makedirs(save_directory, exist_ok=True)
        fine_tune_model(model, df, save_directory)
        upload_to_s3(bucket_name, model, save_directory)

if __name__ == "__main__":
    main()
