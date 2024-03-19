
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import boto3
import os
import logging
from torch.utils.data import Dataset
import awswrangler as wr


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TextDataset(Dataset):
    def __init__(self, tokenizer, texts=None, max_length=256):
        self.tokenizer = tokenizer
        if texts is not None:
            self.texts = texts
        else:
            self.__load()
        self.max_length = max_length

    def __getitem__(self, idx):
        # Tokenize the text at the given index
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze() for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)

    def __load(self):
        BUCKET_NAME = 'webpresence-geocore-geojson-to-parquet-dev'
        self.df = wr.s3.read_parquet(f"s3://{BUCKET_NAME}/", dataset=True)
        selected_columns = ['features_properties_title_en','features_properties_description_en','features_properties_keywords_en']
        self.df = self.df[selected_columns]
        self.df['text'] = self.df.apply(lambda x: f"{x['features_properties_title_en']}\n{x['features_properties_description_en']}\nkeywords:{x['features_properties_keywords_en']}",axis=1 )
        self.df['text'] = self.df.apply(lambda x:self.__preprocess(x))
        self.texts = self.df['text'].tolist() #needs high mem

    
    def __preprocess(self,text):
        return text




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
