
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification,AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments,AutoModelForCausalLM,DataCollatorForLanguageModeling
# import boto3
import os
import logging
from torch.utils.data import Dataset
import sys
# import awswrangler as wr


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_records_into_text(df):
        selected_columns = ['features_properties_title_en','features_properties_description_en','features_properties_keywords_en']
        df = df[selected_columns]
        return df.apply(lambda x: f"{x['features_properties_title_en']}\n{x['features_properties_description_en']}\nkeywords:{x['features_properties_keywords_en']}",axis=1 )
        

class TextDataset(Dataset):
    def __init__(self, tokenizer,df=None, texts=None, max_length=256):
        self.tokenizer = tokenizer
        if texts is not None:
            self.texts = texts
        else:
            self.__load(df)
        self.max_length = max_length

    def __getitem__(self, idx):
        # Tokenize the text at the given index
        #TODO: for long document chunk it to different docs to max_length
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze() for key, val in encoding.items()}

    def __len__(self):
        return len(self.texts)

    def __load(self,df):
        # BUCKET_NAME = 'webpresence-geocore-geojson-to-parquet-dev'
        # self.df = wr.s3.read_parquet(f"s3://{BUCKET_NAME}/", dataset=True)
        self.df=df
        selected_columns = ['features_properties_title_en','features_properties_description_en','features_properties_keywords_en']
        self.df = self.df[selected_columns]
        self.df['text'] = self.df.apply(lambda x: f"{x['features_properties_title_en']}\n{x['features_properties_description_en']}\nkeywords:{x['features_properties_keywords_en']}",axis=1 )
        # self.df['text'] = self.df.apply(lambda x:self.__preprocess(x))
        self.texts = self.df['text'].tolist() #needs high mem

    
    def __preprocess(self,text):
        return text
        
class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)




def fine_tune_model(model_name, save_directory, data_path, num_train_epochs):
    logger.info(f"Starting fine-tuning for {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        model = AutoModelForMaskedLM.from_pretrained(model_name)


    # Tokenize the data
    df = pd.read_parquet(data_path)
    dataset = TextDataset(tokenizer, df=df)
    texts = dataset.texts[:100]
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    
    dataset = MyDataset(encodings)

    # Data collator used for dynamic masking
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results', 
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()

    # Save the model to the specified directory
    model.save_pretrained(save_directory)
    logger.info(f"Finished fine-tuning for {model_name}")

# def upload_to_s3(bucket_name, model_name, directory):
#     s3 = boto3.client('s3')
#     for filename in os.listdir(directory):
#         s3.upload_file(os.path.join(directory, filename), bucket_name, f"{model_name}/{filename}")
#         logger.info(f"Uploaded {filename} to S3 bucket {bucket_name}/{model_name}/{filename}")


def load_saved_model(model_directory):
    """
    Loads a model from the specified directory.

    Args:
        model_directory (str): The directory where the model is saved.

    Returns:
        model: The loaded pre-trained model.
    """
    try:
        # Try to load as a causal language model
        model = AutoModelForCausalLM.from_pretrained(model_directory)
        logger.info(f"Loaded Causal Language Model from {model_directory}")
    except Exception as e1:
        try:
            # If the first attempt fails, try to load as a masked language model
            model = AutoModelForMaskedLM.from_pretrained(model_directory)
            logger.info(f"Loaded Masked Language Model from {model_directory}")
        except Exception as e2:
            logger.error(f"Failed to load model: {e1}, then {e2}")
            return None

    return model


def main():
    # Check if all required arguments are passed
    if len(sys.argv) != 4:
        print("Usage: python finetune.py [data_path] [save_directory] [num_train_epochs]")
        sys.exit(1)

    # Retrieve arguments
    data_path = sys.argv[1]
    save_directory_base = sys.argv[2]
    num_train_epochs = int(sys.argv[3])

    models = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    bucket_name = 'semanticsearch-nlp-finetune'  

    for model in models:
        save_directory = f"{save_directory_base}/{model}-finetune"
        os.makedirs(save_directory, exist_ok=True)
        fine_tune_model(model, save_directory, data_path, num_train_epochs)
        # upload_to_s3(bucket_name, model, save_directory)

if __name__ == "__main__":
    main()
