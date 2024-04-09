# Evaluate Sentence-Transformer Models for Semantic Search Engine

This repository is dedicated to evaluating Sentence-Transformer models optimized for use within a semantic search engine. Our goal is to fine-tune and assess various Sentence-Transformer models to enhance the relevancy and accuracy of search results in semantic search with Amazon OpenSearch.

## Overview

Semantic search engines use understanding beyond keyword matching to interpret the intent and contextual meaning of search queries. By leveraging Sentence-Transformer models, we aim to significantly improve the search experience, offering users more accurate, context-aware results.



## Model Fine-Tuning Script



### Prerequisites

Before running the script, ensure you have the following:
- Python 3.x installed
- Access to a command-line interface (CLI)
- Necessary permissions to run bash scripts and Python scripts on your system
- The required Python environment and dependencies installed

### Setup

1. **Clone the Repository:**
   Ensure that you have the repository cloned to your local machine.

2. **Environment Setup:**
   Ensure that your Python environment is set up correctly and that all required packages are installed. 

   ```bash
   pip install -r finetune/requirement.txt
   ```

3. **Data Preparation:**
   Prepare your training data and note its file path. The data file should be in `.parquet` format for the given Python script.

### Running the Script

To run the script, you'll need to pass three arguments: the path to the training data, the directory where the model should be saved, and the number of training epochs.



1. **Run the Bash Script:**
   Execute the bash script with the required arguments. Here is the format for running the script:

   ```bash
   bash finetune/finetune.sh [path_to_training_data] [model_save_directory] [num_train_epochs]
   ```

   Replace `[path_to_training_data]` with the full path to your training data, `[model_save_directory]` with the path where you want the trained model to be saved, and `[num_train_epochs]` with the number of epochs for training.

   Example:
   ```bash
   bash finetune/finetune.sh  /path/to/training/data /path/to/save/directory 5
   ```

### Notes

- Ensure that the paths provided are accessible and the user has the necessary read/write permissions.
- Adjust the number of epochs according to your model requirements and hardware capabilities.


