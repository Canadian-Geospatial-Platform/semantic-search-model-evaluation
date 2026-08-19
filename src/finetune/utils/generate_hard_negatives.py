from sentence_transformers.util import mine_hard_negatives

def add_hard_negatives(train_dataset, model):
    """
    Mines hard negatives from the training dataset using the provided model and adds them to the dataset.

    Args:
        train_dataset (Dataset): The training dataset containing anchor and document columns.
        model (SentenceTransformer): The model used for mining hard negatives.
    
    Returns:
        Dataset: The training dataset with hard negatives added.
    """
    # Using best parameters as specified in https://www.sbert.net/docs/package_reference/util/hard_negatives.html
    dataset = mine_hard_negatives(
            dataset=train_dataset,
            model=model,
            relative_margin=0.05,         # keep negatives at least 5% of the positive score magnitude below the positive
            num_negatives=1,
            sampling_strategy="top",      # "top" means that we sample the top candidates as negatives
            batch_size=32
        )

    return dataset