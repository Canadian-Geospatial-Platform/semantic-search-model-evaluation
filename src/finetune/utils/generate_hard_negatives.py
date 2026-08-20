from sentence_transformers.util import mine_hard_negatives
import logging

logger = logging.getLogger(__name__)

def add_hard_negatives(train_dataset, model, miningkwargs=None):
    """
    Mines hard negatives from the training dataset using the provided model and adds them to the dataset.

    Args:
        train_dataset (Dataset): The training dataset containing anchor and document columns.
        model (SentenceTransformer): The model used for mining hard negatives.
        miningkwargs (dict, optional): A dictionary containing the parameters for mining hard negatives.
    Returns:
        Dataset: The training dataset with hard negatives added.
    """
    # Using best parameters as specified in https://www.sbert.net/docs/package_reference/util/hard_negatives.html
    if miningkwargs is None:
        miningkwargs = {
            "relative_margin": 0.05,
            "num_negatives": 1,
            "sampling_strategy": "top",
            "range_min": 0,
        }

    logger.info(f"Mining hard negatives with the following parameters: {miningkwargs}")

    dataset = mine_hard_negatives(
            dataset=train_dataset,
            model=model,
            relative_margin=miningkwargs.get("relative_margin", 0.05),
            num_negatives=miningkwargs.get("num_negatives", 1),
            sampling_strategy=miningkwargs.get("sampling_strategy", "top"),
            range_min=miningkwargs.get("range_min", 0),
            batch_size=16
        )

    return dataset