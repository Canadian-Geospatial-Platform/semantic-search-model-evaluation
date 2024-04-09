import pandas as pd
import matplotlib.pyplot as plt
from data_sources import PandasDataSource
from semantic_search import SemanticSearch

def load_test_data(csv_path):
    """ Load test data from CSV """
    return pd.read_csv(csv_path)

def calculate_mrr(semantic_search, test_data):
    """ Calculate Mean Reciprocal Rank (MRR) """
    mrr_total = 0
    for idx, row in test_data.iterrows():
        query = row['query']
        related_documents = eval(row['related_documents']) # Assumes this is a string representation of a list

        results = semantic_search.query(query, k=len(related_documents))
        result_ids = [res[0] for res in results]

        rank = 0
        for doc_id in related_documents:
            if doc_id in result_ids:
                rank = result_ids.index(doc_id) + 1
                break

        if rank > 0:
            mrr_total += 1 / rank

    return mrr_total / len(test_data)

def plot_results(results):
    """ Plot the MRR results """
    models = list(results.keys())
    mrr_scores = list(results.values())

    plt.bar(models, mrr_scores)
    plt.xlabel('Models')
    plt.ylabel('Mean Reciprocal Rank')
    plt.title('Semantic Search Model Benchmarking')
    plt.show()


if __name__ == '__main__':
    # Load test data
    test_data = load_test_data('path_to_test_data.csv')
    #TODO: the model availability on sagemaker
    # Define models to benchmark
    models = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'paraphrase-multilingual-MiniLM-L12-v2'
    ]

    # Assume you have a Pandas data source setup
    data_source = PandasDataSource('s3://your-bucket/your-data.csv')

    # Perform benchmarking
    results = {}
    for model_name in models:
        semantic_search = SemanticSearch(model_name=model_name, data_source=data_source)
        semantic_search.load()
        mrr_score = calculate_mrr(semantic_search, test_data)
        results[model_name] = mrr_score

    # Plot the results
    plot_results(results)
