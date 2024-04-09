from data_sources import DataSource
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

class SemanticSearch:
    def __init__(self, model_name, data_source: DataSource, similarity_measure='cosine', **kwargs):
        self.model_name = model_name
        self.similarity_measure = similarity_measure
        self.data_source = data_source
        self.model = None

    def load(self):
        """ Load the sBERT model and connect to the data source """
        self.model = SentenceTransformer(self.model_name)
        self.data_source.connect()

    def query(self, query_str, k):
        """ Perform a semantic search and return top k results """
        query_embedding = self.model.encode(query_str)
        
        top_k_results = []
        for doc_id, doc_embedding in self.data_source.fetch():
            # Compute similarity score
            score = 1 - cdist([query_embedding], [doc_embedding], metric=self.similarity_measure)
            top_k_results.append((doc_id, score[0][0]))

        # Sort the results by score in descending order and get top k
        top_k_results.sort(key=lambda x: x[1], reverse=True)
        return top_k_results[:k]
