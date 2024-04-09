from abc import ABC, abstractmethod
import pandas as pd
import boto3
import io

class DataSource(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def connect(self):
        """ Connect to the data source """
        pass

    @abstractmethod
    def fetch(self):
        """ Fetch document IDs and embeddings """
        pass
    
##TODO: use the utils for s3 for get and put
class PandasDataSource(DataSource):
    def __init__(self, s3_path, **kwargs):
        self.s3_path = s3_path
        self.s3_client = None
        self.dataframe = None

    def connect(self):
        """ Connect to the S3 bucket and load data into a DataFrame """
        s3_client = boto3.client('s3') #TODO: remove in cloud env
        bucket, key = self.parse_s3_path(self.s3_path) # TODO: remove in the cloud env

        csv_obj = s3_client.get_object(Bucket=bucket, Key=key) #TODO: use utils/s3/open_file_s3
        body = csv_obj['Body']
        csv_string = body.read().decode('utf-8')

        self.dataframe = pd.read_csv(io.StringIO(csv_string))

    def fetch(self):
        """ Yield document IDs and embeddings from the DataFrame """
        for _, row in self.dataframe.iterrows():
            yield row['doc_id'], row['embedding']

    @staticmethod
    def parse_s3_path(s3_path):
        """ Parse S3 path into bucket and key """
        if not s3_path.startswith('s3://'):
            raise ValueError("S3 path must start with 's3://'")
        parts = s3_path[5:].split('/', 1)
        return parts[0], parts[1]
