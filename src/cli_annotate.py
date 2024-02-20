from data_sources import PandasDataSource
from semantic_search import SemanticSearch

def run_cli(semantic_search):
    while True:
        # Get user input
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        # Perform semantic search and get top result
        results = semantic_search.query(user_query, k=1)
        if results:
            top_result_id, top_result_score = results[0]
            print(f"Top result: Document ID {top_result_id} with score {top_result_score}")

            # Get user feedback
            feedback = input("Is this result relevant? (yes/no): ")
            if feedback.lower() == 'yes':
                print("Great! Happy to help.")
            else:
                print("Sorry about that. Let's try another query.")
        else:
            print("No results found for your query.")

        print("-" * 40)

# Setup Semantic Search
data_source = PandasDataSource('s3://your-bucket/your-data.csv')  # Replace with your S3 path
model_name = 'all-MiniLM-L6-v2'  # Replace with your preferred model
semantic_search = SemanticSearch(model_name=model_name, data_source=data_source)
semantic_search.load()

# Run CLI
run_cli(semantic_search)
