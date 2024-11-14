from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        base_url="http://localhost:11435",  # Update if you used a different port
        model="llama3"  # Or whichever model you pulled
    )
