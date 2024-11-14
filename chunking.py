from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from get_embedding_function import get_embedding_function
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
import shutil
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity  # For similarity computation

app = Flask(__name__)

CHROMA_PATH = "chroma"
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the 'uploads' directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check if the uploaded file is a PDF
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to clear old files in the 'uploads' folder
def clear_old_files():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Function to clear Chroma database
def clear_chroma_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"✅ Chroma database at {CHROMA_PATH} has been cleared.")

# Topic-based chunking function
def topic_based_chunking(text, num_topics=3):
    paragraphs = text.split("\n")
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = get_embedding_function.embed_query(paragraphs)
    kmeans = KMeans(n_clusters=num_topics, random_state=0)
    kmeans.fit(embeddings)

    topic_chunks = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in topic_chunks:
            topic_chunks[label] = []
        topic_chunks[label].append(paragraphs[i])

    chunks = [" ".join(topic) for topic in topic_chunks.values()]
    return chunks

# Heading-based chunking function
def heading_based_chunking(text):
    sections = text.split("\n\n")
    chunks = []
    current_chunk = []
    for section in sections:
        if section.isupper():
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        current_chunk.append(section)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Semantic chunking function
def semantic_chunking(text, chunk_size=5):
    sentences = text.split(". ")
    embeddings = get_embedding_function.embed_query(sentence)#SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # embeddings = model.encode(sentences)

    chunks = []
    current_chunk = []
    current_embedding = []

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_embedding.append(embeddings[i])

        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_embedding = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Load documents from a PDF
def load_documents(file_path):
    document_loader = PyMuPDFLoader(file_path)
    return document_loader.load()

# Process file with chunking methods in parallel
def process_file(file_path):
    documents = load_documents(file_path)
    all_topic_chunks = []
    all_heading_chunks = []
    all_semantic_chunks = []

    # Use ThreadPoolExecutor to perform the chunking methods in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for doc in documents:
            text = doc.page_content
            futures.append(executor.submit(chunk_documents, text, doc.metadata))

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            topic_chunks, heading_chunks, semantic_chunks = future.result()
            all_topic_chunks.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in topic_chunks])
            all_heading_chunks.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in heading_chunks])
            all_semantic_chunks.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in semantic_chunks])

    return all_topic_chunks, all_heading_chunks, all_semantic_chunks

# Function to handle chunking of documents
def chunk_documents(text, metadata):
    topic_chunks = topic_based_chunking(text, num_topics=3)
    heading_chunks = heading_based_chunking(text)
    semantic_chunks = semantic_chunking(text, chunk_size=5)

    return topic_chunks, heading_chunks, semantic_chunks

# Add chunks to Chroma
def add_to_chroma(chunks, method_name):
    db = Chroma(persist_directory=f"{CHROMA_PATH}_{method_name}", embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    db.add_documents(chunks_with_ids)
    db.persist()

# Calculate unique IDs for chunks
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

# Query the RAG system
def query_rag(query_text, method_name):
    db = Chroma(persist_directory=f"{CHROMA_PATH}_{method_name}", embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        return f"No results found for {method_name} method."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(base_url="http://localhost:11435", model="llama3")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id") for doc, _ in results]
    
    return {"response": response_text, "sources": sources}

# Function to calculate cosine similarity between two text embeddings
def calculate_similarity(real_answer, chunk_response):
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # real_answer_embedding = model.encode([real_answer])
    embeddings = get_embedding_function.embed_query(real_answer)
    chunk_response_embedding = get_embedding_function.embed_query([chunk_response])
    
    similarity_score = cosine_similarity(real_answer_embedding, chunk_response_embedding)[0][0]
    return similarity_score

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files or 'query' not in request.form or 'answer' not in request.form:
        return jsonify({"error": "No file, query, or answer found in the request"}), 400

    file = request.files['file']
    query = request.form['query']
    real_answer = request.form['answer']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        clear_old_files()
        clear_chroma_db()

        file.save(file_path)

        topic_chunks, heading_chunks, semantic_chunks = process_file(file_path)

        add_to_chroma(topic_chunks, "topic_based")
        add_to_chroma(heading_chunks, "heading_based")
        add_to_chroma(semantic_chunks, "semantic_based")

        topic_response = query_rag(query, "topic_based")
        heading_response = query_rag(query, "heading_based")
        semantic_response = query_rag(query, "semantic_based")

        # Compare the real answer with the responses using similarity
        topic_similarity = calculate_similarity(real_answer, topic_response["response"])
        heading_similarity = calculate_similarity(real_answer, heading_response["response"])
        semantic_similarity = calculate_similarity(real_answer, semantic_response["response"])

        # Determine the most similar answer
        similarities = {
            "Topic-Based": topic_similarity,
            "Heading-Based": heading_similarity,
            "Semantic-Based": semantic_similarity,
        }
        best_method = max(similarities, key=similarities.get)
        best_response = {
            "method": best_method,
            "response": locals()[f"{best_method.lower().replace('-', '_')}_response"]["response"],
            "similarity": similarities[best_method],
        }

        return jsonify({
            "best_response": best_response,
            "topic_response": topic_response,
            "heading_response": heading_response,
            "semantic_response": semantic_response,
        })

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
