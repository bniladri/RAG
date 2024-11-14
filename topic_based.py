import os
import shutil
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

# Function to clear Chroma database using shutil.rmtree
def clear_chroma_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"✅ Chroma database at {CHROMA_PATH} has been cleared.")

# Topic-based chunking function
def topic_based_chunking(text, num_topics=3, chunk_size=2):
    # Split the text into sentences or paragraphs
    paragraphs = text.split("\n")

    # Embed sentences using Sentence-Transformers
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = get_embedding_function.embed_query(paragraphs)

    # Use KMeans clustering to group sentences into topics
    kmeans = KMeans(n_clusters=num_topics, random_state=0)
    kmeans.fit(embeddings)

    # Group paragraphs by topic
    topic_chunks = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in topic_chunks:
            topic_chunks[label] = []
        topic_chunks[label].append(paragraphs[i])

    # Combine paragraphs within each topic
    chunks = []
    for topic in topic_chunks.values():
        chunk = " ".join(topic)
        chunks.append(chunk)

    return chunks

def process_file(file_path):
    documents = load_documents(file_path)
    topic_chunks = split_documents_by_topics(documents)
    x = add_to_chroma(topic_chunks)
    
    if x:
        return True
    return False

def load_documents(file_path):
    document_loader = PyMuPDFLoader(file_path)
    return document_loader.load()

def split_documents_by_topics(documents: list[Document], num_topics=3):
    topic_chunks = []
    for doc in documents:
        # Extract the page content as text and chunk by topics
        page_content = doc.page_content
        topic_chunks_list = topic_based_chunking(page_content, num_topics=num_topics)  # Chunk by topics
        for topic_chunk in topic_chunks_list:
            topic_chunk_doc = Document(page_content=topic_chunk, metadata=doc.metadata)
            topic_chunks.append(topic_chunk_doc)

    return topic_chunks

def add_to_chroma(topic_chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Add topic-based chunks to the Chroma database
    topic_chunks_with_ids = calculate_chunk_ids(topic_chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_topic_chunks = []
    for chunk in topic_chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_topic_chunks.append(chunk)

    if len(new_topic_chunks):
        new_topic_chunk_ids = [chunk.metadata["id"] for chunk in new_topic_chunks]
        db.add_documents(new_topic_chunks, ids=new_topic_chunk_ids)
        db.persist()
    else:
        print("✅ No new topic-based chunks to add")

    return True

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

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        return {"response": "No results found in the database for the query."}

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(base_url="http://localhost:11435", model="llama3")
    try:
        response_text = model.invoke(prompt)
    except Exception as e:
        response_text = "Error while generating response."

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    return {"response": response_text}


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Ensure that the request contains a file and query
    if 'file' not in request.files or 'query' not in request.form:
        return jsonify({"error": "No file or query found in the request"}), 400

    file = request.files['file']
    query = request.form['query']

    # Validate the file
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Clear the old files in the uploads folder
        clear_old_files()

        # Clear the Chroma database
        clear_chroma_db()

        # Save the uploaded file
        file.save(file_path)

        # Process the uploaded PDF
        if process_file(file_path):
            # Query the RAG system
            response = query_rag(query)
            return jsonify(response)
        else:
            return jsonify({"error": "Failed to process the PDF file"}), 500
    else:
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400


if __name__ == '__main__':
    app.run(debug=True)
