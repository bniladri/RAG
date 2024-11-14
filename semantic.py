import os
import shutil
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

CHROMA_PATH = "chroma"
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the uploaded file is a PDF
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Semantic chunking using Sentence-BERT embeddings
def semantic_chunking(text, chunk_size=200):
    sentences = text.split(". ")  # Split text into sentences
    embeddings = get_embedding_function.embed_query(sentences)  # Get embeddings for each sentence

    chunks = []
    current_chunk = []
    current_embedding = []

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_embedding.append(embeddings[i])

        # If chunk reaches the desired size, push it to the chunks list
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_embedding = []

    if current_chunk:  # Add any remaining sentences as a final chunk
        chunks.append(" ".join(current_chunk))

    return chunks

def process_file(file_path):
    documents = load_documents(file_path)
    chunks = split_documents(documents)
    x = add_to_chroma(chunks)
    
    if x:
        return True
    return False

def load_documents(file_path):
    document_loader = PyMuPDFLoader(file_path)
    return document_loader.load()

def split_documents(documents: list[Document]):
    chunks = []
    for doc in documents:
        text_chunks = semantic_chunking(doc.page_content, chunk_size=5)  # Semantic chunking with 5 sentences per chunk
        for text_chunk in text_chunks:
            chunk = Document(page_content=text_chunk, metadata=doc.metadata)
            chunks.append(chunk)
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")
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
