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

# Heading-based chunking function
def heading_based_chunking(text, heading_levels=[1, 2], chunk_size=2):
    # Split the text into lines or paragraphs
    paragraphs = text.split("\n")

    chunks = []
    current_chunk = []
    current_heading = None

    for paragraph in paragraphs:
        # Check if the paragraph is a heading (this could be based on certain keywords, patterns, or font size)
        if is_heading(paragraph, heading_levels):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

            current_heading = paragraph.strip()
        
        current_chunk.append(paragraph.strip())

    if current_chunk:  # Add any remaining content as the final chunk
        chunks.append(" ".join(current_chunk))

    return chunks

def is_heading(text, heading_levels=[1, 2]):
    # Determine if a line is a heading (dummy implementation based on keyword or simple pattern)
    return any(f"Heading {level}" in text for level in heading_levels)

def process_file(file_path):
    documents = load_documents(file_path)
    heading_chunks = split_documents_by_headings(documents)
    x = add_to_chroma(heading_chunks)
    
    if x:
        return True
    return False

def load_documents(file_path):
    document_loader = PyMuPDFLoader(file_path)
    return document_loader.load()

def split_documents_by_headings(documents: list[Document]):
    heading_chunks = []
    for doc in documents:
        # Extract the page content as text and chunk by headings
        page_content = doc.page_content
        heading_chunks_list = heading_based_chunking(page_content, chunk_size=2)  # Chunk by headings
        for heading_chunk in heading_chunks_list:
            heading_chunk_doc = Document(page_content=heading_chunk, metadata=doc.metadata)
            heading_chunks.append(heading_chunk_doc)

    return heading_chunks

def add_to_chroma(heading_chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Add heading-based chunks to the Chroma database
    heading_chunks_with_ids = calculate_chunk_ids(heading_chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_heading_chunks = []
    for chunk in heading_chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_heading_chunks.append(chunk)

    if len(new_heading_chunks):
        new_heading_chunk_ids = [chunk.metadata["id"] for chunk in new_heading_chunks]
        db.add_documents(new_heading_chunks, ids=new_heading_chunk_ids)
        db.persist()
    else:
        print("✅ No new heading-based chunks to add")

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

# Function to calculate context precision
def calculate_context_precision(generated_answer, actual_answer):
    generated_answer_words = set(generated_answer.split())
    actual_answer_words = set(actual_answer.split())
    
    # Precision: proportion of words in the generated answer that are in the actual answer
    precision = len(generated_answer_words.intersection(actual_answer_words)) / len(generated_answer_words) if len(generated_answer_words) > 0 else 0
    return precision

# Function to calculate context recall
def calculate_context_recall(generated_answer, actual_answer):
    generated_answer_words = set(generated_answer.split())
    actual_answer_words = set(actual_answer.split())
    
    # Recall: proportion of words in the actual answer that are in the generated answer
    recall = len(generated_answer_words.intersection(actual_answer_words)) / len(actual_answer_words) if len(actual_answer_words) > 0 else 0
    return recall

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Ensure that the request contains a file, query, and actual_answer
    if 'file' not in request.files or 'query' not in request.form or 'actual_answer' not in request.form:
        return jsonify({"error": "No file, query, or actual_answer found in the request"}), 400

    file = request.files['file']
    query = request.form['query']
    actual_answer = request.form['actual_answer']

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

            # Calculate context precision and recall based on the generated and actual answers
            generated_answer = response["response"]
            context_precision = calculate_context_precision(generated_answer, actual_answer)
            context_recall = calculate_context_recall(generated_answer, actual_answer)

            response["context_precision"] = context_precision
            response["context_recall"] = context_recall
            
            return jsonify(response)
        else:
            return jsonify({"error": "Failed to process the PDF file"}), 500
    else:
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400


if __name__ == '__main__':
    app.run(debug=True)
