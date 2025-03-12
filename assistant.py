import os
import gradio as gr
from dotenv import load_dotenv
import faiss
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT_DIRS = [
    "C:\\Users\\brand\\OneDrive\\Desktop\\AI_Projects\\Sped_Assistant_FastAPI",
    "C:\\Users\\brand\\OneDrive\\Desktop\\AI_Projects\\Sped_Assistant_React"
]
DB_PATH = "vector_db"  # Where FAISS stores embeddings

# Initialize OpenAI LLM
llm = OpenAI(api_key=OPENAI_API_KEY)

def get_project_structure(directories):
    """Extract file paths and contents from multiple directories."""
    project_files = []
    for root_dir in directories:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                if file_path.endswith((".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".md")):  
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        project_files.append({"path": file_path, "content": content})
                    except Exception as e:
                        print(f"Skipping {file_path}: {e}")
    return project_files

def create_vector_db():
    """Create and store vector database from multiple directories."""
    project_data = get_project_structure(PROJECT_DIRS)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Extract content for embeddings
    documents = [doc["content"] for doc in project_data]
    file_paths = [doc["path"] for doc in project_data]
    
    # Convert text to embeddings
    doc_vectors = embeddings.embed_documents(documents)
    
    # Store in FAISS
    index = faiss.IndexFlatL2(len(doc_vectors[0]))  
    index.add(doc_vectors)  
    faiss.write_index(index, f"{DB_PATH}.faiss")  
    
    # Save file paths
    with open(f"{DB_PATH}.pkl", "wb") as f:
        pickle.dump(file_paths, f)

def load_vector_db():
    """Load FAISS vector database."""
    try:
        index = faiss.read_index(f"{DB_PATH}.faiss")
        with open(f"{DB_PATH}.pkl", "rb") as f:
            file_paths = pickle.load(f)
        return index, file_paths
    except:
        return None, None

def search_code(query):
    """Search relevant code snippets from both directories."""
    index, file_paths = load_vector_db()
    if index is None:
        return "Vector database not found. Run 'Create Vector DB' first."
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    query_embedding = embeddings.embed_query(query)

    D, I = index.search([query_embedding], k=3)  
    results = []
    for i in I[0]:
        if i < len(file_paths):
            with open(file_paths[i], "r", encoding="utf-8") as f:
                snippet = f.read()[:500]  
            results.append(f"**File:** {file_paths[i]}\n```{snippet}```\n")
    
    return "\n".join(results) if results else "No relevant code found."

def answer_query(query):
    """Use OpenAI model to answer queries using project context."""
    index, file_paths = load_vector_db()
    if index is None:
        return "Vector database not found. Run 'Create Vector DB' first."
    
    retriever = FAISS.load_local(DB_PATH, OpenAIEmbeddings(api_key=OPENAI_API_KEY)).as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🛠️ AI Coding Assistant (Multi-Directory)")
    
    with gr.Row():
        create_db_btn = gr.Button("🔄 Create Vector DB")
        create_db_btn.click(fn=create_vector_db, inputs=[], outputs=[])

    query_input = gr.Textbox(label="Ask a coding question")
    search_btn = gr.Button("🔍 Search Code")
    answer_btn = gr.Button("🤖 Answer Query")
    
    search_output = gr.Markdown()
    answer_output = gr.Markdown()
    
    search_btn.click(fn=search_code, inputs=[query_input], outputs=[search_output])
    answer_btn.click(fn=answer_query, inputs=[query_input], outputs=[answer_output])

app.launch()