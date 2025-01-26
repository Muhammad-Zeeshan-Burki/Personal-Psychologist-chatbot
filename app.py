import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber

# Initialize the InferenceClient
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Load and preprocess book PDFs
pdf_files = ["Diagnostic and statistical manual of mental disorders _ DSM-5 ( PDFDrive.com ).pdf"]
all_texts = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# Split text into chunks
def chunk_text(text, chunk_size=300):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Prepare embeddings for each book
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
chunked_texts = [chunk_text(text) for text in all_texts]
all_chunks = [chunk for chunks in chunked_texts for chunk in chunks]
embeddings = model.encode(all_chunks, convert_to_tensor=True).detach().cpu().numpy()
index.add(embeddings)

# Function to generate response
def respond(message, history, system_message, max_tokens, temperature, top_p):
    # Step 1: Retrieve relevant chunks based on user message
    query_embedding = model.encode([message], convert_to_tensor=True).detach().cpu().numpy()
    k = 5
    _, indices = index.search(query_embedding, k)
    relevant_chunks = " ".join([all_chunks[idx] for idx in indices[0]])
    
    # Step 2: Create prompt for the model
    prompt = f"{system_message}\n\nUser Query: {message}\n\nRelevant Information: {relevant_chunks}"
    response = ""

    # Step 3: Generate response
    for message in client.chat_completion(
        [{"role": "system", "content": system_message}, {"role": "user", "content": message}],
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Gradio ChatInterface with additional inputs
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful and empathetic mental health assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
