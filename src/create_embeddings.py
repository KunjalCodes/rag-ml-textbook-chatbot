from sentence_transformers import SentenceTransformer
from chunk_text import chunk_text
from load_pdf import load_pdf_text

def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":
    pdf_path = "data/ml_book.pdf"
    text = load_pdf_text(pdf_path)

    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)

    print("Total embeddings created:", len(embeddings))
    print("Embedding vector size:", len(embeddings[0]))
