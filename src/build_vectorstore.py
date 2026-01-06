from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from load_pdf import load_pdf_text
from chunk_text import chunk_text


def build_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory="vectorstore/chroma_db"
    )

    vectorstore.persist()
    return vectorstore


if __name__ == "__main__":
    pdf_path = "data/ml_book.pdf"
    text = load_pdf_text(pdf_path)

    chunks = chunk_text(text)
    vectorstore = build_vectorstore(chunks)

    print("Vector database created and saved successfully!")
