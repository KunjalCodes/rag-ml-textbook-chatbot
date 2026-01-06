from load_pdf import load_pdf_text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


if __name__ == "__main__":
    pdf_path = "data/ml_book.pdf"
    text = load_pdf_text(pdf_path)

    chunks = chunk_text(text)

    print("Total chunks created:", len(chunks))
    print("Sample chunk:\n")
    print(chunks[0])
