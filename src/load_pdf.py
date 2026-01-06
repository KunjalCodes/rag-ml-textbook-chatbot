from pypdf import PdfReader

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


if __name__ == "__main__":
    pdf_path = "data/ml_book.pdf"
    pdf_text = load_pdf_text(pdf_path)

    print("PDF loaded successfully!")
    print("Number of characters:", len(pdf_text))
