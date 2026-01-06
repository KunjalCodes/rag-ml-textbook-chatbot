from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="vectorstore/chroma_db",
        embedding_function=embedding_model
    )

    return vectorstore


def run_chatbot():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="llama3")

    print("🤖 ML Book Chatbot is ready! (type 'exit' to quit)\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            print("Bye 👋")
            break

        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful Machine Learning tutor.
Answer the question ONLY using the context below.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

        print("\n🤖 Answer:\n", response, "\n")


if __name__ == "__main__":
    run_chatbot()
