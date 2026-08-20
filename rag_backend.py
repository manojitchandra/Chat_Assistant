from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torchvision.transforms.v2 import functional as tvF
from webscrap import docs

load_dotenv()


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_rag_chain():
    # Embeddings & Vector Store
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_retries=2,
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a professional customer-support assistant for the company.
Your task is to answer the user's question using ONLY the information contained in the provided context.

STRICT RULES:
1. Do not use information that is not present in the context.
2. Do not reveal internal reasoning. Output ONLY the final answer.
3. Do not say phrases like "Based on the context...".
4. If the answer cannot be found in the context, respond exactly with:
   "I don't have any information related to this."
5. Keep the answer concise and professional.

CONTEXT:
{context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Construct Chain
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


def ask_question(chain, question: str, history: list) -> str:
    """Invokes the RAG chain with user query and session history."""
    return chain.stream({"question": question, "history": history})