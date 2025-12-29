import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class SimpleRAGPipeline:

    def __init__(self, llm_model="llama3.2", embed_model="mxbai-embed-large"):
        # Initialize LLM & Embeddings
        # self.llm = ChatOllama(model=llm_model)
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def ingest_data(self, url: str):
        """
        Load data from a URL, chunk it, and store in vector db.
        Reasoning: Using RecursiveCharacterTextSplitter keeps related text together.
        Overlap ensures context isn't lost at cut points.
        """
        print(f"--- Ingesting data from {url} ---")
        loader = WebBaseLoader(url)
        docs = loader.load()

        # Chunking Strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        print(f"--- Created {len(splits)} chunks ---")

        # Vector Store (In-memory Chroma)
        self.vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.embeddings
        )

        # Retriever Setup (Top-k=3 to balance context window vs relevancy)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def build_chain(self):
        """Constructs the RAG chain."""
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise.
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def run_query(self, query: str):
        """Runs the pipeline and returns answer + source docs."""
        if not self.chain:
            raise ValueError(
                "Pipeline not built. Run ingest_data and build_chain first."
            )

        # We need to manually retrieve to return sources for evaluation
        retrieved_docs = self.retriever.invoke(query)
        formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Invoke LLM
        answer = self.chain.invoke(query)

        return {
            "query": query,
            "answer": answer,
            "source_documents": retrieved_docs,
            "context_str": formatted_context,
        }
