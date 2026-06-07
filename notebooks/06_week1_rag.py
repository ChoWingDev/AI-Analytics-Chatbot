import os
import pickle
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient
from typing import Optional, List

# Load environment variables from a .env file
load_dotenv()

def load_and_parse_pdfs(folder="data/reports"):
    """
    Load all PDF files from `folder`, parse them with unstructured.partition.pdf,
    convert parsed elements with text into langchain_core Document objects,
    and return the list of Documents.
    """
    all_docs = []  # list to accumulate Document objects
    
    # Iterate over files in the folder
    for filename in os.listdir(folder):
        # Only process files with .pdf extension
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            print(f"Parsing: {filename}")  # log of which file is being parsed at the moment
            
            # Use unstructured's partition_pdf to split the PDF into elements.
            # - strategy="hi_res" uses a higher-resolution layout strategy.
            # - infer_table_structure attempts to reconstruct table structure.
            # - extract_image_block_types includes images and tables as blocks.
            elements = partition_pdf(
                filename=filepath,
                strategy="fast",
                infer_table_structure=True,
                extract_image_block_types=["Image", "Table"]
            )
            
            # Convert each parsed element that contains non-empty text into a Document
            for element in elements:
                # If empty text, skip
                if element.text and element.text.strip():
                    metadata = {
                        "source": filename,  # original file name
                        # Try to get page number from element.metadata, defaulting to None if not found
                        "page": getattr(element.metadata, "page_number", None),
                        "element_type": element.category,  # type/category of element
                        "industry": "ecommerce"
                    }
                    # Create a langchain_core Document with the element text and metadata
                    all_docs.append(Document(
                        page_content=element.text.strip(),
                        metadata=metadata
                    ))
        
    # Summary log and return collected documents
    print(f"Total elements extracted: {len(all_docs)}")
    return all_docs

# Save parsed documents to a pickle file
def save_parsed_documents(documents, filename="data/parsed_docs.pkl"):
    """Save parsed documents so no need to re-parse PDFs every time."""
    os.makedirs("data", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(documents, f)
    print(f"Parsed documents saved to {filename}")

# Load previously saved parsed documents
def load_parsed_documents(filename="data/parsed_docs.pkl"):
    """Load previously parsed documents."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            docs = pickle.load(f)
        print(f"Loaded {len(docs)} pre-parsed documents from {filename}")
        return docs
    else:
        print("No saved documents found. Running parsing first.")
        return None

def create_vectorstore(documents):
    """
    Two-level (parent -> child) chunking pipeline:
    - Parent splitter creates larger chunks that preserve broad context (e.g., full sections).
    - Child splitter breaks each parent chunk into smaller, high-recall subchunks that are
      better suited for retrieval and fine-grained embedding.
    Reasoning:
    - Embedding and retrieval work best when chunks are neither too large (diluted signal)
      nor too small (loss of context). Two-level chunking gives both: parent chunks keep
      context and allow retrieving semantically-relevant regions, while child chunks provide
      precise, high-quality passages to embed or use in downstream LLM prompts.
    - Strategy: split documents into parent chunks, then split each parent chunk into child
      chunks and keep child chunks as the final units stored/embedded.
    """
    # Parent splitter: larger chunks to preserve section-level context.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    # Child splitter: smaller chunks focused on high recall and better fit for LLM prompts.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    # Step 1: split original documents into parent chunks
    parent_docs = parent_splitter.split_documents(documents)
    print(f"Created {len(parent_docs)} parent chunks")

    # Step 2: for each parent chunk, create child chunks.
    # Preserve metadata from the parent and add an explicit reference to the parent's id/index
    # So, it's possible to map child chunks back to their parent (such as if we want for hierarchical retrieval).
    child_docs = []
    for idx, parent in enumerate(parent_docs):
        # child_splitter expects a list-like Document; pass single-item list and get list back
        children = child_splitter.split_documents([parent])
        for child in children:
            # Copy parent's metadata and add parent_index for traceability
            if parent.metadata:
                meta = dict(parent.metadata)
            else:
                meta = {}
            meta.update({
                "parent_index": idx,
                # keep original source if present
                "source": meta.get("source"),
                # keep parent element_type if present
                "page": meta.get("page"),
                # keep parent industry
                "industry": meta.get("industry", "ecommerce"),
                "chunk_type": "child"
            })

            new_child = Document(
                page_content=child.page_content,
                metadata=meta
            )
            child_docs.append(new_child)

    print(f"Created {len(child_docs)} child chunks (final units to embed)")

    # Initialize embeddings 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vector store from child chunks.
    # Chroma stores dense vectors and associated metadata on disk,
    # providing fast nearest-neighbor search over embeddings for retrieval tasks.
    # collection_name groups related documents; allowing loading and querying this collection later.
    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        collection_name="ecommerce_reports",
        persist_directory="./chroma_db"
    )

    print("Two-level vector store created")
    return vectorstore


def load_vectorstore():
    """
    Load an existing Chroma vector store from disk without re-embedding.
    Used to skip the expensive embedding step on subsequent runs.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="ecommerce_reports",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    print("Loaded existing vector store from ./chroma_db")
    return vectorstore

class HFInferenceLLM(LLM):
    """
    Minimal LangChain-compatible LLM that calls HuggingFace's chat_completion
    endpoint directly via InferenceClient.
    """
    model_id: str
    hf_token: str
    max_new_tokens: int = 512
    temperature: float = 0.1
 
    @property
    def _llm_type(self) -> str:
        return "hf_inference_client"
 
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # LLM base class converts the prompt value to a plain string before _call,
        # so we receive a clean string here ready to send as a chat message.
        client = InferenceClient(token=self.hf_token)
        response = client.chat_completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

def build_rag_chain(vectorstore):
    """
    Build a simple RAG chain:
      retriever → prompt → LLM → answer

    Design decisions:
    - k=5: retrieve 5 child chunks; enough context without overloading the prompt.
    - Prompt instructs the LLM to cite source + page so answers are traceable.
    - Hallucination guard: LLM is told to say "I don't know" if context is insufficient,
      preventing it from making up industry statistics.
    - Uses Mistral-7B via HuggingFace Inference API.
    """
    # Retriever: convert vectorstore into a callable that returns top-k relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # RAG prompt: inject retrieved context + user question into the LLM
    prompt = ChatPromptTemplate.from_template("""
            You are an ecommerce industry analyst assistant.
            Answer the question using ONLY the context below.
            If the context does not contain enough information, say "I don't have enough data in the loaded reports to answer this confidently."
            Always cite the source document and page number when referencing specific data.

            Context:
            {context}

            Question: {question}

            Answer:"""
                                              )

    # LLM: Mistral-7B via HuggingFace Inference API 
    # HuggingFaceEndpoint streams tokens from HF's hosted inference servers.
    llm = HFInferenceLLM(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        hf_token=os.getenv("HF_TOKEN"),
    )

    def format_docs(docs):
        """
        Format retrieved chunks into a single string for the prompt.
        Each chunk includes its source file and page for citation purposes.
        """
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    # Chain: retrieve → format → prompt → LLM → parse to string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def run_test_queries(rag_chain):
    """
    Week 1 deliverable: prove the RAG chain can answer industry benchmark questions.
    These are the 5 test questions that confirm the system is working end-to-end.
    """
    test_questions = [
        "What was lululemon's ecommerce revenue or net revenue?",
        "What was Aritzia's revenue growth in 2024?",
        "What are Zara's key growth strategies?",
        "What risks does Aritzia identify in its business?",
        "How many stores does Zara operate?",
    ]


    print("\n" + "=" * 60)
    print("WEEK 1 DELIVERABLE — RAG Chain Test")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 40)
        answer = rag_chain.invoke(question)
        print(f"A: {answer}")
        print()


# Run it if executed as a script
if __name__ == "__main__":
    # ── Step 1: Load or parse documents ──────────────────────────────
    docs = load_parsed_documents()

    if docs is None:
        docs = load_and_parse_pdfs()
        save_parsed_documents(docs)

    if len(docs) == 0:
        print("No documents available. Add PDFs to data/reports/ and try again.")
    else:
        # ── Step 2: Build or load vector store ───────────────────────
        # If chroma_db already exists on disk, load it (saves re-embedding time).
        # Otherwise, create it fresh from the parsed documents.
        if os.path.exists("./chroma_db"):
            vectorstore = load_vectorstore()
        else:
            vectorstore = create_vectorstore(docs)

        # ── Step 3: Build RAG chain and run test queries ──────────────
        rag_chain = build_rag_chain(vectorstore)
        run_test_queries(rag_chain)