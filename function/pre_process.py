import os
import glob
from langchain_upstage import UpstageEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_folder="../data/"):
    """
    data_folder 내의 모든 PDF 파일을 로드하여 문서 리스트를 반환합니다.
    """
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    all_docs = []
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)
    return pdf_files, all_docs



def split_documents(all_docs, chunk_size=1000, chunk_overlap=250):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(all_docs)

    return split_documents


def embedding(openai_api_key):
    embeddings = UpstageEmbeddings(api_key=openai_api_key, model="embedding-query", base_url="https://api.upstage.ai/v1")

    return embeddings

def vectorstore(embeddings, split_documents):
    """
    벡터스토어를 생성하고 저장합니다.
    """
    from langchain_community.vectorstores import FAISS
    if os.path.exists("faiss_index_dir"):
        vectorstore = FAISS.load_local("faiss_index_dir", embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local("faiss_index_dir")

    return vectorstore

def sementic_chunker(embeddings, split_documents):
    """
    SemanticChunker를 사용하여 분할된 문서들을 의미 단위로 재구성합니다.
    LangChain Experimental의 SemanticChunker를 사용하여 각 문서들의 의미적 유사성을 고려한 청크를 생성합니다.

    Args:
        embeddings: 임베딩 모델 인스턴스
        split_documents: 분할된 문서들의 리스트 (각 문서는 Document 객체라고 가정하며, .page_content 속성을 가짐)

    Returns:
        semantic_chunks: 의미 단위로 결합된 문서(청크) 리스트
    """
    from langchain_experimental.text_splitter import SemanticChunker
    # SemanticChunker 인스턴스 생성 (필요한 경우 추가 설정 가능)
    chunker = SemanticChunker(embeddings=embeddings)
    # 기존의 chunk 메서드 대신 split_documents 메서드를 사용하여 semantic chunk 생성
    semantic_chunks = chunker.split_documents(split_documents)
    return semantic_chunks

def vectorstore_sementic(embeddings, documents):
    """
    주어진 문서들을 기반으로 벡터스토어를 생성하고 로컬에 저장합니다.
    이미 저장된 인덱스가 있다면 불러오고, 없으면 새로 생성합니다.

    Args:
        embeddings: 임베딩 모델 인스턴스
        documents: 문서 리스트 (여기서는 semantic_chunks를 사용)
    
    Returns:
        vectorstore 인스턴스
    """
    from langchain_community.vectorstores import FAISS
    index_dir = "faiss_index_semantic_dir"
    if os.path.exists(index_dir):
        vectorstore_instance = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore_instance = FAISS.from_documents(documents=documents, embedding=embeddings)
        vectorstore_instance.save_local(index_dir)
    return vectorstore_instance

def retriever(vectorstore):
    """
    벡터스토어에서 검색기를 생성합니다.
    """
    retriever = vectorstore.as_retriever()
    return retriever

def create_models(retriever, prompt, openai_api_key):
    """
    검색기와 프롬프트를 사용하여 모델을 생성합니다.
    """
    from langchain_upstage import ChatUpstage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    llm = ChatUpstage(api_key=openai_api_key, model="solar-pro", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
