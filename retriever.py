# vector_db.py

import os
import pickle
import hashlib
from glob import glob
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# ===== 설정 =====
pdf_path = "./papers/*.pdf"
persist_dir = "chroma_papers"
cache_path = "token_chunk.pkl"
hash_path = "pdf_hash.txt"


# ===== 해시 생성 =====
def hash_files(files):
    h = hashlib.md5()
    for f in sorted(files):
        with open(f, "rb") as file:
            h.update(file.read())
    return h.hexdigest()


# ===== 전역 변수로 선언 (get 함수로 접근 가능하게) =====
db = None
retriever = None


def setup_vector_db():
    global db, retriever

    # ===== 문서 로딩 및 토큰화 =====
    pdf_files = glob(pdf_path)
    regenerate = False

    current_hash = hash_files(pdf_files)
    if os.path.exists(cache_path) and os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            saved_hash = f.read()
        if saved_hash == current_hash:
            print("✅ 캐시된 token_chunk 불러오는 중...")
            with open(cache_path, "rb") as f:
                token_chunk = pickle.load(f)
        else:
            print("📛 PDF 내용이 변경됨. token_chunk 재생성 필요.")
            regenerate = True
    else:
        regenerate = True

    if regenerate:
        print("📄 PDF에서 텍스트 추출 및 분할 수행 중...")
        all_papers = []
        for i, path_paper in enumerate(pdf_files):
            print(f"   → ({i+1}/{len(pdf_files)}) {path_paper} 로딩 중...")
            loader = PyMuPDFLoader(path_paper)
            pages = loader.load()
            source = pages[0].metadata.get("source", path_paper)
            doc = Document(page_content="", metadata={"index": i, "source": source})
            for page in pages:
                doc.page_content += page.page_content
            all_papers.append(doc)

        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=2000,
            chunk_overlap=200,
        )
        token_chunk = token_splitter.split_documents(all_papers)

        with open(cache_path, "wb") as f:
            pickle.dump(token_chunk, f)
        with open(hash_path, "w") as f:
            f.write(current_hash)
        print("✅ token_chunk 저장 완료!")

    # ===== 임베딩 모델 =====
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )

    # ===== Chroma 로딩 or 생성 =====
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        print("✅ 기존 Chroma DB 로드 완료")
    else:
        db = Chroma.from_documents(
            documents=token_chunk,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "l2"},
        )
        print("📦 새로 임베딩하여 Chroma DB 생성 완료")

    retriever = db.as_retriever(search_kwargs={"k": 1})


def get_vector_db():
    return db


def get_retriever():
    return retriever


def get_compressor():
    from server import llm

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""다음 문서 내용을 기반으로 주어진 질문과 관련된 핵심 문장만 추출하세요.
        관련 없는 내용은 제거하세요.

        문서:
        {context}

        질문:
        {question}
        """,
    )

    return LLMChainExtractor.from_llm(llm, prompt=prompt)


def retriever_with_score(query):
    docs, scores = zip(*db.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs


def get_ensemble_retriever():
    # 기존 벡터 retriever 사용
    vector = get_retriever()

    # token_chunk 불러오기
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            token_chunk = pickle.load(f)
    else:
        raise FileNotFoundError(
            "❌ token_chunk.pkl not found. 먼저 setup_vector_db()를 실행해주세요."
        )

    # BM25 retriever 생성
    bm25 = BM25Retriever.from_documents(token_chunk)
    bm25.k = 3  # 검색 문서 수 조절 가능

    # Ensemble 생성
    return EnsembleRetriever(
        retrievers=[vector, bm25], weights=[0.7, 0.3]  # 벡터 70%, BM25 30%
    )


# ===== 직접 실행할 때만 수행 =====
if __name__ == "__main__":
    setup_vector_db()
    result = retriever_with_score("How does Exaone achieve good evaluation results?")
    for doc in result:
        print(doc.page_content[:100], "...", doc.metadata["score"])
