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
            chunk_size=1000,
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

    retriever = db.as_retriever(search_kwargs={"k": 2})


def get_vector_db():
    return db


def get_retriever():
    return retriever


def retriever_with_score(query):
    docs, scores = zip(*db.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs


# ===== 직접 실행할 때만 수행 =====
if __name__ == "__main__":
    setup_vector_db()
    result = retriever_with_score("How does Exaone achieve good evaluation results?")
    for doc in result:
        print(doc.page_content[:100], "...", doc.metadata["score"])
