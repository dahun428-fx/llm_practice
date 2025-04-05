from langchain.prompts import ChatPromptTemplate
from server import llm
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
import os
import pickle
import hashlib
from glob import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# ==== Context 생성용 프롬프트 ====
context_prompt = ChatPromptTemplate(
    [
        (
            "user",
            """
         당신은 문서 분석을 전문으로 하는 AI 어시스턴트입니다.
         주어진 Document 의 일부인 Chunk 에 대해 간결하고 관련성 있는
         짧은 설명을 생성하세요.
         
         # Input Format
         
         -[Document] : '<document>{document}</document>'
         -[Chunk] : '<chunk>{chunk}</chunk>'
         
         아래의 가이드라인을 참고하여,
         이 부분에 대해 간결한 영문 Context을 작성하세요. (1-2문장)
         
         1. 텍스트 부분에서 논의된 주요 주제나 개념을 포함하세요.
         2. 문서 전체의 문맥에서 관련 정보나 비교를 언급하세요.
         3. 가능한 경우, 이 정보가 문서의 전체적인 주제나 목적과 어떻게 연관되는 지를 설명하세요.
         4. 중요한 정보를 제공하는 주요 항목과 수치를 포함하세요.
         
         텍스트 부분의 검색 정확성을 개선하기 위해,
         문서의 전체 맥락에 해당하는 Context 만을 출력하세요.
         답변은 간결하게 작성하세요.
         
         Context :
         
         """,
        )
    ]
)
context_chain = context_prompt | llm | StrOutputParser()

# ===== 설정 =====
pdf_path = "./papers/*.pdf"
persist_dir = "chroma_papers_contextual_retrieval"
cache_path = "token_chunk.pkl"
hash_path = "pdf_hash.txt"


# ===== 해시 생성 =====
def hash_files(files):
    h = hashlib.md5()
    for f in sorted(files):
        with open(f, "rb") as file:
            h.update(file.read())
    return h.hexdigest()


# ===== 전역 변수 =====
db = None
ensemble_retriever = None


def setup_vector_db():
    global db, ensemble_retriever

    pdf_files = glob(pdf_path)
    regenerate = False

    current_hash = hash_files(pdf_files)
    if os.path.exists(cache_path) and os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            saved_hash = f.read()
        if saved_hash == current_hash:
            regenerate = False
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
            encoding_name="cl100k_base", chunk_size=2000, chunk_overlap=200
        )
        token_chunk = token_splitter.split_documents(all_papers)

        print("🧠 각 chunk에 context 생성 중...")
        for i, chunk in tqdm(enumerate(token_chunk), total=len(token_chunk)):
            doc = all_papers[chunk.metadata["index"]].page_content
            try:
                context = context_chain.invoke(
                    {"document": doc, "chunk": chunk.page_content}
                )
            except Exception as e:
                print(f"❌ context 생성 실패: chunk {i} → {e}")
                context = ""
            token_chunk[i].page_content = context + "\n\n" + token_chunk[i].page_content

        with open(cache_path, "wb") as f:
            pickle.dump(token_chunk, f)
        with open(hash_path, "w") as f:
            f.write(current_hash)
        print("✅ token_chunk 저장 완료!")
    else:
        print("✅ 캐시된 token_chunk 불러오는 중...")
        with open(cache_path, "rb") as f:
            token_chunk = pickle.load(f)

    # ===== 임베딩 모델 설정 (항상 필요함) =====
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )

    # ===== Chroma 생성 or 로딩 =====
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        print("✅ 기존 Chroma DB 로드 완료")
    else:
        db = Chroma.from_documents(
            documents=token_chunk,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "l2"},
        )
        print("📦 새로 임베딩하여 Chroma DB 생성 완료")

    # ===== Ensemble Retriever 생성 =====
    bm25_retriever = BM25Retriever.from_documents(token_chunk)
    bm25_retriever.k = 2

    vector_retriever = db.as_retriever(search_kwargs={"k": 2})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


def get_vector_db():
    return db


def get_retriever():
    return ensemble_retriever
