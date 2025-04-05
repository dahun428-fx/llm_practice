# 📊 데이터 흐름도: LangChain 기반 RAG 시스템

이 프로젝트는 PDF 문서를 벡터 DB로 저장하고,  
사용자의 질문을 LLM으로 응답하는 **RAG(Retrieval-Augmented Generation)** 파이프라인입니다.

---

## 📂 데이터 처리 흐름 (벡터 DB 구축)

```mermaid
flowchart TD
    A[📂 PDF 파일] --> B[🧠 텍스트 추출<br>PyMuPDFLoader]
    B --> C[📄 문서 통합<br>+ 메타데이터 추가]
    C --> D[🔪 텍스트 분할<br>RecursiveCharacterTextSplitter]
    D --> E[📦 캐시 저장<br>token_chunk.pkl / hash.txt]
    E --> F[💡 임베딩 생성<br>HuggingFaceEmbeddings]
    F --> G[📊 Chroma 벡터 DB 구축 or 로드]
    G --> H[🔍 Retriever 생성<br>(Top-k 문서 검색기)]

flowchart TD
    Q[❓ 사용자 질문] --> T[🌐 영어 번역<br>translate_chain]
    T --> R[🔎 관련 문서 검색<br>Retriever]
    R --> FMT[📚 문서 포맷팅<br>format_docs]
    FMT --> P[📝 Prompt 구성<br>ChatPromptTemplate]
    P --> LLM[🤖 LLM 응답 생성<br>llm]
    LLM --> A[✅ 최종 응답 출력]

=============================

setup_vector_db()
→ PDF 파일을 불러와 텍스트를 추출하고, 문서를 분할하여 임베딩을 생성한 뒤 Chroma에 저장합니다.

get_retriever()
→ 유사 문서를 검색할 수 있는 Retriever 객체를 반환합니다.

translate_chain
→ 질문을 영어로 번역합니다. (LLM은 영어를 더 잘 이해하기 때문입니다.)

retriever | format_docs
→ 유사 문서를 검색한 뒤 포맷팅합니다.

prompt
→ context와 질문을 Prompt Template에 바인딩합니다.

llm | StrOutputParser()
→ LangChain LLM을 통해 응답을 생성하고 텍스트만 추출합니다.

rag_chain.batch(questions)
→ 여러 개의 질문을 한꺼번에 처리합니다.
