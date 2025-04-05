# 📊 데이터 흐름도: LangChain 기반 RAG 시스템

이 프로젝트는 PDF 문서를 벡터 DB로 저장하고,  
사용자의 질문을 LLM으로 응답하는 **RAG(Retrieval-Augmented Generation)** 파이프라인입니다.

---

## 📂 데이터 처리 흐름 (벡터 DB 구축)

```mermaid
flowchart TD
    A[📂 PDF 파일] --> B[🧠 텍스트 추출 PyMuPDFLoader]
    B --> C[📄 문서 통합 + 메타데이터 추가]
    C --> D[🔪 텍스트 분할 RecursiveCharacterTextSplitter]
    D --> E[📦 캐시 저장 token_chunk.pkl / hash.txt]
    E --> F[💡 임베딩 생성 HuggingFaceEmbeddings]
    F --> G[📊 Chroma 벡터 DB 구축 or 로드]
    G --> H[🔍 Retriever 생성 => Top-k 문서 검색기]





flowchart TD
    Q[❓ 사용자 질문] --> T[🌐 영어 번역 translate_chain]
    T --> R[🔎 관련 문서 검색 Retriever]
    R --> FMT[📚 문서 포맷팅 format_docs]
    FMT --> P[📝 Prompt 구성 ChatPromptTemplate]
    P --> LLM[🤖 LLM 응답 생성 llm]
    LLM --> A[✅ 최종 응답 출력]
