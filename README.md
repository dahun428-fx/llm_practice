# 📊 데이터 흐름도: LangChain 기반 RAG 시스템

이 프로젝트는 PDF 문서를 벡터 DB로 저장하고,  
사용자의 질문을 LLM으로 응답하는 **RAG(Retrieval-Augmented Generation)** 파이프라인입니다.

---

## 📂 전체 데이터 흐름 (벡터 DB 구축 + 질의응답)

> 💡 GitHub에서 Mermaid 다이어그램이 렌더링되지 않을 경우 PNG 이미지로 대체해 주세요.

```mermaid
flowchart TD
    %% 벡터 DB 구축
    A[📂 PDF 파일] --> B[🧠 텍스트 추출\n(PDF → 텍스트)]
    B --> C[📄 문서 통합\n+ 메타데이터 추가]
    C --> D[🔪 텍스트 분할\n(RecursiveCharacterTextSplitter)]
    D --> E[📦 캐시 저장\n(token_chunk.pkl, hash.txt)]
    E --> F[💡 임베딩 생성\n(HuggingFaceEmbeddings)]
    F --> G[📊 Chroma DB 구축 or 로드]
    G --> H[🔍 Retriever 생성\n(Top-k 문서 검색기)]

    %% 질문 처리 흐름
    I[❓ 사용자 질문] --> J[🌐 영어 번역\n(translate_chain)]
    J --> K[🔎 유사 문서 검색\n(Retriever)]
    K --> L[📚 문서 포맷팅\n(format_docs)]
    L --> M[📝 Prompt 구성\n(ChatPromptTemplate)]
    M --> N[🤖 LLM 응답 생성\n(llm + 출력 파서)]
    N --> O[✅ 최종 응답 출력]

    %% 라인 연결
    H -.-> K
