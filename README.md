## 📂 전체 데이터 흐름 (벡터 DB 구축 + 질의응답)

```mermaid
flowchart TD
    %% 벡터 DB 구축
    A[PDF 파일] --> B[텍스트 추출\n(PyMuPDFLoader)]
    B --> C[문서 통합\n+ 메타데이터 추가]
    C --> D[텍스트 분할\n(RecursiveCharacterTextSplitter)]
    D --> E[캐시 저장\n(token_chunk.pkl, hash.txt)]
    E --> F[임베딩 생성\n(HuggingFaceEmbeddings)]
    F --> G[Chroma DB 구축 또는 로드]
    G --> H[Retriever 생성\n(Top-k 문서 검색)]

    %% 질문 처리 흐름
    I[사용자 질문] --> J[질문 영어 번역\n(translate_chain)]
    J --> K[문서 검색\n(Retriever)]
    K --> L[문서 포맷팅\n(format_docs)]
    L --> M[프롬프트 구성\n(ChatPromptTemplate)]
    M --> N[LLM 응답 생성\n(llm)]
    N --> O[최종 응답 출력]

    %% 연결
    H -.-> K
