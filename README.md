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
```
```mermaid
flowchart TD
    Q[❓ 사용자 질문] --> T[🌐 영어 번역 translate_chain]
    T --> R[🔎 관련 문서 검색 Retriever]
    R --> FMT[📚 문서 포맷팅 format_docs]
    FMT --> P[📝 Prompt 구성 ChatPromptTemplate]
    P --> LLM[🤖 LLM 응답 생성 llm]
    LLM --> A[✅ 최종 응답 출력]
```


---

## 🧪 주요 구성 요소 설명

| 구성 요소 | 설명 |
|-----------|------|
| `setup_vector_db()` | PDF → 텍스트 추출 → 분할 → 임베딩 → Chroma 저장 |
| `get_retriever()` | Chroma에서 유사 문서 검색하는 Retriever 객체 생성 |
| `translate_chain` | 한국어 질문을 영어로 번역 |
| `retriever | format_docs` | 관련 문서 검색 후 문자열로 정리 |
| `prompt` | 질문과 context 기반 Prompt 구성 |
| `llm | StrOutputParser()` | LLM에 요청하고, 응답 파싱 |
| `rag_chain.batch(questions)` | 여러 질문을 일괄 처리 |

---

## 💬 예시 질문 목록

```python
questions = [
    "Exaone 언어 모델이 다른 모델과 다른 점은 무엇인가요?",
    "Phi-3 언어 모델은 어떤 데이터로 학습했나요?",
    "Qwen 2 의 다국어 성능은 어떻게 나타났나요?",
    "Gemma 의 스몰 모델은 어떻게 학습했나요?",
]
```

---

## ✅ 예시 응답 형식

```text
Q : Exaone 언어 모델이 다른 모델과 다른 점은 무엇인가요?
A : Exaone은 멀티모달 데이터를 함께 학습하여 한국어와 이미지 이해에 강점을 갖습니다.
--------------------------------------------------
Q : Phi-3 언어 모델은 어떤 데이터로 학습했나요?
A : 고품질 synthetic 교육 데이터를 기반으로 학습되었습니다.
```

---

## 📦 의존 패키지 설치 예시

```bash
pip install langchain langchain-core langchain-community langchain-chroma
pip install langchain-huggingface sentence-transformers
pip install groq  # Groq 모델 사용 시
```

---

## 🖼️ 대체 이미지 삽입 (Mermaid 미지원 뷰어용)

```markdown
![RAG 데이터 흐름도](./images/rag_full_flow.png)
```

---

## 🙌 기여자

- 개발자: 정다훈 (Dahoon Jung)
- 문의: [이메일 주소 또는 GitHub 링크]
