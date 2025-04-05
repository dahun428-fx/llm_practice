from server import llm
from retriever import (
    get_vector_db,
    get_retriever,
    setup_vector_db,
    retriever_with_score,
    get_compressor,
)
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever

import logging

setup_vector_db()
retriver = get_retriever()
prompt = ChatPromptTemplate(
    [
        (
            "user",
            """당신은 QA(Question Answering) 을 수행하는 Assistant 입니다.
        다음의 Context를 이용하여 Question 에 한국어로 답변하세요
        정확한 답변을 제공하세요.
        만약 모든 Context 를 다 확인해도 정보가 없다면,
        "정보가 부족하여 답변할 수 없습니다." 라고 답변하세요
        ---
        Context : {context}
        ---
        Question : {question}
        """,
        )
    ]
)
prompt.pretty_print()


def format_docs(docs):
    return "\n===\n".join(
        [doc.page_content + "\nURL" + doc.metadata["source"] for doc in docs]
    )


questions = [
    "Exaone 언어 모델이 다른 모델과 다른 점은 무엇인가요?",
    "Phi-3 언어 모델은 어떤 데이터로 학습했나요?",
    "Qwen 2 의 다국어 성능은 어떻게 나타났나요?",
    "Gemma 의 스몰 모델은 어떻게 학습했나요?",
]

rewriter_prompt = PromptTemplate(
    template="""당신은 AI 언어 모델 어시스턴트입니다.
                                주어진 사용자 질문을 벡터 데이터베이스에서 관련 문서를
                                검색하기 위해 3가지 다른 영문 버전으로 생성하는 것이
                                당신의 작업입니다.
                                사용자 질문에 대한 여러 관점을 생성함으로써, 
                                당신은 거리 기반 유사성 검색의 한계를 극복할 수 있도록
                                사용자에게 도움을 주는 것이 목표입니다. 이러한 대체 질문들을
                                새로운 줄로 구분하여 제공하세요.
                                ---
                                원본 질문 : {question}"""
)

mutli_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriver,
    llm=llm,
    prompt=rewriter_prompt,
)

# 🔽 압축기 불러오기
compressor = get_compressor()  # 예: LLMChainExtractor.from_llm(llm)

# ✅ 압축 retriever로 감싸기
compressed_retriever = ContextualCompressionRetriever(
    base_retriever=mutli_query_retriever, base_compressor=compressor
)

rag_chain = (
    {
        "context": compressed_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
result = rag_chain.batch(questions)
for i, ans in enumerate(result):
    print(f"Q : {questions[i]}")
    print(f"A : {ans}")
    print("=" * 50)
