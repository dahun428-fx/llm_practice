from server import llm
from retriever import (
    get_vector_db,
    get_retriever,
    setup_vector_db,
    retriever_with_score,
)
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

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

translate_prompt = ChatPromptTemplate(
    [("system", "주어진 질문에 영어로 변환하세요"), ("user", "Question : {question}")]
)
translate_chain = translate_prompt | llm | StrOutputParser()  # 영어로 변환


def format_docs(docs):
    return "\n===\n".join(
        [doc.page_content + "\nURL" + doc.metadata["source"] for doc in docs]
    )


setup_vector_db()
retriver = get_retriever()
rag_chain = (
    {
        "context": translate_chain | retriver | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Exaone 언어 모델이 다른 모델과 다른 점은 무엇인가요?",
    "Phi-3 언어 모델은 어떤 데이터로 학습했나요?",
    "Qwen 2 의 다국어 성능은 어떻게 나타났나요?",
    "Gemma 의 스몰 모델은 어떻게 학습했나요?",
]

result = rag_chain.batch(questions)
for i, ans in enumerate(result):
    print(f"Q : {questions[i]}")
    print(f"A : {ans}")
    print("=" * 50)
