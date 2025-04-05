from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq
import os

load_dotenv()  # .env 파일을 읽어옴
api_key = os.getenv("GROQ_API_KEY")
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

model = "llama-3.3-70b-versatile"


llm = ChatGroq(
    model=model,
    api_key=api_key,
    temperature=0.1,
    max_tokens=1024,
)
