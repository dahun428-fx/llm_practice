from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq
import os
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()  # .env 파일을 읽어옴
# api_key = os.getenv("GROQ_API_KEY")
# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY"),
# )

# model = "llama-3.3-70b-versatile"


# llm = ChatGroq(
#     model=model,
#     api_key=api_key,
#     temperature=0.1,
#     max_tokens=1024,
# )


load_dotenv()  # .env 파일을 읽어옴
api_key = os.getenv("GOOGLE_API_KEY")
model = "gemini-2.0-flash"
client = genai.Client(api_key=api_key)
llm = ChatGoogleGenerativeAI(model=model)
