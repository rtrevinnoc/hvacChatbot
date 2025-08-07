import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from utils.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from utils.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from utils.vectordatabase import VectorDatabase
from utils.openai_utils.chatmodel import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

app = FastAPI()

system_template = """\
Eres un asistente experto en sistemas de calefacción, ventilación y aire acondicionado. Utiliza la información del catálogo de productos para recomendar productos cuando sea necesario.

Realiza preguntas para averiguar lo que busca el usuario cuando no cuentes con información suficiente, ya que puede no saber qué es lo que necesita y debes apoyarlo.

Responde siempre en español. Mantén la conversación sencilla y haz solo una pregunta a la vez.

Contexto:
{context}
"""
system_role_prompt = SystemRolePrompt(system_template)

user_prompt_template = """\
Pregunta:
{question}
"""
user_role_prompt = UserRolePrompt(user_prompt_template)

vector_db = None  # global placeholder
PDF_FILE_PATH = "catalog.pdf"

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    async def arun_pipeline(self, user_query: str):
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)
        context_prompt = "\n".join([context[0] for context in context_list])

        formatted_system_prompt = system_role_prompt.create_message()
        formatted_user_prompt = user_role_prompt.create_message(question=user_query, context=context_prompt)

        response_chunks = []
        async for chunk in self.llm.astream([formatted_system_prompt, formatted_user_prompt]):
            response_chunks.append(chunk)

        return ''.join(response_chunks)

async def prepare_vector_db(file_path):
    print(f"Processing file: {file_path}")
    if file_path.lower().endswith('.pdf'):
        loader = PDFLoader(file_path)
    else:
        loader = TextFileLoader(file_path)

    documents = loader.load_documents()
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_texts(documents)

    print(f"Loaded {len(texts)} text chunks")

    vector_db = VectorDatabase()
    await vector_db.abuild_from_list(texts)
    return vector_db

@app.on_event("startup")
async def startup_event():
    global vector_db
    vector_db = await prepare_vector_db(PDF_FILE_PATH)
    print("Vector DB initialized")

@app.post("/webhook")
async def twilio_webhook(request: Request):
    form = await request.form()
    print("##########################")
    print(form)
    incoming_msg = form.get('Body', '').strip()
    print(incoming_msg)

    async with ChatOpenAI() as chat_openai:
        qa_pipeline = RetrievalAugmentedQAPipeline(llm=chat_openai, vector_db_retriever=vector_db)
        response_text = await qa_pipeline.arun_pipeline(incoming_msg)
        print(response_text)

    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response_text)

    print("##########################")

    return PlainTextResponse(content=str(resp), media_type="application/xml")