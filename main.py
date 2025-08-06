import os
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from utils.text_utils import CharacterTextSplitter, TextFileLoader, PDFLoader
from utils.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from utils.openai_utils.embedding import EmbeddingModel
from utils.vectordatabase import VectorDatabase
from utils.openai_utils.chatmodel import ChatOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on")

# Twilio Flask app
app = Flask(__name__)

# Spanish system prompt specific to HVAC and catalog-based answering
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

# RAG Pipeline Class
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

# Load PDF and Build Vector DB
def prepare_vector_db(file_path):
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
    return asyncio.run(vector_db.abuild_from_list(texts))

# --- Initialization ---
PDF_FILE_PATH = "catalog.pdf"
vector_db = prepare_vector_db(PDF_FILE_PATH)
chat_openai = ChatOpenAI()
qa_pipeline = RetrievalAugmentedQAPipeline(llm=chat_openai, vector_db_retriever=vector_db)

# --- Flask Twilio Webhook ---
@app.route("/webhook", methods=["POST"])
def twilio_webhook():
    try:
        incoming_msg = request.values.get('Body', '').strip()
        response_text = async_to_sync(qa_pipeline.arun_pipeline)(incoming_msg)

        resp = MessagingResponse()
        msg = resp.message()
        msg.body(response_text)

        return Response(str(resp), mimetype="application/xml")

    except Exception as e:
        print(f"Error handling webhook: {str(e)}")
        resp = MessagingResponse()
        resp.message("Lo siento, ocurrió un error procesando tu mensaje.")
        return Response(str(resp), mimetype="application/xml")

# --- Run Flask ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=debug)