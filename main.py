#!/usr/bin/env python3
"""
HVAC RAG System - Fixed Version
A WhatsApp chatbot for HVAC/refrigeration queries with proper response handling
"""

import os
import sys
import torch
import re
from typing import List, Any, Dict, Tuple
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

from dotenv import load_dotenv
load_dotenv()

account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_TOKEN")
debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on")
client = Client(account_sid, auth_token)

class HVACRAGSystem:
    """Fixed HVAC RAG System with proper response handling"""
    
    def __init__(self):
        self.qa_chain = None
        self.vectorstore = None
        self.llm = None
        self.chunk_dir = "chunks"
        self.index_path = "vectorstore_index"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def process_pdf(self, pdf_path: str) -> List[Any]:
        """Process PDF into document chunks"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            print(f"Loaded {len(documents)} pages from PDF")
            
            # Improved text splitting for Spanish content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Larger chunks for better context
                chunk_overlap=200,  # Better overlap
                length_function=len,
                separators=['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ', ''],
                is_separator_regex=False
            )
            
            chunks = text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def create_vectorstore_from_chunks(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
        """Create vector store with better multilingual embeddings"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            documents = []
            for chunk_file in sorted(os.listdir(self.chunk_dir)):
                with open(os.path.join(self.chunk_dir, chunk_file), "r", encoding="utf-8") as file:
                    content = file.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": chunk_file}
                    )
                    documents.append(doc)
            
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(self.index_path)
            print("Vector store created and saved successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def load_or_create_vectorstore(self) -> FAISS:
        """Load existing vector store or create new one"""
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if os.path.exists(self.index_path):
            print("Loading precomputed vector store...")
            try:
                return FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Creating new vector store...")
                return self.create_vectorstore_from_chunks(embedding_model)
        else:
            print("Vector store not found. Creating a new one...")
            return self.create_vectorstore_from_chunks(embedding_model)

    def setup_model(self) -> HuggingFacePipeline:
        try:
            model_name = "tiiuae/falcon-7b"  # or your preferred model
            print(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            pipe = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=300,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Model setup failed: {str(e)}")
            return self.setup_fallback_model()


    def setup_fallback_model(self) -> HuggingFacePipeline:
        try:
            print("Using fallback model: FLAN-T5-Base")
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            pipe = pipeline(
                'text2text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=300,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"All model setups failed: {str(e)}")
            raise Exception("Unable to load any model")

    def classify_question(self, question: str) -> str:
        """Classify question type for appropriate prompt selection"""
        question_lower = question.lower()
        
        patterns = {
            'greeting': r'hola|buenos|buenas|saludos|como estas|que tal|hello|hi',
            'benefits': r'beneficios|ventajas|porqué|valor|precio|bueno|mejor|óptimo|razón|costo|advantages|benefits',
            'technical': r'cómo|implementación|configuración|instalar|configurar|reparar|mantenimiento|problema|falla|how|install|repair',
            'sizing': r'tamaño|dimensión|capacidad|potencia|btu|tonelada|metro|cuarto|espacio|size|capacity',
            'inverter': r'inverter|invertir|sistema inverter|tecnología inverter',
        }
        
        for q_type, pattern in patterns.items():
            if re.search(pattern, question_lower):
                return q_type
        
        return 'general'

    def get_simple_response(self, question: str, question_type: str) -> str:
        """Generate simple responses for common questions without using the model"""
        responses = {
            'greeting': "¡Hola! Soy tu asistente especializado en sistemas HVAC y refrigeración. ¿En qué puedo ayudarte hoy? Puedo responder preguntas sobre instalación, mantenimiento, eficiencia energética y más.",
            
            'inverter': """Los sistemas inverter ofrecen varias ventajas importantes:

🔹 **Eficiencia energética**: Consumen hasta 40% menos energía
🔹 **Control de temperatura**: Mayor precisión y estabilidad
🔹 **Menos ruido**: Operación más silenciosa
🔹 **Arranque suave**: Sin picos de corriente
🔹 **Mayor durabilidad**: Menos desgaste del compresor
🔹 **Confort**: Temperatura más constante

¿Te interesa algún aspecto específico de la tecnología inverter?""",
        }
        
        return responses.get(question_type, None)

    def get_prompt_template(self, question_type: str) -> str:
        """Get improved prompt template based on question type"""
        base_template = """Eres un especialista profesional en sistemas HVAC (calefacción, ventilación y aire acondicionado) con amplia experiencia.

Basándote en el siguiente contexto técnico, proporciona una respuesta clara, útil y profesional.

Contexto técnico: {context}

Pregunta del cliente: {question}

Respuesta profesional (en español, clara y completa):"""

        templates = {
            'benefits': """Eres un especialista en sistemas HVAC. Explica los beneficios y ventajas de manera clara.

Contexto técnico: {context}

Pregunta sobre beneficios: {question}

Respuesta detallada sobre beneficios y ventajas:""",
            
            'technical': """Eres un técnico experto en HVAC. Proporciona información técnica precisa y práctica.

Contexto técnico: {context}

Consulta técnica: {question}

Respuesta técnica detallada:""",
            
            'sizing': """Eres un ingeniero especialista en dimensionamiento de sistemas HVAC.

Contexto técnico: {context}

Consulta sobre dimensionamiento: {question}

Respuesta sobre cálculos y dimensionamiento:""",
        }
        
        return templates.get(question_type, base_template)

    def clean_response(self, response_text: str) -> str:
        """Enhanced response cleaning"""
        if not response_text or len(response_text.strip()) < 10:
            return "Lo siento, no pude generar una respuesta adecuada. ¿Podrías reformular tu pregunta?"
        
        # Remove common prefixes/suffixes
        patterns_to_remove = [
            r'^(Respuesta[:\s]*)',
            r'^(Pregunta[:\s]*)',
            r'^(Usuario[:\s]*)',
            r'^(Asistente[:\s]*)',
            r'^(Human[:\s]*)',
            r'^(Assistant[:\s]*)',
            r'(Contexto[:\s]*.*)$',
        ]
        
        cleaned_text = response_text.strip()
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up formatting
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # Ensure minimum length
        if len(cleaned_text) < 20:
            return "Basándome en la información técnica disponible, te recomiendo contactar a un especialista para una evaluación específica de tu caso."
        
        return cleaned_text

    def format_response(self, result: Dict[str, Any]) -> str:
        """Format response for WhatsApp with better error handling"""
        try:
            response_text = result.get('result', '')
            
            if not response_text or len(response_text.strip()) < 10:
                return "Lo siento, no pude procesar tu consulta correctamente. ¿Podrías ser más específico en tu pregunta sobre HVAC?"
            
            cleaned_response = self.clean_response(response_text)
            
            # Limit response length for WhatsApp
            if len(cleaned_response) > 1400:
                cleaned_response = cleaned_response[:1400] + "...\n\n¿Necesitas información más específica sobre algún punto?"
            
            return cleaned_response
            
        except Exception as e:
            print(f"Error formatting response: {e}")
            return "Hubo un error procesando tu consulta. Por favor, intenta reformular tu pregunta."

    def ask_question(self, question: str) -> str:
        """Process question and return response with better error handling"""
        try:
            # Clean and classify question
            question = question.strip()
            if len(question) < 3:
                return "Por favor, hazme una pregunta más específica sobre sistemas HVAC o refrigeración."
            
            question_type = self.classify_question(question)
            
            # Try simple response first for common questions
            simple_response = self.get_simple_response(question, question_type)
            if simple_response:
                return simple_response
            
            # Use RAG system for complex questions
            if not self.vectorstore or not self.llm:
                return "El sistema está inicializándose. Por favor, intenta en unos momentos."
            
            # Get relevant documents
            retriever = self.vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 3, 'fetch_k': 6}
            )
            
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content[:500] for doc in docs[:3]])
            
            # Create prompt
            prompt_template = self.get_prompt_template(question_type)
            prompt = f"Eres un asistente virtual con conocimiento de sistemas de refrigeración. Responde en español de manera clara y profesional.\n\n{prompt_template.format(context=context, question=question)}"
            
            # Generate response
            try:
                result = self.llm.invoke(prompt)
                
                if isinstance(result, str):
                    response_text = result
                else:
                    response_text = str(result)
                
                # Ensure we have a valid response
                if len(response_text.strip()) < 20:
                    return self.get_fallback_response(question_type)
                
                cleaned_response = self.clean_response(response_text)
                
                # Final validation
                if len(cleaned_response) < 30:
                    return self.get_fallback_response(question_type)
                
                return cleaned_response
                
            except Exception as e:
                print(f"Model generation error: {e}")
                return self.get_fallback_response(question_type)
                
        except Exception as e:
            print(f"Error processing question: {e}")
            return "Disculpa, hubo un problema técnico. ¿Podrías intentar con una pregunta más simple sobre HVAC?"

    def get_fallback_response(self, question_type: str) -> str:
        """Provide fallback responses when model fails"""
        fallbacks = {
            'greeting': "¡Hola! Soy tu asistente de sistemas HVAC. ¿En qué puedo ayudarte?",
            'benefits': "Los sistemas HVAC modernos ofrecen mayor eficiencia energética, mejor control de temperatura y menor impacto ambiental. ¿Qué sistema específico te interesa?",
            'technical': "Para consultas técnicas específicas, necesito más detalles sobre tu instalación. ¿Podrías describir el problema o sistema que tienes?",
            'sizing': "Para calcular el dimensionamiento correcto, necesito conocer el área a climatizar, tipo de uso y condiciones específicas. ¿Puedes proporcionarme estos datos?",
            'inverter': "Los sistemas inverter son más eficientes y ofrecen mejor control de temperatura. ¿Te interesa algún aspecto específico de esta tecnología?",
            'general': "Puedo ayudarte con consultas sobre instalación, mantenimiento, eficiencia energética y selección de equipos HVAC. ¿Qué necesitas saber?"
        }
        
        return fallbacks.get(question_type, fallbacks['general'])

    def initialize_system(self):
        """Initialize the RAG system with better error handling"""
        try:
            print("Inicializando sistema HVAC RAG...")
            
            # Create chunks directory if needed
            if not os.path.exists(self.chunk_dir):
                print("Creando directorio de chunks...")
                os.makedirs(self.chunk_dir)
                
                pdf_path = "catalog.pdf"
                if os.path.exists(pdf_path):
                    chunks = self.process_pdf(pdf_path)
                    if chunks:
                        print(f"Creados {len(chunks)} chunks de documentos")
                        for i, chunk in enumerate(chunks):
                            chunk_file = os.path.join(self.chunk_dir, f"chunk_{i:04d}.txt")
                            with open(chunk_file, "w", encoding="utf-8") as f:
                                f.write(chunk.page_content)
                else:
                    print("Archivo catalog.pdf no encontrado. Sistema funcionará con respuestas predefinidas.")
            
            # Load vector store (optional)
            try:
                print("Cargando vector store...")
                self.vectorstore = self.load_or_create_vectorstore()
            except Exception as e:
                print(f"Warning: No se pudo cargar vector store: {e}")
                self.vectorstore = None
            
            # Load language model (optional)
            try:
                print("Configurando modelo de lenguaje...")
                self.llm = self.setup_model()
            except Exception as e:
                print(f"Warning: No se pudo cargar modelo: {e}")
                self.llm = None
            
            print("Sistema RAG inicializado (con componentes disponibles)!")
            return True
            
        except Exception as e:
            print(f"Error inicializando sistema RAG: {str(e)}")
            return True  # Continue with basic functionality


# Flask app setup
app = Flask(__name__)
rag_system = HVACRAGSystem()

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    """Handle WhatsApp webhook with better error handling"""
    try:
        incoming_msg = request.values.get("Body", "").strip()
        response = MessagingResponse()

        if not incoming_msg:
            response.message("Por favor, envía una pregunta sobre sistemas HVAC o refrigeración.")
        elif incoming_msg.lower() in ['quit', 'exit', 'salir', 'adios']:
            response.message("Sesión terminada. ¡Gracias por usar nuestro chatbot HVAC! Escribe cualquier cosa para comenzar de nuevo.")
        else:
            try:
                reply = rag_system.ask_question(incoming_msg)
                # Ensure we have a valid reply
                if not reply or len(reply.strip()) < 5:
                    reply = "Lo siento, no pude procesar tu consulta. ¿Podrías reformular tu pregunta sobre HVAC?"
                response.message(reply)
            except Exception as e:
                print(f"Error processing question: {e}")
                response.message("Disculpa, hubo un error procesando tu pregunta. Por favor intenta de nuevo con una consulta más específica.")

        return str(response)
        
    except Exception as e:
        print(f"Error in webhook: {e}")
        response = MessagingResponse()
        response.message("Error técnico. Por favor intenta nuevamente.")
        return str(response)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": rag_system.llm is not None,
        "vectorstore_loaded": rag_system.vectorstore is not None
    }

@app.route("/test", methods=["GET"])
def test_endpoint():
    """Test endpoint for debugging"""
    test_question = request.args.get("q", "¿Cuáles son las ventajas de un sistema inverter?")
    try:
        response = rag_system.ask_question(test_question)
        return {"question": test_question, "response": response}
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main function"""
    print("Iniciando chatbot HVAC WhatsApp con Twilio...")
    
    # Initialize system (will work even if some components fail)
    rag_system.initialize_system()
    
    # Start Flask app
    port = int(os.getenv("PORT", 3000))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)

if __name__ == "__main__":
    main()