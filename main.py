#!/usr/bin/env python3
"""
HVAC RAG System - Improved with Better Models
A WhatsApp chatbot for HVAC/refrigeration queries using better language models
"""

import os
import sys
import torch
import re
from typing import List, Any, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
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
    """Improved HVAC RAG System with better models"""
    
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
                chunk_size=512,  # Increased chunk size for better context
                chunk_overlap=128,  # Better overlap for context preservation
                length_function=len,
                separators=['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ', ''],
                is_separator_regex=False
            )
            
            chunks = text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def create_vectorstore_from_chunks(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> FAISS:
        """Create vector store with better multilingual embeddings"""
        try:
            # Better embedding model for Spanish/multilingual content
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
            # Save the vectorstore for future use
            vectorstore.save_local(self.index_path)
            print("Vector store created and saved successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def load_or_create_vectorstore(self) -> FAISS:
        """Load existing vector store or create new one"""
        embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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

    def setup_primary_model(self) -> HuggingFacePipeline:
        """Set up primary model - Microsoft DialoGPT Spanish or Llama-2-7B-Chat"""
        try:
            # Option 1: For GPU with enough VRAM (>= 8GB)
            if self.device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 8e9:
                model_name = "microsoft/DialoGPT-spanish"  # Good for Spanish conversations
                # Alternative: "meta-llama/Llama-2-7b-chat-hf" (requires HF access)
                
                print(f'Loading tokenizer for {model_name}...')
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
                
                print('Loading model with 4-bit quantization...')
                # Use 4-bit quantization to save memory
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                pipe = pipeline(
                    'text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                return HuggingFacePipeline(pipeline=pipe)
            else:
                # Fallback to CPU-friendly model
                return self.setup_cpu_optimized_model()
                
        except Exception as e:
            print(f"Primary model setup failed: {str(e)}")
            return self.setup_cpu_optimized_model()

    def setup_cpu_optimized_model(self) -> HuggingFacePipeline:
        """Set up CPU-optimized model"""
        try:
            # Best CPU option: Flan-T5 or DistilBERT-based models
            model_name = "google/flan-t5-base"  # Good multilingual performance
            # Alternative: "microsoft/DialoGPT-small" for smaller footprint
            
            print(f"Using CPU-optimized model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            pipe = pipeline(
                'text2text-generation' if 't5' in model_name.lower() else 'text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            print(f"CPU model setup failed: {str(e)}")
            return self.setup_fallback_model()

    def setup_fallback_model(self) -> HuggingFacePipeline:
        """Fallback to the most reliable model"""
        try:
            print("Using fallback model: DistilGPT-2")
            model_name = "distilgpt2"  # More capable than GPT-2 but still lightweight
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            pipe = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            print(f"All model setups failed: {str(e)}")
            raise Exception("Unable to load any model")

    def setup_model(self) -> HuggingFacePipeline:
        """Set up the best available language model"""
        print("Setting up language model...")
        
        # Try models in order of preference
        try:
            return self.setup_primary_model()
        except Exception as e:
            print(f"Primary model failed, trying CPU-optimized: {e}")
            try:
                return self.setup_cpu_optimized_model()
            except Exception as e2:
                print(f"CPU model failed, using fallback: {e2}")
                return self.setup_fallback_model()

    def classify_question(self, question: str) -> str:
        """Classify question type for appropriate prompt selection"""
        question_lower = question.lower()
        
        patterns = {
            'greeting': r'hola|buenos|buenas|saludos|como estas|que tal',
            'benefits': r'beneficios|ventajas|porqué|valor|precio|bueno|mejor|óptimo|razón|costo',
            'architecture': r'arquitecto|estructura|componente|diseño|construido|trabajo|infraestructura|diagrama|instalación',
            'features': r'característica|capacidad|habilidad|poder|función|hacer|ofertas|especificaciones',
            'technical': r'cómo|implementación|configuración|instalar|configurar|reparar|mantenimiento|problema|falla',
            'comparison': r'comparar|versus|vs|diferencia|mejor|que|cuál',
            'sizing': r'tamaño|dimensión|capacidad|potencia|btu|tonelada|metro|cuarto|espacio',
        }
        
        for q_type, pattern in patterns.items():
            if re.search(pattern, question_lower):
                return q_type
        
        return 'general'

    def get_prompt_template(self, question_type: str) -> str:
        """Get improved prompt template based on question type"""
        base_context = """
Eres un especialista en sistemas HVAC (calefacción, ventilación y aire acondicionado) con amplia experiencia en refrigeración comercial y residencial.

Responde de manera profesional, clara y útil. Si la información específica no está en el contexto proporcionado, utiliza tu conocimiento general de HVAC pero indica cuando estás usando conocimiento general vs. información del documento.

Para consultas técnicas, haz preguntas de seguimiento sobre:
- Tamaño del espacio (metros cuadrados/cúbicos)
- Tipo de aplicación (residencial/comercial/industrial)  
- Condiciones ambientales
- Presupuesto aproximado
- Eficiencia energética requerida

Contexto del documento:
{context}

Pregunta del usuario: {question}

Respuesta útil y profesional:"""

        templates = {
            'greeting': """
Eres un especialista amigable en sistemas HVAC. Saluda cordialmente y ofrece ayuda.

Pregunta: {question}

Respuesta amigable:""",
            
            'benefits': base_context + "\n\nEnfócate en ventajas económicas, eficiencia energética y beneficios a largo plazo.",
            
            'technical': base_context + "\n\nProporciona detalles técnicos específicos, procedimientos y mejores prácticas.",
            
            'sizing': base_context + "\n\nEnfócate en cálculos de carga térmica, dimensionamiento y recomendaciones específicas. Pregunta por dimensiones del espacio.",
            
            'comparison': base_context + "\n\nCompara objetivamente las opciones, destacando pros y contras de cada una.",
            
            'general': base_context
        }
        
        return templates.get(question_type, templates['general'])

    def get_chain_for_question(self, question: str) -> RetrievalQA:
        """Create QA chain for specific question type"""
        question_type = self.classify_question(question)
        prompt_template = self.get_prompt_template(question_type)
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=['context', 'question']
        )
        
        # Improved retriever settings
        retriever = self.vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={
                'k': 5,  # Retrieve more documents for better context
                'fetch_k': 10  # Consider more documents in initial search
            }
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={
                'prompt': PROMPT,
                'verbose': False
            },
            return_source_documents=True
        )
        
        return qa_chain

    def clean_response(self, response_text: str) -> str:
        """Enhanced response cleaning"""
        patterns_to_remove = [
            r'Pregunta:.*$',
            r'Respuesta:.*$', 
            r'Usuario:.*$',
            r'Asistente:.*$',
            r'Contexto del documento:.*$',
            r'\n\n.*Pregunta.*',
            r'\n\n.*Respuesta.*',
            r'\n\n.*Contexto.*',
            r'Human:.*$',
            r'Assistant:.*$',
        ]
        
        cleaned_text = response_text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.DOTALL)
        
        # Enhanced cleanup
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Remove multiple empty lines
        cleaned_text = re.sub(r'^\s*[\-\*\•]\s*', '', cleaned_text, flags=re.MULTILINE)  # Remove bullet points at start
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def format_response(self, result: Dict[str, Any]) -> str:
        """Format response for WhatsApp"""
        try:
            response_text = result.get('result', 'No se pudo generar una respuesta.')
            cleaned_response = self.clean_response(response_text)
            
            # Limit response length for WhatsApp
            if len(cleaned_response) > 1500:
                cleaned_response = cleaned_response[:1500] + "...\n\nPregunta más específica para obtener información detallada."
            
            return cleaned_response
            
        except Exception as e:
            return f"Error al procesar la respuesta: {str(e)}"

    def preprocess_question(self, question: str) -> str:
        """Enhanced question preprocessing"""
        question = question.strip()
        question = ' '.join(question.split())
        
        # Handle common abbreviations
        abbreviation_map = {
            'hvac': 'calefacción ventilación aire acondicionado',
            'a/c': 'aire acondicionado',
            'btu': 'unidad térmica británica',
        }
        
        question_lower = question.lower()
        for abbr, full_form in abbreviation_map.items():
            question_lower = question_lower.replace(abbr, full_form)
        
        return question_lower

    def ask_question(self, question: str) -> str:
        """Process question and return response"""
        try:
            processed_question = self.preprocess_question(question)
            chain = self.get_chain_for_question(processed_question)
            result = chain.invoke({"query": processed_question})
            return self.format_response(result)
        except Exception as e:
            return f"Error procesando pregunta: {str(e)}"

    def initialize_system(self):
        """Initialize the RAG system"""
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
            
            print("Cargando vector store...")
            self.vectorstore = self.load_or_create_vectorstore()
            if not self.vectorstore:
                return False
            
            print("Configurando modelo de lenguaje...")
            self.llm = self.setup_model()
            if not self.llm:
                return False
            
            print("Sistema RAG inicializado exitosamente!")
            return True
            
        except Exception as e:
            print(f"Error inicializando sistema RAG: {str(e)}")
            return False


# Flask app setup
app = Flask(__name__)
rag_system = HVACRAGSystem()
rag_system.initialize_system()

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    """Handle WhatsApp webhook"""
    incoming_msg = request.values.get("Body", "").strip()
    response = MessagingResponse()

    if incoming_msg.lower() in ['quit', 'exit', 'salir', 'adios']:
        response.message("Sesión terminada. ¡Gracias por usar nuestro chatbot HVAC! Escribe cualquier cosa para comenzar de nuevo.")
    elif not incoming_msg:
        response.message("Por favor, envía una pregunta sobre sistemas HVAC o refrigeración.")
    else:
        try:
            reply = rag_system.ask_question(incoming_msg)
            response.message(reply)
        except Exception as e:
            response.message("Disculpa, hubo un error procesando tu pregunta. Por favor intenta de nuevo.")
            print(f"Error in webhook: {e}")

    return str(response)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": rag_system.llm is not None}

def main():
    """Main function"""
    print("Iniciando chatbot HVAC WhatsApp con Twilio...")
    
    # Initialize system
    if not rag_system.initialize_system():
        print("Error: No se pudo inicializar el sistema.")
        return
    
    # Start Flask app
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=debug)

if __name__ == "__main__":
    main()