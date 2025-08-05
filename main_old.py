#!/usr/bin/env python3
"""
ARO (Azure Red Hat OpenShift) RAG System - Terminal Version
A command-line interface for querying ARO documentation using RAG (Retrieval-Augmented Generation)
"""

import os
import sys
import wget
import torch
import re
from typing import List, Any, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
import os

account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_TOKEN")
debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on")
client = Client(account_sid, auth_token)

class ARORAGSystem:
    """ARO RAG System for terminal-based querying"""
    
    def __init__(self):
        self.qa_chain = None
        self.vectorstore = None
        self.llm = None
        self.chunk_dir = "chunks"
        self.index_path = "vectorstore_index"
        
    # def download_pdf(self, pdf_url: str = "https://learn.microsoft.com/pdf?url=https%3A%2F%2Flearn.microsoft.com%2Fen-us%2Fazure%2Fopenshift%2Ftoc.json") -> str:
    #     """Download the ARO documentation PDF"""
    #     pdf_path = "aro_docs.pdf"
        
    #     if not os.path.exists(pdf_path):
    #         print(f"Downloading PDF from {pdf_url}...")
    #         try:
    #             wget.download(pdf_url, pdf_path)
    #             print("\nDownload complete!")
    #         except Exception as e:
    #             print(f"Error downloading PDF: {e}")
    #             return None
    #     else:
    #         print("PDF file already exists.")
        
    #     return pdf_path

    def process_pdf(self, pdf_path: str) -> List[Any]:
        """Process PDF into document chunks"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            print(f"Loaded {len(documents)} pages from PDF")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=50,
                length_function=len,
                separators=['\n\n', '\n', '. ', '! ', '? ', ', ', ' ', ''],
                is_separator_regex=False
            )
            
            chunks = text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def create_vectorstore_from_chunks(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
        """Create vector store from document chunks"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cuda'},
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
            print("Vector store created successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def load_or_create_vectorstore(self) -> FAISS:
        """Load existing vector store or create new one"""
        embeddings = HuggingFaceEmbeddings()
        if os.path.exists(self.index_path):
            print("Loading precomputed vector store...")
            try:
                return FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading vector store: {e}")
                print("Creating new vector store...")
                return self.create_vectorstore_from_chunks()
        else:
            print("Vector store not found. Creating a new one...")
            return self.create_vectorstore_from_chunks()

    def setup_model(self) -> HuggingFacePipeline:
        """Set up the language model"""
        try:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            print('Loading tokenizer...')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            print('Loading model...')
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            print('Setting up pipeline...')
            pipe = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=128,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Primary model setup failed: {str(e)}")
            return self.setup_fallback_model()

    def setup_fallback_model(self) -> HuggingFacePipeline:
        """Set up fallback model (GPT-2)"""
        try:
            print("\nUsing fallback model: GPT-2")
            model_name = "gpt2"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda",
                torch_dtype=torch.float32
            )
            
            pipe = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Fallback model setup failed: {str(e)}")
            raise

    def classify_question(self, question: str) -> str:
        """Classify question type for appropriate prompt selection"""
        question_lower = question.lower()
        
        patterns = {
            'benefits': r'beneficios|ventajas|porqué|valor|precio|bueno|mejor|óptimo|razón',
            'architecture': r'arquitecto|estructura|componente|diseño|construido|trabajo|infraestructura|diagrama|plantilla',
            'features': r'característica|capacidad|habilidad|poder|función|hacer|ofertas',
            'technical': r'cómo|implementación|configuración|instalar|configurar|desplegar|solución|depurar',
            'comparison': r'comparar|versus|vs|differencia|mejor|que',
        }
        
        for q_type, pattern in patterns.items():
            if re.search(pattern, question_lower):
                return q_type
        
        return 'general'

    def get_prompt_template(self, question_type: str) -> str:
        """Get appropriate prompt template based on question type"""
        base_template = """Eres un asistente de IA especializado en refrigeración. Puedes manejar preguntas técnicas asi como conversación casual.

Para preguntas tecnicas acerca de refrigeración, usa el contexto provisto para responder adecuadamente. Si la información no se encuentra en el contexto, di "No tengo suficiente información para responder correctamente."

Para conversación casual (saludos, charla, etc.), responde naturalmente, luego guía gentilmente la conversación hacia el tema de refrigeración.

Haz preguntas para guiar la conversación hacia la refrigeración. Cuando pregunten por recomendaciones o sugerencias, haz preguntas acerca del caso del usuario y pide especificaciones.
Cuando no tengas suficientes detalles, pregunta por información pertinente para hacer una sugerencia. Pregunta por el tamaño del cuarto, la potencia deseada, la capacidad del cuarto.
Enfocate en hacer preguntas.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
        
        templates = {
            'general': base_template,
            'benefits': base_template + "\nEnfocate en las ventajas y el valor del negocio.",
            'technical': base_template + "\nEnfocate en detalles técnicos y la implementación.",
            'architecture': base_template + "\nEnfocate en componenetes del sistema y estructura.",
            'features': base_template + "\nEnfocate en capacidades y funcionalidades.",
            'comparison': base_template + "\nEnfocate en comparar elementos especificos."
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
        
        retriever = self.vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 3}
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

    def preprocess_question(self, question: str) -> str:
        """Preprocess user question"""
        question = question.strip()
        question = ' '.join(question.split())       
        if not question.endswith('?'):
            question += '?'
        return question

    def clean_response(self, response_text: str) -> str:
        """Clean the response text to remove unwanted patterns"""
        # Remove common patterns that appear in generated text
        patterns_to_remove = [
            r'Pregunta:.*$',  # Remove "Pregunta:" and everything after
            r'Respuesta:.*$',  # Remove "Respuesta:" and everything after  
            r'Usuario:.*$',    # Remove "Usuario:" and everything after
            r'\n\n.*Pregunta.*',  # Remove paragraphs starting with "Pregunta"
            r'\n\n.*Respuesta.*', # Remove paragraphs starting with "Respuesta"

            r'Conversación:.*$',  # Remove "Pregunta:" and everything after
            r'Asistente:.*$',  # Remove "Respuesta:" and everything after  
            r'\n\n.*Conversación.*',  # Remove paragraphs starting with "Pregunta"
            r'\n\n.*Asistente.*', # Remove paragraphs starting with "Respuesta"

        ]
        
        cleaned_text = response_text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.DOTALL)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Remove multiple empty lines
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def format_response(self, result: Dict[str, Any]) -> str:
        """Format response for terminal display"""
        try:
            response_text = result.get('result', 'No response text available')

            # Clean the response text
            cleaned_response = self.clean_response(response_text)
            print("############", cleaned_response)
            
            sources = []
            
            source_docs = result.get('source_documents', [])
            for doc in source_docs:
                if hasattr(doc, 'metadata'):
                    source = doc.metadata.get('source', 'Unknown source')
                    sources.append(f"  - {source}")
                elif isinstance(doc, dict):
                    source = doc.get('metadata', {}).get('source', 'Unknown source')
                    sources.append(f"  - {source}")

            # formatted_response = f"\n{'='*60}\nRESPUESTA:\n{'='*60}\n"
            formatted_response = f"{cleaned_response}\n"
            # formatted_response += f"\n{'='*60}\nSOURCES:\n{'='*60}\n"
            # formatted_response += '\n'.join(sources) if sources else 'No sources available'
            # formatted_response += f"\n{'='*60}\n"
            
            return formatted_response
        except Exception as e:
            return f"Error formatting response: {str(e)}"

    def initialize_system(self):
        """Initialize the RAG system"""
        try:
            print("Starting RAG system initialization...")
            
            # Create chunks directory if it doesn't exist
            if not os.path.exists(self.chunk_dir):
                print("Creating chunks directory...")
                os.makedirs(self.chunk_dir)

                print("Downloading and processing PDF...")
                pdf_path = "catalog.pdf"#self.download_pdf()
                if not pdf_path:
                    return False
                
                chunks = self.process_pdf(pdf_path)
                if not chunks:
                    return False
                    
                print(f"Created {len(chunks)} document chunks")

                print("Saving chunks to files...")
                for i, chunk in enumerate(chunks):
                    chunk_file = os.path.join(self.chunk_dir, f"chunk_{i:04d}.txt")
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        f.write(chunk.page_content)
            
            print("Loading vector store...")
            self.vectorstore = self.load_or_create_vectorstore()
            if not self.vectorstore:
                return False
            
            print("Setting up language model...")
            self.llm = self.setup_model()
            if not self.llm:
                return False
            
            print("RAG system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            return False

    def ask_question(self, question: str) -> str:
        """Ask a question and get response"""
        print("HERE IS A QUESTION")
        try:
            processed_question = self.preprocess_question(question)
            chain = self.get_chain_for_question(processed_question)
            result = chain.invoke({"query": processed_question})
            return self.format_response(result)
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def run_interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*60)
        print("Chatbot HVAC")
        print("="*60)
        print("Haz preguntas de refrigeración")
        print("Escribe 'quit' o 'exit' para salir")
        print("="*60)
        
        while True:
            try:
                question = input("\nIntroduce la pregunta: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Adiós!")
                    break
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                elif not question:
                    print("Por favor introduce una pregunta.")
                    continue
                
                print("\nProcesando pregunta...")
                response = self.ask_question(question)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nTerminando...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    def show_help(self):
        """Show help information"""
        help_text = """
        Chatbot HVAC
        ==================
        
        Este sistema responde preguntas acerca de sistemas de refrigeración.
        """
        print(help_text)


# def main():
#     """Main function"""
#     if len(sys.argv) > 1:
#         if sys.argv[1] in ['-h', '--help']:
#             print("""
#             Chatbot HVAC
#             ===============================
            
#             Usage:
#                 python main.py                    # Interactive mode
#                 python main.py -h, --help         # Show help
            
#             Este sistema responde preguntas acerca de sistemas de refrigeración.
#             """)
#             return
    
#     # Initialize the system
#     rag_system = ARORAGSystem()
    
#     print("Initializing HVAC RAG System...")
#     if not rag_system.initialize_system():
#         print("Failed to initialize system. Exiting.")
#         return
    
#     # Start interactive mode
#     rag_system.run_interactive_mode()

def main():
    print("Running HVAC WhatsApp chatbot on Twilio webhook...")
    app.run(host="0.0.0.0", port=3000, debug=debug)


app = Flask(__name__)
rag_system = ARORAGSystem()
rag_system.initialize_system()
initialized = True

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    global initialized

    incoming_msg = request.values.get("Body", "").strip()
    response = MessagingResponse()

    #if not initialized:
    #    if not rag_system.initialize_system():
    #        response.message("Error initializing system.")
    #        return str(response)
    #    initialized = True

    if incoming_msg.lower() in ['quit', 'exit', 'q']:
        response.message("Sesión terminada. Escribe cualquier cosa para comenzar de nuevo.")
    else:
        reply = rag_system.ask_question(incoming_msg)
        response.message(reply)

    return str(response)

if __name__ == "__main__":
    main()
