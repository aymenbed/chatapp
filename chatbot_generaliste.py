"""
Chatbot Généraliste pour Véhicules Poids Lourds
------------------------------------------------
Ce script implémente un chatbot capable de répondre aux questions générales sur les véhicules poids lourds,
leurs systèmes mécaniques, électriques et hydrauliques en utilisant une architecture RAG
(Retrieval-Augmented Generation).

Technologies utilisées:
- LangChain pour l'orchestration du pipeline RAG
- Ollama Embeddings pour la vectorisation du texte
- FAISS pour le stockage vectoriel
- Groq comme modèle de langage (LLM)
"""

import os
import logging
from typing import List, Dict, Any, Optional

# Bibliothèques LangChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Classe pour charger et prétraiter les documents PDF."""
    
    def __init__(self, documents_dir: str):
        """
        Initialise le chargeur de documents.
        
        Args:
            documents_dir: Chemin vers le répertoire contenant les documents PDF
        """
        self.documents_dir = documents_dir
        logger.info(f"Initialisation du chargeur de documents depuis {documents_dir}")
    
    def load_documents(self) -> List[Any]:
        """
        Charge tous les documents PDF du répertoire spécifié.
        
        Returns:
            Liste des documents chargés
        """
        documents = []
        try:
            pdf_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.pdf')]
            logger.info(f"Fichiers PDF trouvés: {pdf_files}")
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.documents_dir, pdf_file)
                logger.info(f"Chargement du document: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                
            logger.info(f"Total de {len(documents)} pages chargées")
            return documents
        except Exception as e:
            logger.error(f"Erreur lors du chargement des documents: {e}")
            raise

class TextProcessor:
    """Classe pour découper et normaliser le texte des documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialise le processeur de texte.
        
        Args:
            chunk_size: Taille des segments de texte
            chunk_overlap: Chevauchement entre segments
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(f"Initialisation du processeur de texte avec chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Découpe les documents en segments de texte.
        
        Args:
            documents: Liste des documents à découper
            
        Returns:
            Liste des segments de texte
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Documents découpés en {len(chunks)} segments")
            return chunks
        except Exception as e:
            logger.error(f"Erreur lors du découpage des documents: {e}")
            raise

class VectorStore:
    """Classe pour gérer les embeddings et la base vectorielle."""
    
    def __init__(self, embedding_model: str = "deepseek-r1"):
        """
        Initialise la base vectorielle.
        
        Args:
            embedding_model: Nom du modèle Ollama à utiliser pour les embeddings
        """
        self.embedding_model = embedding_model
        try:
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            logger.info(f"Modèle d'embedding initialisé: {embedding_model}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle d'embedding: {e}")
            raise
    
    def create_vector_store(self, chunks: List[Any], persist_directory: Optional[str] = None) -> Any:
        """
        Crée une base vectorielle à partir des segments de texte.
        
        Args:
            chunks: Liste des segments de texte
            persist_directory: Répertoire pour sauvegarder la base vectorielle (optionnel)
            
        Returns:
            Base vectorielle FAISS
        """
        try:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("Base vectorielle FAISS créée avec succès")
            
            if persist_directory:
                vector_store.save_local(persist_directory)
                logger.info(f"Base vectorielle sauvegardée dans {persist_directory}")
                
            return vector_store
        except Exception as e:
            logger.error(f"Erreur lors de la création de la base vectorielle: {e}")
            raise
    
    def load_vector_store(self, persist_directory: str) -> Any:
        """
        Charge une base vectorielle existante.
        
        Args:
            persist_directory: Répertoire contenant la base vectorielle
            
        Returns:
            Base vectorielle FAISS
        """
        try:
            vector_store = FAISS.load_local(persist_directory, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Base vectorielle chargée depuis {persist_directory}")
            return vector_store
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la base vectorielle: {e}")
            raise

class LLMInterface:
    """Classe pour communiquer avec l'API Groq."""
    
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        """
        Initialise l'interface avec le modèle de langage.
        
        Args:
            api_key: Clé API pour GroqCloud
            model_name: Nom du modèle à utiliser
        """
        self.api_key = api_key
        self.model_name = model_name
        try:
            self.llm = ChatGroq(
                api_key=api_key,
                model_name=model_name,
                temperature=0.2,
            )
            logger.info(f"Modèle LLM initialisé: {model_name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle LLM: {e}")
            raise

class RAGPipeline:
    """Classe pour orchestrer le processus RAG complet."""
    
    def __init__(self, vector_store: Any, llm_interface: LLMInterface):
        """
        Initialise le pipeline RAG.
        
        Args:
            vector_store: Base vectorielle pour la recherche
            llm_interface: Interface avec le modèle de langage
        """
        self.vector_store = vector_store
        self.llm = llm_interface.llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Création du template de prompt
        prompt_template = """
        Tu es un assistant spécialisé dans les véhicules poids lourds. Tu réponds aux questions sur les systèmes mécaniques, électriques et hydrauliques des poids lourds.
        
        Utilise uniquement les informations suivantes pour répondre à la question de l'utilisateur:
        {context}
        
        Si tu ne connais pas la réponse, dis simplement que tu ne sais pas. N'invente pas d'information.
        
        Question: {question}
        Réponse en français:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Création de la chaîne de conversation
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            output_key="answer"
        )
        
        logger.info("Pipeline RAG initialisé avec succès")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Traite une requête utilisateur.
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Dictionnaire contenant la réponse et les documents sources
        """
        try:
            logger.info(f"Traitement de la requête: {query}")
            result = self.chain({"question": query})
            logger.info("Réponse générée avec succès")
            return result
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête: {e}")
            raise

class ChatbotGeneraliste:
    """Classe principale du chatbot généraliste."""
    
    def __init__(self, documents_dir: str, api_key: str, vector_store_dir: Optional[str] = None):
        """
        Initialise le chatbot généraliste.
        
        Args:
            documents_dir: Répertoire contenant les documents PDF
            api_key: Clé API pour GroqCloud
            vector_store_dir: Répertoire pour sauvegarder/charger la base vectorielle (optionnel)
        """
        self.documents_dir = documents_dir
        self.api_key = api_key
        self.vector_store_dir = vector_store_dir
        
        # Initialisation des composants
        self.document_loader = DocumentLoader(documents_dir)
        self.text_processor = TextProcessor()
        self.vector_store_manager = VectorStore()
        self.llm_interface = LLMInterface(api_key)
        
        # Vérification de la présence du fichier index.faiss (et non seulement du dossier)
        index_faiss_path = os.path.join(vector_store_dir, "index.faiss") if vector_store_dir else None
        if vector_store_dir and index_faiss_path and os.path.isfile(index_faiss_path):
            self.vector_store = self.vector_store_manager.load_vector_store(vector_store_dir)
        else:
            logger.info("Création d'une nouvelle base vectorielle")
            documents = self.document_loader.load_documents()
            chunks = self.text_processor.split_documents(documents)
            self.vector_store = self.vector_store_manager.create_vector_store(chunks, vector_store_dir)
        
        # Initialisation du pipeline RAG
        self.rag_pipeline = RAGPipeline(self.vector_store, self.llm_interface)
        logger.info("Chatbot généraliste initialisé avec succès")
    
    def ask(self, question: str) -> str:
        """
        Pose une question au chatbot.
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            Réponse du chatbot
        """
        try:
            result = self.rag_pipeline.process_query(question)
            return result["answer"]
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            return f"Désolé, une erreur s'est produite: {str(e)}"

# Exemple d'utilisation
if __name__ == "__main__":
    # Ces valeurs seraient normalement définies dans un fichier de configuration ou des variables d'environnement
    DOCUMENTS_DIR = "./documents/general"
    VECTOR_STORE_DIR = "./vector_store/general"
    API_KEY = "votre_cle_api_groq"  # À remplacer par votre clé API
    
    # Création des répertoires nécessaires
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    # Initialisation du chatbot
    chatbot = ChatbotGeneraliste(DOCUMENTS_DIR, API_KEY, VECTOR_STORE_DIR)
    
    # Exemple de question
    question = "Comment fonctionne le système de freinage d'un camion ?"
    print(f"Question: {question}")
    reponse = chatbot.ask(question)
    print(f"Réponse: {reponse}")
