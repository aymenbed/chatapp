"""
Chatbot Spécialiste en Lubrification pour Véhicules Poids Lourds
---------------------------------------------------------------
Ce script implémente un chatbot spécialisé capable de répondre aux questions techniques et détaillées
sur les systèmes de lubrification des véhicules poids lourds en utilisant une architecture RAG
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

class SpecializedDocumentLoader:
    """Classe pour charger et prétraiter les documents PDF spécialisés en lubrification."""
    
    def __init__(self, documents_dir: str):
        """
        Initialise le chargeur de documents spécialisés.
        
        Args:
            documents_dir: Chemin vers le répertoire contenant les documents PDF spécialisés
        """
        self.documents_dir = documents_dir
        logger.info(f"Initialisation du chargeur de documents spécialisés depuis {documents_dir}")
    
    def load_documents(self) -> List[Any]:
        """
        Charge tous les documents PDF spécialisés du répertoire spécifié.
        
        Returns:
            Liste des documents chargés
        """
        documents = []
        try:
            pdf_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.pdf')]
            logger.info(f"Fichiers PDF spécialisés trouvés: {pdf_files}")
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.documents_dir, pdf_file)
                logger.info(f"Chargement du document spécialisé: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
                
            logger.info(f"Total de {len(documents)} pages spécialisées chargées")
            return documents
        except Exception as e:
            logger.error(f"Erreur lors du chargement des documents spécialisés: {e}")
            raise

class TechnicalTextProcessor:
    """Classe pour découper et normaliser le texte technique des documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialise le processeur de texte technique.
        
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
        logger.info(f"Initialisation du processeur de texte technique avec chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Découpe les documents techniques en segments de texte.
        
        Args:
            documents: Liste des documents techniques à découper
            
        Returns:
            Liste des segments de texte technique
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Documents techniques découpés en {len(chunks)} segments")
            return chunks
        except Exception as e:
            logger.error(f"Erreur lors du découpage des documents techniques: {e}")
            raise

class LubricationVectorStore:
    """Classe pour gérer les embeddings et la base vectorielle spécialisée en lubrification."""
    
    def __init__(self, embedding_model: str = "deepseek-r1"):
        """
        Initialise la base vectorielle spécialisée.
        
        Args:
            embedding_model: Nom du modèle Ollama à utiliser pour les embeddings
        """
        self.embedding_model = embedding_model
        try:
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            logger.info(f"Modèle d'embedding pour lubrification initialisé: {embedding_model}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle d'embedding pour lubrification: {e}")
            raise
    
    def create_vector_store(self, chunks: List[Any], persist_directory: Optional[str] = None) -> Any:
        """
        Crée une base vectorielle spécialisée à partir des segments de texte technique.
        
        Args:
            chunks: Liste des segments de texte technique
            persist_directory: Répertoire pour sauvegarder la base vectorielle (optionnel)
            
        Returns:
            Base vectorielle FAISS spécialisée
        """
        try:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("Base vectorielle FAISS spécialisée créée avec succès")
            
            if persist_directory:
                vector_store.save_local(persist_directory)
                logger.info(f"Base vectorielle spécialisée sauvegardée dans {persist_directory}")
                
            return vector_store
        except Exception as e:
            logger.error(f"Erreur lors de la création de la base vectorielle spécialisée: {e}")
            raise
    
    def load_vector_store(self, persist_directory: str) -> Any:
        """
        Charge une base vectorielle spécialisée existante.
        
        Args:
            persist_directory: Répertoire contenant la base vectorielle spécialisée
            
        Returns:
            Base vectorielle FAISS spécialisée
        """
        try:
            vector_store = FAISS.load_local(persist_directory, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Base vectorielle spécialisée chargée depuis {persist_directory}")
            return vector_store
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la base vectorielle spécialisée: {e}")
            raise

class TechnicalLLMInterface:
    """Classe pour communiquer avec l'API Groq pour des questions techniques."""
    
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192"):
        """
        Initialise l'interface technique avec le modèle de langage.
        
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
                temperature=0.1,  # Température plus basse pour des réponses plus précises
            )
            logger.info(f"Modèle LLM technique initialisé: {model_name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle LLM technique: {e}")
            raise

class SpecializedRAGPipeline:
    """Classe pour orchestrer le processus RAG spécialisé en lubrification."""
    
    def __init__(self, vector_store: Any, llm_interface: TechnicalLLMInterface):
        """
        Initialise le pipeline RAG spécialisé.
        
        Args:
            vector_store: Base vectorielle spécialisée pour la recherche
            llm_interface: Interface technique avec le modèle de langage
        """
        self.vector_store = vector_store
        self.llm = llm_interface.llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Création du template de prompt spécialisé
        prompt_template = """
        Tu es un expert technique spécialisé dans les systèmes de lubrification des véhicules poids lourds. 
        Tu réponds aux questions techniques et détaillées sur les lubrifiants, les composants des systèmes 
        de lubrification, les normes et standards, et les procédures de maintenance.
        
        Utilise uniquement les informations suivantes pour répondre à la question technique de l'utilisateur:
        {context}
        
        Si tu ne connais pas la réponse, dis simplement que tu ne sais pas. N'invente pas d'information technique.
        Lorsque tu mentionnes des spécifications ou des normes, sois précis et cite-les correctement.
        
        Question technique: {question}
        Réponse technique en français:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Création de la chaîne de conversation spécialisée
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 7}),  # Plus de contexte pour les questions techniques
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            output_key="answer"
        )
        
        logger.info("Pipeline RAG spécialisé initialisé avec succès")
    
    def process_technical_query(self, query: str) -> Dict[str, Any]:
        """
        Traite une requête technique utilisateur.
        
        Args:
            query: Question technique de l'utilisateur
            
        Returns:
            Dictionnaire contenant la réponse technique et les documents sources
        """
        try:
            logger.info(f"Traitement de la requête technique: {query}")
            result = self.chain({"question": query})
            logger.info("Réponse technique générée avec succès")
            return result
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête technique: {e}")
            raise

class ChatbotSpecialiste:
    """Classe principale du chatbot spécialiste en lubrification."""
    
    def __init__(self, documents_dir: str, api_key: str, vector_store_dir: Optional[str] = None):
        """
        Initialise le chatbot spécialiste en lubrification.
        
        Args:
            documents_dir: Répertoire contenant les documents PDF spécialisés
            api_key: Clé API pour GroqCloud
            vector_store_dir: Répertoire pour sauvegarder/charger la base vectorielle spécialisée (optionnel)
        """
        self.documents_dir = documents_dir
        self.api_key = api_key
        self.vector_store_dir = vector_store_dir
        
        # Initialisation des composants spécialisés
        self.document_loader = SpecializedDocumentLoader(documents_dir)
        self.text_processor = TechnicalTextProcessor()
        self.vector_store_manager = LubricationVectorStore()
        self.llm_interface = TechnicalLLMInterface(api_key)
        
        # Chargement ou création de la base vectorielle spécialisée
        if vector_store_dir and os.path.exists(vector_store_dir):
            logger.info(f"Chargement de la base vectorielle spécialisée existante depuis {vector_store_dir}")
            self.vector_store = self.vector_store_manager.load_vector_store(vector_store_dir)
        else:
            logger.info("Création d'une nouvelle base vectorielle spécialisée")
            documents = self.document_loader.load_documents()
            chunks = self.text_processor.split_documents(documents)
            self.vector_store = self.vector_store_manager.create_vector_store(chunks, vector_store_dir)
        
        # Initialisation du pipeline RAG spécialisé
        self.rag_pipeline = SpecializedRAGPipeline(self.vector_store, self.llm_interface)
        logger.info("Chatbot spécialiste en lubrification initialisé avec succès")
    
    def ask_technical(self, question: str) -> str:
        """
        Pose une question technique au chatbot spécialiste.
        
        Args:
            question: Question technique de l'utilisateur
            
        Returns:
            Réponse technique du chatbot
        """
        try:
            result = self.rag_pipeline.process_technical_query(question)
            return result["answer"]
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse technique: {e}")
            return f"Désolé, une erreur technique s'est produite: {str(e)}"

# Exemple d'utilisation
if __name__ == "__main__":
    # Ces valeurs seraient normalement définies dans un fichier de configuration ou des variables d'environnement
    DOCUMENTS_DIR = "./documents/specialise"
    VECTOR_STORE_DIR = "./vector_store/specialise"
    API_KEY = "votre_cle_api_groq"  # À remplacer par votre clé API
    
    # Création des répertoires nécessaires
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    # Initialisation du chatbot spécialiste
    chatbot = ChatbotSpecialiste(DOCUMENTS_DIR, API_KEY, VECTOR_STORE_DIR)
    
    # Exemple de question technique
    question = "Quelles sont les spécifications de viscosité recommandées pour l'huile de transmission d'un camion opérant dans des conditions hivernales ?"
    print(f"Question technique: {question}")
    reponse = chatbot.ask_technical(question)
    print(f"Réponse technique: {reponse}")
