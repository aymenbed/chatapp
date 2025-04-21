"""
Interface Utilisateur pour les Chatbots de Maintenance Prédictive
----------------------------------------------------------------
Ce script implémente une interface utilisateur avec Gradio pour les deux chatbots :
- Chatbot Généraliste pour les questions générales sur les véhicules poids lourds
- Chatbot Spécialiste pour les questions techniques sur les systèmes de lubrification

L'interface permet à l'utilisateur de choisir le chatbot à utiliser et d'interagir avec lui.
"""

import os
from dotenv import load_dotenv
import gradio as gr
from chatbot_generaliste import ChatbotGeneraliste
from chatbot_specialiste import ChatbotSpecialiste
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration des chemins et de l'API
load_dotenv()
DOCUMENTS_DIR_GENERAL = "./documents/general"
DOCUMENTS_DIR_SPECIALISE = "./documents/specialise"
VECTOR_STORE_DIR_GENERAL = "./vector_store/general"
VECTOR_STORE_DIR_SPECIALISE = "./vector_store/specialise"
API_KEY = os.getenv("GROQ_API_KEY", "votre_cle_api_groq")  # À remplacer par votre clé API ou définir comme variable d'environnement

# Création des répertoires nécessaires
os.makedirs(DOCUMENTS_DIR_GENERAL, exist_ok=True)
os.makedirs(DOCUMENTS_DIR_SPECIALISE, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR_GENERAL, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR_SPECIALISE, exist_ok=True)

# Initialisation des chatbots
try:
    chatbot_generaliste = ChatbotGeneraliste(DOCUMENTS_DIR_GENERAL, API_KEY, VECTOR_STORE_DIR_GENERAL)
    chatbot_specialiste = ChatbotSpecialiste(DOCUMENTS_DIR_SPECIALISE, API_KEY, VECTOR_STORE_DIR_SPECIALISE)
    logger.info("Chatbots initialisés avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation des chatbots: {e}")
    # Nous continuons malgré l'erreur pour permettre à l'interface de se lancer
    # Les erreurs seront gérées lors de l'utilisation des chatbots

# Historique des conversations
conversation_history_general = []
conversation_history_specialise = []

def respond_general(message, history):
    """
    Fonction pour traiter les questions adressées au chatbot généraliste.
    
    Args:
        message: Question de l'utilisateur
        history: Historique de la conversation
        
    Returns:
        Réponse du chatbot généraliste
    """
    try:
        logger.info(f"Question au chatbot généraliste: {message}")
        response = chatbot_generaliste.ask(message)
        logger.info("Réponse généraliste générée avec succès")
        conversation_history_general.append((message, response))
        return response
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse généraliste: {e}")
        return f"Désolé, une erreur s'est produite: {str(e)}\n\nVeuillez vérifier que les documents PDF sont bien présents dans le répertoire {DOCUMENTS_DIR_GENERAL} et que la clé API Groq est valide."

def respond_specialise(message, history):
    """
    Fonction pour traiter les questions adressées au chatbot spécialiste.
    
    Args:
        message: Question technique de l'utilisateur
        history: Historique de la conversation
        
    Returns:
        Réponse technique du chatbot spécialiste
    """
    try:
        logger.info(f"Question technique au chatbot spécialiste: {message}")
        response = chatbot_specialiste.ask_technical(message)
        logger.info("Réponse technique générée avec succès")
        conversation_history_specialise.append((message, response))
        return response
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse technique: {e}")
        return f"Désolé, une erreur technique s'est produite: {str(e)}\n\nVeuillez vérifier que les documents PDF spécialisés sont bien présents dans le répertoire {DOCUMENTS_DIR_SPECIALISE} et que la clé API Groq est valide."

def clear_history_general():
    """
    Fonction pour effacer l'historique de conversation du chatbot généraliste.
    
    Returns:
        None
    """
    global conversation_history_general
    conversation_history_general = []
    if hasattr(chatbot_generaliste, 'rag_pipeline') and hasattr(chatbot_generaliste.rag_pipeline, 'memory'):
        chatbot_generaliste.rag_pipeline.memory.clear()
    logger.info("Historique du chatbot généraliste effacé")
    return None

def clear_history_specialise():
    """
    Fonction pour effacer l'historique de conversation du chatbot spécialiste.
    
    Returns:
        None
    """
    global conversation_history_specialise
    conversation_history_specialise = []
    if hasattr(chatbot_specialiste, 'rag_pipeline') and hasattr(chatbot_specialiste.rag_pipeline, 'memory'):
        chatbot_specialiste.rag_pipeline.memory.clear()
    logger.info("Historique du chatbot spécialiste effacé")
    return None

def upload_document_general(files):
    """
    Fonction pour télécharger des documents PDF pour le chatbot généraliste.
    
    Args:
        files: Liste des fichiers téléchargés
        
    Returns:
        Message de confirmation
    """
    try:
        file_paths = []
        for file in files:
            if file.name.endswith('.pdf'):
                dest_path = os.path.join(DOCUMENTS_DIR_GENERAL, os.path.basename(file.name))
                with open(dest_path, 'wb') as f:
                    f.write(file.read())
                file_paths.append(dest_path)
                logger.info(f"Document généraliste téléchargé: {dest_path}")
        
        # Réinitialisation du chatbot pour prendre en compte les nouveaux documents
        global chatbot_generaliste
        chatbot_generaliste = ChatbotGeneraliste(DOCUMENTS_DIR_GENERAL, API_KEY, VECTOR_STORE_DIR_GENERAL)
        
        return f"{len(file_paths)} document(s) téléchargé(s) avec succès pour le chatbot généraliste."
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des documents généralistes: {e}")
        return f"Erreur lors du téléchargement des documents: {str(e)}"

def upload_document_specialise(files):
    """
    Fonction pour télécharger des documents PDF pour le chatbot spécialiste.
    
    Args:
        files: Liste des fichiers téléchargés
        
    Returns:
        Message de confirmation
    """
    try:
        file_paths = []
        for file in files:
            if file.name.endswith('.pdf'):
                dest_path = os.path.join(DOCUMENTS_DIR_SPECIALISE, os.path.basename(file.name))
                with open(dest_path, 'wb') as f:
                    f.write(file.read())
                file_paths.append(dest_path)
                logger.info(f"Document spécialisé téléchargé: {dest_path}")
        
        # Réinitialisation du chatbot pour prendre en compte les nouveaux documents
        global chatbot_specialiste
        chatbot_specialiste = ChatbotSpecialiste(DOCUMENTS_DIR_SPECIALISE, API_KEY, VECTOR_STORE_DIR_SPECIALISE)
        
        return f"{len(file_paths)} document(s) téléchargé(s) avec succès pour le chatbot spécialiste."
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des documents spécialisés: {e}")
        return f"Erreur lors du téléchargement des documents: {str(e)}"

# Création de l'interface Gradio
with gr.Blocks(title="Chatbots de Maintenance Prédictive pour Véhicules Poids Lourds") as demo:
    gr.Markdown("""
    # Chatbots de Maintenance Prédictive pour Véhicules Poids Lourds
    
    Cette application propose deux chatbots spécialisés :
    
    1. **Chatbot Généraliste** : Répond aux questions générales sur les véhicules poids lourds, leurs systèmes mécaniques, électriques et hydrauliques.
    
    2. **Chatbot Spécialiste** : Répond spécifiquement aux questions techniques et détaillées sur les systèmes de lubrification.
    
    Choisissez l'onglet correspondant au type de question que vous souhaitez poser.
    """)
    
    with gr.Tabs():
        with gr.TabItem("Chatbot Généraliste"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot_general_interface = gr.Chatbot(label="Conversation avec le Chatbot Généraliste")
                    msg_general = gr.Textbox(
                        placeholder="Posez une question générale sur les véhicules poids lourds...",
                        label="Votre question"
                    )
                    with gr.Row():
                        submit_general = gr.Button("Envoyer")
                        clear_general = gr.Button("Effacer la conversation")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Téléchargement de documents")
                    upload_general = gr.File(
                        file_types=[".pdf"],
                        file_count="multiple",
                        label="Télécharger des documents PDF généraux"
                    )
                    upload_general_button = gr.Button("Télécharger")
                    upload_general_output = gr.Textbox(label="Statut du téléchargement")
        
        with gr.TabItem("Chatbot Spécialiste en Lubrification"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot_specialise_interface = gr.Chatbot(label="Conversation avec le Chatbot Spécialiste")
                    msg_specialise = gr.Textbox(
                        placeholder="Posez une question technique sur les systèmes de lubrification...",
                        label="Votre question technique"
                    )
                    with gr.Row():
                        submit_specialise = gr.Button("Envoyer")
                        clear_specialise = gr.Button("Effacer la conversation")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Téléchargement de documents")
                    upload_specialise = gr.File(
                        file_types=[".pdf"],
                        file_count="multiple",
                        label="Télécharger des documents PDF spécialisés"
                    )
                    upload_specialise_button = gr.Button("Télécharger")
                    upload_specialise_output = gr.Textbox(label="Statut du téléchargement")
    
    # Configuration des événements
    submit_general.click(respond_general, inputs=[msg_general, chatbot_general_interface], outputs=chatbot_general_interface)
    msg_general.submit(respond_general, inputs=[msg_general, chatbot_general_interface], outputs=chatbot_general_interface)
    clear_general.click(clear_history_general, outputs=chatbot_general_interface)
    
    submit_specialise.click(respond_specialise, inputs=[msg_specialise, chatbot_specialise_interface], outputs=chatbot_specialise_interface)
    msg_specialise.submit(respond_specialise, inputs=[msg_specialise, chatbot_specialise_interface], outputs=chatbot_specialise_interface)
    clear_specialise.click(clear_history_specialise, outputs=chatbot_specialise_interface)
    
    upload_general_button.click(upload_document_general, inputs=upload_general, outputs=upload_general_output)
    upload_specialise_button.click(upload_document_specialise, inputs=upload_specialise, outputs=upload_specialise_output)

# Lancement de l'interface
if __name__ == "__main__":
    try:
        logger.info("Démarrage de l'interface utilisateur")
        demo.launch(share=True)
    except Exception as e:
        logger.error(f"Erreur lors du lancement de l'interface: {e}")
