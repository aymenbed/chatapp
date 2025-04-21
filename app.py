import os
import streamlit as st
from chatbot_generaliste import ChatbotGeneraliste
from chatbot_specialiste import ChatbotSpecialiste
from dotenv import load_dotenv

# Configuration des chemins et de l'API
load_dotenv()
DOCUMENTS_DIR_GENERAL = "./documents/general"
DOCUMENTS_DIR_SPECIALISE = "./documents/specialise"
VECTOR_STORE_DIR_GENERAL = "./vector_store/general"
VECTOR_STORE_DIR_SPECIALISE = "./vector_store/specialise"
API_KEY = os.getenv("GROQ_API_KEY", "votre_cle_api_groq")

# Création des répertoires nécessaires
os.makedirs(DOCUMENTS_DIR_GENERAL, exist_ok=True)
os.makedirs(DOCUMENTS_DIR_SPECIALISE, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR_GENERAL, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR_SPECIALISE, exist_ok=True)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbots de Maintenance Prédictive",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et description
st.title("Chatbots de Maintenance Prédictive pour Véhicules Poids Lourds")
st.markdown("""
Cette application propose deux chatbots spécialisés :

1. **Chatbot Généraliste** : Répond aux questions générales sur les véhicules poids lourds, leurs systèmes mécaniques, électriques et hydrauliques.

2. **Chatbot Spécialiste** : Répond spécifiquement aux questions techniques et détaillées sur les systèmes de lubrification.

Choisissez le type de chatbot dans la barre latérale.
""")

# Barre latérale pour la sélection du chatbot
st.sidebar.title("Configuration")
chatbot_type = st.sidebar.radio(
    "Choisissez un chatbot :",
    ["Généraliste", "Spécialiste en Lubrification"]
)

# Initialisation des sessions state pour les historiques de conversation
if 'history_general' not in st.session_state:
    st.session_state.history_general = []
if 'history_specialise' not in st.session_state:
    st.session_state.history_specialise = []

# Fonction pour initialiser les chatbots
@st.cache_resource
def load_chatbot_generaliste():
    try:
        return ChatbotGeneraliste(DOCUMENTS_DIR_GENERAL, API_KEY, VECTOR_STORE_DIR_GENERAL)
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du chatbot généraliste: {e}")
        return None

@st.cache_resource
def load_chatbot_specialiste():
    try:
        return ChatbotSpecialiste(DOCUMENTS_DIR_SPECIALISE, API_KEY, VECTOR_STORE_DIR_SPECIALISE)
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du chatbot spécialiste: {e}")
        return None

# Fonction pour télécharger des documents
def upload_documents(uploaded_files, target_dir, chatbot_type):
    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith('.pdf'):
                with open(os.path.join(target_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
                st.success(f"Fichier '{file.name}' téléchargé avec succès!")
            else:
                st.error(f"Le fichier '{file.name}' n'est pas un PDF.")
        
        # Réinitialisation du cache pour recharger le chatbot
        if chatbot_type == "Généraliste":
            st.cache_resource.clear()
            load_chatbot_generaliste()
        else:
            st.cache_resource.clear()
            load_chatbot_specialiste()
        
        return True
    return False

# Interface en fonction du type de chatbot sélectionné
if chatbot_type == "Généraliste":
    # Téléchargement de documents pour le chatbot généraliste
    st.sidebar.header("Téléchargement de documents")
    uploaded_files = st.sidebar.file_uploader(
        "Télécharger des documents PDF généraux",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.sidebar.button("Télécharger les documents"):
        upload_documents(uploaded_files, DOCUMENTS_DIR_GENERAL, "Généraliste")
    
    # Initialisation du chatbot généraliste
    chatbot_generaliste = load_chatbot_generaliste()
    
    # Zone de chat
    st.header("Conversation avec le Chatbot Généraliste")
    
    # Affichage de l'historique
    for question, answer in st.session_state.history_general:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
    
    # Zone de saisie de la question
    if question := st.chat_input("Posez une question générale sur les véhicules poids lourds..."):
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            if chatbot_generaliste:
                with st.spinner("Réflexion en cours..."):
                    try:
                        answer = chatbot_generaliste.ask(question)
                        st.write(answer)
                        st.session_state.history_general.append((question, answer))
                    except Exception as e:
                        st.error(f"Erreur: {e}")
            else:
                st.error("Le chatbot généraliste n'a pas pu être initialisé. Veuillez vérifier les documents et la clé API.")
    
    # Bouton pour effacer l'historique
    if st.sidebar.button("Effacer l'historique de conversation"):
        st.session_state.history_general = []
        st.experimental_rerun()

else:  # Chatbot Spécialiste
    # Téléchargement de documents pour le chatbot spécialiste
    st.sidebar.header("Téléchargement de documents")
    uploaded_files = st.sidebar.file_uploader(
        "Télécharger des documents PDF spécialisés",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.sidebar.button("Télécharger les documents"):
        upload_documents(uploaded_files, DOCUMENTS_DIR_SPECIALISE, "Spécialiste")
    
    # Initialisation du chatbot spécialiste
    chatbot_specialiste = load_chatbot_specialiste()
    
    # Zone de chat
    st.header("Conversation avec le Chatbot Spécialiste en Lubrification")
    
    # Affichage de l'historique
    for question, answer in st.session_state.history_specialise:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
    
    # Zone de saisie de la question
    if question := st.chat_input("Posez une question technique sur les systèmes de lubrification..."):
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            if chatbot_specialiste:
                with st.spinner("Analyse technique en cours..."):
                    try:
                        answer = chatbot_specialiste.ask_technical(question)
                        st.write(answer)
                        st.session_state.history_specialise.append((question, answer))
                    except Exception as e:
                        st.error(f"Erreur: {e}")
            else:
                st.error("Le chatbot spécialiste n'a pas pu être initialisé. Veuillez vérifier les documents et la clé API.")
    
    # Bouton pour effacer l'historique
    if st.sidebar.button("Effacer l'historique de conversation"):
        st.session_state.history_specialise = []
        st.experimental_rerun()

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info("""
**Note**: Pour un fonctionnement optimal, veuillez télécharger des documents PDF pertinents pour chaque chatbot et définir une clé API Groq valide.
""")
