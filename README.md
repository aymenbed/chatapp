# Documentation du Projet de Chatbots pour la Maintenance Prédictive des Systèmes de Lubrification des Poids Lourds

## Vue d'ensemble

Ce projet implémente deux chatbots spécialisés utilisant l'architecture RAG (Retrieval-Augmented Generation) pour répondre aux questions sur les véhicules poids lourds et leurs systèmes de lubrification :

1. **Chatbot Généraliste** : Répond aux questions générales sur les véhicules poids lourds, leurs systèmes mécaniques, électriques et hydrauliques.
2. **Chatbot Spécialiste** : Répond spécifiquement aux questions techniques et détaillées sur les systèmes de lubrification.

## Technologies utilisées

- **LangChain** : Framework pour le développement d'applications basées sur les LLM
- **Ollama Embeddings** : Service d'embeddings pour la vectorisation du texte
- **FAISS** : Bibliothèque pour le stockage et la recherche vectorielle efficace
- **Groq** : API de modèle de langage pour la génération de réponses
- **Gradio** : Bibliothèque pour la création d'interfaces utilisateur

## Structure du projet

```
chatbots/
├── chatbot_generaliste.py     # Implémentation du chatbot généraliste
├── chatbot_specialiste.py     # Implémentation du chatbot spécialiste
├── interface_utilisateur.py   # Interface utilisateur Gradio
├── requirements.txt           # Liste des dépendances
├── documents/                 # Répertoire pour les documents PDF
│   ├── general/               # Documents pour le chatbot généraliste
│   └── specialise/            # Documents pour le chatbot spécialiste
└── vector_store/              # Répertoire pour les bases vectorielles
    ├── general/               # Base vectorielle du chatbot généraliste
    └── specialise/            # Base vectorielle du chatbot spécialiste
```

## Installation

1. Clonez ce dépôt ou téléchargez les fichiers
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
   ou directement :
   ```bash
   pip install langchain langchain-groq langchain-community pypdf langchain-chroma langchain-ollama faiss-cpu gradio streamlit
   ```
3. Créez les répertoires nécessaires :
   ```bash
   mkdir -p documents/general documents/specialise vector_store/general vector_store/specialise
   ```
4. Ajoutez vos documents PDF dans les répertoires correspondants :
   - Documents généraux sur les véhicules poids lourds dans `documents/general/`
   - Documents spécialisés sur les systèmes de lubrification dans `documents/specialise/`

## Configuration

1. Obtenez une clé API Groq en vous inscrivant sur [GroqCloud](https://console.groq.com/)
2. Définissez votre clé API comme variable d'environnement :
   ```bash
   export GROQ_API_KEY="votre_cle_api_groq"
   ```
   ou modifiez directement la variable `API_KEY` dans les fichiers Python

## Utilisation

### Lancement de l'interface utilisateur

```bash
python interface_utilisateur.py
```

L'interface sera accessible dans votre navigateur à l'adresse indiquée (généralement http://127.0.0.1:7860).

### Utilisation des chatbots individuellement

Pour utiliser le chatbot généraliste :

```python
from chatbot_generaliste import ChatbotGeneraliste

# Initialisation
chatbot = ChatbotGeneraliste(
    documents_dir="./documents/general",
    api_key="votre_cle_api_groq",
    vector_store_dir="./vector_store/general"
)

# Poser une question
reponse = chatbot.ask("Comment fonctionne le système de freinage d'un camion ?")
print(reponse)
```

Pour utiliser le chatbot spécialiste :

```python
from chatbot_specialiste import ChatbotSpecialiste

# Initialisation
chatbot = ChatbotSpecialiste(
    documents_dir="./documents/specialise",
    api_key="votre_cle_api_groq",
    vector_store_dir="./vector_store/specialise"
)

# Poser une question technique
reponse = chatbot.ask_technical("Quelles sont les spécifications de viscosité recommandées pour l'huile de transmission ?")
print(reponse)
```

## Description des composants

### Chatbot Généraliste

Le chatbot généraliste est implémenté dans `chatbot_generaliste.py` et comprend les classes suivantes :

- **DocumentLoader** : Charge les documents PDF généraux
- **TextProcessor** : Découpe les documents en segments de texte
- **VectorStore** : Gère les embeddings et la base vectorielle
- **LLMInterface** : Communique avec l'API Groq
- **RAGPipeline** : Orchestre le processus RAG complet
- **ChatbotGeneraliste** : Classe principale qui intègre tous les composants

### Chatbot Spécialiste

Le chatbot spécialiste est implémenté dans `chatbot_specialiste.py` et comprend les classes suivantes :

- **SpecializedDocumentLoader** : Charge les documents PDF spécialisés en lubrification
- **TechnicalTextProcessor** : Découpe les documents techniques en segments
- **LubricationVectorStore** : Gère les embeddings et la base vectorielle spécialisée
- **TechnicalLLMInterface** : Communique avec l'API Groq pour des questions techniques
- **SpecializedRAGPipeline** : Orchestre le processus RAG spécialisé
- **ChatbotSpecialiste** : Classe principale qui intègre tous les composants spécialisés

### Interface Utilisateur

L'interface utilisateur est implémentée dans `interface_utilisateur.py` et offre les fonctionnalités suivantes :

- Onglets pour choisir entre les deux chatbots
- Zones de texte pour poser des questions
- Affichage des conversations
- Boutons pour effacer l'historique
- Fonctionnalité de téléchargement de documents PDF
- Gestion des erreurs

## Personnalisation

### Modification des paramètres de découpage du texte

Pour modifier la taille des segments et le chevauchement, modifiez les paramètres dans les constructeurs des classes `TextProcessor` et `TechnicalTextProcessor` :

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Modifier cette valeur
    chunk_overlap=200,  # Modifier cette valeur
    length_function=len,
    is_separator_regex=False,
)
```

### Modification du modèle d'embedding

Pour utiliser un modèle d'embedding différent, modifiez le paramètre dans les constructeurs des classes `VectorStore` et `LubricationVectorStore` :

```python
self.embeddings = OllamaEmbeddings(model="deepseek-r1")  # Remplacer par un autre modèle
```

### Modification du modèle LLM

Pour utiliser un modèle LLM différent, modifiez le paramètre dans les constructeurs des classes `LLMInterface` et `TechnicalLLMInterface` :

```python
self.llm = ChatGroq(
    api_key=api_key,
    model_name="llama3-8b-8192",  # Remplacer par un autre modèle
    temperature=0.2,  # Ajuster la température
)
```

## Dépannage

### Problèmes courants

1. **Erreur d'API Groq** : Vérifiez que votre clé API est valide et correctement définie.
2. **Aucun document trouvé** : Assurez-vous d'avoir ajouté des documents PDF dans les répertoires appropriés.
3. **Erreur de mémoire** : Si vous rencontrez des erreurs de mémoire, essayez de réduire la taille des chunks ou le nombre de documents.
4. **Réponses imprécises** : Ajoutez plus de documents pertinents pour améliorer la qualité des réponses.

### Logs

Les logs sont configurés pour être affichés dans la console. Pour les sauvegarder dans un fichier, ajoutez un handler de fichier à la configuration de logging :

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbots.log"),
        logging.StreamHandler()
    ]
)
```

## Améliorations futures

1. Ajout d'une fonctionnalité de recherche d'images et de schémas techniques
2. Intégration d'une base de données pour stocker l'historique des conversations
3. Ajout de la prise en charge de formats de documents supplémentaires (DOCX, TXT, etc.)
4. Implémentation d'un système de feedback utilisateur pour améliorer les réponses
5. Optimisation des performances de recherche vectorielle

## Licence

Ce projet est fourni à titre éducatif dans le cadre d'un projet de fin d'études.

## Contact

Pour toute question ou suggestion, veuillez contacter [votre nom et email].
