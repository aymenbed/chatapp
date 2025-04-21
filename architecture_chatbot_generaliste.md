# Architecture du Chatbot Généraliste pour Véhicules Poids Lourds

## Vue d'ensemble

Le chatbot généraliste est conçu pour répondre aux questions générales sur les véhicules poids lourds, leurs systèmes mécaniques, électriques et hydrauliques. Il utilise une architecture RAG (Retrieval-Augmented Generation) pour fournir des réponses précises et contextuelles basées sur une variété de documents techniques.

## Composants principaux

### 1. Gestion des documents
- **Source de données**: Documents PDF techniques généraux sur les véhicules poids lourds
- **Chargement**: Utilisation de `PyPDFLoader` de LangChain pour charger les documents
- **Prétraitement**: Extraction du texte et des métadonnées pertinentes

### 2. Traitement du texte
- **Découpage**: Utilisation de `RecursiveCharacterTextSplitter` avec:
  - `chunk_size=1000`: Taille des segments de texte
  - `chunk_overlap=200`: Chevauchement entre segments pour maintenir le contexte
- **Normalisation**: Nettoyage du texte (suppression des caractères spéciaux, normalisation des espaces)

### 3. Vectorisation et stockage
- **Embeddings**: Utilisation d'`OllamaEmbeddings` avec le modèle `deepseek-r1`
- **Base de données vectorielle**: Implémentation avec `Chroma` ou `FAISS`
  - Indexation efficace pour la recherche rapide
  - Métadonnées associées pour le contexte (source, page, etc.)

### 4. Modèle de langage
- **LLM**: Utilisation de `ChatGroq` via l'API GroqCloud
- **Configuration**:
  - Température: 0.2 (pour des réponses plus déterministes)
  - Langue de sortie: Français uniquement
  - Contexte système: Instructions spécifiques pour répondre aux questions techniques

### 5. Pipeline RAG
- **Requête utilisateur**: Prétraitement et vectorisation
- **Recherche**: Récupération des chunks les plus pertinents (top-k=5)
- **Contexte**: Construction du prompt avec les informations récupérées
- **Génération**: Production de la réponse par le LLM
- **Post-traitement**: Formatage et vérification de la réponse

### 6. Gestion de la mémoire
- **Historique des conversations**: Stockage des échanges précédents
- **Contexte persistant**: Utilisation de `ConversationBufferMemory` de LangChain
- **Fenêtre glissante**: Limitation à 5 échanges pour éviter la surcharge du contexte

## Diagramme de flux

```
[Documents PDF] → [PyPDFLoader] → [Texte brut]
                                      ↓
[Texte brut] → [RecursiveCharacterTextSplitter] → [Chunks de texte]
                                                      ↓
[Chunks de texte] → [OllamaEmbeddings] → [Vecteurs]
                                             ↓
[Vecteurs] → [Chroma/FAISS] → [Base de données vectorielle]
                                  ↓
[Question utilisateur] → [Prétraitement] → [Vectorisation]
                                               ↓
[Vectorisation] → [Recherche similitude] → [Chunks pertinents]
                                               ↓
[Chunks pertinents] + [Historique] → [Construction du prompt]
                                           ↓
[Prompt] → [ChatGroq] → [Réponse brute]
                             ↓
[Réponse brute] → [Post-traitement] → [Réponse finale]
```

## Considérations techniques

### Performance
- Optimisation de la taille des chunks pour équilibrer précision et contexte
- Configuration de l'index vectoriel pour des recherches rapides
- Mise en cache des embeddings pour éviter les recalculs

### Robustesse
- Gestion des exceptions pour les requêtes mal formées
- Fallback en cas d'échec de récupération de contexte pertinent
- Mécanismes de retry pour les appels API

### Extensibilité
- Architecture modulaire permettant l'ajout de nouvelles sources de documents
- Possibilité de changer de modèle d'embedding ou de LLM
- Interface standardisée pour l'intégration avec d'autres systèmes

## Implémentation

L'implémentation suivra une approche modulaire avec des classes distinctes pour:
- `DocumentLoader`: Gestion du chargement et prétraitement des documents
- `TextProcessor`: Découpage et normalisation du texte
- `VectorStore`: Gestion des embeddings et de la base vectorielle
- `LLMInterface`: Communication avec l'API Groq
- `RAGPipeline`: Orchestration du processus complet
- `ConversationManager`: Gestion de l'historique et du contexte

Cette architecture permettra de répondre efficacement aux questions générales sur les véhicules poids lourds tout en maintenant une base technique solide et extensible.
