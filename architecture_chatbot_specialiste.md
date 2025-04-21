# Architecture du Chatbot Spécialiste en Lubrification pour Véhicules Poids Lourds

## Vue d'ensemble

Le chatbot spécialiste est conçu pour répondre spécifiquement aux questions techniques et détaillées sur les systèmes de lubrification des véhicules poids lourds. Contrairement au chatbot généraliste, il se concentre exclusivement sur les connaissances liées à la lubrification, offrant ainsi une expertise plus approfondie dans ce domaine spécifique.

## Composants principaux

### 1. Gestion des documents
- **Source de données**: Documents PDF spécialisés uniquement sur les systèmes de lubrification
- **Chargement**: Utilisation de `PyPDFLoader` de LangChain pour charger les documents
- **Prétraitement**: Extraction du texte avec attention particulière aux tableaux et données techniques spécifiques à la lubrification

### 2. Traitement du texte
- **Découpage**: Utilisation de `RecursiveCharacterTextSplitter` avec:
  - `chunk_size=1000`: Taille des segments de texte
  - `chunk_overlap=200`: Chevauchement entre segments pour maintenir le contexte
- **Normalisation**: Préservation des termes techniques spécifiques à la lubrification et des valeurs numériques importantes

### 3. Vectorisation et stockage
- **Embeddings**: Utilisation d'`OllamaEmbeddings` avec le modèle `deepseek-r1`
- **Base de données vectorielle**: Implémentation avec `Chroma` ou `FAISS`
  - Optimisation pour la recherche de termes techniques spécifiques
  - Métadonnées enrichies incluant des informations sur les types de lubrifiants, composants, et normes

### 4. Modèle de langage
- **LLM**: Utilisation de `ChatGroq` via l'API GroqCloud
- **Configuration**:
  - Température: 0.1 (pour des réponses plus précises et techniques)
  - Langue de sortie: Français uniquement
  - Contexte système: Instructions spécifiques pour répondre avec précision aux questions techniques sur la lubrification

### 5. Pipeline RAG
- **Requête utilisateur**: Prétraitement avec reconnaissance des termes techniques de lubrification
- **Recherche**: Récupération des chunks les plus pertinents (top-k=7) pour assurer une couverture complète des informations techniques
- **Contexte**: Construction du prompt avec les informations récupérées et ajout de directives spécifiques pour les réponses techniques
- **Génération**: Production de la réponse par le LLM avec attention particulière à la précision technique
- **Post-traitement**: Vérification de la cohérence technique et des unités de mesure

### 6. Gestion de la mémoire
- **Historique des conversations**: Stockage des échanges précédents avec focus sur les détails techniques
- **Contexte persistant**: Utilisation de `ConversationBufferMemory` de LangChain
- **Fenêtre glissante**: Limitation à 5 échanges pour éviter la surcharge du contexte

## Diagramme de flux

```
[Documents PDF spécialisés] → [PyPDFLoader] → [Texte brut]
                                                 ↓
[Texte brut] → [RecursiveCharacterTextSplitter] → [Chunks de texte]
                                                      ↓
[Chunks de texte] → [OllamaEmbeddings] → [Vecteurs]
                                             ↓
[Vecteurs] → [Chroma/FAISS] → [Base de données vectorielle spécialisée]
                                  ↓
[Question technique] → [Prétraitement spécialisé] → [Vectorisation]
                                                        ↓
[Vectorisation] → [Recherche similitude] → [Chunks techniques pertinents]
                                                ↓
[Chunks pertinents] + [Historique] → [Construction du prompt technique]
                                            ↓
[Prompt] → [ChatGroq] → [Réponse technique brute]
                              ↓
[Réponse brute] → [Vérification technique] → [Réponse technique finale]
```

## Spécificités du chatbot spécialiste

### Base de connaissances spécialisée
- Concentration sur les documents traitant exclusivement des systèmes de lubrification
- Inclusion des normes ISO et SAE spécifiques aux lubrifiants (ISO 4406, ISO 3448, SAE J300, etc.)
- Intégration des informations sur les composants typiques des systèmes de lubrification (pompes, filtres, refroidisseurs, etc.)

### Traitement des requêtes techniques
- Reconnaissance des termes techniques spécifiques à la lubrification (viscosité, contamination, additifs, etc.)
- Capacité à interpréter les questions sur les intervalles de maintenance, les spécifications des lubrifiants, et les symptômes de défaillance
- Détection des références aux normes et standards de l'industrie

### Réponses spécialisées
- Formulation de réponses techniques précises avec les valeurs et unités appropriées
- Inclusion des références aux normes pertinentes lorsque applicable
- Capacité à expliquer les concepts techniques de manière claire tout en restant précis

## Considérations techniques

### Performance
- Optimisation de la recherche vectorielle pour les termes techniques spécifiques
- Indexation spéciale pour les références aux normes et standards
- Mise en cache des requêtes fréquentes sur les spécifications des lubrifiants

### Robustesse
- Gestion des questions ambiguës avec demande de précisions
- Mécanisme de fallback vers des informations générales en cas d'absence d'information spécifique
- Vérification de la cohérence technique des réponses générées

### Extensibilité
- Possibilité d'ajouter de nouvelles sources de données techniques
- Structure permettant la mise à jour des informations sur les normes et standards
- Interface pour l'intégration avec des bases de données de spécifications de lubrifiants

## Implémentation

L'implémentation suivra une approche modulaire similaire au chatbot généraliste, mais avec des classes spécialisées:
- `SpecializedDocumentLoader`: Gestion du chargement des documents techniques sur la lubrification
- `TechnicalTextProcessor`: Traitement spécifique du texte technique
- `LubricationVectorStore`: Base vectorielle optimisée pour les informations de lubrification
- `TechnicalLLMInterface`: Communication avec l'API Groq avec paramètres adaptés
- `SpecializedRAGPipeline`: Pipeline RAG adapté aux questions techniques
- `TechnicalConversationManager`: Gestion de l'historique avec focus sur les détails techniques

Cette architecture permettra au chatbot spécialiste de fournir des réponses précises, techniques et pertinentes sur les systèmes de lubrification des véhicules poids lourds, en s'appuyant sur une base de connaissances spécialisée et des mécanismes optimisés pour le traitement des informations techniques.
