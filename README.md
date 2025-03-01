<div align="center">

# NLP EXPLORATION

![Python Version=3.10](https://img.shields.io/badge/Python-v3.10-darkgreen)
![Virtual Environment Name=conda:comp_ling_env](https://img.shields.io/badge/VirtualEnv-Conda:_comp__ling__env-lightblue)
![Docustring format=Sphinx Style](https://img.shields.io/badge/DocuString-Sphinx_Style-violet)

This repository is my personal dedication to exploring various computational linguistic solutions.
</div>

## Table of Contents:

<ol>
<li> Setup.
<li> <strong>Necessary Considerations</strong> (If forked, it is <strong>vital</strong> you review the section especially if you plan to submit a pull request).
<li> Resources.
<li> Focus.
</ol>

## Setup

This repo is managed using a Conda environment - enter the following code in your terminal to initialize your environment with the appropriate packages:

<pre>
conda env create -f environment.yml
</pre>

You should get confirmation that the environment titled <strong>"comp_ling_env"</strong> has been created. Proceed with activating it accordingly:

<pre>
conda activate comp_ling_env
</pre>

</br>
The above operations are performed via the <strong>environment.yml file</strong> therefore any changes (i.e. addition/subtraction of packages to/from the environment) should be recorded in this file.

<br>
</br>
You can use the command below for updating your environment's packages accordingly:

</br>
<pre>
conda env update -f environment.yml
</pre>

Please ensure you <strong>confirm the host for said package</strong> prior to modifying the YAML file (<strong>Conda packages have highest priority</strong>, therefore if a package is found e.g. on both PyPI and Conda, place it within the Conda section).

Certain packages have been left unspecified intentionally e.g.<code>kagglehub</code> - kindly take this into consideration when modifying the YAML file.

## Necessary Considerations

You should go through this section when <strong>initializing this repo</strong>, each time you <strong>start making modifications</strong>, and prior to <strong>committing an update</strong>.

<ol>
<li> Review the badges at the top of this README especially before initialization.
<li> Ensure you follow the <strong>package preservation guidelines</strong> outlined in the <a href='#setup'>Setup</a> section.
</ol>

## Resources

The following documents can be found within the <code>\resources</code> directory:

<ol>
<li> <a href="resources/Speech and Language Processing (12-Jan-2025) - Daniel Jurafsky & James H. Martin.pdf"><strong>Speech and Language Processing (12-Jan-2025)</strong></a> by Daniel Jurafsky & James H. Martin.
<li> <a href="resources/Foundations of Large Language Models - (16-Jan-2025).pdf"><strong>Foundations of Large Language Models (16-Jan-2025)</strong></a> by Tong Xiao & Jingbo Zhu.
<li> <a href="resources/EmoBench-M_Benchmarking_Emotional_Intelligence_for_Multimodal_Large_Language_Models - (6-Feb-2025).pdf"><strong>EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models (6-Feb-2025)</strong></a> by He Hu et al.
<li> <a href="resources/A_Statistical_Framework_for_Ranking_LLM-Based_Chatbots - (24-Dec-2024).pdf"><strong>A Statistical Framework for Ranking LLM-Based Chatbots (24-Dec-2024)</strong></a> by Siavash Ameli et al.
<li> <a href="resources/Mathematics_for_inference_ML - (13-Jan-2017).pdf"><strong>Imperial College: 'Mathematics for Inference and Machine Learning' (13-Jan-2017)</strong></a> instructed by Marc Deisenroth & Stefanos Zafeiriou.
</ol>

Consider reading through the above-listed resources in the order it's been displayed.

## Focus

This section provides a brief overview outlining major tasks and methodologies used to process and analyze human language, which I aim to explore long-term in this repository.

<ol>
<li> Text Processing & Tokenization
<li> Syntax & Parsing Solutions
<li> Semantic Analysis
<li> Machine Translation (MT)
<li> Speech Processing
<li> Information Retrieval & Search
<li> Sentiment Analysis & Emotion Detection
<li> Text Generation & Summarization
<li> Question Answering (QA)
<li> Conversational AI & Chatbots
<li> Fake News & Misinformation Detection
<li> Text Classification & Topic Modeling
<li> Ethical NLP & Bias Mitigation
<li> Multimodal NLP (Text + Image/Audio)
</ol>

## Breakdown by Tasks/Methodologies

<ol type="1">

### <li> Text Processing & Tokenization

#### - Summary:

<ol type="A">
<li> <strong>Sentence Segmentation</strong> – Splitting text into sentences.
<li> <strong>Word Tokenization</strong> – Dividing text into individual words or tokens.
<li> <strong>Part-of-Speech (POS) Tagging</strong> – Identifying word types (e.g., noun, verb).
<li> <strong>Lemmatization & Stemming</strong> – Reducing words to their base or root form.
<li> <strong>Named Entity Recognition (NER)</strong> – Identifying proper names (e.g., people, places, organizations).
<li> <strong>Dependency Parsing</strong> – Analyzing grammatical structures between words.
</ol>

#### - Tools:

<ol type="I">
<li> NLTK (Python)
<li> SpaCy
<li> Stanford NLP
<li> CoreNLP
<li> Stanza
</ol>

### <li> Syntax & Parsing Solutions

#### - Summary

<ol type="A">
<li> <strong>Constituency Parsing</strong> - Breaking sentences into tree structures
<li> <strong>Dependency Parsing</strong> – Understanding how words relate in a sentence.
<li> <strong>Morphological Analysis</strong> – Analyzing word formations.
</ol>

#### - Tools:

<ol type="I">
<li> Berkeley Parser
<li> SpaCy Dependency Parser
<li> MaltParser
<li> AllenNLP
</ol>

### <li> Semantic Analysis

#### - Summary

<ol type="A">
<li> <strong>Word Sense Disambiguation (WSD)</strong> – Understanding word meanings in context.
<li> <strong>Semantic Role Labeling (SRL)</strong> – Identifying actions and their participants.
<li> <strong>Named Entity Linking (NEL)</strong> – Connecting entities to a knowledge base.
<li> <strong>Coreference Resolution</strong> – Identifying when different words refer to the same entity.
<li> <strong>Frame Semantics</strong> – Understanding meanings through conceptual structures.
</ol>

#### - Tools:

<ol type="I">
<li> WordNet
<li> BabelNet
<li> FrameNet
<li> PropBank
<li> VerbNet
</ol>

### <li> Machine Translation (MT)

#### - Summary

<ol type="A">
<li> <strong>Rule-Based Machine Translation (RBMT)</strong> – Uses linguistic rules for translation.
<li> <strong>Statistical Machine Translation (SMT)</strong> – Uses probability models (e.g., IBM Model).
<li> <strong>Neural Machine Translation (NMT)</strong> – Uses deep learning models for translation.
</ol>

#### - Tools:

<ol type="I">
<li> Google Translate API
<li> DeepL
<li> MarianNMT
<li> OpenNMT
<li> Fairseq (Meta AI)
</ol>

### <li> Speech Processing

#### - Summary

<ol type="A">
<li> <strong>Speech-to-Text (ASR)</strong> – Converts spoken words into text.
<li> <strong>Text-to-Speech (TTS)</strong> – Converts text into spoken words.
<li> <strong>Phonetic Analysis</strong> – Analyzing pronunciation and speech patterns.
</ol>

#### - Tools:

<ol type="I">
<li> CMU Sphinx
<li> Kaldi
<li> Google Speech-to-Text API
<li> Mozilla DeepSpeech
<li> Festival Speech Synthesis System
</ol>

### <li> Information Retrieval & Search

#### - Summary

<ol type="A">
<li> <strong>Text Indexing</strong> – Creating structured representations for search.
<li> <strong>Query Expansion</strong> – Enhancing search queries with synonyms.
<li> <strong>Document Ranking</strong> – Ordering search results by relevance.
</ol>

#### - Tools:

<ol type="I">
<li> Elasticsearch
<li> Apache Solr
<li> Lucene
<li> BM25 Ranking Algorithm
</ol>

### <li> Sentiment Analysis & Emotion Detection

#### - Summary

<ol type="A">
<li> <strong>Lexicon-Based Sentiment Analysis</strong> – Uses predefined word lists.
<li> <strong>Machine Learning Sentiment Analysis</strong> – Trained on labeled data.
<li> <strong>Emotion Classification</strong> – Detecting emotions (happy, sad, angry).
</ol>

#### - Tools:

<ol type="I">
<li> VADER (for social media)
<li> TextBlob
<li> BERT-based sentiment models
<li> LIWC (Linguistic Inquiry and Word Count)
</ol>

### <li> Text Generation & Summarization

#### - Summary

<ol type="A">
<li> <strong>Extractive Summarization</strong> – Identifies key sentences from text.
<li> <strong>Abstractive Summarization</strong> – Generates new, concise text.
<li> <strong>Text Completion</strong> – Predicts missing or next words.
</ol>

#### - Tools:

<ol type="I">
<li> GPT-based models (GPT-3, GPT-4, ChatGPT)
<li> BART (Facebook AI)
<li> T5 (Google AI)
<li> LexRank (Graph-based summarization)
</ol>

### <li> Question Answering (QA)

#### - Summary

<ol type="A">
<li> <strong>Fact-Based QA</strong> – Retrieves answers from knowledge bases.
<li> <strong>Open-Domain QA</strong> – Answers questions using large-scale text corpora.
<li> <strong>Contextual QA</strong> – Uses deep learning models to generate responses.
</ol>

#### - Tools:

<ol type="I">
<li> Haystack (QA Framework)
<li> Google BERT for QA
<li> DrQA (Facebook Research)
<li> RAG (Retrieval-Augmented Generation)
</ol>

### <li> Conversational AI & Chatbots

#### - Summary

<ol type="A">
<li> <strong>Rule-Based Chatbots</strong> – Uses decision trees and predefined rules.
<li> <strong>AI-Driven Chatbots</strong> – Uses deep learning and NLP models.
<li> <strong>Dialogue Management</strong> – Handles multi-turn conversations.
</ol>

#### - Tools:

<ol type="I">
<li> Rasa
<li> Dialogflow (Google)
<li> Microsoft Bot Framework
<li> Amazon Lex
<li> LlamaIndex (LangChain for retrieval-based chatbots)
</ol>

### <li> Fake News & Misinformation Detection

#### - Summary

<ol type="A">
<li> <strong>Stance Detection</strong> – Determines if a claim agrees or disagrees with a statement.
<li> <strong>Fact-Checking NLP Models</strong> – Verifies facts using knowledge bases.
<li> <strong>Propaganda Detection</strong> – Identifies biased or misleading content.
</ol>

#### - Tools:

<ol type="I">
<li> FactCheck.org API
<li> Hugging Face Fake News Classifiers
<li> ClaimBuster (Fact-Checking AI)
</ol>

### <li> Text Classification & Topic Modeling

#### - Summary

<ol type="A">
<li> <strong>Supervised Classification</strong> – Assigns categories using labeled training data.
<li> <strong>Unsupervised Topic Modeling</strong> – Finds hidden topics in text.
</ol>

#### - Tools:

<ol type="I">
<li> Latent Dirichlet Allocation (LDA)
<li> Non-Negative Matrix Factorization (NMF)
<li> BERT-based Classifiers
<li> Scikit-learn Naive Bayes for text classification
</ol>

### <li> Ethical NLP & Bias Mitigation

#### - Summary

<ol type="A">
<li> <strong>Bias Detection in Text</strong> – Identifies gender, racial, or political bias.
<li> <strong>Hate Speech Detection</strong> – Recognizes offensive or harmful language.
<li> <strong>Fairness-Aware NLP</strong> – Ensures models do not reinforce discrimination.
</ol>

#### - Tools:

<ol type="I">
<li> Fairness Indicators (Google AI)
<li> Hugging Face Bias Detection Models
<li> Perspective API (Google Jigsaw)
</ol>

### <li> Multimodal NLP (Text + Image/Audio)

#### - Summary

<ol type="A">
<li> <strong>Image Captioning</strong> – Generates text descriptions for images.
<li> <strong>Visual Question Answering (VQA)</strong> – Answers questions about images.
<li> <strong>Speech & Text Fusion Models</strong> – Analyzes speech with textual context.
</ol>

#### - Tools:

<ol type="I">
<li> CLIP (OpenAI)
<li> BLIP (Bootstrapped Language-Image Pretraining)
<li> Whisper (OpenAI for Speech Recognition)
<li> LAVIS (Multimodal AI Library)
</ol>

</ol>