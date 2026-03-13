This repository contains code, intermediate outputs, and visualizations for a Cultural Analytics project on discourse change in National Socialist entertainment films between 1936 and 1945.

The project uses automatically generated dialogue transcripts, linguistic preprocessing, and keyness analysis to compare three historical phases of NS film discourse. The aim is to identify which lexical fields become statistically characteristic in each phase and how these changes relate to broader political and cultural shifts. The work is situated in the context of Cultural Analytics and the quantitative analysis of film and video data. 

## Project focus

The analysis examines dialogue transcripts from NS entertainment films and compares them across three phases:

- Phase 1: 1936–1939
- Phase 2: 1940–1942
- Phase 3: 1943–1945

The final interpretation developed from the project log describes a movement from militarization and everyday pre-war vocabulary in Phase 1, to state, monarchy, and authority vocabulary in Phase 2, and finally to private, artistic, and escapist vocabulary in Phase 3. [file:1]

## Method

The workflow consists of the following steps:

1. Audio extraction and transcription with `faster-whisper`.
2. Text cleaning and timestamp removal.
3. Linguistic preprocessing with `spaCy` (lemmatization, POS filtering, stopword removal).
4. Phase assignment of films via filename mapping.
5. Quantitative comparison of phases using:
   - TF-IDF
   - Log-likelihood keyness
6. Validation and interpretation through:
   - bar plots
   - word clouds
   - trend plots
   - source CSVs showing which films contribute to specific keywords

The project log documents key methodological decisions, including the use of `faster-whisper large-v3-turbo`, German spaCy preprocessing, lowercasing after lemmatization, frequency thresholds, and a minimum-film filter to reduce single-film dominance. [file:1]

## Main outputs
Typical outputs of the pipeline include:
    
    keynessphase1.csv, keynessphase2.csv, keynessphase3.csv
    
    tfidfphase1.csv, tfidfphase2.csv, tfidfphase3.csv
    
    quellenphase1.csv, quellenphase2.csv, quellenphase3.csv

    keynessbarplot.png, wordclouds.png, heatmap.png , keywordstrength.png, wordtrends.png

These outputs were explicitly developed during the project workflow and used for validation and interpretation.

## Installation
Create an environment and install the required packages:

    bash
    pip install -r requirements.txt
    python -m spacy download de_core_news_sm
  
  A minimal setup mentioned in the project log includes:

    bash
    pip install faster-whisper spacy pandas scikit-learn matplotlib wordcloud
 This setup was used for transcription and keyness analysis in the project workflow. [file:1]

## Usage
1. Transcription
   
      bash
      python scripts/transcribe.py
   
      The transcription workflow was based on batch processing with faster-whisper, skipping files that had already been processed.

3. Analysis
   
      bash
      python scripts/analyze_keyness.py
   
The analysis script:

reads transcripts, assigns films to phases, preprocesses the text, computes TF-IDF and log-likelihood keyness, exports CSV files, and creates visualizations. 

## Data note
This repository may not contain the full raw film or audio data for copyright and storage reasons. It is designed to provide the scripts, mappings, processed outputs, and documentation necessary to understand and reproduce the analytical workflow as far as possible.


### MIT License
