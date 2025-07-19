#!/bin/bash

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK data
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True) 
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
"

# Download spaCy model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p data/raw data/processed data/external
mkdir -p models/sentiment models/topics
mkdir -p reports logs visualizations/topics

# Run the data pipeline to generate sample data
echo "Generating sample data and training models..."
python run_pipeline.py --skip-data-generation=false

echo "Build completed successfully!"