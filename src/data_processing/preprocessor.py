"""
Data Preprocessing Module for Customer Feedback Analytics
Handles text cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
import yaml
import logging
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
import pickle

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Advanced text preprocessing for customer reviews"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['preprocessing']
        
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep some important negation words
        self.stop_words -= {'not', 'no', 'nor', 'neither', 'never', 'none'}
        
        # Load spaCy model for advanced processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s\'\-\.\,\!\?]')
        
        # Contractions dictionary
        self.contractions = {
            "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
            "shouldn't": "should not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
            "don't": "do not", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "can't": "cannot",
            "couldn't": "could not", "won't": "will not", "i'm": "i am",
            "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would",
            "they'd": "they would", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "let's": "let us", "that's": "that is",
            "who's": "who is", "what's": "what is", "here's": "here is",
            "there's": "there is", "where's": "where is", "when's": "when is",
            "why's": "why is", "how's": "how is"
        }
        
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""
        text = text.lower()
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Expand contractions first
        text = self.expand_contractions(text)
        
        # Remove URLs, emails, phone numbers
        if self.preprocessing_config['remove_urls']:
            text = self.url_pattern.sub(' ', text)
        if self.preprocessing_config['remove_emails']:
            text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Handle special characters
        if self.preprocessing_config['remove_special_chars']:
            text = self.special_char_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Lowercase
        if self.preprocessing_config['lowercase']:
            text = text.lower()
        
        return text.strip()
    
    def handle_negations(self, text: str) -> str:
        """Handle negations by appending _NOT to following words"""
        if not self.preprocessing_config['handle_negations']:
            return text
        
        negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing', 'nowhere'}
        punctuation = {'.', ',', '!', '?', ';', ':'}
        
        words = text.split()
        result = []
        negate = False
        
        for word in words:
            if word in negation_words:
                negate = True
                result.append(word)
            elif any(p in word for p in punctuation):
                negate = False
                result.append(word)
            elif negate:
                result.append(f"{word}_NOT")
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords while preserving important terms"""
        if not self.preprocessing_config['remove_stopwords']:
            return text
        
        words = word_tokenize(text)
        return ' '.join([w for w in words if w.lower() not in self.stop_words or '_NOT' in w])
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using WordNet lemmatizer"""
        if not self.preprocessing_config['lemmatize']:
            return text
        
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(w) for w in words])
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from reviews"""
        logger.info("Extracting features...")
        
        # Text statistics
        df['char_count'] = df['review_text'].str.len()
        df['word_count'] = df['review_text'].str.split().str.len()
        df['avg_word_length'] = df['review_text'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
        )
        
        # Punctuation features
        df['exclamation_count'] = df['review_text'].str.count('!')
        df['question_count'] = df['review_text'].str.count('\?')
        df['caps_ratio'] = df['review_text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        # Sentiment indicators
        df['positive_word_count'] = df['cleaned_text'].apply(self.count_positive_words)
        df['negative_word_count'] = df['cleaned_text'].apply(self.count_negative_words)
        
        # TextBlob features
        df['textblob_polarity'] = df['review_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        df['textblob_subjectivity'] = df['review_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )
        
        # Time-based features
        df['review_hour'] = pd.to_datetime(df['review_date']).dt.hour
        df['review_dayofweek'] = pd.to_datetime(df['review_date']).dt.dayofweek
        df['review_month'] = pd.to_datetime(df['review_date']).dt.month
        df['is_weekend'] = df['review_dayofweek'].isin([5, 6]).astype(int)
        
        return df
    
    def count_positive_words(self, text: str) -> int:
        """Count positive sentiment words"""
        positive_words = {
            'excellent', 'perfect', 'amazing', 'wonderful', 'fantastic', 'great',
            'love', 'best', 'awesome', 'incredible', 'outstanding', 'superb',
            'brilliant', 'beautiful', 'gorgeous', 'fabulous', 'magnificent'
        }
        return sum(1 for word in str(text).lower().split() if word in positive_words)
    
    def count_negative_words(self, text: str) -> int:
        """Count negative sentiment words"""
        negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'poor', 'worst', 'hate',
            'disgusting', 'disappointed', 'useless', 'waste', 'broken', 'defective',
            'unacceptable', 'pathetic', 'ridiculous', 'garbage', 'trash'
        }
        return sum(1 for word in str(text).lower().split() if word in negative_words)
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the entire dataset"""
        logger.info("Starting preprocessing pipeline...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Basic cleaning
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['review_text'].apply(self.clean_text)
        
        # Handle negations
        logger.info("Handling negations...")
        df['cleaned_text'] = df['cleaned_text'].apply(self.handle_negations)
        
        # Remove stopwords
        logger.info("Removing stopwords...")
        df['cleaned_text'] = df['cleaned_text'].apply(self.remove_stopwords)
        
        # Lemmatization
        logger.info("Lemmatizing text...")
        df['cleaned_text'] = df['cleaned_text'].apply(self.lemmatize_text)
        
        # Extract features
        df = self.extract_features(df)
        
        # Filter by length constraints
        min_length = self.preprocessing_config['min_review_length']
        max_length = self.preprocessing_config['max_review_length']
        
        original_count = len(df)
        df = df[(df['word_count'] >= min_length) & (df['word_count'] <= max_length)]
        filtered_count = original_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} reviews based on length constraints")
        
        # Save preprocessed data
        output_path = self.config['data']['processed_data_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")
        
        # Generate preprocessing report
        self.generate_preprocessing_report(df)
        
        return df
    
    def generate_preprocessing_report(self, df: pd.DataFrame):
        """Generate preprocessing statistics report"""
        print("\n=== Preprocessing Report ===")
        print(f"Total reviews after preprocessing: {len(df)}")
        print(f"\nText statistics:")
        print(f"  Average word count: {df['word_count'].mean():.1f}")
        print(f"  Average character count: {df['char_count'].mean():.1f}")
        print(f"  Average word length: {df['avg_word_length'].mean():.1f}")
        print(f"\nSentiment indicators:")
        print(f"  Reviews with positive words: {(df['positive_word_count'] > 0).sum()}")
        print(f"  Reviews with negative words: {(df['negative_word_count'] > 0).sum()}")
        print(f"  Average TextBlob polarity: {df['textblob_polarity'].mean():.3f}")
        print(f"\nFeature statistics:")
        print(f"  Reviews with exclamations: {(df['exclamation_count'] > 0).sum()}")
        print(f"  Reviews with questions: {(df['question_count'] > 0).sum()}")
        print(f"  Weekend reviews: {df['is_weekend'].sum()} ({df['is_weekend'].mean()*100:.1f}%)")


def main():
    """Main preprocessing pipeline"""
    # Load raw data
    try:
        df = pd.read_csv("data/raw/customer_reviews.csv")
        logger.info(f"Loaded {len(df)} reviews")
    except FileNotFoundError:
        logger.error("Raw data not found. Please run data_generator.py first.")
        return
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Run preprocessing
    preprocessed_df = preprocessor.preprocess_dataset(df)
    
    # Show sample preprocessed reviews
    print("\n=== Sample Preprocessed Reviews ===")
    for _, row in preprocessed_df.sample(3).iterrows():
        print(f"\nOriginal: {row['review_text'][:100]}...")
        print(f"Cleaned: {row['cleaned_text'][:100]}...")
        print(f"Features: Words={row['word_count']}, Polarity={row['textblob_polarity']:.3f}")


if __name__ == "__main__":
    main()