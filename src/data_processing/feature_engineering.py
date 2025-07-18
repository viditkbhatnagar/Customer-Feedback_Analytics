"""
Feature Engineering Module for Customer Feedback Analytics
Advanced feature extraction for improved model performance
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import spacy
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
import yaml
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for customer reviews"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Some features will be disabled.")
            self.nlp = None
        
        # Initialize VADER sentiment analyzer
        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon', quiet=True)
            self.vader = SentimentIntensityAnalyzer()
        
        # Compile regex patterns
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE
        )
        
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        
    def extract_text_statistics(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics"""
        if pd.isna(text) or text == "":
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'unique_word_ratio': 0,
                'capital_ratio': 0,
                'digit_ratio': 0,
                'whitespace_ratio': 0
            }
        
        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentence_count = max(1, sentences)
        
        # Averages
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Ratios
        unique_words = set(words)
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0
        
        capital_count = sum(1 for c in text if c.isupper())
        capital_ratio = capital_count / char_count if char_count > 0 else 0
        
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / char_count if char_count > 0 else 0
        
        whitespace_count = sum(1 for c in text if c.isspace())
        whitespace_ratio = whitespace_count / char_count if char_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_word_ratio': unique_word_ratio,
            'capital_ratio': capital_ratio,
            'digit_ratio': digit_ratio,
            'whitespace_ratio': whitespace_ratio
        }
    
    def extract_punctuation_features(self, text: str) -> Dict[str, int]:
        """Extract punctuation-based features"""
        if pd.isna(text):
            return {
                'exclamation_count': 0,
                'question_count': 0,
                'ellipsis_count': 0,
                'comma_count': 0,
                'period_count': 0,
                'total_punctuation': 0,
                'punctuation_diversity': 0
            }
        
        features = {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'ellipsis_count': text.count('...'),
            'comma_count': text.count(','),
            'period_count': text.count('.'),
            'total_punctuation': sum(1 for c in text if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        }
        
        # Punctuation diversity
        punctuation_types = set(c for c in text if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        features['punctuation_diversity'] = len(punctuation_types)
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment-based features using multiple methods"""
        if pd.isna(text) or text == "":
            return {
                'textblob_polarity': 0,
                'textblob_subjectivity': 0,
                'vader_positive': 0,
                'vader_negative': 0,
                'vader_neutral': 0,
                'vader_compound': 0,
                'positive_word_count': 0,
                'negative_word_count': 0,
                'sentiment_word_ratio': 0
            }
        
        # TextBlob sentiment
        blob = TextBlob(str(text))
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # Sentiment word counts
        positive_words = {
            'excellent', 'perfect', 'love', 'wonderful', 'amazing', 'good', 'great',
            'fantastic', 'happy', 'awesome', 'best', 'beautiful', 'nice', 'super'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor',
            'disappointing', 'useless', 'broken', 'failed', 'sucks', 'disgusting'
        }
        
        words_lower = text.lower().split()
        positive_count = sum(1 for word in words_lower if word in positive_words)
        negative_count = sum(1 for word in words_lower if word in negative_words)
        
        total_words = len(words_lower)
        sentiment_word_ratio = (positive_count + negative_count) / total_words if total_words > 0 else 0
        
        return {
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'vader_compound': vader_scores['compound'],
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'sentiment_word_ratio': sentiment_word_ratio
        }
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability metrics"""
        if pd.isna(text) or text == "" or len(text.split()) < 5:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'automated_readability_index': 0,
                'coleman_liau_index': 0,
                'linsear_write_formula': 0,
                'dale_chall_readability': 0,
                'text_standard': 0
            }
        
        try:
            features = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'linsear_write_formula': textstat.linsear_write_formula(text),
                'dale_chall_readability': textstat.dale_chall_readability_score(text),
                'text_standard': textstat.text_standard(text, float_output=True)
            }
        except:
            features = {key: 0 for key in [
                'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
                'automated_readability_index', 'coleman_liau_index', 'linsear_write_formula',
                'dale_chall_readability', 'text_standard'
            ]}
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, any]:
        """Extract advanced linguistic features using spaCy"""
        if pd.isna(text) or text == "" or not self.nlp:
            return {
                'noun_count': 0,
                'verb_count': 0,
                'adj_count': 0,
                'adv_count': 0,
                'entity_count': 0,
                'entity_types': '',
                'dependency_depth': 0,
                'pos_diversity': 0
            }
        
        # Process text with spaCy
        doc = self.nlp(text[:1000])  # Limit length for performance
        
        # POS tag counts
        pos_counts = Counter(token.pos_ for token in doc)
        
        # Named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_types = Counter(ent.label_ for ent in doc.ents)
        
        # Dependency parsing depth (simplified)
        dependency_depths = []
        for sent in doc.sents:
            depths = {}
            for token in sent:
                if token.dep_ == "ROOT":
                    depths[token] = 0
                else:
                    depths[token] = depths.get(token.head, 0) + 1
            if depths:
                dependency_depths.append(max(depths.values()))
        
        avg_dependency_depth = np.mean(dependency_depths) if dependency_depths else 0
        
        return {
            'noun_count': pos_counts.get('NOUN', 0),
            'verb_count': pos_counts.get('VERB', 0),
            'adj_count': pos_counts.get('ADJ', 0),
            'adv_count': pos_counts.get('ADV', 0),
            'entity_count': len(entities),
            'entity_types': ','.join(entity_types.keys()),
            'dependency_depth': avg_dependency_depth,
            'pos_diversity': len(pos_counts)
        }
    
    def extract_special_patterns(self, text: str) -> Dict[str, int]:
        """Extract special patterns like URLs, emails, etc."""
        if pd.isna(text):
            return {
                'url_count': 0,
                'email_count': 0,
                'hashtag_count': 0,
                'mention_count': 0,
                'emoji_count': 0,
                'repeated_char_count': 0,
                'all_caps_word_count': 0
            }
        
        features = {
            'url_count': len(self.url_pattern.findall(text)),
            'email_count': len(self.email_pattern.findall(text)),
            'hashtag_count': len(self.hashtag_pattern.findall(text)),
            'mention_count': len(self.mention_pattern.findall(text)),
            'emoji_count': len(self.emoji_pattern.findall(text))
        }
        
        # Repeated characters (e.g., "sooooo good")
        repeated_pattern = re.compile(r'(.)\1{2,}')
        features['repeated_char_count'] = len(repeated_pattern.findall(text))
        
        # All caps words
        words = text.split()
        all_caps_words = [w for w in words if w.isupper() and len(w) > 1]
        features['all_caps_word_count'] = len(all_caps_words)
        
        return features
    
    def extract_temporal_features(self, review_date) -> Dict[str, int]:
        """Extract temporal features from review date"""
        if pd.isna(review_date):
            return {
                'review_hour': 12,
                'review_dayofweek': 3,
                'review_day': 15,
                'review_month': 6,
                'is_weekend': 0,
                'is_holiday_season': 0,
                'days_since_epoch': 0
            }
        
        date = pd.to_datetime(review_date)
        
        features = {
            'review_hour': date.hour,
            'review_dayofweek': date.dayofweek,
            'review_day': date.day,
            'review_month': date.month,
            'is_weekend': 1 if date.dayofweek >= 5 else 0,
            'is_holiday_season': 1 if date.month in [11, 12, 1] else 0,
            'days_since_epoch': (date - pd.Timestamp('2020-01-01')).days
        }
        
        return features
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from the dataframe"""
        logger.info("Starting comprehensive feature extraction...")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Text statistics
        logger.info("Extracting text statistics...")
        text_stats = df_features['review_text'].apply(self.extract_text_statistics)
        for col in text_stats.iloc[0].keys():
            df_features[f'text_{col}'] = text_stats.apply(lambda x: x[col])
        
        # Punctuation features
        logger.info("Extracting punctuation features...")
        punct_features = df_features['review_text'].apply(self.extract_punctuation_features)
        for col in punct_features.iloc[0].keys():
            df_features[f'punct_{col}'] = punct_features.apply(lambda x: x[col])
        
        # Sentiment features
        logger.info("Extracting sentiment features...")
        sent_features = df_features['review_text'].apply(self.extract_sentiment_features)
        for col in sent_features.iloc[0].keys():
            df_features[f'sent_{col}'] = sent_features.apply(lambda x: x[col])
        
        # Readability features
        logger.info("Extracting readability features...")
        read_features = df_features['review_text'].apply(self.extract_readability_features)
        for col in read_features.iloc[0].keys():
            df_features[f'read_{col}'] = read_features.apply(lambda x: x[col])
        
        # Linguistic features (if spaCy is available)
        if self.nlp:
            logger.info("Extracting linguistic features...")
            ling_features = df_features['review_text'].apply(self.extract_linguistic_features)
            for col in ling_features.iloc[0].keys():
                df_features[f'ling_{col}'] = ling_features.apply(lambda x: x[col])
        
        # Special patterns
        logger.info("Extracting special patterns...")
        pattern_features = df_features['review_text'].apply(self.extract_special_patterns)
        for col in pattern_features.iloc[0].keys():
            df_features[f'pattern_{col}'] = pattern_features.apply(lambda x: x[col])
        
        # Temporal features
        logger.info("Extracting temporal features...")
        if 'review_date' in df_features.columns:
            temp_features = df_features['review_date'].apply(self.extract_temporal_features)
            for col in temp_features.iloc[0].keys():
                df_features[f'temp_{col}'] = temp_features.apply(lambda x: x[col])
        
        # Interaction features
        logger.info("Creating interaction features...")
        
        # Rating-sentiment alignment
        df_features['rating_sentiment_aligned'] = (
            ((df_features['rating'] >= 4) & (df_features.get('predicted_sentiment', '') == 'positive')) |
            ((df_features['rating'] <= 2) & (df_features.get('predicted_sentiment', '') == 'negative')) |
            ((df_features['rating'] == 3) & (df_features.get('predicted_sentiment', '') == 'neutral'))
        ).astype(int)
        
        # Review quality score
        df_features['review_quality_score'] = (
            df_features['text_word_count'] * 0.3 +
            df_features['text_unique_word_ratio'] * 50 +
            df_features['sent_textblob_subjectivity'] * 20 +
            (1 - df_features['text_capital_ratio']) * 30
        ) / 100
        
        # Engagement potential
        df_features['engagement_potential'] = (
            df_features['punct_exclamation_count'] * 2 +
            df_features['punct_question_count'] * 3 +
            df_features['pattern_emoji_count'] * 2 +
            df_features.get('helpful_count', 0) * 0.5
        )
        
        logger.info(f"Feature extraction complete. Total features: {len(df_features.columns)}")
        
        return df_features
    
    def select_top_features(self, df: pd.DataFrame, target_col: str, n_features: int = 50) -> List[str]:
        """Select top features using mutual information"""
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        
        # Get numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and identifier columns
        exclude_cols = [target_col, 'review_id', 'product_id', 'review_date']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        
        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Get top features
        feature_scores = pd.DataFrame({
            'feature': feature_cols,
            'score': mi_scores
        }).sort_values('score', ascending=False)
        
        top_features = feature_scores.head(n_features)['feature'].tolist()
        
        logger.info(f"Selected top {len(top_features)} features based on mutual information")
        
        return top_features


def main():
    """Example usage of feature engineering"""
    # This would typically be called as part of the preprocessing pipeline
    pass


if __name__ == "__main__":
    main()