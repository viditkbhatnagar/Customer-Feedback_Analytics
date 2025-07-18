"""
Topic Modeling and Extraction Module for Customer Feedback Analytics
Implements LDA, BERTopic, and keyword extraction techniques
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import yake
import yaml
import pickle
import os
import logging
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicExtractor:
    """Advanced topic modeling and keyword extraction system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize topic extractor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.topic_config = self.config['models']['topic_modeling']
        self.stop_words = set(stopwords.words('english'))
        
        # Create directories
        self.models_dir = "models/topics"
        self.visualizations_dir = "visualizations/topics"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        self.topics = {}
        self.keywords = {}
        
    def extract_keywords_yake(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE algorithm"""
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # max n-gram size
            dedupLim=0.7,
            dedupFunc='seqm',
            windowsSize=1,
            top=max_keywords
        )
        
        keywords = kw_extractor.extract_keywords(text)
        return [(kw[0], kw[1]) for kw in keywords]
    
    def extract_keywords_tfidf(self, texts: List[str], max_keywords: int = 20) -> Dict[str, float]:
        """Extract keywords using TF-IDF"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = avg_tfidf.argsort()[-max_keywords:][::-1]
        keywords = {feature_names[i]: avg_tfidf[i] for i in top_indices}
        
        return keywords
    
    def train_lda_model(self, texts: List[str], n_topics: int = None) -> Dict:
        """Train LDA topic model"""
        logger.info("Training LDA model...")
        
        if n_topics is None:
            n_topics = self.topic_config['num_topics']
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=tuple(self.topic_config['n_gram_range']),
            stop_words='english',
            min_df=5,
            max_df=0.8
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Train LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online',
            learning_offset=50.,
            max_iter=10
        )
        
        doc_topics = lda.fit_transform(doc_term_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'scores': top_scores,
                'word_scores': list(zip(top_words, top_scores))
            })
        
        # Calculate coherence score
        texts_for_coherence = [text.split() for text in texts[:1000]]  # Sample for efficiency
        dictionary = corpora.Dictionary(texts_for_coherence)
        corpus = [dictionary.doc2bow(text) for text in texts_for_coherence]
        
        # Get topic words for coherence
        topics_words = [[word for word, _ in topic['word_scores'][:10]] for topic in topics]
        
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=texts_for_coherence,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Save model
        with open(os.path.join(self.models_dir, 'lda_model.pkl'), 'wb') as f:
            pickle.dump(lda, f)
        with open(os.path.join(self.models_dir, 'lda_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        self.topics['lda'] = topics
        
        return {
            'model': lda,
            'topics': topics,
            'doc_topics': doc_topics,
            'coherence_score': coherence_score,
            'feature_names': feature_names
        }
    
    def train_bertopic_model(self, texts: List[str], n_topics: int = None) -> Dict:
        """Train BERTopic model"""
        logger.info("Training BERTopic model...")
        
        if n_topics is None:
            n_topics = self.topic_config['num_topics']
        
        try:
            # Initialize BERTopic
            topic_model = BERTopic(
                language='english',
                calculate_probabilities=True,
                n_gram_range=tuple(self.topic_config['n_gram_range']),
                min_topic_size=self.topic_config['min_topic_size'],
                diversity=self.topic_config['diversity']
            )
            
            # Fit model
            topics, probs = topic_model.fit_transform(texts)
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            
            # Extract topics
            topics_list = []
            for topic_id in range(len(topic_info) - 1):  # Exclude outlier topic
                if topic_id == -1:
                    continue
                topic_words = topic_model.get_topic(topic_id)
                if topic_words:
                    topics_list.append({
                        'topic_id': topic_id,
                        'words': [word for word, _ in topic_words[:10]],
                        'scores': [score for _, score in topic_words[:10]],
                        'word_scores': topic_words[:10]
                    })
            
            # Save model
            topic_model.save(os.path.join(self.models_dir, "bertopic_model"))
            
            self.topics['bertopic'] = topics_list
            
            return {
                'model': topic_model,
                'topics': topics_list,
                'doc_topics': topics,
                'topic_info': topic_info
            }
            
        except Exception as e:
            logger.warning(f"BERTopic training failed: {e}")
            logger.info("Please ensure you have sentence-transformers installed for BERTopic")
            return {}
    
    def extract_category_topics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Extract topics for each product category"""
        logger.info("Extracting category-specific topics...")
        
        category_topics = {}
        
        for category in df['category'].unique():
            logger.info(f"Processing category: {category}")
            
            # Filter reviews for category
            category_df = df[df['category'] == category]
            texts = category_df['cleaned_text'].tolist()
            
            if len(texts) < 50:  # Skip if too few reviews
                continue
            
            # Extract keywords
            keywords = self.extract_keywords_tfidf(texts, max_keywords=15)
            
            # Get sentiment-specific topics
            positive_texts = category_df[category_df['predicted_sentiment'] == 'positive']['cleaned_text'].tolist()
            negative_texts = category_df[category_df['predicted_sentiment'] == 'negative']['cleaned_text'].tolist()
            
            positive_keywords = self.extract_keywords_tfidf(positive_texts, max_keywords=10) if len(positive_texts) > 10 else {}
            negative_keywords = self.extract_keywords_tfidf(negative_texts, max_keywords=10) if len(negative_texts) > 10 else {}
            
            category_topics[category] = {
                'total_reviews': len(texts),
                'keywords': keywords,
                'positive_keywords': positive_keywords,
                'negative_keywords': negative_keywords,
                'sentiment_distribution': category_df['predicted_sentiment'].value_counts().to_dict()
            }
        
        self.keywords = category_topics
        return category_topics
    
    def identify_trending_topics(self, df: pd.DataFrame, window_days: int = 30) -> Dict:
        """Identify trending topics over time"""
        logger.info("Identifying trending topics...")
        
        df['review_date'] = pd.to_datetime(df['review_date'])
        end_date = df['review_date'].max()
        start_date = end_date - pd.Timedelta(days=window_days)
        
        # Recent reviews
        recent_df = df[df['review_date'] >= start_date]
        older_df = df[df['review_date'] < start_date]
        
        # Extract keywords for both periods
        recent_keywords = self.extract_keywords_tfidf(recent_df['cleaned_text'].tolist())
        older_keywords = self.extract_keywords_tfidf(older_df['cleaned_text'].tolist())
        
        # Calculate trending score
        trending_topics = {}
        for keyword, recent_score in recent_keywords.items():
            older_score = older_keywords.get(keyword, 0.001)  # Small value to avoid division by zero
            trend_score = (recent_score - older_score) / older_score
            
            if trend_score > 0.5:  # 50% increase threshold
                trending_topics[keyword] = {
                    'trend_score': trend_score,
                    'recent_score': recent_score,
                    'older_score': older_score,
                    'percentage_change': trend_score * 100
                }
        
        # Sort by trend score
        trending_topics = dict(sorted(trending_topics.items(), 
                                    key=lambda x: x[1]['trend_score'], 
                                    reverse=True)[:20])
        
        return trending_topics
    
    def visualize_topics(self, topics: List[Dict], model_name: str):
        """Create visualizations for topics"""
        logger.info(f"Creating visualizations for {model_name}...")
        
        # Topic word distribution
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, topic in enumerate(topics[:10]):
            ax = axes[idx]
            words = topic['words'][:10]
            scores = topic['scores'][:10]
            
            ax.barh(words, scores)
            ax.set_title(f"Topic {topic['topic_id']}")
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, f'{model_name}_topics.png'))
        plt.close()
        
        # Word cloud for top topics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, topic in enumerate(topics[:4]):
            ax = axes[idx]
            word_freq = dict(topic['word_scores'])
            
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f"Topic {topic['topic_id']} Word Cloud")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, f'{model_name}_wordclouds.png'))
        plt.close()
    
    def generate_topic_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive topic analysis report"""
        report = {
            'summary': {},
            'lda_results': {},
            'bertopic_results': {},
            'category_analysis': {},
            'trending_topics': {},
            'insights': []
        }
        
        # Train LDA
        lda_results = self.train_lda_model(df['cleaned_text'].tolist())
        if lda_results:
            report['lda_results'] = {
                'num_topics': len(lda_results['topics']),
                'coherence_score': lda_results['coherence_score'],
                'top_topics': lda_results['topics'][:5]
            }
            self.visualize_topics(lda_results['topics'], 'lda')
        
        # Train BERTopic
        bertopic_results = self.train_bertopic_model(df['cleaned_text'].tolist()[:5000])  # Sample for efficiency
        if bertopic_results:
            report['bertopic_results'] = {
                'num_topics': len(bertopic_results['topics']),
                'top_topics': bertopic_results['topics'][:5]
            }
            self.visualize_topics(bertopic_results['topics'], 'bertopic')
        
        # Category analysis
        category_topics = self.extract_category_topics(df)
        report['category_analysis'] = category_topics
        
        # Trending topics
        trending = self.identify_trending_topics(df)
        report['trending_topics'] = trending
        
        # Generate insights
        insights = []
        
        # Find categories with most negative feedback
        negative_categories = []
        for category, data in category_topics.items():
            sentiment_dist = data['sentiment_distribution']
            if sentiment_dist.get('negative', 0) > sentiment_dist.get('positive', 0) * 0.3:
                negative_categories.append({
                    'category': category,
                    'negative_ratio': sentiment_dist.get('negative', 0) / data['total_reviews']
                })
        
        if negative_categories:
            worst_category = max(negative_categories, key=lambda x: x['negative_ratio'])
            insights.append(f"ALERT: {worst_category['category']} has {worst_category['negative_ratio']*100:.1f}% negative reviews")
        
        # Identify common complaints
        all_negative_keywords = []
        for cat_data in category_topics.values():
            all_negative_keywords.extend(list(cat_data['negative_keywords'].keys()))
        
        common_complaints = Counter(all_negative_keywords).most_common(5)
        if common_complaints:
            insights.append(f"Top complaints across all categories: {', '.join([c[0] for c in common_complaints])}")
        
        # Trending issues
        if trending:
            top_trending = list(trending.keys())[:3]
            insights.append(f"Trending topics (last 30 days): {', '.join(top_trending)}")
        
        report['insights'] = insights
        
        # Save report
        with open(os.path.join(self.models_dir, 'topic_analysis_report.pkl'), 'wb') as f:
            pickle.dump(report, f)
        
        return report


def main():
    """Main topic extraction pipeline"""
    # Load data with predictions
    try:
        df = pd.read_csv("data/processed/sentiment_predictions.csv")
        logger.info(f"Loaded {len(df)} reviews with sentiment predictions")
    except FileNotFoundError:
        logger.error("Sentiment predictions not found. Please run sentiment_analyzer.py first.")
        return
    
    # Initialize extractor
    extractor = TopicExtractor()
    
    # Generate comprehensive report
    report = extractor.generate_topic_report(df)
    
    # Print summary
    print("\n=== Topic Analysis Summary ===")
    
    if report['lda_results']:
        print(f"\nLDA Model:")
        print(f"  Number of topics: {report['lda_results']['num_topics']}")
        print(f"  Coherence score: {report['lda_results']['coherence_score']:.3f}")
        print("\n  Top 3 Topics:")
        for topic in report['lda_results']['top_topics'][:3]:
            top_words = ', '.join(topic['words'][:5])
            print(f"    Topic {topic['topic_id']}: {top_words}")
    
    print("\n=== Category Analysis ===")
    for category, data in list(report['category_analysis'].items())[:3]:
        print(f"\n{category}:")
        print(f"  Total reviews: {data['total_reviews']}")
        print(f"  Top keywords: {', '.join(list(data['keywords'].keys())[:5])}")
        if data['negative_keywords']:
            print(f"  Main complaints: {', '.join(list(data['negative_keywords'].keys())[:3])}")
    
    print("\n=== Trending Topics (Last 30 days) ===")
    for topic, data in list(report['trending_topics'].items())[:5]:
        print(f"  {topic}: +{data['percentage_change']:.1f}% change")
    
    print("\n=== Key Insights ===")
    for insight in report['insights']:
        print(f"  • {insight}")


if __name__ == "__main__":
    main()