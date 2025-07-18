"""
Customer Feedback Analytics Source Package
Main source code for the analytics system
"""

__version__ = "1.0.0"
__author__ = "Customer Analytics Team"

# Make key modules easily importable
from .data_processing.preprocessor import TextPreprocessor
from .data_processing.feature_engineering import FeatureEngineer
from .models.sentiment_analyzer import SentimentAnalyzer
from .models.topic_extractor import TopicExtractor
from .models.model_evaluator import ModelEvaluator
from .utils.business_insights import BusinessInsightsGenerator
from .visualization.charts import ChartGenerator
from .visualization.dashboard_components import DashboardComponents

__all__ = [
    'TextPreprocessor',
    'FeatureEngineer',
    'SentimentAnalyzer',
    'TopicExtractor',
    'ModelEvaluator',
    'BusinessInsightsGenerator',
    'ChartGenerator',
    'DashboardComponents'
]