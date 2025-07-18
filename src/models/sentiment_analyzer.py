"""
Sentiment Analysis Module for Customer Feedback Analytics
Implements multiple sentiment analysis approaches: Traditional ML, Deep Learning, and Transformers
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import yaml
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataset(Dataset):
    """PyTorch Dataset for review data"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class LSTMSentimentModel(nn.Module):
    """LSTM model for sentiment classification"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(LSTMSentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output

class SentimentAnalyzer:
    """Multi-model sentiment analysis system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize sentiment analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['sentiment']
        self.models = {}
        self.label_encoder = LabelEncoder()
        
        # Create models directory
        self.models_dir = "models/sentiment"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Use true_sentiment for training if available, otherwise derive from rating
        if 'true_sentiment' in df.columns:
            y = df['true_sentiment'].map({'positive': 2, 'negative': 0, 'neutral': 1})
        else:
            # Derive sentiment from rating
            df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
            y = df['sentiment'].map({'positive': 2, 'negative': 0, 'neutral': 1})
        
        X = df['cleaned_text'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train.values, y_test.values
    
    def train_tfidf_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train traditional ML models with TF-IDF features"""
        logger.info("Training TF-IDF based models...")
        
        # Create TF-IDF vectorizer
        tfidf_config = self.model_config['tfidf']
        vectorizer = TfidfVectorizer(
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            min_df=tfidf_config['min_df'],
            max_df=tfidf_config['max_df']
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Save vectorizer
        with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        results = {}
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_tfidf, y_train)
        rf_pred = rf_model.predict(X_test_tfidf)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': rf_accuracy,
            'predictions': rf_pred,
            'report': classification_report(y_test, rf_pred, target_names=['negative', 'neutral', 'positive'])
        }
        
        # Save model
        with open(os.path.join(self.models_dir, 'random_forest.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        
        # Train SVM
        logger.info("Training SVM...")
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(X_train_tfidf, y_train)
        svm_pred = svm_model.predict(X_test_tfidf)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        
        self.models['svm'] = svm_model
        results['svm'] = {
            'accuracy': svm_accuracy,
            'predictions': svm_pred,
            'report': classification_report(y_test, svm_pred, target_names=['negative', 'neutral', 'positive'])
        }
        
        # Save model
        with open(os.path.join(self.models_dir, 'svm.pkl'), 'wb') as f:
            pickle.dump(svm_model, f)
        
        return results
    
    def train_transformer_model(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train transformer-based model"""
        logger.info("Training Transformer model...")
        
        # Use DistilBERT for efficiency
        model_name = self.model_config['transformer']['model_name']
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3
            )
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            logger.info("Skipping transformer training. Install transformers and torch for this feature.")
            return {}
        
        # Create datasets
        train_dataset = ReviewDataset(X_train, y_train, tokenizer)
        test_dataset = ReviewDataset(X_test, y_test, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.models_dir, 'transformer'),
            num_train_epochs=self.model_config['transformer']['epochs'],
            per_device_train_batch_size=self.model_config['transformer']['batch_size'],
            per_device_eval_batch_size=self.model_config['transformer']['batch_size'],
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # Train model
        trainer.train()
        
        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model.save_pretrained(os.path.join(self.models_dir, 'transformer'))
        tokenizer.save_pretrained(os.path.join(self.models_dir, 'transformer'))
        
        self.models['transformer'] = model
        
        return {
            'transformer': {
                'accuracy': accuracy,
                'predictions': y_pred,
                'report': classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
            }
        }
    
    def predict(self, texts: List[str], model_name: str = 'random_forest') -> np.ndarray:
        """Predict sentiment for new texts"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        if model_name in ['random_forest', 'svm']:
            # Load vectorizer
            with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
                vectorizer = pickle.load(f)
            
            X = vectorizer.transform(texts)
            predictions = self.models[model_name].predict(X)
            
        elif model_name == 'transformer':
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.models_dir, 'transformer'))
            model = self.models['transformer']
            
            predictions = []
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
            
            predictions = np.array(predictions)
        
        # Convert predictions to labels
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return np.array([label_map[p] for p in predictions])
    
    def predict_with_confidence(self, texts: List[str], model_name: str = 'random_forest') -> Tuple[np.ndarray, np.ndarray]:
        """Predict sentiment with confidence scores"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if model_name in ['random_forest', 'svm']:
            # Load vectorizer
            with open(os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
                vectorizer = pickle.load(f)
            
            X = vectorizer.transform(texts)
            predictions = self.models[model_name].predict(X)
            probabilities = self.models[model_name].predict_proba(X)
            confidence = np.max(probabilities, axis=1)
            
        elif model_name == 'transformer':
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.models_dir, 'transformer'))
            model = self.models['transformer']
            
            predictions = []
            confidence = []
            
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = torch.max(probs).item()
                
                predictions.append(pred)
                confidence.append(conf)
            
            predictions = np.array(predictions)
            confidence = np.array(confidence)
        
        # Convert predictions to labels
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predictions = np.array([label_map[p] for p in predictions])
        
        return predictions, confidence
    
    def compare_models(self, results: Dict) -> pd.DataFrame:
        """Compare performance of different models"""
        comparison = []
        
        for model_name, result in results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Negative Precision': float(result['report'].split('\n')[2].split()[1]),
                'Neutral Precision': float(result['report'].split('\n')[3].split()[1]),
                'Positive Precision': float(result['report'].split('\n')[4].split()[1]),
            })
        
        return pd.DataFrame(comparison).sort_values('Accuracy', ascending=False)


def main():
    """Main training pipeline"""
    # Load preprocessed data
    try:
        df = pd.read_csv("data/processed/preprocessed_reviews.csv")
        logger.info(f"Loaded {len(df)} preprocessed reviews")
    except FileNotFoundError:
        logger.error("Preprocessed data not found. Please run preprocessor.py first.")
        return
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
    
    # Train models
    tfidf_results = analyzer.train_tfidf_models(X_train, X_test, y_train, y_test)
    
    # Try to train transformer (optional, requires GPU for best performance)
    try:
        transformer_results = analyzer.train_transformer_model(X_train, X_test, y_train, y_test)
        all_results = {**tfidf_results, **transformer_results}
    except Exception as e:
        logger.warning(f"Transformer training failed: {e}")
        all_results = tfidf_results
    
    # Compare models
    comparison_df = analyzer.compare_models(all_results)
    print("\n=== Model Comparison ===")
    print(comparison_df)
    
    # Save predictions
    best_model = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
    logger.info(f"Best model: {best_model}")
    
    # Generate predictions for all data
    predictions, confidence = analyzer.predict_with_confidence(df['cleaned_text'].values, best_model)
    
    df['predicted_sentiment'] = predictions
    df['confidence_score'] = confidence
    
    # Save predictions
    output_path = analyzer.config['data']['predictions_path']
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Print sample predictions
    print("\n=== Sample Predictions ===")
    for _, row in df.sample(5).iterrows():
        print(f"\nReview: {row['review_text'][:100]}...")
        print(f"True Sentiment: {row.get('true_sentiment', 'N/A')}")
        print(f"Predicted: {row['predicted_sentiment']} (Confidence: {row['confidence_score']:.2f})")


if __name__ == "__main__":
    main()