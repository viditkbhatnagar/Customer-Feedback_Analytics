"""
Model Evaluation Module for Customer Feedback Analytics
Comprehensive model performance evaluation and comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and comparison system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model evaluator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.evaluation_config = self.config['evaluation']
        self.results = {}
        self.models = {}
        
        # Create directories
        self.results_dir = "results/evaluation"
        self.plots_dir = "results/plots"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None,
                      model_name: str = "model") -> Dict[str, Any]:
        """Evaluate a single model comprehensively"""
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        results['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        results['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        results['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        # ROC AUC if probabilities are provided
        if y_proba is not None and len(np.unique(y_true)) == 2:
            results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            results['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
        
        # Store results
        self.results[model_name] = results
        
        logger.info(f"Evaluation complete for {model_name}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"F1-Score (macro): {results['f1_macro']:.4f}")
        
        return results
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on a model"""
        # Create stratified folds
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Metrics to evaluate
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        cv_results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'confidence_interval': (scores.mean() - 2 * scores.std(), 
                                      scores.mean() + 2 * scores.std())
            }
        
        return cv_results
    
    def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, save: bool = True) -> go.Figure:
        """Create interactive confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        labels = ['Negative', 'Neutral', 'Positive']
        
        # Create text annotations
        text = []
        for i in range(len(labels)):
            row_text = []
            for j in range(len(labels)):
                row_text.append(f"{cm[i,j]}<br>{cm_normalized[i,j]:.2%}")
            text.append(row_text)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            text=text,
            texttemplate='%{text}',
            textfont={"size": 14},
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Proportion")
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=600,
            height=500
        )
        
        if save:
            fig.write_html(os.path.join(self.plots_dir, f'confusion_matrix_{model_name}.html'))
        
        return fig
    
    def create_roc_curves(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                         save: bool = True) -> go.Figure:
        """Create ROC curves for multiple models"""
        fig = go.Figure()
        
        for model_name, (y_true, y_score, _) in models_data.items():
            # For multiclass, we need to handle each class
            if len(np.unique(y_true)) > 2:
                # Convert to binary for positive class
                y_binary = (y_true == 2).astype(int)  # Positive class
                if y_score.ndim > 1:
                    y_score_binary = y_score[:, 2]  # Positive class probabilities
                else:
                    y_score_binary = y_score
            else:
                y_binary = y_true
                y_score_binary = y_score
            
            fpr, tpr, _ = roc_curve(y_binary, y_score_binary)
            auc = roc_auc_score(y_binary, y_score_binary)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc:.3f})',
                line=dict(width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=600,
            hovermode='closest'
        )
        
        if save:
            fig.write_html(os.path.join(self.plots_dir, 'roc_curves_comparison.html'))
        
        return fig
    
    def create_precision_recall_curves(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                     save: bool = True) -> go.Figure:
        """Create precision-recall curves for multiple models"""
        fig = go.Figure()
        
        for model_name, (y_true, y_score, _) in models_data.items():
            # For multiclass, handle positive class
            if len(np.unique(y_true)) > 2:
                y_binary = (y_true == 2).astype(int)  # Positive class
                if y_score.ndim > 1:
                    y_score_binary = y_score[:, 2]
                else:
                    y_score_binary = y_score
            else:
                y_binary = y_true
                y_score_binary = y_score
            
            precision, recall, _ = precision_recall_curve(y_binary, y_score_binary)
            avg_precision = average_precision_score(y_binary, y_score_binary)
            
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode='lines',
                name=f'{model_name} (AP = {avg_precision:.3f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700,
            height=600,
            hovermode='closest'
        )
        
        if save:
            fig.write_html(os.path.join(self.plots_dir, 'precision_recall_curves.html'))
        
        return fig
    
    def create_model_comparison_plot(self, save: bool = True) -> go.Figure:
        """Create comprehensive model comparison visualization"""
        if not self.results:
            logger.warning("No results to compare")
            return None
        
        # Prepare data
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Metrics', 'Per-Class F1 Scores', 
                          'Model Rankings', 'Confidence Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # 1. Overall metrics comparison
        for metric in metrics:
            values = [self.results[model][metric] for model in models]
            fig.add_trace(
                go.Bar(name=metric, x=models, y=values),
                row=1, col=1
            )
        
        # 2. Per-class F1 scores
        classes = ['Negative', 'Neutral', 'Positive']
        for i, class_name in enumerate(classes):
            values = [self.results[model]['f1_per_class'][i] for model in models]
            fig.add_trace(
                go.Bar(name=class_name, x=models, y=values),
                row=1, col=2
            )
        
        # 3. Model rankings
        ranking_data = []
        for metric in metrics:
            metric_values = [(model, self.results[model][metric]) for model in models]
            metric_values.sort(key=lambda x: x[1], reverse=True)
            for rank, (model, value) in enumerate(metric_values, 1):
                ranking_data.append({
                    'Model': model,
                    'Metric': metric,
                    'Rank': rank,
                    'Value': value
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        for model in models:
            model_ranks = ranking_df[ranking_df['Model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_ranks['Metric'],
                    y=model_ranks['Rank'],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Comprehensive Model Comparison",
            showlegend=True,
            height=800,
            width=1200
        )
        
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="F1 Score", row=1, col=2)
        fig.update_yaxes(title_text="Rank", autorange="reversed", row=2, col=1)
        
        if save:
            fig.write_html(os.path.join(self.plots_dir, 'model_comparison.html'))
        
        return fig
    
    def create_error_analysis(self, df: pd.DataFrame, y_true: np.ndarray, 
                            y_pred: np.ndarray, model_name: str) -> pd.DataFrame:
        """Analyze prediction errors"""
        # Create error dataframe
        error_df = df.copy()
        error_df['true_label'] = y_true
        error_df['predicted_label'] = y_pred
        error_df['is_error'] = y_true != y_pred
        
        # Map numeric labels to names
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        error_df['true_sentiment'] = error_df['true_label'].map(label_map)
        error_df['predicted_sentiment'] = error_df['predicted_label'].map(label_map)
        
        # Focus on errors
        errors_only = error_df[error_df['is_error']]
        
        # Error analysis
        error_analysis = {
            'total_errors': len(errors_only),
            'error_rate': len(errors_only) / len(df) * 100,
            'errors_by_true_class': errors_only['true_sentiment'].value_counts().to_dict(),
            'errors_by_predicted_class': errors_only['predicted_sentiment'].value_counts().to_dict(),
            'errors_by_category': errors_only['category'].value_counts().to_dict(),
            'avg_confidence_on_errors': errors_only.get('confidence_score', pd.Series()).mean()
        }
        
        # Common error patterns
        error_patterns = []
        
        # Pattern 1: High rating but negative prediction
        pattern1 = errors_only[(errors_only['rating'] >= 4) & (errors_only['predicted_sentiment'] == 'negative')]
        if len(pattern1) > 0:
            error_patterns.append({
                'pattern': 'High rating but negative prediction',
                'count': len(pattern1),
                'percentage': len(pattern1) / len(errors_only) * 100
            })
        
        # Pattern 2: Low rating but positive prediction
        pattern2 = errors_only[(errors_only['rating'] <= 2) & (errors_only['predicted_sentiment'] == 'positive')]
        if len(pattern2) > 0:
            error_patterns.append({
                'pattern': 'Low rating but positive prediction',
                'count': len(pattern2),
                'percentage': len(pattern2) / len(errors_only) * 100
            })
        
        # Save error analysis
        error_report = {
            'model_name': model_name,
            'error_analysis': error_analysis,
            'error_patterns': error_patterns,
            'sample_errors': errors_only.head(10)[['review_text', 'rating', 'true_sentiment', 'predicted_sentiment']].to_dict('records')
        }
        
        # Save to file
        with open(os.path.join(self.results_dir, f'error_analysis_{model_name}.pkl'), 'wb') as f:
            pickle.dump(error_report, f)
        
        return errors_only
    
    def calculate_prediction_confidence_calibration(self, y_true: np.ndarray, 
                                                  y_proba: np.ndarray,
                                                  n_bins: int = 10) -> Dict[str, Any]:
        """Calculate prediction confidence calibration"""
        # Get predicted class and confidence
        y_pred = np.argmax(y_proba, axis=1)
        confidences = np.max(y_proba, axis=1)
        accuracies = y_pred == y_true
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Calculate calibration
        calibration_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                calibration_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'prop_in_bin': prop_in_bin,
                    'accuracy': accuracy_in_bin,
                    'avg_confidence': avg_confidence_in_bin,
                    'count': in_bin.sum()
                })
        
        # Calculate ECE (Expected Calibration Error)
        ece = sum(abs(d['accuracy'] - d['avg_confidence']) * d['prop_in_bin'] 
                 for d in calibration_data)
        
        return {
            'calibration_data': calibration_data,
            'ece': ece,
            'avg_confidence': confidences.mean(),
            'avg_accuracy': accuracies.mean()
        }
    
    def generate_evaluation_report(self, model_name: str) -> str:
        """Generate comprehensive evaluation report"""
        if model_name not in self.results:
            return "No results found for this model"
        
        results = self.results[model_name]
        
        report = f"""
# Model Evaluation Report: {model_name}

## Overall Performance
- **Accuracy**: {results['accuracy']:.4f}
- **Precision (macro)**: {results['precision_macro']:.4f}
- **Recall (macro)**: {results['recall_macro']:.4f}
- **F1-Score (macro)**: {results['f1_macro']:.4f}

## Per-Class Performance
"""
        
        classes = ['Negative', 'Neutral', 'Positive']
        for i, class_name in enumerate(classes):
            report += f"""
### {class_name}
- Precision: {results['precision_per_class'][i]:.4f}
- Recall: {results['recall_per_class'][i]:.4f}
- F1-Score: {results['f1_per_class'][i]:.4f}
"""
        
        report += """
## Confusion Matrix
```
"""
        cm = results['confusion_matrix']
        report += f"         Predicted\n"
        report += f"         Neg  Neu  Pos\n"
        for i, class_name in enumerate(['Neg', 'Neu', 'Pos']):
            report += f"Actual {class_name}  "
            for j in range(3):
                report += f"{cm[i,j]:4d} "
            report += "\n"
        report += "```\n"
        
        return report
    
    def save_all_results(self):
        """Save all evaluation results"""
        # Save results dictionary
        with open(os.path.join(self.results_dir, 'all_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        # Generate summary report
        summary = []
        for model_name, results in self.results.items():
            summary.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision_macro'],
                'Recall': results['recall_macro'],
                'F1-Score': results['f1_macro']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(self.results_dir, 'model_comparison_summary.csv'), index=False)
        
        logger.info(f"All results saved to {self.results_dir}")


def main():
    """Example usage of model evaluator"""
    # This would typically be called after model training
    pass


if __name__ == "__main__":
    main()