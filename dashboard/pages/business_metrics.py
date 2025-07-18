"""
Business Insights and Recommendations Generator
Analyzes sentiment and topic data to generate actionable business insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BusinessMetric:
    """Data class for business metrics"""
    name: str
    value: float
    benchmark: float
    trend: str  # 'improving', 'declining', 'stable'
    priority: str  # 'high', 'medium', 'low'
    recommendation: str

class BusinessInsightsGenerator:
    """Generate business insights and recommendations from customer feedback"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the insights generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.business_rules = self.config['business_rules']
        self.analysis_config = self.config['analysis']
        self.insights = []
        self.metrics = []
        self.recommendations = []
        
    def calculate_customer_satisfaction_index(self, df: pd.DataFrame) -> float:
        """Calculate overall customer satisfaction index (CSI)"""
        # Weighted calculation based on ratings and sentiment
        rating_weight = self.business_rules['sentiment_weights']['rating_weight']
        text_weight = self.business_rules['sentiment_weights']['text_weight']
        
        # Normalize ratings to 0-1 scale
        rating_score = (df['rating'] - 1) / 4  # Convert 1-5 to 0-1
        
        # Convert sentiment to numeric
        sentiment_score = df['predicted_sentiment'].map({
            'positive': 1.0,
            'neutral': 0.5,
            'negative': 0.0
        })
        
        # Calculate weighted CSI
        csi = (rating_score * rating_weight + sentiment_score * text_weight).mean()
        
        return csi * 100  # Convert to percentage
    
    def identify_problem_products(self, df: pd.DataFrame, threshold: float = 0.3) -> List[Dict]:
        """Identify products with high negative sentiment"""
        product_sentiment = df.groupby('product_id').agg({
            'predicted_sentiment': lambda x: (x == 'negative').sum() / len(x),
            'rating': 'mean',
            'review_text': 'count'
        }).rename(columns={'review_text': 'review_count'})
        
        # Filter products with significant review volume
        product_sentiment = product_sentiment[product_sentiment['review_count'] >= 10]
        
        # Identify problematic products
        problem_products = product_sentiment[
            product_sentiment['predicted_sentiment'] > threshold
        ].sort_values('predicted_sentiment', ascending=False)
        
        results = []
        for product_id, data in problem_products.head(10).iterrows():
            # Get sample negative reviews
            negative_reviews = df[
                (df['product_id'] == product_id) & 
                (df['predicted_sentiment'] == 'negative')
            ]['review_text'].head(3).tolist()
            
            results.append({
                'product_id': product_id,
                'negative_rate': data['predicted_sentiment'],
                'avg_rating': data['rating'],
                'review_count': data['review_count'],
                'sample_complaints': negative_reviews
            })
        
        return results
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in customer feedback"""
        df['review_date'] = pd.to_datetime(df['review_date'])
        
        # Weekly sentiment trends
        weekly_sentiment = df.set_index('review_date').resample('W').agg({
            'predicted_sentiment': lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0,
            'rating': 'mean',
            'review_text': 'count'
        }).rename(columns={
            'predicted_sentiment': 'positive_rate',
            'review_text': 'review_count'
        })
        
        # Calculate trend
        recent_weeks = weekly_sentiment.tail(4)['positive_rate'].mean()
        previous_weeks = weekly_sentiment.iloc[-8:-4]['positive_rate'].mean()
        trend = 'improving' if recent_weeks > previous_weeks else 'declining'
        
        # Identify anomalies
        z_scores = np.abs(stats.zscore(weekly_sentiment['positive_rate'].dropna()))
        anomaly_weeks = weekly_sentiment.index[z_scores > 2].tolist()
        
        # Seasonal patterns
        df['month'] = df['review_date'].dt.month
        monthly_sentiment = df.groupby('month')['predicted_sentiment'].apply(
            lambda x: (x == 'positive').sum() / len(x)
        )
        
        peak_months = monthly_sentiment.nlargest(3).index.tolist()
        low_months = monthly_sentiment.nsmallest(3).index.tolist()
        
        return {
            'trend': trend,
            'recent_positive_rate': recent_weeks,
            'previous_positive_rate': previous_weeks,
            'anomaly_weeks': anomaly_weeks,
            'peak_months': peak_months,
            'low_months': low_months,
            'weekly_data': weekly_sentiment
        }
    
    def analyze_category_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by product category"""
        category_metrics = df.groupby('category').agg({
            'predicted_sentiment': [
                lambda x: (x == 'positive').sum() / len(x),
                lambda x: (x == 'negative').sum() / len(x),
                'count'
            ],
            'rating': ['mean', 'std'],
            'confidence_score': 'mean',
            'helpful_count': 'sum'
        })
        
        category_metrics.columns = [
            'positive_rate', 'negative_rate', 'review_count',
            'avg_rating', 'rating_std', 'avg_confidence', 'total_helpful'
        ]
        
        # Calculate engagement score
        category_metrics['engagement_score'] = (
            category_metrics['total_helpful'] / category_metrics['review_count']
        )
        
        # Identify best and worst performers
        best_categories = category_metrics.nlargest(3, 'positive_rate')
        worst_categories = category_metrics.nlargest(3, 'negative_rate')
        
        return {
            'metrics': category_metrics,
            'best_performers': best_categories,
            'worst_performers': worst_categories
        }
    
    def calculate_topic_impact(self, df: pd.DataFrame, topic_report: Dict) -> Dict:
        """Calculate business impact of identified topics"""
        topic_impacts = {}
        
        # Analyze trending topics impact
        if 'trending_topics' in topic_report:
            for topic, data in topic_report['trending_topics'].items():
                # Find reviews mentioning this topic
                topic_reviews = df[df['cleaned_text'].str.contains(topic, case=False, na=False)]
                
                if len(topic_reviews) > 0:
                    impact = {
                        'mention_count': len(topic_reviews),
                        'avg_rating': topic_reviews['rating'].mean(),
                        'sentiment_distribution': topic_reviews['predicted_sentiment'].value_counts().to_dict(),
                        'affected_categories': topic_reviews['category'].value_counts().head(3).to_dict(),
                        'trend_score': data['trend_score']
                    }
                    topic_impacts[topic] = impact
        
        return topic_impacts
    
    def generate_actionable_recommendations(self, df: pd.DataFrame, 
                                          topic_report: Dict) -> List[Dict]:
        """Generate specific actionable recommendations"""
        recommendations = []
        
        # 1. Customer Satisfaction Analysis
        csi = self.calculate_customer_satisfaction_index(df)
        target_csi = self.analysis_config['satisfaction_goal'] * 100
        
        if csi < target_csi:
            gap = target_csi - csi
            recommendations.append({
                'category': 'Customer Satisfaction',
                'priority': 'HIGH',
                'issue': f'Customer Satisfaction Index ({csi:.1f}%) is {gap:.1f}% below target',
                'recommendation': 'Implement immediate quality improvement program',
                'expected_impact': f'Increase customer retention by {gap*0.5:.0f}%',
                'timeline': '30 days'
            })
        
        # 2. Problem Products
        problem_products = self.identify_problem_products(df)
        if problem_products:
            top_problem = problem_products[0]
            recommendations.append({
                'category': 'Product Quality',
                'priority': 'HIGH',
                'issue': f"Product {top_problem['product_id']} has {top_problem['negative_rate']*100:.0f}% negative reviews",
                'recommendation': 'Conduct quality audit and consider product redesign',
                'expected_impact': f"Reduce returns by 20-30%",
                'timeline': '60 days'
            })
        
        # 3. Category Performance
        category_analysis = self.analyze_category_performance(df)
        worst_category = category_analysis['worst_performers'].index[0]
        worst_negative_rate = category_analysis['worst_performers'].iloc[0]['negative_rate']
        
        if worst_negative_rate > 0.25:
            recommendations.append({
                'category': 'Category Management',
                'priority': 'MEDIUM',
                'issue': f'{worst_category} category has {worst_negative_rate*100:.0f}% negative reviews',
                'recommendation': 'Review supplier quality standards and enhance QC processes',
                'expected_impact': 'Improve category NPS by 15-20 points',
                'timeline': '45 days'
            })
        
        # 4. Trending Issues
        topic_impacts = self.calculate_topic_impact(df, topic_report)
        for topic, impact in sorted(topic_impacts.items(), 
                                   key=lambda x: x[1]['trend_score'], 
                                   reverse=True)[:3]:
            if impact['avg_rating'] < 3:
                recommendations.append({
                    'category': 'Emerging Issues',
                    'priority': 'MEDIUM',
                    'issue': f'"{topic}" mentioned in {impact["mention_count"]} reviews with avg rating {impact["avg_rating"]:.1f}',
                    'recommendation': f'Address {topic} issues across affected products',
                    'expected_impact': 'Prevent escalation to major customer concern',
                    'timeline': '14 days'
                })
        
        # 5. Temporal Patterns
        temporal_analysis = self.analyze_temporal_patterns(df)
        if temporal_analysis['trend'] == 'declining':
            decline_pct = (temporal_analysis['previous_positive_rate'] - 
                          temporal_analysis['recent_positive_rate']) * 100
            recommendations.append({
                'category': 'Trend Alert',
                'priority': 'HIGH',
                'issue': f'Customer satisfaction declining by {decline_pct:.1f}% over past month',
                'recommendation': 'Launch customer win-back campaign and service recovery program',
                'expected_impact': 'Reverse negative trend within 30 days',
                'timeline': 'Immediate'
            })
        
        return recommendations
    
    def calculate_financial_impact(self, df: pd.DataFrame, 
                                 avg_order_value: float = 75.0) -> Dict:
        """Calculate estimated financial impact of customer feedback"""
        total_customers = len(df['review_id'].unique())
        
        # Estimate based on sentiment
        sentiment_distribution = df['predicted_sentiment'].value_counts(normalize=True)
        
        # Research shows negative reviews impact 2.5x more than positive
        negative_impact_multiplier = 2.5
        
        # Calculate potential lost revenue from negative reviews
        negative_customers = sentiment_distribution.get('negative', 0) * total_customers
        lost_revenue = negative_customers * avg_order_value * negative_impact_multiplier
        
        # Calculate potential revenue from improving satisfaction
        improvement_potential = 0.1  # 10% improvement target
        potential_revenue = total_customers * improvement_potential * avg_order_value
        
        # ROI calculation
        intervention_cost = 50000  # Estimated cost of improvement program
        roi = ((potential_revenue - intervention_cost) / intervention_cost) * 100
        
        return {
            'total_customers_analyzed': total_customers,
            'estimated_lost_revenue': lost_revenue,
            'improvement_potential': potential_revenue,
            'intervention_cost': intervention_cost,
            'expected_roi': roi,
            'payback_period_days': int(intervention_cost / (potential_revenue / 365))
        }
    
    def generate_executive_summary(self, df: pd.DataFrame, 
                                 topic_report: Dict) -> Dict:
        """Generate comprehensive executive summary"""
        # Key metrics
        csi = self.calculate_customer_satisfaction_index(df)
        total_reviews = len(df)
        avg_rating = df['rating'].mean()
        
        # Sentiment distribution
        sentiment_dist = df['predicted_sentiment'].value_counts(normalize=True).to_dict()
        
        # Temporal analysis
        temporal = self.analyze_temporal_patterns(df)
        
        # Category analysis
        category = self.analyze_category_performance(df)
        
        # Financial impact
        financial = self.calculate_financial_impact(df)
        
        # Generate recommendations
        recommendations = self.generate_actionable_recommendations(df, topic_report)
        
        summary = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'analysis_period': f"{df['review_date'].min().strftime('%Y-%m-%d')} to {df['review_date'].max().strftime('%Y-%m-%d')}",
            'key_metrics': {
                'customer_satisfaction_index': f"{csi:.1f}%",
                'total_reviews_analyzed': f"{total_reviews:,}",
                'average_rating': f"{avg_rating:.2f}/5.0",
                'positive_sentiment_rate': f"{sentiment_dist.get('positive', 0)*100:.1f}%",
                'response_rate_trend': temporal['trend']
            },
            'financial_impact': {
                'estimated_revenue_at_risk': f"${financial['estimated_lost_revenue']:,.0f}",
                'improvement_opportunity': f"${financial['improvement_potential']:,.0f}",
                'expected_roi': f"{financial['expected_roi']:.0f}%",
                'payback_period': f"{financial['payback_period_days']} days"
            },
            'top_insights': self._extract_top_insights(df, topic_report),
            'recommendations': recommendations[:5],  # Top 5 recommendations
            'next_steps': [
                "Review and prioritize recommendations with leadership team",
                "Allocate resources for high-priority initiatives",
                "Establish KPIs and monitoring dashboard",
                "Schedule 30-day progress review"
            ]
        }
        
        return summary
    
    def _extract_top_insights(self, df: pd.DataFrame, topic_report: Dict) -> List[str]:
        """Extract top insights from analysis"""
        insights = []
        
        # Insight 1: Overall satisfaction
        csi = self.calculate_customer_satisfaction_index(df)
        if csi < 70:
            insights.append(f"⚠️ Customer satisfaction critically low at {csi:.1f}%")
        elif csi > 85:
            insights.append(f"✅ Strong customer satisfaction at {csi:.1f}%")
        
        # Insight 2: Trend direction
        temporal = self.analyze_temporal_patterns(df)
        if temporal['trend'] == 'declining':
            insights.append("📉 Customer sentiment showing declining trend over past month")
        else:
            insights.append("📈 Customer sentiment improving over past month")
        
        # Insight 3: Category issues
        category = self.analyze_category_performance(df)
        worst_cat = category['worst_performers'].index[0]
        worst_rate = category['worst_performers'].iloc[0]['negative_rate']
        insights.append(f"🔴 {worst_cat} category needs attention ({worst_rate*100:.0f}% negative)")
        
        # Insight 4: Trending topics
        if topic_report.get('trending_topics'):
            top_trend = list(topic_report['trending_topics'].keys())[0]
            insights.append(f"🔥 '{top_trend}' emerging as key customer concern")
        
        # Insight 5: Verified purchase impact
        verified_sentiment = df[df['verified_purchase']]['predicted_sentiment'].value_counts(normalize=True)
        unverified_sentiment = df[~df['verified_purchase']]['predicted_sentiment'].value_counts(normalize=True)
        
        if verified_sentiment.get('positive', 0) < unverified_sentiment.get('positive', 0) - 0.1:
            insights.append("⚡ Verified purchasers significantly less satisfied than unverified")
        
        return insights[:5]  # Return top 5 insights
    
    def export_report(self, summary: Dict, output_dir: str = "reports"):
        """Export executive summary to various formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        import json
        with open(os.path.join(output_dir, 'executive_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create PDF-ready markdown
        markdown_report = self._generate_markdown_report(summary)
        with open(os.path.join(output_dir, 'executive_summary.md'), 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Executive summary exported to {output_dir}")
    
    def _generate_markdown_report(self, summary: Dict) -> str:
        """Generate markdown formatted report"""
        md = f"""# Customer Feedback Analytics - Executive Summary

**Report Date:** {summary['report_date']}  
**Analysis Period:** {summary['analysis_period']}

## Executive Overview

### Key Performance Indicators
- **Customer Satisfaction Index:** {summary['key_metrics']['customer_satisfaction_index']}
- **Total Reviews Analyzed:** {summary['key_metrics']['total_reviews_analyzed']}
- **Average Rating:** {summary['key_metrics']['average_rating']}
- **Positive Sentiment Rate:** {summary['key_metrics']['positive_sentiment_rate']}
- **Trend:** {summary['key_metrics']['response_rate_trend'].upper()}

### Financial Impact Assessment
- **Revenue at Risk:** {summary['financial_impact']['estimated_revenue_at_risk']}
- **Improvement Opportunity:** {summary['financial_impact']['improvement_opportunity']}
- **Expected ROI:** {summary['financial_impact']['expected_roi']}
- **Payback Period:** {summary['financial_impact']['payback_period']}

## Key Insights
"""
        for insight in summary['top_insights']:
            md += f"- {insight}\n"
        
        md += "\n## Strategic Recommendations\n\n"
        
        for i, rec in enumerate(summary['recommendations'], 1):
            md += f"""### {i}. {rec['category']} [{rec['priority']}]
**Issue:** {rec['issue']}  
**Recommendation:** {rec['recommendation']}  
**Expected Impact:** {rec['expected_impact']}  
**Timeline:** {rec['timeline']}

"""
        
        md += "## Next Steps\n"
        for step in summary['next_steps']:
            md += f"- {step}\n"
        
        return md


def main():
    """Generate business insights and recommendations"""
    # Load data
    try:
        df = pd.read_csv("data/processed/sentiment_predictions.csv")
        with open("models/topics/topic_analysis_report.pkl", 'rb') as f:
            topic_report = pickle.load(f)
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Generate insights
    generator = BusinessInsightsGenerator()
    
    # Generate executive summary
    summary = generator.generate_executive_summary(df, topic_report)
    
    # Export report
    generator.export_report(summary)
    
    # Print summary
    print("\n=== EXECUTIVE SUMMARY ===")
    print(f"\nAnalysis Period: {summary['analysis_period']}")
    print("\nKey Metrics:")
    for metric, value in summary['key_metrics'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print("\nFinancial Impact:")
    for metric, value in summary['financial_impact'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value}")
    
    print("\nTop Insights:")
    for insight in summary['top_insights']:
        print(f"  {insight}")
    
    print("\nTop Recommendations:")
    for rec in summary['recommendations'][:3]:
        print(f"\n  [{rec['priority']}] {rec['category']}")
        print(f"  Issue: {rec['issue']}")
        print(f"  Action: {rec['recommendation']}")
        print(f"  Impact: {rec['expected_impact']}")


if __name__ == "__main__":
    main()