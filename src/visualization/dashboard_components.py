"""
Dashboard Components Module for Customer Feedback Analytics
Reusable UI components for Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
from datetime import datetime, timedelta

class DashboardComponents:
    """Reusable components for the Streamlit dashboard"""
    
    def __init__(self):
        """Initialize dashboard components"""
        self.init_custom_css()
    
    @staticmethod
    def init_custom_css():
        """Initialize custom CSS styling"""
        st.markdown("""
        <style>
        /* Import custom CSS from style.css */
        .metric-container {
            background: linear-gradient(135deg, #ffffff 0%, #f0f2f6 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        .alert-box {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border-left-color: #ffc107;
            color: #856404;
        }
        
        .alert-success {
            background-color: #d4edda;
            border-left-color: #28a745;
            color: #155724;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }
        
        .info-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .recommendation-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid;
            margin-bottom: 1rem;
        }
        
        .priority-high {
            border-left-color: #E74C3C;
        }
        
        .priority-medium {
            border-left-color: #F39C12;
        }
        
        .priority-low {
            border-left-color: #95A5A6;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_metric_row(metrics: List[Dict[str, Any]], columns: Optional[int] = None):
        """Create a row of metric cards"""
        if columns is None:
            columns = len(metrics)
        
        cols = st.columns(columns)
        
        for i, metric in enumerate(metrics):
            with cols[i % columns]:
                DashboardComponents.create_metric_card(
                    title=metric.get('title', ''),
                    value=metric.get('value', ''),
                    delta=metric.get('delta'),
                    delta_color=metric.get('delta_color', 'normal'),
                    help_text=metric.get('help')
                )
    
    @staticmethod
    def create_metric_card(title: str, value: str, 
                          delta: Optional[str] = None,
                          delta_color: str = "normal",
                          help_text: Optional[str] = None):
        """Create a single metric card"""
        if help_text:
            st.metric(label=title, value=value, delta=delta, 
                     delta_color=delta_color, help=help_text)
        else:
            st.metric(label=title, value=value, delta=delta, 
                     delta_color=delta_color)
    
    @staticmethod
    def create_alert_box(message: str, alert_type: str = "info"):
        """Create an alert/notification box"""
        alert_classes = {
            'warning': 'alert-warning',
            'success': 'alert-success',
            'danger': 'alert-danger',
            'info': 'alert-info'
        }
        
        icons = {
            'warning': '⚠️',
            'success': '✅',
            'danger': '❌',
            'info': 'ℹ️'
        }
        
        alert_class = alert_classes.get(alert_type, 'alert-info')
        icon = icons.get(alert_type, 'ℹ️')
        
        st.markdown(f"""
        <div class="alert-box {alert_class}">
            <strong>{icon} {message}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_info_card(title: str, content: str, icon: Optional[str] = None):
        """Create an information card"""
        icon_html = f"{icon} " if icon else ""
        
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin-top: 0; color: #333;">{icon_html}{title}</h4>
            <p style="margin-bottom: 0; color: #666;">{content}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_recommendation_card(recommendation: Dict[str, Any]):
        """Create a recommendation card"""
        priority_class = f"priority-{recommendation.get('priority', 'medium').lower()}"
        
        st.markdown(f"""
        <div class="recommendation-card {priority_class}">
            <h4 style="margin: 0; color: #333;">
                {recommendation.get('icon', '💡')} {recommendation.get('category', 'Recommendation')} - 
                {recommendation.get('priority', 'MEDIUM')} Priority
            </h4>
            <p style="margin: 0.5rem 0;"><strong>Issue:</strong> {recommendation.get('issue', '')}</p>
            <p style="margin: 0.5rem 0;"><strong>Action:</strong> {recommendation.get('action', '')}</p>
            <p style="margin: 0.5rem 0;"><strong>Expected Impact:</strong> {recommendation.get('impact', '')}</p>
            <p style="margin: 0.5rem 0;"><strong>Timeline:</strong> {recommendation.get('timeline', '')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_filter_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
        """Create standardized filter sidebar"""
        st.sidebar.header("🔍 Filters")
        
        filters = {}
        
        # Date range filter
        if 'review_date' in df.columns:
            date_range = st.sidebar.date_input(
                "📅 Date Range",
                value=(df['review_date'].min(), df['review_date'].max()),
                min_value=df['review_date'].min(),
                max_value=df['review_date'].max(),
                key="date_filter"
            )
            filters['date_range'] = date_range
        
        # Category filter
        if 'category' in df.columns:
            categories = st.sidebar.multiselect(
                "🏷️ Categories",
                options=sorted(df['category'].unique()),
                default=sorted(df['category'].unique()),
                key="category_filter"
            )
            filters['categories'] = categories
        
        # Sentiment filter
        if 'predicted_sentiment' in df.columns:
            sentiments = st.sidebar.multiselect(
                "😊 Sentiments",
                options=sorted(df['predicted_sentiment'].unique()),
                default=sorted(df['predicted_sentiment'].unique()),
                key="sentiment_filter"
            )
            filters['sentiments'] = sentiments
        
        # Rating filter
        if 'rating' in df.columns:
            rating_range = st.sidebar.slider(
                "⭐ Rating Range",
                min_value=int(df['rating'].min()),
                max_value=int(df['rating'].max()),
                value=(int(df['rating'].min()), int(df['rating'].max())),
                key="rating_filter"
            )
            filters['rating_range'] = rating_range
        
        # Confidence threshold
        if 'confidence_score' in df.columns:
            confidence_threshold = st.sidebar.slider(
                "🎯 Min Confidence Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="confidence_filter"
            )
            filters['confidence_threshold'] = confidence_threshold
        
        # Verified purchase filter
        if 'verified_purchase' in df.columns:
            verified_only = st.sidebar.checkbox(
                "✅ Verified Purchases Only",
                value=False,
                key="verified_filter"
            )
            filters['verified_only'] = verified_only
        
        return filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Date filter
        if 'date_range' in filters and 'review_date' in df.columns:
            date_range = filters['date_range']
            if len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['review_date'].dt.date >= date_range[0]) &
                    (filtered_df['review_date'].dt.date <= date_range[1])
                ]
        
        # Category filter
        if 'categories' in filters and 'category' in df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
        
        # Sentiment filter
        if 'sentiments' in filters and 'predicted_sentiment' in df.columns:
            filtered_df = filtered_df[filtered_df['predicted_sentiment'].isin(filters['sentiments'])]
        
        # Rating filter
        if 'rating_range' in filters and 'rating' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['rating'] >= filters['rating_range'][0]) &
                (filtered_df['rating'] <= filters['rating_range'][1])
            ]
        
        # Confidence filter
        if 'confidence_threshold' in filters and 'confidence_score' in df.columns:
            filtered_df = filtered_df[filtered_df['confidence_score'] >= filters['confidence_threshold']]
        
        # Verified purchase filter
        if filters.get('verified_only', False) and 'verified_purchase' in df.columns:
            filtered_df = filtered_df[filtered_df['verified_purchase'] == True]
        
        return filtered_df
    
    @staticmethod
    def create_download_button(df: pd.DataFrame, filename: str = "export.csv", 
                             button_text: str = "📥 Download Data"):
        """Create a download button for dataframe"""
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label=button_text,
            data=csv,
            file_name=f"{filename}_{timestamp}.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def create_progress_indicator(current: int, total: int, label: str = "Progress"):
        """Create a progress indicator"""
        progress = current / total if total > 0 else 0
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(progress)
        with col2:
            st.write(f"{current}/{total}")
        
        st.caption(f"{label}: {progress*100:.1f}%")
    
    @staticmethod
    def create_expandable_review(review: pd.Series, show_confidence: bool = True):
        """Create an expandable review display"""
        sentiment_emojis = {
            'positive': '🟢',
            'negative': '🔴',
            'neutral': '🟡'
        }
        
        sentiment = review.get('predicted_sentiment', 'unknown')
        emoji = sentiment_emojis.get(sentiment, '⚪')
        rating = int(review.get('rating', 0))
        stars = '⭐' * rating
        
        title = f"{emoji} {stars} | {review.get('product_name', 'Product')} | {sentiment.upper()}"
        
        if show_confidence and 'confidence_score' in review:
            title += f" ({review['confidence_score']:.0%} confidence)"
        
        with st.expander(title):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Review:** {review.get('review_text', '')}")
                st.write(f"**Category:** {review.get('category', '')}")
                st.write(f"**Date:** {review.get('review_date', '')}")
            
            with col2:
                if review.get('verified_purchase', False):
                    st.write("✅ Verified Purchase")
                if 'helpful_count' in review:
                    st.write(f"👍 {review['helpful_count']} found helpful")
    
    @staticmethod
    def create_kpi_dashboard(kpis: Dict[str, Dict[str, Any]]):
        """Create a KPI dashboard section"""
        cols = st.columns(len(kpis))
        
        for col, (kpi_name, kpi_data) in zip(cols, kpis.items()):
            with col:
                value = kpi_data.get('value', 'N/A')
                target = kpi_data.get('target', '')
                status = kpi_data.get('status', '')
                trend = kpi_data.get('trend', '')
                
                st.markdown(f"""
                <div class="metric-container" style="text-align: center;">
                    <h5 style="color: #666; margin: 0;">{kpi_name}</h5>
                    <h2 style="color: #333; margin: 0.5rem 0;">{value}</h2>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Target: {target}</p>
                    <p style="font-size: 1.5rem; margin: 0;">{status} {trend}</p>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_comparison_table(data: pd.DataFrame, title: str = "Comparison Table",
                               highlight_best: bool = True):
        """Create a formatted comparison table"""
        st.subheader(title)
        
        if highlight_best:
            # Highlight best values in each numeric column
            def highlight_max(s):
                if s.dtype in ['float64', 'int64']:
                    is_max = s == s.max()
                    return ['background-color: #d4edda' if v else '' for v in is_max]
                return ['' for _ in s]
            
            styled_df = data.style.apply(highlight_max)
        else:
            styled_df = data.style
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
    
    @staticmethod
    def create_insights_section(insights: List[str], title: str = "💡 Key Insights"):
        """Create an insights section"""
        st.subheader(title)
        
        for insight in insights:
            if insight.startswith("⚠️"):
                DashboardComponents.create_alert_box(insight[2:], "warning")
            elif insight.startswith("✅"):
                DashboardComponents.create_alert_box(insight[2:], "success")
            elif insight.startswith("❌"):
                DashboardComponents.create_alert_box(insight[2:], "danger")
            else:
                st.info(insight)


def main():
    """Example usage of dashboard components"""
    # This would typically be imported and used in the dashboard pages
    pass


if __name__ == "__main__":
    main()