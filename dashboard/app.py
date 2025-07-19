"""
Customer Feedback Analytics Dashboard v
Main Streamlit application for interactive analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import yaml
import pickle
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add pages directory to Python path
pages_path = os.path.join(os.path.dirname(__file__), 'pages')
if pages_path not in sys.path:
    sys.path.insert(0, pages_path)

# Page configuration
st.set_page_config(
    page_title="Customer Feedback Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    h3 {
        color: #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        # Load main dataset with predictions
        df = pd.read_csv("data/processed/sentiment_predictions.csv")
        df['review_date'] = pd.to_datetime(df['review_date'])
        
        # Load configuration
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Load topic analysis report
        with open("models/topics/topic_analysis_report.pkl", 'rb') as f:
            topic_report = pickle.load(f)
        
        return df, config, topic_report
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def create_sentiment_distribution_chart(df):
    """Create sentiment distribution pie chart"""
    sentiment_counts = df['predicted_sentiment'].value_counts()
    
    colors = {
        'positive': '#2ECC71',
        'negative': '#E74C3C',
        'neutral': '#95A5A6'
    }
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Sentiment Distribution",
        color_discrete_map=colors,
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    return fig

def create_sentiment_timeline(df):
    """Create sentiment trend over time"""
    # Aggregate by week
    df['week'] = df['review_date'].dt.to_period('W').dt.to_timestamp()
    
    weekly_sentiment = df.groupby(['week', 'predicted_sentiment']).size().reset_index(name='count')
    weekly_total = df.groupby('week').size().reset_index(name='total')
    
    weekly_sentiment = weekly_sentiment.merge(weekly_total, on='week')
    weekly_sentiment['percentage'] = (weekly_sentiment['count'] / weekly_sentiment['total']) * 100
    
    fig = px.line(
        weekly_sentiment,
        x='week',
        y='percentage',
        color='predicted_sentiment',
        title="Sentiment Trends Over Time",
        labels={'percentage': 'Percentage (%)', 'week': 'Week'},
        color_discrete_map={
            'positive': '#2ECC71',
            'negative': '#E74C3C',
            'neutral': '#95A5A6'
        }
    )
    
    fig.update_layout(
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Percentage of Reviews (%)"
    )
    
    return fig

def create_category_sentiment_heatmap(df):
    """Create heatmap of sentiment by category"""
    category_sentiment = pd.crosstab(
        df['category'],
        df['predicted_sentiment'],
        normalize='index'
    ) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=category_sentiment.values,
        x=category_sentiment.columns,
        y=category_sentiment.index,
        colorscale='RdYlGn',
        text=np.round(category_sentiment.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 12},
        hoverongaps=False,
        hovertemplate='Category: %{y}<br>Sentiment: %{x}<br>Percentage: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Sentiment Distribution by Product Category",
        xaxis_title="Sentiment",
        yaxis_title="Category",
        height=400
    )
    
    return fig

def create_rating_sentiment_comparison(df):
    """Create comparison between ratings and predicted sentiment"""
    comparison = pd.crosstab(df['rating'], df['predicted_sentiment'])
    
    fig = go.Figure()
    
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment in comparison.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=comparison.index,
                y=comparison[sentiment],
                marker_color={
                    'positive': '#2ECC71',
                    'negative': '#E74C3C',
                    'neutral': '#95A5A6'
                }[sentiment]
            ))
    
    fig.update_layout(
        title="Rating vs Predicted Sentiment Comparison",
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
        barmode='stack',
        hovermode='x unified'
    )
    
    return fig

def create_wordcloud(text_series, title="Word Cloud"):
    """Create word cloud visualization"""
    text = ' '.join(text_series.dropna().astype(str))
    
    if not text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    return fig

def display_insights(df, topic_report):
    """Display key business insights"""
    st.header("💡 Key Insights & Recommendations")
    
    # Create insight cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Customer satisfaction score
        positive_ratio = (df['predicted_sentiment'] == 'positive').sum() / len(df)
        satisfaction_score = positive_ratio * 100
        
        st.metric(
            "Customer Satisfaction Score",
            f"{satisfaction_score:.1f}%",
            delta=f"{satisfaction_score - 75:.1f}% vs target (75%)",
            delta_color="normal" if satisfaction_score >= 75 else "inverse"
        )
    
    with col2:
        # Average confidence score
        avg_confidence = df['confidence_score'].mean()
        st.metric(
            "Model Confidence",
            f"{avg_confidence:.2%}",
            help="Average confidence in sentiment predictions"
        )
    
    with col3:
        # Reviews needing attention
        urgent_reviews = df[
            (df['predicted_sentiment'] == 'negative') & 
            (df['confidence_score'] > 0.8)
        ]
        st.metric(
            "Reviews Needing Attention",
            len(urgent_reviews),
            help="High-confidence negative reviews"
        )
    
    # Business recommendations
    st.subheader("📋 Business Recommendations")
    
    # Find problematic categories
    category_sentiment = df.groupby('category')['predicted_sentiment'].apply(
        lambda x: (x == 'negative').sum() / len(x)
    ).sort_values(ascending=False)
    
    worst_categories = category_sentiment.head(3)
    
    recommendations = []
    
    for category, negative_ratio in worst_categories.items():
        if negative_ratio > 0.3:
            recommendations.append(
                f"**{category}**: {negative_ratio*100:.1f}% negative reviews - "
                f"Immediate quality review recommended"
            )
    
    # Add insights from topic report
    if 'insights' in topic_report:
        for insight in topic_report['insights']:
            recommendations.append(f"**Trending Alert**: {insight}")
    
    for rec in recommendations[:5]:
        st.warning(rec)

def main():
    """Main dashboard application"""
    st.title("🛍️ Customer Feedback Analytics Dashboard")
    st.markdown("**Real-time insights from customer reviews to drive business decisions**")
    
    # Load data
    df, config, topic_report = load_data()
    
    if df is None:
        st.error("Unable to load data. Please ensure all data files are generated.")
        st.info("Run: python run_pipeline.py")
        return
    
    # Sidebar navigation - MAIN NAVIGATION
    st.sidebar.title("🧭 Navigation")
    page_options = [
        "📊 Overview",
        "💼 Business Metrics", 
        "🎭 Sentiment Analysis",
        "🔍 Topic Insights"
    ]
    
    selected_page = st.sidebar.selectbox("Select Page", page_options)
    
    # Sidebar filters (only for Overview page)
    if selected_page == "📊 Overview":
        st.sidebar.header("Filters")
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['review_date'].min(), df['review_date'].max()),
            min_value=df['review_date'].min(),
            max_value=df['review_date'].max()
        )
        
        # Category filter
        categories = st.sidebar.multiselect(
            "Select Categories",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
        
        # Sentiment filter
        sentiments = st.sidebar.multiselect(
            "Select Sentiments",
            options=df['predicted_sentiment'].unique(),
            default=df['predicted_sentiment'].unique()
        )
        
        # Confidence threshold
        confidence_threshold = st.sidebar.slider(
            "Minimum Confidence Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Apply filters
        filtered_df = df[
            (df['review_date'].dt.date >= date_range[0]) &
            (df['review_date'].dt.date <= date_range[1]) &
            (df['category'].isin(categories)) &
            (df['predicted_sentiment'].isin(sentiments)) &
            (df['confidence_score'] >= confidence_threshold)
        ]
    else:
        filtered_df = df
    
    # PAGE ROUTING - Call appropriate page functions
    if selected_page == "📊 Overview":
        # Display overview metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", f"{len(filtered_df):,}")
        with col2:
            st.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f}")
        with col3:
            st.metric("Categories", len(filtered_df['category'].unique()))
        with col4:
            verified_pct = filtered_df['verified_purchase'].mean() * 100
            st.metric("Verified Purchases", f"{verified_pct:.1f}%")
        
        # Main content tabs for overview
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Sentiment Analysis",
            "🔍 Topic Insights",
            "📈 Trends & Patterns",
            "🔎 Review Explorer",
            "💡 Recommendations"
        ])
        
        with tab1:
            st.header("Sentiment Analysis Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_sentiment_distribution_chart(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_rating_sentiment_comparison(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment by category heatmap
            st.subheader("Sentiment Analysis by Category")
            fig = create_category_sentiment_heatmap(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Word clouds by sentiment
            st.subheader("Word Clouds by Sentiment")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_reviews = filtered_df[filtered_df['predicted_sentiment'] == 'positive']['cleaned_text']
                if len(positive_reviews) > 0:
                    fig = create_wordcloud(positive_reviews, "Positive Reviews")
                    if fig:
                        st.pyplot(fig)
            
            with col2:
                negative_reviews = filtered_df[filtered_df['predicted_sentiment'] == 'negative']['cleaned_text']
                if len(negative_reviews) > 0:
                    fig = create_wordcloud(negative_reviews, "Negative Reviews")
                    if fig:
                        st.pyplot(fig)
            
            with col3:
                neutral_reviews = filtered_df[filtered_df['predicted_sentiment'] == 'neutral']['cleaned_text']
                if len(neutral_reviews) > 0:
                    fig = create_wordcloud(neutral_reviews, "Neutral Reviews")
                    if fig:
                        st.pyplot(fig)
        
        with tab2:
            if topic_report and 'lda_results' in topic_report:
                st.header("🔍 Topic Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Topics (LDA Model)")
                    if topic_report['lda_results']:
                        for topic in topic_report['lda_results']['top_topics'][:5]:
                            words = ', '.join(topic['words'][:5])
                            st.write(f"**Topic {topic['topic_id']}:** {words}")
                            
                        st.metric(
                            "Model Coherence Score",
                            f"{topic_report['lda_results']['coherence_score']:.3f}",
                            help="Higher coherence indicates better topic quality"
                        )
                
                with col2:
                    st.subheader("Trending Topics (Last 30 Days)")
                    if topic_report.get('trending_topics'):
                        trending_df = pd.DataFrame([
                            {'Topic': topic, 'Change': data['percentage_change']}
                            for topic, data in list(topic_report['trending_topics'].items())[:10]
                        ])
                        
                        fig = px.bar(
                            trending_df,
                            x='Change',
                            y='Topic',
                            orientation='h',
                            title="Top Trending Topics",
                            labels={'Change': 'Percentage Change (%)'},
                            color='Change',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Topic analysis data not available")
        
        with tab3:
            st.header("Trends & Patterns Analysis")
            
            # Sentiment timeline
            fig = create_sentiment_timeline(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Reviews volume over time
            daily_reviews = filtered_df.groupby(filtered_df['review_date'].dt.date).size().reset_index(name='count')
            
            fig = px.line(
                daily_reviews,
                x='review_date',
                y='count',
                title="Daily Review Volume",
                labels={'review_date': 'Date', 'count': 'Number of Reviews'}
            )
            fig.add_trace(
                go.Scatter(
                    x=daily_reviews['review_date'],
                    y=daily_reviews['count'].rolling(7).mean(),
                    mode='lines',
                    name='7-day Moving Average',
                    line=dict(color='red', width=2)
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("Review Explorer")
            
            # Search functionality
            search_term = st.text_input("Search reviews", placeholder="Enter keywords to search...")
            
            # Advanced filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    options=['review_date', 'rating', 'confidence_score', 'helpful_count'],
                    index=0
                )
            
            with col2:
                sort_order = st.radio("Order", ["Descending", "Ascending"])
            
            with col3:
                num_reviews = st.number_input("Number of reviews to display", min_value=5, max_value=100, value=20)
            
            # Filter reviews
            display_df = filtered_df.copy()
            
            if search_term:
                display_df = display_df[
                    display_df['review_text'].str.contains(search_term, case=False, na=False) |
                    display_df['cleaned_text'].str.contains(search_term, case=False, na=False)
                ]
            
            # Sort
            display_df = display_df.sort_values(
                by=sort_by,
                ascending=(sort_order == "Ascending")
            ).head(num_reviews)
            
            # Display reviews
            for idx, row in display_df.iterrows():
                with st.expander(
                    f"{'⭐' * int(row['rating'])} | {row['product_name']} | "
                    f"{row['predicted_sentiment'].upper()} ({row['confidence_score']:.0%} confidence)"
                ):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Review:** {row['review_text']}")
                        st.write(f"**Category:** {row['category']}")
                        st.write(f"**Date:** {row['review_date'].strftime('%Y-%m-%d')}")
                    
                    with col2:
                        sentiment_color = {
                            'positive': '🟢',
                            'negative': '🔴',
                            'neutral': '🟡'
                        }
                        st.write(f"**Sentiment:** {sentiment_color.get(row['predicted_sentiment'], '⚪')}")
                        st.write(f"**Helpful:** {row['helpful_count']} votes")
                        if row['verified_purchase']:
                            st.write("✅ Verified Purchase")
        
        with tab5:
            display_insights(filtered_df, topic_report)
    
    elif selected_page == "💼 Business Metrics":
        try:
            from business_metrics import render_business_metrics_page
            render_business_metrics_page(filtered_df)
        except ImportError:
            st.error("Business Metrics page not available. Please check business_metrics.py file.")
    
    elif selected_page == "🎭 Sentiment Analysis":
        try:
            from sentiment_analysis import render_sentiment_analysis_page
            render_sentiment_analysis_page(filtered_df)
        except ImportError:
            st.error("Sentiment Analysis page not available. Please check sentiment_analysis.py file.")
    
    elif selected_page == "🔍 Topic Insights":
        try:
            from topic_insights import render_topic_insights_page
            render_topic_insights_page(filtered_df)
        except ImportError:
            st.error("Topic Insights page not available. Please check topic_insights.py file.")

if __name__ == "__main__":
    main()