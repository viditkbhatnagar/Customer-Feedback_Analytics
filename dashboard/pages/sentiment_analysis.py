"""
Sentiment Analysis Page for Customer Feedback Dashboard
Provides detailed sentiment analysis views and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta
import pickle
import yaml

def load_config():
    """Load configuration"""
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def create_sentiment_gauge(df):
    """Create a gauge chart for overall sentiment score"""
    positive_ratio = (df['predicted_sentiment'] == 'positive').sum() / len(df)
    sentiment_score = positive_ratio * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Customer Sentiment Score", 'font': {'size': 24}},
        delta = {'reference': 75, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#E74C3C'},
                {'range': [50, 75], 'color': '#F39C12'},
                {'range': [75, 100], 'color': '#2ECC71'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_sentiment_distribution_donut(df):
    """Create an enhanced donut chart for sentiment distribution"""
    sentiment_counts = df['predicted_sentiment'].value_counts()
    
    colors = {
        'positive': '#2ECC71',
        'negative': '#E74C3C', 
        'neutral': '#95A5A6'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.6,
        marker_colors=[colors.get(x, '#333') for x in sentiment_counts.index],
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Add center text
    total_reviews = len(df)
    fig.add_annotation(
        text=f'<b>{total_reviews:,}</b><br>Total Reviews',
        x=0.5, y=0.5,
        font=dict(size=20),
        showarrow=False
    )
    
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=True,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_sentiment_by_rating_sunburst(df):
    """Create sunburst chart showing sentiment breakdown by rating"""
    # Prepare data
    sunburst_data = []
    
    for rating in sorted(df['rating'].unique()):
        rating_df = df[df['rating'] == rating]
        sunburst_data.append({
            'labels': f'Rating {rating}',
            'parents': '',
            'values': len(rating_df)
        })
        
        for sentiment in ['positive', 'negative', 'neutral']:
            count = len(rating_df[rating_df['predicted_sentiment'] == sentiment])
            if count > 0:
                sunburst_data.append({
                    'labels': sentiment.capitalize(),
                    'parents': f'Rating {rating}',
                    'values': count
                })
    
    sunburst_df = pd.DataFrame(sunburst_data)
    
    fig = px.sunburst(
        sunburst_df,
        names='labels',
        parents='parents',
        values='values',
        title='Sentiment Distribution by Rating',
        color='labels',
        color_discrete_map={
            'Positive': '#2ECC71',
            'Negative': '#E74C3C',
            'Neutral': '#95A5A6'
        }
    )
    
    fig.update_layout(height=500)
    return fig

def create_confidence_distribution(df):
    """Create confidence score distribution by sentiment"""
    fig = go.Figure()
    
    colors = {
        'positive': '#2ECC71',
        'negative': '#E74C3C',
        'neutral': '#95A5A6'
    }
    
    for sentiment in ['positive', 'negative', 'neutral']:
        data = df[df['predicted_sentiment'] == sentiment]['confidence_score']
        
        fig.add_trace(go.Violin(
            y=data,
            name=sentiment.capitalize(),
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[sentiment],
            opacity=0.6,
            x0=sentiment
        ))
    
    fig.update_layout(
        title="Model Confidence Distribution by Sentiment",
        yaxis_title="Confidence Score",
        showlegend=False,
        height=400
    )
    
    return fig

def create_sentiment_heatmap_by_time(df):
    """Create heatmap showing sentiment patterns by time"""
    # Extract time features
    df['hour'] = pd.to_datetime(df['review_date']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['review_date']).dt.dayofweek
    
    # Calculate positive sentiment rate by hour and day
    heatmap_data = df.groupby(['dayofweek', 'hour'])['predicted_sentiment'].apply(
        lambda x: (x == 'positive').sum() / len(x) * 100
    ).reset_index()
    
    # Pivot for heatmap
    heatmap_pivot = heatmap_data.pivot(index='dayofweek', columns='hour', values='predicted_sentiment')
    
    # Day names
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=[day_names[i] for i in heatmap_pivot.index],
        colorscale='RdYlGn',
        text=np.round(heatmap_pivot.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Positive %")
    ))
    
    fig.update_layout(
        title="Sentiment Patterns by Day and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    
    return fig

def create_sentiment_word_clouds(df):
    """Create word clouds for each sentiment"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors_map = {
        'positive': 'Greens',
        'negative': 'Reds',
        'neutral': 'Greys'
    }
    
    for idx, (sentiment, ax) in enumerate(zip(['positive', 'negative', 'neutral'], axes)):
        text = ' '.join(df[df['predicted_sentiment'] == sentiment]['cleaned_text'].dropna().astype(str))
        
        if text.strip():
            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap=colors_map[sentiment],
                max_words=50,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(text)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'{sentiment.capitalize()} Reviews', fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(f'{sentiment.capitalize()} Reviews', fontsize=16, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_misclassification_analysis(df):
    """Analyze potential misclassifications"""
    # Find reviews where rating and sentiment might not align
    misaligned = df[
        ((df['rating'] >= 4) & (df['predicted_sentiment'] == 'negative')) |
        ((df['rating'] <= 2) & (df['predicted_sentiment'] == 'positive'))
    ].copy()
    
    if len(misaligned) > 0:
        fig = px.scatter(
            misaligned,
            x='rating',
            y='confidence_score',
            color='predicted_sentiment',
            size='word_count',
            hover_data=['review_text'],
            title=f"Potential Misclassifications ({len(misaligned)} reviews)",
            labels={'confidence_score': 'Model Confidence', 'rating': 'User Rating'},
            color_discrete_map={
                'positive': '#2ECC71',
                'negative': '#E74C3C',
                'neutral': '#95A5A6'
            }
        )
        
        fig.update_layout(height=400)
        return fig
    else:
        return None

# ADD THIS TO THE END OF YOUR EXISTING dashboard/pages/sentiment_analysis.py FILE
# Replace the existing render_sentiment_analysis_page function with this:

def render_sentiment_analysis_page(df):
    """Main function to render the sentiment analysis page"""
    st.header("🎭 Sentiment Analysis Deep Dive")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        positive_rate = (df['predicted_sentiment'] == 'positive').sum() / len(df) * 100
        st.metric(
            "Positive Rate",
            f"{positive_rate:.1f}%",
            delta=f"{positive_rate - 60:.1f}%",
            delta_color="normal"
        )
    
    with col2:
        avg_confidence = df['confidence_score'].mean()
        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.2%}",
            help="Average model confidence across all predictions"
        )
    
    with col3:
        high_confidence_negative = len(
            df[(df['predicted_sentiment'] == 'negative') & (df['confidence_score'] > 0.8)]
        )
        st.metric(
            "High-Confidence Negative",
            high_confidence_negative,
            help="Negative reviews with >80% confidence"
        )
    
    with col4:
        sentiment_rating_correlation = df['rating'].corr(
            df['predicted_sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
        )
        st.metric(
            "Rating-Sentiment Correlation",
            f"{sentiment_rating_correlation:.2f}",
            help="How well ratings align with predicted sentiment"
        )
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🕐 Temporal Analysis", 
        "💭 Word Analysis",
        "🔍 Quality Check"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment gauge
            fig = create_sentiment_gauge(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment distribution donut
            fig = create_sentiment_distribution_donut(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = create_confidence_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment by rating sunburst
            fig = create_sentiment_by_rating_sunburst(df)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Time-based analysis
        st.subheader("Temporal Sentiment Patterns")
        
        # Sentiment heatmap
        fig = create_sentiment_heatmap_by_time(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment trend over time
        daily_sentiment = df.groupby([pd.to_datetime(df['review_date']).dt.date, 'predicted_sentiment']).size().reset_index(name='count')
        daily_sentiment['review_date'] = pd.to_datetime(daily_sentiment['review_date'])
        
        fig = px.line(
            daily_sentiment,
            x='review_date',
            y='count',
            color='predicted_sentiment',
            title="Daily Sentiment Volume Trends",
            color_discrete_map={
                'positive': '#2ECC71',
                'negative': '#E74C3C',
                'neutral': '#95A5A6'
            }
        )
        
        fig.update_layout(
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Number of Reviews"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Word Cloud Analysis by Sentiment")
        
        # Generate word clouds
        fig = create_sentiment_word_clouds(df)
        st.pyplot(fig)
        
        # Top keywords by sentiment
        st.subheader("Top Keywords by Sentiment")
        
        col1, col2, col3 = st.columns(3)
        
        for col, sentiment in zip([col1, col2, col3], ['positive', 'negative', 'neutral']):
            with col:
                st.write(f"**{sentiment.capitalize()} Keywords**")
                
                # Simple keyword extraction
                sentiment_text = ' '.join(df[df['predicted_sentiment'] == sentiment]['cleaned_text'].dropna())
                words = sentiment_text.split()
                word_freq = pd.Series(words).value_counts().head(10)
                
                for word, count in word_freq.items():
                    if len(word) > 3:  # Filter short words
                        st.write(f"• {word}: {count}")
    
    with tab4:
        st.subheader("Model Quality Analysis")
        
        # Misclassification analysis
        misclass_fig = create_misclassification_analysis(df)
        if misclass_fig:
            st.plotly_chart(misclass_fig, use_container_width=True)
        else:
            st.info("No obvious misclassifications detected!")
        
        # Confidence threshold analysis
        st.subheader("Prediction Confidence Analysis")
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        coverage = []
        accuracy_proxy = []
        
        for threshold in thresholds:
            high_conf = df[df['confidence_score'] >= threshold]
            coverage.append(len(high_conf) / len(df) * 100)
            
            # Proxy for accuracy: alignment between rating and sentiment
            if len(high_conf) > 0:
                aligned = (
                    ((high_conf['rating'] >= 4) & (high_conf['predicted_sentiment'] == 'positive')) |
                    ((high_conf['rating'] <= 2) & (high_conf['predicted_sentiment'] == 'negative')) |
                    ((high_conf['rating'] == 3) & (high_conf['predicted_sentiment'] == 'neutral'))
                ).sum() / len(high_conf) * 100
                accuracy_proxy.append(aligned)
            else:
                accuracy_proxy.append(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=coverage,
            mode='lines+markers',
            name='Coverage %',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=accuracy_proxy,
            mode='lines+markers',
            name='Alignment %',
            line=dict(color='green', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Model Performance vs Confidence Threshold",
            xaxis_title="Confidence Threshold",
            yaxis=dict(title="Coverage %", side='left'),
            yaxis2=dict(title="Rating-Sentiment Alignment %", overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample reviews by confidence level
        st.subheader("Sample Reviews by Confidence Level")
        
        conf_level = st.select_slider(
            "Select confidence range",
            options=['Very Low (0-20%)', 'Low (20-40%)', 'Medium (40-60%)', 
                    'High (60-80%)', 'Very High (80-100%)'],
            value='High (60-80%)'
        )
        
        # Map selection to confidence range
        conf_ranges = {
            'Very Low (0-20%)': (0, 0.2),
            'Low (20-40%)': (0.2, 0.4),
            'Medium (40-60%)': (0.4, 0.6),
            'High (60-80%)': (0.6, 0.8),
            'Very High (80-100%)': (0.8, 1.0)
        }
        
        low, high = conf_ranges[conf_level]
        sample_df = df[
            (df['confidence_score'] >= low) & 
            (df['confidence_score'] < high)
        ].sample(min(5, len(df)))
        
        for _, row in sample_df.iterrows():
            sentiment_color = {
                'positive': '🟢',
                'negative': '🔴',
                'neutral': '🟡'
            }
            
            with st.expander(
                f"{sentiment_color[row['predicted_sentiment']]} "
                f"{row['predicted_sentiment'].upper()} "
                f"(Confidence: {row['confidence_score']:.1%}) | "
                f"Rating: {'⭐' * int(row['rating'])}"
            ):
                st.write(f"**Review:** {row['review_text']}")
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Product:** {row['product_name']}")


# This function will be called from the main dashboard
def show():
    """Entry point for the sentiment analysis page"""
    # This function would be called from the main app
    # For now, we'll just define it
    pass