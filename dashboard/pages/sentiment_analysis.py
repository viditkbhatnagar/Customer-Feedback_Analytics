import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def show_sentiment_analysis(df):
    """Display sentiment analysis visualizations and insights"""
    st.header("😊 Sentiment Analysis Dashboard")
    
    # Overall sentiment distribution
    st.subheader("📊 Overall Sentiment Distribution")
    
    sentiment_counts = df['predicted_sentiment'].value_counts()
    sentiment_percentages = df['predicted_sentiment'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#FFD700'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Metrics
        st.metric("Positive Reviews", f"{sentiment_percentages['positive']:.1f}%")
        st.metric("Negative Reviews", f"{sentiment_percentages['negative']:.1f}%")
        st.metric("Neutral Reviews", f"{sentiment_percentages['neutral']:.1f}%")
        
        # Sentiment score
        sentiment_score = (sentiment_percentages['positive'] - sentiment_percentages['negative'])
        st.metric("Sentiment Score", f"{sentiment_score:.1f}", help="Positive % - Negative %")
    
    # Sentiment by Category
    st.subheader("📂 Sentiment by Category")
    
    category_sentiment = pd.crosstab(df['category'], df['predicted_sentiment'], normalize='index') * 100
    
    fig_category = px.bar(
        category_sentiment.reset_index(),
        x='category',
        y=['positive', 'negative', 'neutral'],
        title="Sentiment Distribution by Category",
        labels={'value': 'Percentage', 'category': 'Category'},
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#FFD700'
        }
    )
    fig_category.update_layout(barmode='stack')
    st.plotly_chart(fig_category, use_container_width=True)
    
    # Sentiment over time
    st.subheader("📈 Sentiment Trends Over Time")
    
    # Monthly sentiment trends
    df['year_month'] = df['review_date'].dt.to_period('M')
    monthly_sentiment = df.groupby(['year_month', 'predicted_sentiment']).size().unstack(fill_value=0)
    monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
    
    fig_trend = go.Figure()
    
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment in monthly_sentiment_pct.columns:
            fig_trend.add_trace(go.Scatter(
                x=monthly_sentiment_pct.index.astype(str),
                y=monthly_sentiment_pct[sentiment],
                mode='lines+markers',
                name=sentiment.capitalize(),
                line=dict(color={'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#FFD700'}[sentiment])
            ))
    
    fig_trend.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Month",
        yaxis_title="Percentage",
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Rating vs Sentiment Analysis
    st.subheader("⭐ Rating vs Sentiment Correlation")
    
    rating_sentiment = pd.crosstab(df['rating'], df['predicted_sentiment'], normalize='index') * 100
    
    fig_rating = px.imshow(
        rating_sentiment.T,
        title="Rating vs Sentiment Heatmap",
        labels={'x': 'Rating', 'y': 'Sentiment', 'color': 'Percentage'},
        aspect='auto'
    )
    st.plotly_chart(fig_rating, use_container_width=True)
    
    # Confidence Analysis
    st.subheader("🎯 Prediction Confidence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence distribution
        fig_conf = px.histogram(
            df,
            x='confidence_score',
            nbins=20,
            title="Prediction Confidence Distribution",
            labels={'confidence_score': 'Confidence Score', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Confidence by sentiment
        fig_conf_sent = px.box(
            df,
            x='predicted_sentiment',
            y='confidence_score',
            title="Confidence by Sentiment",
            color='predicted_sentiment',
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#FFD700'
            }
        )
        st.plotly_chart(fig_conf_sent, use_container_width=True)
    
    # Word Analysis
    st.subheader("💭 Word Analysis")
    
    # Most common words by sentiment
    sentiment_choice = st.selectbox("Select Sentiment for Word Analysis", ['positive', 'negative', 'neutral'])
    
    filtered_reviews = df[df['predicted_sentiment'] == sentiment_choice]['cleaned_text']
    
    if len(filtered_reviews) > 0:
        # Word frequency
        all_words = ' '.join(filtered_reviews).split()
        word_freq = Counter(all_words)
        common_words = word_freq.most_common(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Word frequency chart
            words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
            fig_words = px.bar(
                words_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title=f"Top 20 Words in {sentiment_choice.capitalize()} Reviews"
            )
            fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            # Word cloud
            if len(all_words) > 0:
                wordcloud = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    colormap={'positive': 'Greens', 'negative': 'Reds', 'neutral': 'Blues'}[sentiment_choice]
                ).generate(' '.join(all_words))
                
                fig_cloud, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'{sentiment_choice.capitalize()} Reviews Word Cloud')
                st.pyplot(fig_cloud)
    
    # Sample Reviews
    st.subheader("📝 Sample Reviews")
    
    sample_sentiment = st.selectbox("Select Sentiment for Sample Reviews", ['positive', 'negative', 'neutral'])
    sample_reviews = df[df['predicted_sentiment'] == sample_sentiment].sample(min(5, len(df)))
    
    for idx, row in sample_reviews.iterrows():
        with st.expander(f"Review {idx} - Rating: {row['rating']}/5 - Confidence: {row['confidence_score']:.2f}"):
            st.write(f"**Category**: {row['category']}")
            st.write(f"**Date**: {row['review_date'].strftime('%Y-%m-%d')}")
            st.write(f"**Review**: {row['review_text']}")
            st.write(f"**Predicted Sentiment**: {row['predicted_sentiment']}")
    
    # Sentiment Insights
    st.subheader("🔍 Key Insights")
    
    insights = []
    
    # Dominant sentiment
    dominant_sentiment = sentiment_counts.index[0]
    insights.append(f"📊 **Dominant Sentiment**: {dominant_sentiment.capitalize()} ({sentiment_percentages[dominant_sentiment]:.1f}%)")
    
    # Best and worst categories
    category_pos = df.groupby('category')['predicted_sentiment'].apply(lambda x: (x == 'positive').mean() * 100).sort_values(ascending=False)
    if len(category_pos) > 0:
        best_cat = category_pos.index[0]
        worst_cat = category_pos.index[-1]
        insights.append(f"🏆 **Best Category**: {best_cat} ({category_pos[best_cat]:.1f}% positive)")
        insights.append(f"⚠️ **Needs Attention**: {worst_cat} ({category_pos[worst_cat]:.1f}% positive)")
    
    # Confidence insights
    low_confidence = df[df['confidence_score'] < 0.7]
    if len(low_confidence) > 0:
        insights.append(f"🎯 **Low Confidence Predictions**: {len(low_confidence)} reviews ({len(low_confidence)/len(df)*100:.1f}%) have confidence < 70%")
    
    # Rating-sentiment mismatch
    mismatch = df[((df['rating'] >= 4) & (df['predicted_sentiment'] == 'negative')) | 
                  ((df['rating'] <= 2) & (df['predicted_sentiment'] == 'positive'))]
    if len(mismatch) > 0:
        insights.append(f"🤔 **Rating-Sentiment Mismatch**: {len(mismatch)} reviews show conflicting rating and sentiment")
    
    for insight in insights:
        st.info(insight)
    
    # Export options
    st.subheader("📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Sentiment Summary"):
            summary = {
                'Total Reviews': len(df),
                'Positive': f"{sentiment_percentages['positive']:.1f}%",
                'Negative': f"{sentiment_percentages['negative']:.1f}%",
                'Neutral': f"{sentiment_percentages['neutral']:.1f}%",
                'Sentiment Score': f"{sentiment_score:.1f}",
                'Average Confidence': f"{df['confidence_score'].mean():.2f}"
            }
            st.json(summary)
    
    with col2:
        if st.button("Export Low Confidence Reviews"):
            low_conf_reviews = df[df['confidence_score'] < 0.7][['review_text', 'predicted_sentiment', 'confidence_score']]
            st.dataframe(low_conf_reviews)
