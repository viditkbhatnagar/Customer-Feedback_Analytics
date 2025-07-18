import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx
import pickle
import os

def show_topic_insights(df, topic_report):
    """Display topic modeling insights and analysis"""
    st.header("💡 Topic Insights & Analysis")
    
    if topic_report is None:
        st.warning("⚠️ No topic analysis available. Please run topic extraction first.")
        if st.button("Run Topic Analysis Now"):
            st.info("Please run: `python src/models/topic_extractor.py`")
        return
    
    # Topic Model Results
    st.subheader("🔍 Topic Model Results")
    
    # Show available models
    available_models = []
    if topic_report.get('lda_results'):
        available_models.append('LDA')
    if topic_report.get('clustering_results'):
        available_models.append('Clustering')
    if topic_report.get('bertopic_results'):
        available_models.append('BERTopic')
    
    if available_models:
        selected_model = st.selectbox("Select Topic Model", available_models)
        
        # Get model results
        if selected_model == 'LDA':
            model_results = topic_report['lda_results']
        elif selected_model == 'Clustering':
            model_results = topic_report['clustering_results']
        else:
            model_results = topic_report['bertopic_results']
        
        # Display model metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Topics", model_results['num_topics'])
        
        with col2:
            if 'coherence_score' in model_results:
                st.metric("Coherence Score", f"{model_results['coherence_score']:.3f}")
            elif 'perplexity' in model_results:
                st.metric("Perplexity", f"{model_results['perplexity']:.1f}")
        
        with col3:
            st.metric("Model Type", selected_model)
        
        # Topic visualization
        st.subheader(f"📊 {selected_model} Topics")
        
        topics = model_results['top_topics']
        
        # Topic selection
        topic_names = [f"Topic {topic['topic_id']}: {', '.join(topic['words'][:3])}" 
                      for topic in topics]
        selected_topic_idx = st.selectbox("Select Topic to Explore", range(len(topics)), 
                                         format_func=lambda x: topic_names[x])
        
        selected_topic = topics[selected_topic_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Topic words bar chart
            words = selected_topic['words'][:10]
            scores = selected_topic['scores'][:10]
            
            fig_words = px.bar(
                x=scores,
                y=words,
                orientation='h',
                title=f"Top Words in Topic {selected_topic['topic_id']}",
                labels={'x': 'Score', 'y': 'Words'}
            )
            fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            # Topic word cloud
            word_freq = dict(selected_topic['word_scores'][:20])
            if word_freq:
                wordcloud = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(word_freq)
                
                fig_cloud, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f'Topic {selected_topic["topic_id"]} Word Cloud')
                st.pyplot(fig_cloud)
        
        # All topics overview
        st.subheader("🗂️ All Topics Overview")
        
        topics_df = pd.DataFrame([
            {
                'Topic ID': topic['topic_id'],
                'Top Words': ', '.join(topic['words'][:5]),
                'Word Count': len(topic['words'])
            }
            for topic in topics
        ])
        
        st.dataframe(topics_df, use_container_width=True)
    
    # Category Analysis
    st.subheader("📂 Category-based Topic Analysis")
    
    if 'category_analysis' in topic_report:
        category_analysis = topic_report['category_analysis']
        
        # Category selection
        categories = list(category_analysis.keys())
        selected_category = st.selectbox("Select Category for Analysis", categories)
        
        if selected_category in category_analysis:
            cat_data = category_analysis[selected_category]
            
            # Category metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Reviews", cat_data['total_reviews'])
            
            with col2:
                sentiment_dist = cat_data['sentiment_distribution']
                positive_pct = sentiment_dist.get('positive', 0) / cat_data['total_reviews'] * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            with col3:
                negative_pct = sentiment_dist.get('negative', 0) / cat_data['total_reviews'] * 100
                st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
            
            # Keywords analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔑 Top Keywords**")
                if 'tfidf_keywords' in cat_data and cat_data['tfidf_keywords']:
                    keywords_df = pd.DataFrame(
                        list(cat_data['tfidf_keywords'].items()),
                        columns=['Keyword', 'Score']
                    ).head(10)
                    st.dataframe(keywords_df)
                else:
                    st.write("No keywords available")
            
            with col2:
                st.write("**😊 Positive Keywords**")
                if cat_data['positive_keywords']:
                    pos_keywords_df = pd.DataFrame(
                        list(cat_data['positive_keywords'].items()),
                        columns=['Keyword', 'Score']
                    ).head(10)
                    st.dataframe(pos_keywords_df)
                else:
                    st.write("No positive keywords available")
            
            # Negative keywords
            if cat_data['negative_keywords']:
                st.write("**😞 Negative Keywords (Issues to Address)**")
                neg_keywords_df = pd.DataFrame(
                    list(cat_data['negative_keywords'].items()),
                    columns=['Keyword', 'Score']
                ).head(10)
                st.dataframe(neg_keywords_df)
                
                # Visualization of negative keywords
                fig_neg = px.bar(
                    neg_keywords_df,
                    x='Score',
                    y='Keyword',
                    orientation='h',
                    title=f"Main Issues in {selected_category}",
                    color='Score',
                    color_continuous_scale='Reds'
                )
                fig_neg.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_neg, use_container_width=True)
    
    # Trending Topics
    st.subheader("📈 Trending Topics")
    
    if 'trending_topics' in topic_report and topic_report['trending_topics']:
        trending = topic_report['trending_topics']
        
        # Trending topics chart
        trending_df = pd.DataFrame([
            {
                'Topic': topic,
                'Trend Score': data['trend_score'],
                'Percentage Change': data['percentage_change'],
                'Recent Count': data.get('recent_count', 0),
                'Older Count': data.get('older_count', 0)
            }
            for topic, data in trending.items()
        ]).head(10)
        
        fig_trending = px.bar(
            trending_df,
            x='Percentage Change',
            y='Topic',
            orientation='h',
            title="Trending Topics (Last 30 Days)",
            labels={'Percentage Change': 'Percentage Change (%)', 'Topic': 'Topic'},
            color='Percentage Change',
            color_continuous_scale='RdYlGn'
        )
        fig_trending.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_trending, use_container_width=True)
        
        # Trending topics table
        st.write("**📊 Trending Topics Details**")
        st.dataframe(trending_df, use_container_width=True)
    else:
        st.info("No trending topics identified in the current dataset.")
    
    # Cross-Category Analysis
    st.subheader("🔄 Cross-Category Topic Analysis")
    
    if 'category_analysis' in topic_report:
        # Compare categories
        category_comparison = []
        
        for category, data in topic_report['category_analysis'].items():
            sentiment_dist = data['sentiment_distribution']
            total = data['total_reviews']
            
            category_comparison.append({
                'Category': category,
                'Total Reviews': total,
                'Positive %': sentiment_dist.get('positive', 0) / total * 100,
                'Negative %': sentiment_dist.get('negative', 0) / total * 100,
                'Neutral %': sentiment_dist.get('neutral', 0) / total * 100
            })
        
        comparison_df = pd.DataFrame(category_comparison)
        
        # Category comparison chart
        fig_comparison = px.bar(
            comparison_df,
            x='Category',
            y=['Positive %', 'Negative %', 'Neutral %'],
            title="Sentiment Distribution Across Categories",
            labels={'value': 'Percentage', 'Category': 'Category'},
            color_discrete_map={
                'Positive %': '#2E8B57',
                'Negative %': '#DC143C',
                'Neutral %': '#FFD700'
            }
        )
        fig_comparison.update_layout(barmode='stack')
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Category ranking
        st.write("**🏆 Category Performance Ranking**")
        ranking_df = comparison_df.sort_values('Positive %', ascending=False)
        st.dataframe(ranking_df, use_container_width=True)
    
    # Key Insights
    st.subheader("🎯 Key Insights")
    
    if 'insights' in topic_report:
        insights = topic_report['insights']
        
        for insight in insights:
            if 'ALERT' in insight:
                st.error(insight)
            elif 'complaint' in insight.lower():
                st.warning(insight)
            elif 'trending' in insight.lower():
                st.info(insight)
            else:
                st.success(insight)
    
    # Action Items
    st.subheader("📋 Recommended Actions")
    
    action_items = []
    
    # Based on trending topics
    if 'trending_topics' in topic_report and topic_report['trending_topics']:
        top_trending = list(topic_report['trending_topics'].keys())[:3]
        action_items.append(f"🔍 **Investigate trending topics**: {', '.join(top_trending)}")
    
    # Based on category analysis
    if 'category_analysis' in topic_report:
        worst_categories = []
        for category, data in topic_report['category_analysis'].items():
            sentiment_dist = data['sentiment_distribution']
            negative_pct = sentiment_dist.get('negative', 0) / data['total_reviews'] * 100
            if negative_pct > 30:
                worst_categories.append(category)
        
        if worst_categories:
            action_items.append(f"⚠️ **Immediate attention needed**: {', '.join(worst_categories)} categories")
    
    # General recommendations
    action_items.extend([
        "📊 **Regular monitoring**: Set up automated alerts for sentiment changes",
        "🔄 **Feedback loop**: Implement systematic response to negative feedback",
        "📈 **Track improvements**: Monitor topic trends after implementing changes",
        "🎯 **Focus areas**: Prioritize categories with highest negative sentiment"
    ])
    
    for action in action_items:
        st.info(action)
    
    # Export functionality
    st.subheader("📥 Export Topic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Topic Summary"):
            if topic_report:
                st.json({
                    'total_topics': len(topics) if 'topics' in locals() else 0,
                    'categories_analyzed': len(topic_report.get('category_analysis', {})),
                    'trending_topics_count': len(topic_report.get('trending_topics', {})),
                    'key_insights': topic_report.get('insights', [])
                })
    
    with col2:
        if st.button("Export Trending Topics"):
            if 'trending_topics' in topic_report:
                st.dataframe(trending_df)
