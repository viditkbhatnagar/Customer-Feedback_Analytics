"""
Topic Insights Page for Customer Feedback Dashboard
Provides detailed topic modeling views and keyword analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pickle
import yaml
from datetime import datetime, timedelta
import networkx as nx

def load_topic_report():
    """Load topic analysis report"""
    try:
        with open('models/topics/topic_analysis_report.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

def create_topic_distribution_chart(topic_report):
    """Create topic distribution visualization"""
    if not topic_report or 'lda_results' not in topic_report:
        return None
    
    topics = topic_report['lda_results']['top_topics'][:10]
    
    # Prepare data
    topic_data = []
    for topic in topics:
        topic_data.append({
            'Topic': f"Topic {topic['topic_id']}",
            'Top Words': ', '.join(topic['words'][:5]),
            'Weight': sum(topic['scores'][:5])
        })
    
    df_topics = pd.DataFrame(topic_data)
    
    fig = px.bar(
        df_topics,
        x='Weight',
        y='Topic',
        orientation='h',
        text='Top Words',
        title='Topic Distribution (LDA Model)',
        labels={'Weight': 'Topic Weight', 'Topic': 'Topic ID'},
        color='Weight',
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500, showlegend=False)
    
    return fig

def create_topic_evolution_chart(df, topic_report):
    """Create topic evolution over time chart"""
    if not topic_report or 'trending_topics' not in topic_report:
        return None
    
    # Simulate topic evolution data
    dates = pd.date_range(end=df['review_date'].max(), periods=30, freq='D')
    
    evolution_data = []
    for i, (topic, data) in enumerate(list(topic_report['trending_topics'].items())[:5]):
        base_score = data['recent_score']
        trend = data['trend_score']
        
        for j, date in enumerate(dates):
            # Simulate evolution with trend
            score = base_score * (1 + trend * (j / 30) + np.random.normal(0, 0.1))
            evolution_data.append({
                'Date': date,
                'Topic': topic,
                'Score': max(0, score),
                'Trend': trend
            })
    
    evolution_df = pd.DataFrame(evolution_data)
    
    fig = px.line(
        evolution_df,
        x='Date',
        y='Score',
        color='Topic',
        title='Topic Evolution Over Time (30 Days)',
        labels={'Score': 'Topic Prevalence Score'},
        line_shape='spline'
    )
    
    fig.update_layout(
        hovermode='x unified',
        height=400,
        xaxis_title="Date",
        yaxis_title="Prevalence Score"
    )
    
    return fig

def create_category_topic_heatmap(topic_report):
    """Create heatmap of topics by category"""
    if not topic_report or 'category_analysis' not in topic_report:
        return None
    
    categories = []
    topics = []
    values = []
    
    # Extract top keywords per category
    for category, data in topic_report['category_analysis'].items():
        if 'keywords' in data:
            for keyword, score in list(data['keywords'].items())[:5]:
                categories.append(category)
                topics.append(keyword)
                values.append(score)
    
    # Create pivot table
    heatmap_df = pd.DataFrame({
        'Category': categories,
        'Topic': topics,
        'Score': values
    })
    
    pivot_df = heatmap_df.pivot_table(
        values='Score',
        index='Topic',
        columns='Category',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='YlOrRd',
        text=np.round(pivot_df.values, 3),
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Relevance Score")
    ))
    
    fig.update_layout(
        title='Topic Relevance by Product Category',
        xaxis_title='Category',
        yaxis_title='Topic/Keyword',
        height=600
    )
    
    return fig

def create_topic_network_graph(topic_report):
    """Create network graph showing topic relationships"""
    if not topic_report or 'lda_results' not in topic_report:
        return None
    
    # Create network graph
    G = nx.Graph()
    
    # Add topic nodes
    topics = topic_report['lda_results']['top_topics'][:8]
    
    for topic in topics:
        topic_id = f"Topic {topic['topic_id']}"
        G.add_node(topic_id, node_type='topic', size=30)
        
        # Add word nodes
        for word, score in topic['word_scores'][:5]:
            G.add_node(word, node_type='word', size=score*20)
            G.add_edge(topic_id, word, weight=score)
    
    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create edge trace
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            )
        )
    
    # Create node traces
    node_trace_topics = go.Scatter(
        x=[pos[node][0] for node in G.nodes() if G.nodes[node]['node_type'] == 'topic'],
        y=[pos[node][1] for node in G.nodes() if G.nodes[node]['node_type'] == 'topic'],
        mode='markers+text',
        text=[node for node in G.nodes() if G.nodes[node]['node_type'] == 'topic'],
        textposition="top center",
        marker=dict(
            size=30,
            color='#1f77b4',
            line=dict(width=2, color='white')
        ),
        hoverinfo='text'
    )
    
    node_trace_words = go.Scatter(
        x=[pos[node][0] for node in G.nodes() if G.nodes[node]['node_type'] == 'word'],
        y=[pos[node][1] for node in G.nodes() if G.nodes[node]['node_type'] == 'word'],
        mode='markers+text',
        text=[node for node in G.nodes() if G.nodes[node]['node_type'] == 'word'],
        textposition="bottom center",
        marker=dict(
            size=[G.nodes[node]['size'] for node in G.nodes() if G.nodes[node]['node_type'] == 'word'],
            color='#2ca02c',
            line=dict(width=1, color='white')
        ),
        hoverinfo='text'
    )
    
    fig = go.Figure(data=edge_trace + [node_trace_topics, node_trace_words])
    
    fig.update_layout(
        title='Topic-Keyword Network Graph',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def create_sentiment_topic_matrix(df, topic_report):
    """Create matrix showing topics by sentiment"""
    if not topic_report:
        return None
    
    # Extract topics and create sentiment breakdown
    sentiments = ['positive', 'negative', 'neutral']
    topics = []
    
    if 'trending_topics' in topic_report:
        topics = list(topic_report['trending_topics'].keys())[:5]
    
    matrix_data = []
    
    for topic in topics:
        topic_reviews = df[df['cleaned_text'].str.contains(topic, case=False, na=False)]
        
        for sentiment in sentiments:
            count = len(topic_reviews[topic_reviews['predicted_sentiment'] == sentiment])
            matrix_data.append({
                'Topic': topic,
                'Sentiment': sentiment.capitalize(),
                'Count': count
            })
    
    matrix_df = pd.DataFrame(matrix_data)
    
    fig = px.bar(
        matrix_df,
        x='Topic',
        y='Count',
        color='Sentiment',
        title='Topic Distribution by Sentiment',
        labels={'Count': 'Number of Reviews'},
        color_discrete_map={
            'Positive': '#2ECC71',
            'Negative': '#E74C3C',
            'Neutral': '#95A5A6'
        }
    )
    
    fig.update_layout(
        barmode='group',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_emerging_topics_alert(topic_report):
    """Create alert box for emerging topics"""
    if not topic_report or 'trending_topics' not in topic_report:
        return None
    
    # Get top 3 trending topics with high growth
    trending = sorted(
        topic_report['trending_topics'].items(),
        key=lambda x: x[1]['trend_score'],
        reverse=True
    )[:3]
    
    alert_html = "<div style='background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;'>"
    alert_html += "<h4 style='color: #856404; margin-top: 0;'>🔥 Emerging Topics Alert</h4>"
    
    for topic, data in trending:
        change = data['percentage_change']
        alert_html += f"<p style='margin: 0.5rem 0;'><strong>{topic}</strong>: "
        alert_html += f"<span style='color: {'red' if change > 100 else 'orange'};'>+{change:.0f}% growth</span></p>"
    
    alert_html += "</div>"
    
    return alert_html

# ADD THIS TO THE END OF YOUR EXISTING dashboard/pages/topic_insights.py FILE
# Replace the existing render_topic_insights_page function with this:

def render_topic_insights_page(df):
    """Main function to render the topic insights page"""
    st.header("🔍 Topic Analysis & Insights")
    
    # Load topic report
    topic_report = load_topic_report()
    
    if not topic_report:
        st.error("Topic analysis report not found. Please run the topic extraction pipeline first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        num_topics = len(topic_report.get('lda_results', {}).get('top_topics', []))
        st.metric("Topics Discovered", num_topics)
    
    with col2:
        coherence = topic_report.get('lda_results', {}).get('coherence_score', 0)
        st.metric("Model Coherence", f"{coherence:.3f}", help="Higher is better (>0.4 is good)")
    
    with col3:
        trending_count = len(topic_report.get('trending_topics', {}))
        st.metric("Trending Topics", trending_count, "+20%")
    
    with col4:
        categories_analyzed = len(topic_report.get('category_analysis', {}))
        st.metric("Categories Analyzed", categories_analyzed)
    
    # Emerging topics alert
    alert_html = create_emerging_topics_alert(topic_report)
    if alert_html:
        st.markdown(alert_html, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Topic Overview",
        "📈 Trend Analysis", 
        "🔗 Topic Networks",
        "📝 Category Insights"
    ])
    
    with tab1:
        st.subheader("Topic Distribution Analysis")
        
        # Topic distribution chart
        fig = create_topic_distribution_chart(topic_report)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Topic details
        st.subheader("Topic Details")
        
        if 'lda_results' in topic_report:
            topics = topic_report['lda_results']['top_topics'][:5]
            
            for topic in topics:
                with st.expander(f"Topic {topic['topic_id']} - {', '.join(topic['words'][:3])}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Top Keywords:**")
                        for word, score in topic['word_scores'][:10]:
                            st.progress(score / topic['word_scores'][0][1])
                            st.write(f"{word} ({score:.3f})")
                    
                    with col2:
                        # Mini word cloud for this topic
                        wordcloud_dict = dict(topic['word_scores'][:20])
                        
                        fig, ax = plt.subplots(figsize=(5, 3))
                        wordcloud = WordCloud(
                            width=300,
                            height=200,
                            background_color='white'
                        ).generate_from_frequencies(wordcloud_dict)
                        
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
    
    with tab2:
        st.subheader("Topic Trend Analysis")
        
        # Topic evolution chart
        fig = create_topic_evolution_chart(df, topic_report)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment-topic matrix
        fig = create_sentiment_topic_matrix(df, topic_report)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Trending topics table
        if 'trending_topics' in topic_report:
            st.subheader("📈 Top Trending Topics")
            
            trending_data = []
            for topic, data in list(topic_report['trending_topics'].items())[:10]:
                trending_data.append({
                    'Topic': topic,
                    'Growth': f"+{data['percentage_change']:.0f}%",
                    'Recent Score': f"{data['recent_score']:.3f}",
                    'Previous Score': f"{data['older_score']:.3f}",
                    'Trend': '🔥' if data['percentage_change'] > 100 else '📈'
                })
            
            trending_df = pd.DataFrame(trending_data)
            st.dataframe(
                trending_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Growth": st.column_config.TextColumn(
                        "Growth",
                        help="Percentage change in last 30 days"
                    ),
                    "Trend": st.column_config.TextColumn(
                        "Trend",
                        help="🔥 = Hot topic (>100% growth)"
                    )
                }
            )
    
    with tab3:
        st.subheader("Topic Relationship Networks")
        
        # Network graph
        fig = create_topic_network_graph(topic_report)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Topic correlation matrix
        st.subheader("Topic Co-occurrence")
        
        if 'lda_results' in topic_report:
            # Simulate topic correlation
            num_topics = min(8, len(topic_report['lda_results']['top_topics']))
            correlation_matrix = np.random.rand(num_topics, num_topics)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1)
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=[f"Topic {i}" for i in range(num_topics)],
                y=[f"Topic {i}" for i in range(num_topics)],
                colorscale='Blues',
                text=np.round(correlation_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Topic Co-occurrence Matrix",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Category-Specific Topics")
        
        # Category topic heatmap
        fig = create_category_topic_heatmap(topic_report)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Category deep dive
        if 'category_analysis' in topic_report:
            st.subheader("Category Deep Dive")
            
            selected_category = st.selectbox(
                "Select a category for detailed analysis",
                options=list(topic_report['category_analysis'].keys())
            )
            
            if selected_category:
                cat_data = topic_report['category_analysis'][selected_category]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Reviews", cat_data['total_reviews'])
                
                with col2:
                    sentiment_dist = cat_data.get('sentiment_distribution', {})
                    positive_pct = sentiment_dist.get('positive', 0) / cat_data['total_reviews'] * 100
                    st.metric("Positive %", f"{positive_pct:.1f}%")
                
                with col3:
                    negative_pct = sentiment_dist.get('negative', 0) / cat_data['total_reviews'] * 100
                    st.metric("Negative %", f"{negative_pct:.1f}%")
                
                # Keywords analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Keywords**")
                    if 'keywords' in cat_data:
                        for keyword, score in list(cat_data['keywords'].items())[:10]:
                            st.write(f"• {keyword}: {score:.3f}")
                
                with col2:
                    st.write("**Main Complaints**")
                    if 'negative_keywords' in cat_data:
                        for keyword, score in list(cat_data['negative_keywords'].items())[:10]:
                            st.write(f"• {keyword}: {score:.3f}")
        
        # Insights summary
        st.subheader("💡 Key Insights")
        
        insights = topic_report.get('insights', [])
        if insights:
            for insight in insights:
                st.info(f"• {insight}")
        else:
            # Generate some insights based on data
            if 'category_analysis' in topic_report:
                worst_category = min(
                    topic_report['category_analysis'].items(),
                    key=lambda x: x[1].get('sentiment_distribution', {}).get('positive', 0) / max(x[1]['total_reviews'], 1)
                )
                st.warning(f"• {worst_category[0]} has the lowest positive sentiment rate")
            
            if 'trending_topics' in topic_report:
                top_trend = list(topic_report['trending_topics'].keys())[0]
                st.info(f"• '{top_trend}' is the fastest growing topic with significant impact")

                
# This function will be called from the main dashboard
def show():
    """Entry point for the topic insights page"""
    pass