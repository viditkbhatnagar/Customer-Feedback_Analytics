import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def show_business_metrics(df, topic_report):
    """Display business metrics and KPIs"""
    st.header("📈 Business Metrics & KPIs")
    
    # Revenue Impact Analysis
    st.subheader("💰 Revenue Impact Analysis")
    
    # Calculate business metrics
    total_reviews = len(df)
    positive_reviews = len(df[df['predicted_sentiment'] == 'positive'])
    negative_reviews = len(df[df['predicted_sentiment'] == 'negative'])
    neutral_reviews = len(df[df['predicted_sentiment'] == 'neutral'])
    
    # Customer satisfaction score
    satisfaction_score = (positive_reviews + (neutral_reviews * 0.5)) / total_reviews * 100
    
    # Net Promoter Score approximation
    promoters = len(df[df['rating'] >= 4])
    detractors = len(df[df['rating'] <= 2])
    nps = (promoters - detractors) / total_reviews * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Customer Satisfaction Score",
            f"{satisfaction_score:.1f}%",
            help="Percentage of positive + (neutral × 0.5) reviews"
        )
    
    with col2:
        st.metric(
            "Net Promoter Score",
            f"{nps:.1f}",
            help="Percentage of promoters (4-5 stars) minus detractors (1-2 stars)"
        )
    
    with col3:
        avg_confidence = df['confidence_score'].mean() * 100
        st.metric(
            "Prediction Confidence",
            f"{avg_confidence:.1f}%",
            help="Average confidence of sentiment predictions"
        )
    
    # Category Performance
    st.subheader("📊 Category Performance")
    
    category_metrics = df.groupby('category').agg({
        'rating': ['mean', 'count'],
        'predicted_sentiment': lambda x: (x == 'positive').mean() * 100
    }).round(2)
    
    category_metrics.columns = ['Avg Rating', 'Review Count', 'Positive %']
    category_metrics = category_metrics.sort_values('Positive %', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category rating chart
        fig_rating = px.bar(
            x=category_metrics.index,
            y=category_metrics['Avg Rating'],
            title="Average Rating by Category",
            labels={'x': 'Category', 'y': 'Average Rating'}
        )
        fig_rating.update_layout(showlegend=False)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        # Category sentiment distribution
        fig_sentiment = px.bar(
            x=category_metrics.index,
            y=category_metrics['Positive %'],
            title="Positive Sentiment % by Category",
            labels={'x': 'Category', 'y': 'Positive Sentiment %'}
        )
        fig_sentiment.update_layout(showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Time Series Analysis
    st.subheader("📅 Time Series Analysis")
    
    # Monthly trends
    df['year_month'] = df['review_date'].dt.to_period('M')
    monthly_trends = df.groupby('year_month').agg({
        'rating': 'mean',
        'predicted_sentiment': lambda x: (x == 'positive').mean() * 100
    }).reset_index()
    
    monthly_trends['year_month'] = monthly_trends['year_month'].astype(str)
    
    fig_trends = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Rating Over Time', 'Positive Sentiment % Over Time'),
        vertical_spacing=0.1
    )
    
    # Rating trend
    fig_trends.add_trace(
        go.Scatter(
            x=monthly_trends['year_month'],
            y=monthly_trends['rating'],
            mode='lines+markers',
            name='Avg Rating',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Sentiment trend
    fig_trends.add_trace(
        go.Scatter(
            x=monthly_trends['year_month'],
            y=monthly_trends['predicted_sentiment'],
            mode='lines+markers',
            name='Positive %',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig_trends.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Business Impact Summary
    st.subheader("💼 Business Impact Summary")
    
    # Calculate potential revenue impact
    if negative_reviews > 0:
        potential_lost_customers = negative_reviews * 0.7  # Assume 70% of negative reviews lose customers
        estimated_clv = 100  # Customer lifetime value assumption
        potential_revenue_loss = potential_lost_customers * estimated_clv
        
        st.warning(f"⚠️ **Potential Revenue Impact**: ${potential_revenue_loss:,.0f}")
        st.write(f"Based on {negative_reviews} negative reviews with 70% churn probability")
    
    # Top performing categories
    best_category = category_metrics.iloc[0]
    worst_category = category_metrics.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🏆 **Best Performing Category**: {best_category.name}")
        st.write(f"- Average Rating: {best_category['Avg Rating']:.1f}/5")
        st.write(f"- Positive Sentiment: {best_category['Positive %']:.1f}%")
        st.write(f"- Review Count: {best_category['Review Count']}")
    
    with col2:
        st.error(f"⚠️ **Needs Attention**: {worst_category.name}")
        st.write(f"- Average Rating: {worst_category['Avg Rating']:.1f}/5")
        st.write(f"- Positive Sentiment: {worst_category['Positive %']:.1f}%")
        st.write(f"- Review Count: {worst_category['Review Count']}")
    
    # Recommendations
    st.subheader("🎯 Business Recommendations")
    
    recommendations = []
    
    # Rating-based recommendations
    if category_metrics['Avg Rating'].min() < 3.5:
        worst_rating_cat = category_metrics.nsmallest(1, 'Avg Rating').index[0]
        recommendations.append(f"🔧 **Immediate Action Required**: Investigate quality issues in {worst_rating_cat} category (Rating: {category_metrics.loc[worst_rating_cat, 'Avg Rating']:.1f}/5)")
    
    # Sentiment-based recommendations
    if category_metrics['Positive %'].min() < 50:
        worst_sentiment_cat = category_metrics.nsmallest(1, 'Positive %').index[0]
        recommendations.append(f"📈 **Improve Customer Experience**: Focus on {worst_sentiment_cat} category (Only {category_metrics.loc[worst_sentiment_cat, 'Positive %']:.1f}% positive sentiment)")
    
    # Volume-based recommendations
    if category_metrics['Review Count'].std() > category_metrics['Review Count'].mean():
        recommendations.append("📊 **Marketing Opportunity**: Some categories have significantly fewer reviews - consider targeted campaigns")
    
    # Confidence-based recommendations
    if avg_confidence < 75:
        recommendations.append("🎯 **Data Quality**: Low prediction confidence suggests need for more training data or model improvement")
    
    for rec in recommendations:
        st.info(rec)
    
    # Export functionality
    st.subheader("📥 Export Data")
    
    if st.button("Generate Business Report"):
        report_data = {
            'Total Reviews': total_reviews,
            'Customer Satisfaction Score': f"{satisfaction_score:.1f}%",
            'Net Promoter Score': f"{nps:.1f}",
            'Best Category': best_category.name,
            'Worst Category': worst_category.name,
            'Recommendations': recommendations
        }
        
        st.json(report_data)
        st.success("✅ Business report generated successfully!")
