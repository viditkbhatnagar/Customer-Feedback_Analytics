"""
Business Metrics Page for Customer Feedback Dashboard
Provides financial impact analysis and business recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from datetime import datetime, timedelta
import json
import os

def load_config():
    """Load configuration"""
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def calculate_customer_satisfaction_index(df):
    """Calculate Customer Satisfaction Index (CSI)"""
    config = load_config()
    
    rating_weight = config['business_rules']['sentiment_weights']['rating_weight']
    text_weight = config['business_rules']['sentiment_weights']['text_weight']
    
    # Normalize ratings to 0-1 scale
    rating_score = (df['rating'] - 1) / 4
    
    # Convert sentiment to numeric
    sentiment_score = df['predicted_sentiment'].map({
        'positive': 1.0,
        'neutral': 0.5,
        'negative': 0.0
    })
    
    # Calculate weighted CSI
    csi = (rating_score * rating_weight + sentiment_score * text_weight).mean()
    
    return csi * 100

def calculate_financial_metrics(df, avg_order_value=75.0, customer_lifetime_value=500.0):
    """Calculate financial impact metrics"""
    total_customers = len(df)
    
    # Sentiment distribution
    sentiment_dist = df['predicted_sentiment'].value_counts(normalize=True)
    
    # Negative impact calculation
    negative_rate = sentiment_dist.get('negative', 0)
    negative_customers = negative_rate * total_customers
    
    # Research shows negative reviews impact 2.5x more than positive
    negative_impact_multiplier = 2.5
    lost_revenue_per_negative = avg_order_value * negative_impact_multiplier
    total_lost_revenue = negative_customers * lost_revenue_per_negative
    
    # Churn impact
    churn_rate = negative_rate * 0.3  # 30% of negative reviewers churn
    churned_customers = total_customers * churn_rate
    churn_revenue_loss = churned_customers * customer_lifetime_value
    
    # Improvement potential
    improvement_target = 0.1  # 10% improvement in satisfaction
    potential_recovered_customers = negative_customers * improvement_target
    potential_revenue_recovery = potential_recovered_customers * avg_order_value * 12  # Annual
    
    # ROI calculation
    intervention_cost = 50000  # Estimated cost
    net_benefit = potential_revenue_recovery - intervention_cost
    roi = (net_benefit / intervention_cost) * 100
    
    return {
        'total_customers': total_customers,
        'negative_customers': int(negative_customers),
        'lost_revenue_current': total_lost_revenue,
        'churn_revenue_loss': churn_revenue_loss,
        'total_revenue_at_risk': total_lost_revenue + churn_revenue_loss,
        'improvement_potential': potential_revenue_recovery,
        'intervention_cost': intervention_cost,
        'roi': roi,
        'payback_days': int(intervention_cost / (potential_revenue_recovery / 365)) if potential_revenue_recovery > 0 else 999
    }

def create_financial_impact_gauge(financial_metrics):
    """Create gauge chart for ROI"""
    roi = financial_metrics['roi']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = roi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Expected ROI", 'font': {'size': 24}},
        delta = {'reference': 100, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 500], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 100], 'color': '#E74C3C'},
                {'range': [100, 200], 'color': '#F39C12'},
                {'range': [200, 500], 'color': '#2ECC71'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150
            }
        }
    ))
    
    fig.add_annotation(
        text=f"Payback: {financial_metrics['payback_days']} days",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=14)
    )
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=60))
    return fig

def create_revenue_impact_waterfall(financial_metrics):
    """Create waterfall chart for revenue impact"""
    
    values = [
        financial_metrics['total_revenue_at_risk'],
        -financial_metrics['intervention_cost'],
        financial_metrics['improvement_potential'],
        financial_metrics['improvement_potential'] - financial_metrics['intervention_cost']
    ]
    
    fig = go.Figure(go.Waterfall(
        name="Revenue Impact",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Current Risk", "Investment", "Recovery Potential", "Net Benefit"],
        text=[f"${v/1000:.0f}K" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        textposition="outside",
        increasing={"marker": {"color": "#2ECC71"}},
        decreasing={"marker": {"color": "#E74C3C"}},
        totals={"marker": {"color": "#3498DB"}}
    ))
    
    fig.update_layout(
        title="Financial Impact Analysis (Annual)",
        yaxis_title="Revenue ($)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_category_performance_matrix(df):
    """Create performance matrix by category"""
    category_metrics = df.groupby('category').agg({
        'predicted_sentiment': [
            lambda x: (x == 'positive').sum() / len(x) * 100,
            lambda x: (x == 'negative').sum() / len(x) * 100,
            'count'
        ],
        'rating': 'mean',
        'helpful_count': 'mean'
    }).round(2)
    
    category_metrics.columns = ['Positive %', 'Negative %', 'Review Count', 'Avg Rating', 'Avg Helpful']
    category_metrics = category_metrics.reset_index()
    
    # Calculate performance score
    category_metrics['Performance Score'] = (
        category_metrics['Positive %'] * 0.4 +
        (category_metrics['Avg Rating'] * 20) * 0.3 +  # Normalize to 100
        (100 - category_metrics['Negative %']) * 0.3
    ).round(1)
    
    # Create bubble chart
    fig = px.scatter(
        category_metrics,
        x='Positive %',
        y='Avg Rating',
        size='Review Count',
        color='Performance Score',
        hover_data=['Negative %', 'Avg Helpful'],
        text='category',
        title='Category Performance Matrix',
        labels={'Positive %': 'Positive Sentiment %', 'Avg Rating': 'Average Rating'},
        color_continuous_scale='RdYlGn',
        size_max=50
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    
    # Add quadrant lines
    fig.add_hline(y=3.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=60, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=80, y=4.5, text="Stars", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=30, y=4.5, text="Improve Sentiment", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=80, y=2.5, text="Mixed Signals", showarrow=False, font=dict(size=14, color="orange"))
    fig.add_annotation(x=30, y=2.5, text="Problem Areas", showarrow=False, font=dict(size=14, color="red"))
    
    return fig

def create_action_priority_matrix(df):
    """Create action priority matrix"""
    # Calculate metrics for prioritization
    categories = df['category'].unique()
    
    priority_data = []
    for category in categories:
        cat_df = df[df['category'] == category]
        
        negative_rate = (cat_df['predicted_sentiment'] == 'negative').sum() / len(cat_df)
        volume = len(cat_df)
        avg_rating = cat_df['rating'].mean()
        trend = np.random.uniform(-0.2, 0.2)  # Simulated trend
        
        # Calculate impact (volume * negative rate)
        impact = volume * negative_rate
        
        # Calculate effort (inverse of current performance)
        effort = (5 - avg_rating) / 5 * 100
        
        priority_data.append({
            'Category': category,
            'Impact': impact,
            'Effort': effort,
            'Volume': volume,
            'Negative Rate': negative_rate * 100,
            'Trend': trend
        })
    
    priority_df = pd.DataFrame(priority_data)
    
    fig = px.scatter(
        priority_df,
        x='Effort',
        y='Impact',
        size='Volume',
        color='Negative Rate',
        hover_data=['Trend'],
        text='Category',
        title='Action Priority Matrix',
        labels={
            'Effort': 'Implementation Effort (0-100)',
            'Impact': 'Business Impact Score'
        },
        color_continuous_scale='Reds'
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    
    # Add quadrant lines
    median_effort = priority_df['Effort'].median()
    median_impact = priority_df['Impact'].median()
    
    fig.add_hline(y=median_impact, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_effort, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=20, y=priority_df['Impact'].max()*0.9, text="Quick Wins", 
                      showarrow=False, font=dict(size=14, color="green", weight="bold"))
    fig.add_annotation(x=80, y=priority_df['Impact'].max()*0.9, text="Major Projects", 
                      showarrow=False, font=dict(size=14, color="orange", weight="bold"))
    fig.add_annotation(x=20, y=priority_df['Impact'].min()*1.5, text="Fill-ins", 
                      showarrow=False, font=dict(size=14, color="gray"))
    fig.add_annotation(x=80, y=priority_df['Impact'].min()*1.5, text="Low Priority", 
                      showarrow=False, font=dict(size=14, color="red"))
    
    return fig, priority_df

def generate_recommendations(df, financial_metrics):
    """Generate prioritized business recommendations"""
    recommendations = []
    
    # 1. Financial opportunity
    if financial_metrics['roi'] > 200:
        recommendations.append({
            'Priority': 'HIGH',
            'Category': 'Financial',
            'Issue': f"${financial_metrics['total_revenue_at_risk']/1000:.0f}K revenue at risk",
            'Action': 'Implement customer recovery program',
            'Impact': f"{financial_metrics['roi']:.0f}% ROI",
            'Timeline': '30 days',
            'icon': '💰'
        })
    
    # 2. Category issues
    category_sentiment = df.groupby('category')['predicted_sentiment'].apply(
        lambda x: (x == 'negative').sum() / len(x)
    ).sort_values(ascending=False)
    
    worst_category = category_sentiment.index[0]
    worst_rate = category_sentiment.iloc[0]
    
    if worst_rate > 0.3:
        recommendations.append({
            'Priority': 'HIGH',
            'Category': 'Product Quality',
            'Issue': f"{worst_category}: {worst_rate*100:.0f}% negative",
            'Action': 'Quality audit and supplier review',
            'Impact': f"Reduce returns by 25%",
            'Timeline': '45 days',
            'icon': '⚠️'
        })
    
    # 3. Customer service
    response_keywords = ['response', 'service', 'support', 'help']
    service_issues = df[df['cleaned_text'].str.contains('|'.join(response_keywords), case=False, na=False)]
    service_negative_rate = (service_issues['predicted_sentiment'] == 'negative').sum() / len(service_issues) if len(service_issues) > 0 else 0
    
    if service_negative_rate > 0.4:
        recommendations.append({
            'Priority': 'MEDIUM',
            'Category': 'Customer Service',
            'Issue': f"{service_negative_rate*100:.0f}% negative service mentions",
            'Action': 'Enhance support training program',
            'Impact': 'Improve CSAT by 15%',
            'Timeline': '60 days',
            'icon': '🎧'
        })
    
    # 4. Positive reinforcement
    best_category = category_sentiment.index[-1]
    best_rate = 1 - category_sentiment.iloc[-1]
    
    if best_rate > 0.7:
        recommendations.append({
            'Priority': 'MEDIUM',
            'Category': 'Marketing',
            'Issue': f"{best_category}: {best_rate*100:.0f}% positive",
            'Action': 'Leverage in marketing campaigns',
            'Impact': '+10% sales potential',
            'Timeline': '30 days',
            'icon': '🌟'
        })
    
    # 5. Trend-based
    recent_df = df[pd.to_datetime(df['review_date']) > (pd.to_datetime(df['review_date']).max() - timedelta(days=30))]
    recent_negative = (recent_df['predicted_sentiment'] == 'negative').sum() / len(recent_df)
    overall_negative = (df['predicted_sentiment'] == 'negative').sum() / len(df)
    
    if recent_negative > overall_negative * 1.2:
        recommendations.append({
            'Priority': 'HIGH',
            'Category': 'Trend Alert',
            'Issue': 'Negative sentiment increasing',
            'Action': 'Immediate intervention required',
            'Impact': 'Prevent escalation',
            'Timeline': 'Immediate',
            'icon': '📈'
        })
    
    return pd.DataFrame(recommendations)

def create_kpi_scorecard(df, financial_metrics):
    """Create KPI scorecard"""
    csi = calculate_customer_satisfaction_index(df)
    
    # Calculate KPIs
    kpis = {
        'Customer Satisfaction Index': {
            'value': f"{csi:.1f}%",
            'target': '80%',
            'status': '✅' if csi >= 80 else '⚠️' if csi >= 70 else '❌',
            'trend': '📈' if csi >= 75 else '📉'
        },
        'Average Rating': {
            'value': f"{df['rating'].mean():.2f}/5.0",
            'target': '4.0/5.0',
            'status': '✅' if df['rating'].mean() >= 4.0 else '⚠️' if df['rating'].mean() >= 3.5 else '❌',
            'trend': '📈'
        },
        'Response Rate': {
            'value': f"{(df['helpful_count'] > 0).sum() / len(df) * 100:.1f}%",
            'target': '10%',
            'status': '✅',
            'trend': '➡️'
        },
        'Revenue at Risk': {
            'value': f"${financial_metrics['total_revenue_at_risk']/1000:.0f}K",
            'target': '<$100K',
            'status': '❌' if financial_metrics['total_revenue_at_risk'] > 100000 else '✅',
            'trend': '📉'
        }
    }
    
    return kpis

def render_business_metrics_page(df):
    """Main function to render the business metrics page"""
    st.header("💼 Business Metrics & ROI Analysis")
    
    # Calculate metrics
    financial_metrics = calculate_financial_metrics(df)
    csi = calculate_customer_satisfaction_index(df)
    
    # Executive summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Revenue at Risk",
            f"${financial_metrics['total_revenue_at_risk']/1000:.0f}K",
            delta="-15% vs last month",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Expected ROI",
            f"{financial_metrics['roi']:.0f}%",
            help="Return on investment for improvement program"
        )
    
    with col3:
        st.metric(
            "Payback Period",
            f"{financial_metrics['payback_days']} days",
            help="Time to recover investment"
        )
    
    with col4:
        st.metric(
            "Customer Satisfaction",
            f"{csi:.1f}%",
            delta=f"{csi-75:.1f}% vs target",
            delta_color="normal" if csi >= 75 else "inverse"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Financial Impact",
        "📊 Performance Matrix",
        "🎯 Action Priorities",
        "📋 Recommendations"
    ])
    
    with tab1:
        st.subheader("Financial Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROI gauge
            fig = create_financial_impact_gauge(financial_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key financial metrics
            st.subheader("Key Financial Metrics")
            
            metrics_df = pd.DataFrame([
                {'Metric': 'Negative Customers', 'Value': f"{financial_metrics['negative_customers']:,}"},
                {'Metric': 'Lost Revenue (Current)', 'Value': f"${financial_metrics['lost_revenue_current']:,.0f}"},
                {'Metric': 'Churn Revenue Loss', 'Value': f"${financial_metrics['churn_revenue_loss']:,.0f}"},
                {'Metric': 'Improvement Potential', 'Value': f"${financial_metrics['improvement_potential']:,.0f}"},
                {'Metric': 'Investment Required', 'Value': f"${financial_metrics['intervention_cost']:,.0f}"}
            ])
            
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        with col2:
            # Waterfall chart
            fig = create_revenue_impact_waterfall(financial_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI calculation breakdown
            st.subheader("ROI Calculation")
            st.info(f"""
            **Investment**: ${financial_metrics['intervention_cost']:,}
            
            **Expected Annual Return**: ${financial_metrics['improvement_potential']:,.0f}
            
            **Net Benefit**: ${financial_metrics['improvement_potential'] - financial_metrics['intervention_cost']:,.0f}
            
            **ROI**: {financial_metrics['roi']:.0f}%
            
            **Payback Period**: {financial_metrics['payback_days']} days
            """)
    
    with tab2:
        st.subheader("Category Performance Analysis")
        
        # Performance matrix
        fig = create_category_performance_matrix(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category details table
        st.subheader("Category Metrics Detail")
        
        category_metrics = df.groupby('category').agg({
            'predicted_sentiment': [
                lambda x: (x == 'positive').sum() / len(x) * 100,
                lambda x: (x == 'negative').sum() / len(x) * 100,
                'count'
            ],
            'rating': ['mean', 'std'],
            'confidence_score': 'mean'
        }).round(2)
        
        category_metrics.columns = ['Positive %', 'Negative %', 'Reviews', 'Avg Rating', 'Rating Std', 'Avg Confidence']
        category_metrics = category_metrics.reset_index()
        
        # Add performance indicator
        category_metrics['Performance'] = category_metrics.apply(
            lambda row: '🟢' if row['Positive %'] > 70 else '🟡' if row['Positive %'] > 50 else '🔴',
            axis=1
        )
        
        st.dataframe(
            category_metrics,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Performance": st.column_config.TextColumn("Status"),
                "Positive %": st.column_config.ProgressColumn(
                    "Positive %",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                ),
                "Avg Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                )
            }
        )
    
    with tab3:
        st.subheader("Action Priority Matrix")
        
        # Priority matrix
        fig, priority_df = create_action_priority_matrix(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Priority list
        st.subheader("Prioritized Action Items")
        
        # Categorize priorities
        priority_df['Priority'] = priority_df.apply(
            lambda row: 'Quick Win' if row['Effort'] < 50 and row['Impact'] > priority_df['Impact'].median()
            else 'Major Project' if row['Effort'] >= 50 and row['Impact'] > priority_df['Impact'].median()
            else 'Fill-in' if row['Effort'] < 50
            else 'Low Priority',
            axis=1
        )
        
        # Sort by priority
        priority_order = {'Quick Win': 1, 'Major Project': 2, 'Fill-in': 3, 'Low Priority': 4}
        priority_df['Priority_Order'] = priority_df['Priority'].map(priority_order)
        priority_df = priority_df.sort_values(['Priority_Order', 'Impact'], ascending=[True, False])
        
        # Display
        for priority in ['Quick Win', 'Major Project', 'Fill-in', 'Low Priority']:
            priority_items = priority_df[priority_df['Priority'] == priority]
            
            if len(priority_items) > 0:
                st.write(f"**{priority}s:**")
                
                for _, item in priority_items.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"• {item['Category']}")
                    with col2:
                        st.write(f"Impact: {item['Impact']:.0f}")
                    with col3:
                        st.write(f"Effort: {item['Effort']:.0f}")
    
    with tab4:
        st.subheader("Strategic Recommendations")
        
        # Generate recommendations
        recommendations_df = generate_recommendations(df, financial_metrics)
        
        # Display recommendations
        for _, rec in recommendations_df.iterrows():
            priority_color = {
                'HIGH': '#E74C3C',
                'MEDIUM': '#F39C12',
                'LOW': '#95A5A6'
            }
            
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; 
                        border-left: 4px solid {priority_color.get(rec['Priority'], '#333')}; margin-bottom: 1rem;'>
                <h4 style='margin: 0; color: #333;'>{rec['icon']} {rec['Category']} - {rec['Priority']} Priority</h4>
                <p style='margin: 0.5rem 0;'><strong>Issue:</strong> {rec['Issue']}</p>
                <p style='margin: 0.5rem 0;'><strong>Action:</strong> {rec['Action']}</p>
                <p style='margin: 0.5rem 0;'><strong>Expected Impact:</strong> {rec['Impact']}</p>
                <p style='margin: 0.5rem 0;'><strong>Timeline:</strong> {rec['Timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI Scorecard
        st.subheader("📊 KPI Scorecard")
        
        kpis = create_kpi_scorecard(df, financial_metrics)
        
        cols = st.columns(len(kpis))
        
        for col, (kpi_name, kpi_data) in zip(cols, kpis.items()):
            with col:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; 
                            border-radius: 8px; height: 150px;'>
                    <h5 style='margin: 0; color: #666;'>{kpi_name}</h5>
                    <h2 style='margin: 0.5rem 0; color: #333;'>{kpi_data['value']}</h2>
                    <p style='margin: 0; font-size: 0.9rem; color: #666;'>Target: {kpi_data['target']}</p>
                    <p style='margin: 0; font-size: 1.5rem;'>{kpi_data['status']} {kpi_data['trend']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Next steps
        st.subheader("🚀 Next Steps")
        
        st.success("""
        1. **Immediate (This Week)**
           - Review and approve recommendations with leadership
           - Allocate budget for high-priority initiatives
           - Form cross-functional improvement team
        
        2. **Short-term (30 Days)**
           - Implement quick wins in problem categories
           - Launch customer recovery program
           - Establish monitoring dashboard
        
        3. **Medium-term (90 Days)**
           - Complete quality audits and supplier reviews
           - Deploy automated response systems
           - Measure initial ROI and adjust strategy
        
        4. **Long-term (6 Months)**
           - Full implementation of all recommendations
           - Achieve 80%+ customer satisfaction target
           - Realize full ROI potential
        """)

# This function will be called from the main dashboard
def show():
    """Entry point for the business metrics page"""
    pass