"""
Visualization Charts Module for Customer Feedback Analytics
Reusable chart components for dashboard and reports
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ChartGenerator:
    """Generate various types of charts for the analytics dashboard"""
    
    def __init__(self, color_scheme: Optional[Dict[str, str]] = None):
        """Initialize chart generator with color scheme"""
        self.colors = color_scheme or {
            'positive': '#2ECC71',
            'negative': '#E74C3C',
            'neutral': '#95A5A6',
            'primary': '#1f77b4',
            'secondary': '#2ca02c',
            'accent': '#ff7f0e',
            'background': '#f0f2f6'
        }
        
    def create_sentiment_pie_chart(self, df: pd.DataFrame, 
                                  title: str = "Sentiment Distribution") -> go.Figure:
        """Create an enhanced pie chart for sentiment distribution"""
        sentiment_counts = df['predicted_sentiment'].value_counts()
        
        # Create pie chart with custom styling
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=.4,
            marker=dict(
                colors=[self.colors.get(label.lower(), '#333') for label in sentiment_counts.index],
                line=dict(color='white', width=2)
            ),
            textfont=dict(size=14),
            textposition='outside',
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>' +
                         'Count: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        )])
        
        # Add center annotation
        total_reviews = len(df)
        fig.add_annotation(
            text=f'<b>{total_reviews:,}</b><br>Reviews',
            x=0.5, y=0.5,
            font=dict(size=20),
            showarrow=False
        )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            showlegend=True,
            margin=dict(t=80, b=40, l=40, r=40),
            height=400
        )
        
        return fig
    
    def create_sentiment_timeline(self, df: pd.DataFrame, 
                                date_column: str = 'review_date',
                                window: str = 'W') -> go.Figure:
        """Create sentiment trend over time"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Resample by window
        sentiment_over_time = df.set_index(date_column).groupby('predicted_sentiment').resample(window).size().reset_index(name='count')
        
        # Calculate percentages
        total_by_date = sentiment_over_time.groupby(date_column)['count'].sum()
        sentiment_over_time['percentage'] = sentiment_over_time.apply(
            lambda x: x['count'] / total_by_date[x[date_column]] * 100, axis=1
        )
        
        # Create line chart
        fig = px.line(
            sentiment_over_time,
            x=date_column,
            y='percentage',
            color='predicted_sentiment',
            title=f'Sentiment Trends Over Time ({window})',
            labels={'percentage': 'Percentage (%)', date_column: 'Date'},
            color_discrete_map={
                'positive': self.colors['positive'],
                'negative': self.colors['negative'],
                'neutral': self.colors['neutral']
            },
            line_shape='spline'
        )
        
        # Add markers
        fig.update_traces(mode='lines+markers', marker=dict(size=6))
        
        # Update layout
        fig.update_layout(
            hovermode='x unified',
            height=400,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_category_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing sentiment by category"""
        # Calculate sentiment percentages by category
        heatmap_data = pd.crosstab(
            df['category'],
            df['predicted_sentiment'],
            normalize='index'
        ) * 100
        
        # Create custom colorscale
        colorscale = [
            [0, self.colors['negative']],
            [0.5, self.colors['neutral']],
            [1, self.colors['positive']]
        ]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate='Category: %{y}<br>' +
                         'Sentiment: %{x}<br>' +
                         'Percentage: %{z:.1f}%<br>' +
                         '<extra></extra>',
            colorbar=dict(title="Percentage (%)")
        ))
        
        fig.update_layout(
            title='Sentiment Distribution by Product Category',
            xaxis_title='Sentiment',
            yaxis_title='Category',
            height=400 + len(heatmap_data.index) * 30,  # Dynamic height
            margin=dict(l=150)  # More space for category names
        )
        
        return fig
    
    def create_word_cloud(self, text_series: pd.Series, 
                         sentiment: Optional[str] = None,
                         max_words: int = 100) -> plt.Figure:
        """Create word cloud from text data"""
        # Combine all text
        text = ' '.join(text_series.dropna().astype(str))
        
        if not text.strip():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=20)
            ax.axis('off')
            return fig
        
        # Color schemes based on sentiment
        color_maps = {
            'positive': 'Greens',
            'negative': 'Reds',
            'neutral': 'Greys',
            None: 'viridis'
        }
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=color_maps.get(sentiment, 'viridis'),
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        title = f'Word Cloud'
        if sentiment:
            title += f' - {sentiment.capitalize()} Reviews'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_rating_sentiment_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create stacked bar chart comparing ratings vs sentiment"""
        # Create crosstab
        comparison = pd.crosstab(df['rating'], df['predicted_sentiment'])
        
        # Create stacked bar chart
        fig = go.Figure()
        
        sentiments = ['negative', 'neutral', 'positive']
        colors = [self.colors['negative'], self.colors['neutral'], self.colors['positive']]
        
        for sentiment, color in zip(sentiments, colors):
            if sentiment in comparison.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.capitalize(),
                    x=comparison.index,
                    y=comparison[sentiment],
                    marker_color=color,
                    hovertemplate='Rating: %{x}<br>' +
                                 f'{sentiment.capitalize()}: %{{y}}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title='Rating vs Sentiment Distribution',
            xaxis_title='Rating',
            yaxis_title='Number of Reviews',
            barmode='stack',
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_confidence_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create violin plot for confidence score distribution"""
        fig = go.Figure()
        
        sentiments = df['predicted_sentiment'].unique()
        
        for sentiment in sentiments:
            data = df[df['predicted_sentiment'] == sentiment]['confidence_score']
            
            fig.add_trace(go.Violin(
                y=data,
                name=sentiment.capitalize(),
                box_visible=True,
                meanline_visible=True,
                fillcolor=self.colors.get(sentiment, '#333'),
                opacity=0.6,
                x0=sentiment,
                hovertemplate='Sentiment: %{x}<br>' +
                            'Confidence: %{y:.2f}<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Model Confidence Distribution by Sentiment',
            yaxis_title='Confidence Score',
            showlegend=False,
            height=400,
            yaxis=dict(range=[0, 1.05])
        )
        
        return fig
    
    def create_time_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create heatmap showing review patterns by time"""
        df = df.copy()
        df['hour'] = pd.to_datetime(df['review_date']).dt.hour
        df['dayofweek'] = pd.to_datetime(df['review_date']).dt.dayofweek
        
        # Calculate average sentiment score by hour and day
        df['sentiment_score'] = df['predicted_sentiment'].map({
            'positive': 1, 'neutral': 0, 'negative': -1
        })
        
        heatmap_data = df.pivot_table(
            values='sentiment_score',
            index='dayofweek',
            columns='hour',
            aggfunc='mean'
        )
        
        # Day names
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=[day_names[i] for i in heatmap_data.index],
            colorscale='RdYlGn',
            text=np.round(heatmap_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Avg Sentiment<br>Score"),
            hovertemplate='Day: %{y}<br>' +
                         'Hour: %{x}<br>' +
                         'Avg Score: %{z:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Average Sentiment by Day and Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        return fig
    
    def create_bubble_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create bubble chart for multi-dimensional analysis"""
        # Aggregate by category
        bubble_data = df.groupby('category').agg({
            'rating': 'mean',
            'predicted_sentiment': lambda x: (x == 'positive').sum() / len(x) * 100,
            'review_id': 'count',
            'helpful_count': 'sum'
        }).reset_index()
        
        bubble_data.columns = ['Category', 'Avg Rating', 'Positive %', 'Review Count', 'Total Helpful']
        bubble_data['Engagement'] = bubble_data['Total Helpful'] / bubble_data['Review Count']
        
        fig = px.scatter(
            bubble_data,
            x='Positive %',
            y='Avg Rating',
            size='Review Count',
            color='Engagement',
            hover_name='Category',
            hover_data={
                'Review Count': ':,',
                'Total Helpful': ':,',
                'Engagement': ':.2f'
            },
            title='Category Performance Overview',
            labels={
                'Positive %': 'Positive Sentiment %',
                'Avg Rating': 'Average Rating'
            },
            color_continuous_scale='Viridis',
            size_max=60
        )
        
        # Add quadrant lines
        fig.add_hline(y=3.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            height=500,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[1, 5])
        )
        
        return fig
    
    def create_sunburst_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create sunburst chart for hierarchical data"""
        # Prepare hierarchical data
        sunburst_data = []
        
        # Root
        sunburst_data.append({
            'labels': 'All Reviews',
            'parents': '',
            'values': len(df),
            'color': self.colors['primary']
        })
        
        # Categories
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            sunburst_data.append({
                'labels': category,
                'parents': 'All Reviews',
                'values': len(cat_df),
                'color': self.colors['secondary']
            })
            
            # Sentiments within categories
            for sentiment in ['positive', 'negative', 'neutral']:
                count = len(cat_df[cat_df['predicted_sentiment'] == sentiment])
                if count > 0:
                    sunburst_data.append({
                        'labels': f'{sentiment.capitalize()}',
                        'parents': category,
                        'values': count,
                        'color': self.colors[sentiment]
                    })
        
        sunburst_df = pd.DataFrame(sunburst_data)
        
        fig = px.sunburst(
            sunburst_df,
            names='labels',
            parents='parents',
            values='values',
            title='Review Distribution Hierarchy',
            color='color',
            color_discrete_map="identity"
        )
        
        fig.update_layout(
            height=600,
            margin=dict(t=80, b=0, l=0, r=0)
        )
        
        return fig
    
    def create_metric_card(self, title: str, value: str, 
                          delta: Optional[str] = None,
                          delta_color: str = "normal") -> str:
        """Create HTML for a metric card"""
        delta_html = ""
        if delta:
            color = "#2ECC71" if delta_color == "normal" else "#E74C3C"
            arrow = "↑" if delta.startswith("+") else "↓"
            delta_html = f'<p style="color: {color}; margin: 0;">{arrow} {delta}</p>'
        
        return f"""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: #666; margin: 0; font-size: 0.9rem;">{title}</h4>
            <h2 style="color: #333; margin: 0.5rem 0; font-size: 2rem;">{value}</h2>
            {delta_html}
        </div>
        """


def main():
    """Example usage of chart generator"""
    # This would typically be imported and used in the dashboard
    pass


if __name__ == "__main__":
    main()