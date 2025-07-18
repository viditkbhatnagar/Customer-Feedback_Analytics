"""
Data Generator for Customer Feedback Analytics
Generates realistic synthetic customer review data for e-commerce
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import yaml
import os
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerReviewGenerator:
    """Generates synthetic customer review data with realistic patterns"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the generator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.categories = self.config['data']['categories']
        self.num_reviews = self.config['data']['num_reviews']
        
        # Review templates by sentiment
        self.positive_templates = [
            "Absolutely love this {product}! {positive_aspect}. Highly recommend!",
            "Best {product} I've ever purchased. {positive_aspect}. Worth every penny.",
            "Exceeded my expectations! {positive_aspect}. Would buy again.",
            "{positive_aspect}. Great {product}, very satisfied with my purchase.",
            "Amazing quality and {positive_aspect}. 5 stars!",
            "Perfect {product}! {positive_aspect}. Couldn't be happier.",
            "Fantastic purchase. {positive_aspect}. Excellent value for money.",
            "{positive_aspect}. This {product} is exactly what I needed!"
        ]
        
        self.negative_templates = [
            "Terrible {product}. {negative_aspect}. Very disappointed.",
            "Worst {product} ever. {negative_aspect}. Want my money back!",
            "{negative_aspect}. This {product} is a complete waste of money.",
            "Do not buy! {negative_aspect}. Extremely poor quality.",
            "Horrible experience. {negative_aspect}. Would not recommend.",
            "{negative_aspect}. This {product} broke after just a few days.",
            "Complete disaster. {negative_aspect}. Save your money!",
            "Awful {product}. {negative_aspect}. Returning immediately."
        ]
        
        self.neutral_templates = [
            "The {product} is okay. {neutral_aspect}. Nothing special.",
            "Average {product}. {neutral_aspect}. Gets the job done.",
            "{neutral_aspect}. It's a decent {product} for the price.",
            "Not bad, not great. {neutral_aspect}. Just average.",
            "The {product} works as expected. {neutral_aspect}.",
            "{neutral_aspect}. Standard {product}, nothing remarkable.",
            "It's fine. {neutral_aspect}. Meets basic requirements.",
            "Acceptable {product}. {neutral_aspect}. Could be better."
        ]
        
        # Product types by category
        self.products_by_category = {
            "Electronics": ["laptop", "smartphone", "headphones", "tablet", "smartwatch", "camera", "speaker"],
            "Fashion": ["shirt", "dress", "shoes", "jacket", "jeans", "bag", "watch"],
            "Home & Kitchen": ["blender", "coffee maker", "vacuum", "cookware set", "knife set", "toaster"],
            "Sports & Outdoors": ["running shoes", "yoga mat", "bike", "tent", "backpack", "fitness tracker"],
            "Beauty & Personal Care": ["moisturizer", "shampoo", "makeup kit", "perfume", "hair dryer"],
            "Books": ["novel", "textbook", "cookbook", "self-help book", "biography"],
            "Toys & Games": ["board game", "puzzle", "action figure", "lego set", "video game"],
            "Health & Household": ["vitamins", "first aid kit", "air purifier", "humidifier", "scale"]
        }
        
        # Aspects by sentiment
        self.positive_aspects = [
            "excellent build quality", "fast shipping", "great customer service",
            "amazing performance", "beautiful design", "easy to use",
            "incredible value", "works perfectly", "super comfortable",
            "outstanding features", "very durable", "looks fantastic"
        ]
        
        self.negative_aspects = [
            "broke after a week", "terrible quality", "doesn't work as advertised",
            "cheap materials", "poor customer service", "arrived damaged",
            "complete waste of money", "stopped working", "uncomfortable to use",
            "missing parts", "false advertising", "horrible design"
        ]
        
        self.neutral_aspects = [
            "works as described", "standard quality", "nothing special",
            "average performance", "basic features", "acceptable for the price",
            "meets expectations", "decent quality", "okay for casual use",
            "functional but basic", "does what it's supposed to", "fair value"
        ]
        
    def add_typos(self, text: str, rate: float) -> str:
        """Add realistic typos to text"""
        if random.random() > rate:
            return text
            
        words = text.split()
        num_typos = max(1, int(len(words) * rate))
        
        for _ in range(num_typos):
            if not words:
                break
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            
            if len(word) > 3:
                typo_type = random.choice(['swap', 'duplicate', 'missing'])
                if typo_type == 'swap' and len(word) > 1:
                    i = random.randint(0, len(word) - 2)
                    word = word[:i] + word[i+1] + word[i] + word[i+2:]
                elif typo_type == 'duplicate':
                    i = random.randint(0, len(word) - 1)
                    word = word[:i] + word[i] + word[i:]
                else:  # missing
                    i = random.randint(0, len(word) - 1)
                    word = word[:i] + word[i+1:]
                    
                words[idx] = word
                
        return ' '.join(words)
    
    def add_slang(self, text: str, rate: float) -> str:
        """Add internet slang and informal language"""
        if random.random() > rate:
            return text
            
        slang_replacements = {
            "great": ["awesome", "lit", "fire", "dope"],
            "bad": ["trash", "garbage", "sucks"],
            "good": ["cool", "sick", "rad"],
            "very": ["super", "hella", "totally"],
            "love": ["luv", "❤️", "adore"],
            "hate": ["can't stand", "despise", "h8"],
            "money": ["$$", "cash", "bucks"],
            "!": ["!!!", "!!!!", "!1!"]
        }
        
        for word, replacements in slang_replacements.items():
            if word in text.lower():
                text = text.replace(word, random.choice(replacements))
                
        # Add some emojis
        if random.random() < 0.3:
            emojis = ["😊", "😍", "😡", "😭", "🔥", "💯", "👍", "👎", "⭐"]
            text += f" {random.choice(emojis)}"
            
        return text
    
    def generate_review_text(self, sentiment: str, category: str) -> Tuple[str, str]:
        """Generate review text based on sentiment and category"""
        product = random.choice(self.products_by_category[category])
        
        if sentiment == "positive":
            template = random.choice(self.positive_templates)
            aspect = random.choice(self.positive_aspects)
        elif sentiment == "negative":
            template = random.choice(self.negative_templates)
            aspect = random.choice(self.negative_aspects)
        else:
            template = random.choice(self.neutral_templates)
            aspect = random.choice(self.neutral_aspects)
            
        review = template.format(product=product, 
                               positive_aspect=aspect,
                               negative_aspect=aspect,
                               neutral_aspect=aspect)
        
        # Add variations
        if random.random() < 0.3:
            # Add personal experience
            experiences = [
                " Been using it for a month now.",
                " Bought this for my spouse.",
                " This is my second purchase.",
                " Compared to my previous one, this is better.",
                " Using it daily.",
                " Got this on sale."
            ]
            review += random.choice(experiences)
            
        return review, product
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset"""
        logger.info(f"Generating {self.num_reviews} customer reviews...")
        
        reviews = []
        
        # Generate date range (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        for i in range(self.num_reviews):
            # Determine sentiment distribution (60% positive, 25% negative, 15% neutral)
            rand = random.random()
            if rand < 0.6:
                sentiment = "positive"
                rating = random.choice([4, 5, 5])  # More 5s
            elif rand < 0.85:
                sentiment = "negative"
                rating = random.choice([1, 1, 2])  # More 1s
            else:
                sentiment = "neutral"
                rating = 3
                
            category = random.choice(self.categories)
            review_text, product = self.generate_review_text(sentiment, category)
            
            # Apply data quality issues
            review_text = self.add_typos(review_text, self.config['data']['typo_rate'])
            review_text = self.add_slang(review_text, self.config['data']['slang_rate'])
            
            # Generate metadata
            review_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            
            verified_purchase = random.random() < 0.75  # 75% verified
            helpful_count = np.random.poisson(5) if sentiment != "neutral" else np.random.poisson(2)
            
            review_length = len(review_text.split())
            
            reviews.append({
                'review_id': f'R{i+1:06d}',
                'product_id': f'P{random.randint(1, 1000):04d}',
                'product_name': product.title(),
                'category': category,
                'rating': rating,
                'review_text': review_text,
                'review_date': review_date,
                'verified_purchase': verified_purchase,
                'helpful_count': helpful_count,
                'review_length': review_length,
                'true_sentiment': sentiment  # For validation
            })
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1} reviews...")
        
        df = pd.DataFrame(reviews)
        
        # Add some seasonal patterns
        df['month'] = df['review_date'].dt.month
        df['is_holiday_season'] = df['month'].isin([11, 12, 1])  # Nov, Dec, Jan
        
        # Save the dataset
        output_dir = os.path.dirname(self.config['data']['raw_data_path'])
        os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(self.config['data']['raw_data_path'], index=False)
        logger.info(f"Dataset saved to {self.config['data']['raw_data_path']}")
        
        # Generate summary statistics
        self.generate_summary_stats(df)
        
        return df
    
    def generate_summary_stats(self, df: pd.DataFrame):
        """Generate and display summary statistics"""
        print("\n=== Dataset Summary ===")
        print(f"Total reviews: {len(df)}")
        print(f"Date range: {df['review_date'].min()} to {df['review_date'].max()}")
        print("\nRating distribution:")
        print(df['rating'].value_counts().sort_index())
        print("\nCategory distribution:")
        print(df['category'].value_counts())
        print("\nSentiment distribution:")
        print(df['true_sentiment'].value_counts())
        print(f"\nAverage review length: {df['review_length'].mean():.1f} words")
        print(f"Verified purchases: {df['verified_purchase'].sum()} ({df['verified_purchase'].mean()*100:.1f}%)")


if __name__ == "__main__":
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    # Check if config exists, if not use defaults
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using default configuration.")
        # You would need to create a default config here
    
    generator = CustomerReviewGenerator()
    df = generator.generate_dataset()
    
    print("\nSample reviews:")
    for _, row in df.sample(5).iterrows():
        print(f"\nCategory: {row['category']} | Rating: {row['rating']} | Sentiment: {row['true_sentiment']}")
        print(f"Review: {row['review_text']}")