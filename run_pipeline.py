"""
Automated Pipeline Runner for Customer Feedback Analytics
Runs the complete end-to-end pipeline from data generation to insights
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime
import yaml
import argparse
from tqdm import tqdm
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Orchestrates the complete analytics pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline runner"""
        self.start_time = datetime.now()
        self.config_path = config_path
        self.steps_completed = []
        self.errors = []
        
        # Create necessary directories
        self.setup_directories()
        
        # Load configuration
        self.load_config()
        
    def setup_directories(self):
        """Create required directory structure"""
        directories = [
            'data/raw', 'data/processed', 'data/external',
            'models/sentiment', 'models/topics',
            'reports', 'logs', 'visualizations/topics',
            'dashboard/assets'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Directory structure created successfully")
    
    def load_config(self):
        """Load or create configuration file"""
        if not os.path.exists(self.config_path):
            logger.warning("Config file not found. Creating default configuration...")
            self.create_default_config()
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'data': {
                'raw_data_path': 'data/raw/customer_reviews.csv',
                'processed_data_path': 'data/processed/preprocessed_reviews.csv',
                'predictions_path': 'data/processed/sentiment_predictions.csv',
                'num_reviews': 10000,
                'categories': [
                    'Electronics', 'Fashion', 'Home & Kitchen',
                    'Sports & Outdoors', 'Beauty & Personal Care',
                    'Books', 'Toys & Games', 'Health & Household'
                ],
                'typo_rate': 0.15,
                'slang_rate': 0.20,
                'mixed_expression_rate': 0.10
            },
            'preprocessing': {
                'min_review_length': 10,
                'max_review_length': 1000,
                'remove_stopwords': True,
                'lemmatize': True,
                'handle_negations': True,
                'remove_urls': True,
                'remove_emails': True,
                'remove_special_chars': True,
                'lowercase': True
            },
            'models': {
                'sentiment': {
                    'tfidf': {
                        'max_features': 5000,
                        'ngram_range': [1, 3],
                        'min_df': 5,
                        'max_df': 0.95
                    },
                    'transformer': {
                        'model_name': 'distilbert-base-uncased',
                        'max_length': 256,
                        'learning_rate': 2e-5,
                        'batch_size': 16,
                        'epochs': 3
                    }
                },
                'topic_modeling': {
                    'num_topics': 10,
                    'min_topic_size': 50,
                    'n_gram_range': [1, 3],
                    'diversity': 0.3
                }
            },
            'analysis': {
                'confidence_threshold': 0.8,
                'trend_window_days': 30,
                'satisfaction_goal': 0.8
            },
            'business_rules': {
                'sentiment_weights': {
                    'rating_weight': 0.3,
                    'text_weight': 0.7
                }
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        self.config = default_config
        logger.info(f"Default configuration created at {self.config_path}")
    
    def check_dependencies(self):
        """Check if all required packages are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'nltk', 'spacy',
            'transformers', 'streamlit', 'plotly', 'yaml',
            'bertopic', 'yake', 'tqdm'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Please run: pip install -r requirements.txt")
            return False
            
        # Check NLTK data
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        
        # Check spaCy model
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except:
            logger.info("Downloading spaCy model...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        
        logger.info("All dependencies satisfied ✓")
        return True
    
    def run_step(self, step_name: str, module_path: str, description: str):
        """Run a single pipeline step"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {description}")
        logger.info(f"{'='*60}")
        
        try:
            # Import and run the module
            module_dir = os.path.dirname(module_path)
            module_name = os.path.basename(module_path).replace('.py', '')
            
            sys.path.insert(0, module_dir)
            module = __import__(module_name)
            
            # Run main function if exists
            if hasattr(module, 'main'):
                module.main()
            else:
                logger.warning(f"No main() function found in {module_path}")
            
            self.steps_completed.append(step_name)
            logger.info(f"✓ {step_name} completed successfully")
            
        except Exception as e:
            error_msg = f"Error in {step_name}: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            raise
        finally:
            sys.path.pop(0)
    
    def run_pipeline(self, skip_data_generation: bool = False):
        """Run the complete analytics pipeline"""
        logger.info("\n" + "="*60)
        logger.info("CUSTOMER FEEDBACK ANALYTICS PIPELINE")
        logger.info("="*60)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Pipeline steps
        steps = [
            {
                'name': 'data_generation',
                'path': 'src/data_processing/data_generator.py',
                'description': 'Generating synthetic customer review data',
                'skip': skip_data_generation
            },
            {
                'name': 'preprocessing',
                'path': 'src/data_processing/preprocessor.py',
                'description': 'Preprocessing and cleaning review text',
                'skip': False
            },
            {
                'name': 'sentiment_analysis',
                'path': 'src/models/sentiment_analyzer.py',
                'description': 'Training sentiment analysis models',
                'skip': False
            },
            {
                'name': 'topic_extraction',
                'path': 'src/models/topic_extractor.py',
                'description': 'Extracting topics and keywords',
                'skip': False
            },
            {
                'name': 'business_insights',
                'path': 'src/utils/business_insights.py',
                'description': 'Generating business insights and recommendations',
                'skip': False
            }
        ]
        
        # Progress bar
        with tqdm(total=len([s for s in steps if not s.get('skip')]), 
                  desc="Pipeline Progress") as pbar:
            
            for step in steps:
                if step.get('skip'):
                    logger.info(f"Skipping {step['name']} (as requested)")
                    continue
                    
                self.run_step(
                    step['name'],
                    step['path'],
                    step['description']
                )
                pbar.update(1)
                time.sleep(1)  # Brief pause between steps
        
        # Generate summary
        self.generate_pipeline_summary()
        
        return len(self.errors) == 0
    
    def generate_pipeline_summary(self):
        """Generate summary of pipeline execution"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration: {duration}")
        logger.info(f"\nSteps Completed: {len(self.steps_completed)}")
        for step in self.steps_completed:
            logger.info(f"  ✓ {step}")
        
        if self.errors:
            logger.error(f"\nErrors Encountered: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  ✗ {error}")
        else:
            logger.info("\n✅ All steps completed successfully!")
        
        # Check output files
        logger.info("\nOutput Files Generated:")
        output_files = [
            ('Raw Data', self.config['data']['raw_data_path']),
            ('Processed Data', self.config['data']['processed_data_path']),
            ('Predictions', self.config['data']['predictions_path']),
            ('Executive Summary', 'reports/executive_summary.md'),
            ('Topic Report', 'models/topics/topic_analysis_report.pkl')
        ]
        
        for name, path in output_files:
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024  # KB
                logger.info(f"  ✓ {name}: {path} ({size:.1f} KB)")
            else:
                logger.warning(f"  ✗ {name}: {path} (not found)")
    
    def launch_dashboard(self):
        """Launch the Streamlit dashboard"""
        logger.info("\n" + "="*60)
        logger.info("LAUNCHING DASHBOARD")
        logger.info("="*60)
        
        dashboard_path = "dashboard/app.py"
        
        if not os.path.exists(dashboard_path):
            logger.error(f"Dashboard file not found: {dashboard_path}")
            return
        
        logger.info("Starting Streamlit dashboard...")
        logger.info("Dashboard will open in your default browser")
        logger.info("Press Ctrl+C to stop the dashboard")
        
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
        except KeyboardInterrupt:
            logger.info("\nDashboard stopped by user")
        except Exception as e:
            logger.error(f"Error launching dashboard: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Customer Feedback Analytics Pipeline Runner"
    )
    parser.add_argument(
        '--skip-data-generation',
        action='store_true',
        help='Skip data generation step (use existing data)'
    )
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Launch dashboard without running pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PipelineRunner(config_path=args.config)
    
    if args.dashboard_only:
        # Just launch dashboard
        runner.launch_dashboard()
    else:
        # Run pipeline
        success = runner.run_pipeline(skip_data_generation=args.skip_data_generation)
        
        if success:
            # Ask user if they want to launch dashboard
            logger.info("\n" + "="*60)
            response = input("\nPipeline completed! Launch dashboard? (y/n): ")
            if response.lower() == 'y':
                runner.launch_dashboard()
            else:
                logger.info("Dashboard can be launched later with: streamlit run dashboard/app.py")
        else:
            logger.error("\nPipeline failed. Please check the errors above.")
            sys.exit(1)


if __name__ == "__main__":
    main()