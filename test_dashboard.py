"""
Quick test script to verify dashboard setup
Run this before launching the dashboard to check for issues
"""

import os
import sys
import pandas as pd

def test_data_files():
    """Test if all required data files exist"""
    print("🔍 Testing Data Files...")
    
    required_files = [
        "data/processed/sentiment_predictions.csv",
        "models/topics/topic_analysis_report.pkl",
        "config/config.yaml"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def test_page_imports():
    """Test if page modules can be imported"""
    print("\n🔍 Testing Page Imports...")
    
    # Add pages directory to path
    pages_path = os.path.join("dashboard", "pages")
    if pages_path not in sys.path:
        sys.path.insert(0, pages_path)
    
    try:
        from business_metrics import render_business_metrics_page
        print("✅ business_metrics.py - Import successful")
    except Exception as e:
        print(f"❌ business_metrics.py - Import failed: {e}")
        return False
    
    try:
        from sentiment_analysis import render_sentiment_analysis_page
        print("✅ sentiment_analysis.py - Import successful")
    except Exception as e:
        print(f"❌ sentiment_analysis.py - Import failed: {e}")
        return False
    
    try:
        from topic_insights import render_topic_insights_page
        print("✅ topic_insights.py - Import successful")
    except Exception as e:
        print(f"❌ topic_insights.py - Import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if data can be loaded properly"""
    print("\n🔍 Testing Data Loading...")
    
    try:
        # Test main dataset
        df = pd.read_csv("data/processed/sentiment_predictions.csv")
        print(f"✅ Main dataset loaded: {len(df)} rows")
        
        # Check required columns
        required_columns = [
            'predicted_sentiment', 'confidence_score', 'rating', 
            'category', 'review_text', 'review_date'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
        
        # Test topic report
        import pickle
        with open("models/topics/topic_analysis_report.pkl", 'rb') as f:
            topic_report = pickle.load(f)
        print("✅ Topic report loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_dashboard_files():
    """Test if dashboard files exist"""
    print("\n🔍 Testing Dashboard Files...")
    
    required_files = [
        "dashboard/app.py",
        "dashboard/pages/__init__.py",
        "dashboard/pages/business_metrics.py",
        "dashboard/pages/sentiment_analysis.py",
        "dashboard/pages/topic_insights.py"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🚀 Dashboard Readiness Test")
    print("=" * 50)
    
    tests = [
        ("Data Files", test_data_files),
        ("Dashboard Files", test_dashboard_files),
        ("Data Loading", test_data_loading),
        ("Page Imports", test_page_imports)
    ]
    
    all_passed = True
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all_passed:
        print("\n🎉 All tests passed! Your dashboard should work correctly.")
        print("\nTo launch dashboard:")
        print("streamlit run dashboard/app.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above before launching the dashboard.")
        print("\nCommon fixes:")
        print("- Run the pipeline: python run_pipeline.py")
        print("- Update your page files with the provided code")
        print("- Create missing __init__.py file in dashboard/pages/")

if __name__ == "__main__":
    main()