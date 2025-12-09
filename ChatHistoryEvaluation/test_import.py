#!/usr/bin/env python3
"""
Simple test script to verify module imports work correctly.
"""

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import the module
try:
    from utils.evaluation_utils import ChatHistoryEvaluator

    print("✅ Module imported successfully")

    # Try to initialize evaluator with sample data
    print("Testing evaluator initialization...")
    evaluator = ChatHistoryEvaluator(
        chat_data_path="baton-export-2025-11-24-nofullstop.json", corpus_ratio=0.67
    )
    print(f"✅ Evaluator initialized. Test set size: {len(evaluator.test_df)}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
