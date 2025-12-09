#!/usr/bin/env python3
"""
Simple test script to verify evaluation framework works correctly.
"""

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from utils.evaluation_utils import ChatHistoryEvaluator

    print("✅ Module imported successfully")

    # Try to initialize evaluator with sample data
    print("Testing evaluator initialization...")
    evaluator = ChatHistoryEvaluator(
        chat_data_path="baton-export-2025-11-24-nofullstop.json", corpus_ratio=0.67
    )
    print(f"✅ Evaluator initialized. Test set size: {len(evaluator.test_df)}")

    # Test a single partial utterance
    test_utterance = evaluator.test_df.iloc[0]["content"]
    print(f"Testing with utterance: '{test_utterance}'")

    # Test partial utterance creation
    partial = evaluator.create_prefix_partial(test_utterance, 3)
    print(f"Partial utterance (prefix_3): '{partial}'")

    # Test retrieval
    examples = evaluator.retrieve_lexical_examples(partial)
    print(f"Retrieved {len(examples)} examples using lexical search")

    # Test generation
    context = f"Time: {evaluator.test_df.iloc[0]['timestamp']}"
    proposal = evaluator.generate_with_lexical_retrieval(partial, context)
    print(f"Generated proposal: '{proposal}'")

    # Test evaluation metrics
    embed_sim = evaluator.calculate_embedding_similarity(test_utterance, proposal)
    print(f"Embedding similarity: {embed_sim:.3f}")

    print("✅ All tests completed successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
