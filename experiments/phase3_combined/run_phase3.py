#!/usr/bin/env python3
"""
Phase 3: Combined Experiments Runner

Integration of real chat data with synthetic social graphs and user profiles.
"""

import argparse
import sys
import importlib
from pathlib import Path

# Add project root to path for lib imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.llm_clients import create_llm_client
from lib.utils import load_env


def main():
    """Main entry point for Phase 3 experiments."""
    # Load environment
    load_env()

    parser = argparse.ArgumentParser(description="Run Phase 3 Combined Experiments")
    parser.add_argument("--provider", type=str, choices=["gemini", "openai"],
                       default="openai", help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, help="Output path for results")

    args = parser.parse_args()

    experiments_dir = Path(__file__).parent / "experiments"
    sys.path.insert(0, str(experiments_dir))

    print("=== Phase 3: Combined Experiments ===")
    print(f"LLM Provider: {args.provider or 'default'}")
    print(f"Model: {args.model or 'default'}")

    # Run profile-enhanced experiment if available
    profile_script = experiments_dir / "profile_enhanced.py"
    if profile_script.exists():
        print("\nRunning Profile-Enhanced Experiment...")
        try:
            profile_module = importlib.import_module("profile_enhanced")
            ProfileEnhancedExperiment = getattr(profile_module, "ProfileEnhancedExperiment")
            profile_config = args.config or str(Path(__file__).parent / "configs" / "profile_enhanced.yaml")
            exp = ProfileEnhancedExperiment(profile_config, args.provider, args.model)
            results = exp.run_experiment()
            exp.save_results(results, args.output)
            print(f"[ok] Profile-Enhanced completed. Results: {len(results.get('enhanced_scores', []))} evaluations")
        except Exception as e:
            print(f"[error] Profile-Enhanced failed: {e}")

    # Run social context experiment if available
    social_script = experiments_dir / "social_context.py"
    if social_script.exists():
        print("\nRunning Social Context Experiment...")
        try:
            social_module = importlib.import_module("social_context")
            SocialContextExperiment = getattr(social_module, "SocialContextExperiment")
            exp = SocialContextExperiment(args.config, args.provider, max_chats=5)
            results = exp.run_experiment()
            exp.save_results(results, args.output)
            print(f"[ok] Social Context completed. Results: {len(results.get('social_scores', []))} evaluations")
        except Exception as e:
            print(f"[error] Social Context failed: {e}")

    # Simple test if experiments aren't available
    if not profile_script.exists() and not social_script.exists():
        print("\nRunning simple Phase 3 test...")
        client = create_llm_client(args.provider, args.model)
        print(f"[ok] LLM client created: {client.provider_name}:{client.model_name}")

        # Test profile context
        profile_context = """
User: Dave (MND patient)
- Needs: Direct communication, medical assistance
- Time: 2:00 PM
- Location: Living room
"""
        prompt = f"{profile_context}\nUser says: 'I need'\nComplete based on context:"
        response = client.generate(prompt, temperature=0.2)
        print(f"[ok] Profile-aware generation: {response[:50]}...")

    print("\n[done] Phase 3 experiments completed!")


if __name__ == "__main__":
    main()
