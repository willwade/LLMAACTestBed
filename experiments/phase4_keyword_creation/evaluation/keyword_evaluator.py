"""
Keyword Evaluator for Phase 4 Experiments

Evaluates LLM predictions for keyword-to-utterance generation
with varying levels of contextual enhancement.
"""

from typing import Any

import pandas as pd

from .context_builder import ContextBuilder


class KeywordEvaluator:
    """Evaluates LLM predictions for keyword combinations."""

    def __init__(self, llm_client, logger=None):
        """
        Initialize keyword evaluator.

        Args:
            llm_client: LLM client for generation and scoring
            logger: Optional logger instance
        """
        self.llm_client = llm_client
        self.logger = logger
        self.context_builder = ContextBuilder(logger)

    def _extract_keywords(self, row) -> list[str]:
        """Extract keywords from a DataFrame row."""
        keywords = []
        for col in ["Key word", "Key Word2", "Key Word3"]:
            if pd.notna(row[col]) and str(row[col]).strip() != "N/A":
                keywords.append(str(row[col]).strip())
        return keywords

    def _create_baseline_prompt(self, keywords: list[str]) -> str:
        """Create prompt for baseline testing (keywords only)."""
        keywords_str = ", ".join(keywords)

        return f"""You are an AAC assistant communicating for Dwayne, an MND patient.

KEYWORDS: {keywords_str}

Based only on these keywords, give ONE concise, actionable utterance that states exactly what Dwayne wants done. Include the specific action/object implied by the keywords. Do not add pleasantries or labels. Output only the utterance."""

    def _create_contextual_prompt(self, keywords: list[str], context: dict[str, Any]) -> str:
        """Create prompt with contextual information."""
        keywords_str = ", ".join(keywords)

        # Build context string
        context_parts = []

        if "location" in context:
            context_parts.append(f"LOCATION: {context['location']}")

        if "people_present" in context:
            context_parts.append(f"PEOPLE PRESENT: {context['people_present']}")

        if "time_of_day" in context:
            context_parts.append(f"TIME: {context['time_of_day']}")

        if "relationship_context" in context:
            context_parts.append(f"RELATIONSHIPS: {context['relationship_context']}")

        if "recent_activities" in context:
            context_parts.append(f"RECENT ACTIVITIES: {context['recent_activities']}")

        if "medical_context" in context:
            context_parts.append(f"HEALTH STATUS: {context['medical_context']}")

        context_str = (
            "\n".join(context_parts) if context_parts else "No additional context available"
        )

        return f"""You are an AAC assistant communicating for Dwayne, an MND patient.

CONTEXT:
{context_str}

KEYWORDS: {keywords_str}

Write ONE concise, actionable utterance that states exactly what Dwayne wants done, using the context to pick the specific action/object (e.g., what to apply, move, fetch). Keep it telegraphic but complete. Do not add pleasantries or labels. Output only the utterance."""

    def _evaluate_prediction(self, prediction: str, target: str) -> int:
        """Evaluate prediction against target using LLM judge."""
        try:
            score = self.llm_client.judge_similarity(target, prediction)
            return int(min(10, max(1, score)))  # Ensure score is within 1-10 range
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error evaluating prediction: {e}")
            return 5  # Default middle score

    def run_baseline_test(self, keywords_df, sample_size: int | None = None) -> dict[str, Any]:
        """Run Part 1: Baseline testing with keywords only."""
        if sample_size:
            keywords_df = keywords_df.head(sample_size)

        results: dict[str, Any] = {
            "experiment_type": "baseline_keywords_only",
            "total_tests": len(keywords_df),
            "predictions": [],
            "scores": [],
            "target_texts": [],
            "keyword_combinations": [],
        }

        for _, row in keywords_df.iterrows():
            # Extract keywords and target
            keywords = self._extract_keywords(row)
            target = str(row["Instruction"]).strip()

            if not keywords or not target:
                continue

            # Generate prediction
            prompt = self._create_baseline_prompt(keywords)
            prediction = self.llm_client.generate(prompt, temperature=0.2)

            # Evaluate prediction
            score = self._evaluate_prediction(prediction, target)

            # Store results
            results["predictions"].append(prediction)
            results["scores"].append(score)
            results["target_texts"].append(target)
            results["keyword_combinations"].append(keywords)

            if self.logger:
                self.logger.debug(f"Baseline test: {keywords} -> Score: {score}/10")

        return results

    def run_contextual_test(
        self, keywords_df, social_graph, sample_size: int | None = None
    ) -> dict[str, Any]:
        """Run Part 2: Contextual enhancement testing."""
        if sample_size:
            keywords_df = keywords_df.head(sample_size)

        # Define context levels
        context_levels = [
            "location_only",
            "location_people",
            "location_equipment",
            "social_relationships",
            "full_context",
        ]

        results: dict[str, Any] = {
            "experiment_type": "contextual_enhancement",
            "total_tests": len(keywords_df),
            "context_levels": {},
            "improvements_by_level": {},
        }

        for level in context_levels:
            level_results: dict[str, list[Any]] = {
                "predictions": [],
                "scores": [],
                "target_texts": [],
                "keyword_combinations": [],
                "contexts_used": [],
            }

            for _, row in keywords_df.iterrows():
                # Extract keywords and target
                keywords = self._extract_keywords(row)
                target = str(row["Instruction"]).strip()

                if not keywords or not target:
                    continue

                # Build context based on level
                context = self.context_builder.build_context(
                    level=level, social_graph=social_graph, keywords=keywords
                )

                # Generate prediction
                prompt = self._create_contextual_prompt(keywords, context)
                prediction = self.llm_client.generate(prompt, temperature=0.2)

                # Evaluate prediction
                score = self._evaluate_prediction(prediction, target)

                # Store results
                level_results["predictions"].append(prediction)
                level_results["scores"].append(score)
                level_results["target_texts"].append(target)
                level_results["keyword_combinations"].append(keywords)
                level_results["contexts_used"].append(context)

            results["context_levels"][level] = level_results

        # Calculate improvements compared to baseline
        if "baseline_scores" in results.get("context_levels", {}).get("location_only", {}).get(
            "scores", []
        ):
            baseline_scores = results["context_levels"]["location_only"]["scores"]
            for level in context_levels[1:]:  # Skip baseline
                if level in results["context_levels"]:
                    level_scores = results["context_levels"][level]["scores"]
                    if len(level_scores) == len(baseline_scores):
                        improvements = [
                            level_scores[i] - baseline_scores[i] for i in range(len(level_scores))
                        ]
                        results["improvements_by_level"][level] = {
                            "mean_improvement": sum(improvements) / len(improvements),
                            "positive_improvements": sum(1 for imp in improvements if imp > 0),
                            "improvement_rate": sum(1 for imp in improvements if imp > 0)
                            / len(improvements),
                        }

        return results

    def run_single_keyword_test(
        self, keywords_df, social_graph, sample_size: int | None = None
    ) -> dict[str, Any]:
        """Run Part 3: Single keyword testing with optimal context."""
        if sample_size:
            keywords_df = keywords_df.head(sample_size)

        # Identify high-value single keywords
        high_value_keywords = [
            "Pain",
            "Scratch",
            "Move",
            "Sick",
            "Help",
            "Chair",
            "Transfer",
            "Feed",
        ]

        results: dict[str, Any] = {
            "experiment_type": "single_keyword_optimal_context",
            "total_tests": 0,
            "keyword_results": {},
            "effectiveness_by_keyword": {},
        }

        for keyword in high_value_keywords:
            keyword_results: dict[str, list[Any]] = {
                "predictions": [],
                "scores": [],
                "target_texts": [],
                "contexts_used": [],
            }

            # Find rows with this keyword
            matching_rows = keywords_df[
                keywords_df[["Key word", "Key Word2", "Key Word3"]].apply(
                    lambda row, kw=keyword: kw in row.values, axis=1
                )
            ]

            for _, row in matching_rows.iterrows():
                target = str(row["Instruction"]).strip()
                if not target:
                    continue

                # Use optimal context (full context)
                context = self.context_builder.build_context(
                    level="full_context", social_graph=social_graph, keywords=[keyword]
                )

                # Generate prediction
                prompt = self._create_contextual_prompt([keyword], context)
                prediction = self.llm_client.generate(prompt, temperature=0.2)

                # Evaluate prediction
                score = self._evaluate_prediction(prediction, target)

                # Store results
                keyword_results["predictions"].append(prediction)
                keyword_results["scores"].append(score)
                keyword_results["target_texts"].append(target)
                keyword_results["contexts_used"].append(context)

                results["total_tests"] = int(results["total_tests"]) + 1

            if keyword_results["scores"]:
                results["keyword_results"][keyword] = keyword_results
                results["effectiveness_by_keyword"][keyword] = {
                    "total_tests": len(keyword_results["scores"]),
                    "mean_score": sum(keyword_results["scores"]) / len(keyword_results["scores"]),
                    "max_score": max(keyword_results["scores"]),
                    "success_rate": sum(1 for s in keyword_results["scores"] if s >= 7)
                    / len(keyword_results["scores"])
                    * 100,
                }

        return results
