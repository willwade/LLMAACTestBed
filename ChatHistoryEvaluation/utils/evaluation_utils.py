"""
Utility functions for chat history-driven LLM evaluation.

This module provides core functions for evaluating LLM-based text completion systems
using real user chat history data.
"""

import json
import math
import os
import re
import warnings
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# LLM and embedding libraries
import llm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class ChatHistoryEvaluator:
    """
    A class for evaluating LLM-based text completion systems using chat history.
    """

    def __init__(
        self,
        chat_data_path: str,
        api_key: Optional[str] = None,
        corpus_ratio: float = 0.67,
        model_name: str = "gemini-2.0-flash-exp",
    ):
        """
        Initialize evaluator, models, and dataset split.

        Args:
            chat_data_path: Path to the chat history JSON file.
            api_key: API key for the LLM (if not set as environment variable).
            corpus_ratio: Fraction of data to use as corpus (default 0.67).
            model_name: Name of the generative model to use.
        """
        self.chat_data_path = chat_data_path
        self.corpus_ratio = corpus_ratio
        self.model_name = model_name
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Configure API key if provided
        if api_key:
            os.environ["LLM_GEMINI_KEY"] = api_key

        # Initialize models
        self._init_models()

        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        self._split_data()

    def _init_models(self):
        """Initialize generative and judge LLM handles; fail fast on configuration errors."""
        try:
            self.generative_model = llm.get_model(self.model_name)
            self.judge_model = llm.get_model(self.model_name)
            print(f"✅ Models initialized: {self.model_name}")
        except Exception as e:
            raise Exception(f"Failed to initialize models: {e}")

    def _load_data(self):
        """Load chat history JSON from disk into memory."""
        try:
            with open(self.chat_data_path, "r") as f:
                self.chat_data = json.load(f)
            print(
                f"✅ Loaded {len(self.chat_data['sentences'])} utterances from chat history."
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Chat history file not found: {self.chat_data_path}"
            )
        except Exception as e:
            raise Exception(f"Error loading chat data: {e}")

    def _preprocess_data(self):
        """Convert raw chat data to a structured, time-sorted DataFrame with metadata."""
        sentences = self.chat_data["sentences"]
        processed_data = []

        for sentence in sentences:
            content = sentence["content"]

            # Extract timestamp from metadata
            if sentence.get("metadata") and len(sentence["metadata"]) > 0:
                timestamp_str = None
                latitude = longitude = None

                for meta in sentence["metadata"]:
                    if "timestamp" in meta:
                        timestamp_str = meta["timestamp"]
                        latitude = meta.get("latitude")
                        longitude = meta.get("longitude")
                        break

                # Convert timestamp to datetime
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        timestamp = datetime.strptime(
                            timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ"
                        )
                else:
                    timestamp = None
            else:
                timestamp = latitude = longitude = None

            # Calculate utterance length
            word_count = len(content.split()) if content else 0

            processed_data.append(
                {
                    "content": content,
                    "timestamp": timestamp,
                    "latitude": latitude,
                    "longitude": longitude,
                    "word_count": word_count,
                    "uuid": sentence.get("uuid"),
                    "anonymous_uuid": sentence.get("anonymousUUID"),
                }
            )

        # Create DataFrame and clean data
        self.chat_df = pd.DataFrame(processed_data)
        self.chat_df = self.chat_df.dropna(subset=["content", "timestamp"])
        self.chat_df = self.chat_df.sort_values("timestamp").reset_index(drop=True)

        # Add time features
        self.chat_df["hour"] = self.chat_df["timestamp"].dt.hour
        self.chat_df["day_of_week"] = self.chat_df["timestamp"].dt.dayofweek

        print(f"✅ Processed data: {len(self.chat_df)} utterances with timestamps.")
        print(
            f"Date range: {self.chat_df['timestamp'].min()} to {self.chat_df['timestamp'].max()}"
        )
        print(
            f"Average utterance length: {self.chat_df['word_count'].mean():.1f} words"
        )

    def _split_data(self):
        """Split data chronologically into corpus (conditioning) and test (evaluation) sets."""
        split_idx = int(len(self.chat_df) * self.corpus_ratio)

        self.corpus_df = self.chat_df.iloc[:split_idx].reset_index(drop=True)
        self.test_df = self.chat_df.iloc[split_idx:].reset_index(drop=True)

        print(f"Corpus (for conditioning): {len(self.corpus_df)} utterances")
        print(f"Test (for evaluation): {len(self.test_df)} utterances")

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute haversine distance between two lat/lon pairs in km."""
        R = 6371  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _filter_corpus_by_context(
        self,
        timestamp: Optional[datetime],
        latitude: Optional[float],
        longitude: Optional[float],
        context_filter: str = "none",
        time_window_hours: float = 3.0,
        geo_radius_km: float = 10.0,
    ) -> pd.DataFrame:
        """
        Filter the corpus using temporal and/or location signals.

        - time: keep corpus utterances within +/- time_window_hours of the sample timestamp.
        - geo: keep corpus utterances within geo_radius_km of the sample lat/lon.
        - time_geo: apply both filters (intersection).
        """
        if not context_filter or context_filter == "none":
            return self.corpus_df

        filtered = self.corpus_df
        applied_filter = False

        if context_filter in {"time", "time_geo"} and timestamp is not None:
            time_deltas = (
                filtered["timestamp"]
                .dropna()
                .apply(lambda t: abs((t - timestamp).total_seconds()) / 3600.0)
            )
            filtered = filtered.loc[time_deltas[time_deltas <= time_window_hours].index]
            applied_filter = True

        if context_filter in {"geo", "time_geo"} and latitude is not None and longitude is not None:
            geo_mask = []
            for _, row in filtered.iterrows():
                if row["latitude"] is None or row["longitude"] is None:
                    geo_mask.append(False)
                    continue
                distance = self._haversine_km(
                    latitude, longitude, row["latitude"], row["longitude"]
                )
                geo_mask.append(distance <= geo_radius_km)
            filtered = filtered[pd.Series(geo_mask, index=filtered.index)]
            applied_filter = True

        # Fallback to full corpus if filtering removed everything
        if applied_filter and filtered.empty:
            return self.corpus_df

        return filtered.reset_index(drop=True)

    @staticmethod
    def create_prefix_partial(text: str, n_words: int = 3) -> str:
        """
        Create a partial utterance by taking the first N words.
        """
        words = text.split()
        if len(words) <= n_words:
            return text
        return " ".join(words[:n_words])

    @staticmethod
    def create_keyword_partial(text: str, n_keywords: int = 2) -> str:
        """
        Create a partial utterance by extracting the most salient keywords
        using simple word frequency.
        """
        words = text.split()
        if len(words) <= n_keywords:
            return text

        # Simple keyword extraction using word frequency
        word_freq = Counter([word.lower() for word in words if len(word) > 2])

        # Get the most frequent words
        top_words = [word for word, count in word_freq.most_common(n_keywords)]

        # Find these words in the original order
        partial_words = []
        for word in words:
            if word.lower() in top_words and len(partial_words) < n_keywords:
                partial_words.append(word)

        # If we couldn't find enough unique words, fall back to first N words
        if len(partial_words) < n_keywords:
            return ChatHistoryEvaluator.create_prefix_partial(text, n_keywords)

        return " ".join(partial_words)

    @staticmethod
    def create_random_partial(text: str, min_words: int = 1, max_words: int = 3) -> str:
        """
        Create a partial utterance by selecting random words.
        """
        words = text.split()
        n_words = min(len(words), np.random.randint(min_words, max_words + 1))

        # Select random indices
        indices = np.random.choice(len(words), size=n_words, replace=False)
        indices = sorted(indices)  # Maintain original order

        return " ".join([words[i] for i in indices])

    def retrieve_lexical_examples(
        self, partial_text: str, top_k: int = 3, corpus_df: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Retrieve corpus examples containing exact word overlaps with the partial text.
        """
        matching_examples = []
        corpus = corpus_df if corpus_df is not None else self.corpus_df
        partial_words = set(partial_text.lower().split())

        for _, row in corpus.iterrows():
            content = row["content"]
            content_words = set(content.lower().split())

            # Check if there's any overlap
            overlap = len(partial_words.intersection(content_words))

            if overlap > 0:
                matching_examples.append(
                    {
                        "content": content,
                        "overlap": overlap,
                        "timestamp": row["timestamp"],
                    }
                )

        # Sort by overlap count and take top_k
        matching_examples.sort(key=lambda x: x["overlap"], reverse=True)
        return matching_examples[:top_k]

    def retrieve_tfidf_examples(
        self, partial_text: str, top_k: int = 3, corpus_df: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Retrieve corpus examples using TF-IDF similarity over the corpus.
        """
        corpus = corpus_df if corpus_df is not None else self.corpus_df
        corpus_contents = corpus["content"].tolist()

        all_texts = corpus_contents + [partial_text]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        partial_vector = tfidf_matrix[-1]  # Last item is our partial text
        corpus_vectors = tfidf_matrix[:-1]  # All but last are corpus items

        similarities = cosine_similarity(partial_vector, corpus_vectors).flatten()

        # Get the top_k most similar examples
        top_indices = similarities.argsort()[-top_k:][::-1]

        examples = []
        for idx in top_indices:
            examples.append(
                {
                    "content": corpus_contents[idx],
                    "similarity": similarities[idx],
                    "timestamp": corpus.iloc[idx]["timestamp"],
                }
            )

        return examples

    def retrieve_embedding_examples(
        self, partial_text: str, top_k: int = 3, corpus_df: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Retrieve corpus examples using embedding similarity.
        """
        corpus = corpus_df if corpus_df is not None else self.corpus_df
        corpus_contents = corpus["content"].tolist()
        corpus_embeddings = self.embedding_model.encode(corpus_contents)

        # Generate embedding for partial text
        partial_embedding = self.embedding_model.encode([partial_text])

        # Calculate similarity
        similarities = cosine_similarity(partial_embedding, corpus_embeddings).flatten()

        # Get the top_k most similar examples
        top_indices = similarities.argsort()[-top_k:][::-1]

        examples = []
        for idx in top_indices:
            examples.append(
                {
                    "content": corpus_contents[idx],
                    "similarity": similarities[idx],
                    "timestamp": self.corpus_df.iloc[idx]["timestamp"],
                }
            )

        return examples

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_completion(
        self,
        partial_text: str,
        examples: Optional[List[Dict]] = None,
        context: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        """
        Generate a completion for a partial utterance using an LLM, optionally conditioned
        on retrieved examples, metadata context, and recent conversation.
        """
        system_prompt = "You are an intelligent AAC text completion system. Complete the user's partial text based on provided context and examples."

        user_prompt = f"Complete the following partial text: '{partial_text}'\n\n"

        # Add context if provided
        if context:
            user_prompt += f"Context: {context}\n\n"

        if conversation_context:
            user_prompt += f"Recent conversation context:\n{conversation_context}\n\n"

        # Add examples if provided
        if examples:
            user_prompt += "Here are some examples of similar utterances:\n"
            for i, example in enumerate(examples, 1):
                user_prompt += f"{i}. {example['content']}\n"
            user_prompt += "\n"

        user_prompt += "Provide a completion that matches the user's likely intent. Only return the completed text, no explanation."

        try:
            response = self.generative_model.prompt(
                user_prompt, system=system_prompt, temperature=0.2
            )
            return response.text().strip()
        except Exception as e:
            print(f"Error generating completion: {e}")
            return partial_text  # Fallback to returning partial text

    def generate_with_lexical_retrieval(
        self,
        partial_text: str,
        context: Optional[str] = None,
        corpus_df: Optional[pd.DataFrame] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        """Generate a completion using exact-match retrieval examples."""
        corpus = corpus_df if corpus_df is not None else self.corpus_df
        examples = self.retrieve_lexical_examples(partial_text, corpus_df=corpus)
        return self.generate_completion(
            partial_text, examples, context, conversation_context
        )

    def generate_with_tfidf_retrieval(
        self,
        partial_text: str,
        context: Optional[str] = None,
        corpus_df: Optional[pd.DataFrame] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        """Generate a completion using TF-IDF retrieval examples."""
        corpus = corpus_df if corpus_df is not None else self.corpus_df
        examples = self.retrieve_tfidf_examples(partial_text, corpus_df=corpus)
        return self.generate_completion(
            partial_text, examples, context, conversation_context
        )

    def generate_with_embedding_retrieval(
        self,
        partial_text: str,
        context: Optional[str] = None,
        corpus_df: Optional[pd.DataFrame] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        """Generate a completion using embedding-based retrieval examples."""
        corpus = corpus_df if corpus_df is not None else self.corpus_df
        examples = self.retrieve_embedding_examples(partial_text, corpus_df=corpus)
        return self.generate_completion(
            partial_text, examples, context, conversation_context
        )

    @staticmethod
    def generate_with_context_only(
        self,
        partial_text: str,
        context: str,
        corpus_df: Optional[pd.DataFrame] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        """Generate a completion using only contextual information (no corpus examples)."""
        system_prompt = "You are an intelligent AAC text completion system."
        user_prompt = (
            f"Complete the following partial text: '{partial_text}'\n\n"
            f"Context: {context}\n\n"
        )

        if conversation_context:
            user_prompt += f"Recent conversation context:\n{conversation_context}\n\n"

        user_prompt += "Provide a completion that matches the user's likely intent."

        try:
            response = self.generative_model.prompt(
                user_prompt, system=system_prompt, temperature=0.2
            )
            return response.text().strip()
        except Exception as e:
            print(f"Error generating context-only completion: {e}")
            return partial_text  # Fallback to returning partial text

    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using sentence embeddings.
        """
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def judge_similarity(self, target: str, proposal: str) -> int:
        """
        Use an LLM to judge semantic similarity between a target and proposal.
        """
        system_prompt = "You are an expert evaluator of AAC text completions."

        user_prompt = f"""
        Compare these two phrases:
        1. TARGET (what the user actually said): "{target}"
        2. PROPOSAL (what the system predicted): "{proposal}"

        Rate semantic similarity on a scale of 1 to 10:
        1 = Completely unrelated or harmful
        5 = Vaguely related but incorrect
        10 = Perfect match in meaning and intent

        Return ONLY an integer score, nothing else.
        """

        try:
            response = self.judge_model.prompt(
                user_prompt, system=system_prompt, temperature=0.0
            )
            # Extract integer score from the response
            score_text = response.text().strip()
            score = "".join(filter(str.isdigit, score_text))
            return int(score) if score else 0
        except Exception as e:
            print(f"Error judging similarity: {e}")
            return 0  # Fallback to minimum score

    @staticmethod
    def calculate_character_accuracy(target: str, proposal: str) -> float:
        """
        Calculate character-level accuracy between target and proposal.
        """
        if not target or not proposal:
            return 0.0

        # Use Levenshtein distance approximated by sequence matcher
        import difflib

        similarity = difflib.SequenceMatcher(None, target, proposal).ratio()
        return similarity

    @staticmethod
    def calculate_word_accuracy(target: str, proposal: str) -> float:
        """
        Calculate word-level accuracy between target and proposal.
        """
        if not target or not proposal:
            return 0.0

        target_words = set(target.lower().split())
        proposal_words = set(proposal.lower().split())

        if not target_words:
            return 0.0

        intersection = target_words.intersection(proposal_words)
        return len(intersection) / len(target_words)

    def run_evaluation(
        self,
        partial_methods: Dict[str, Callable],
        generation_methods: Dict[str, Callable],
        evaluation_metrics: Dict[str, Callable],
        sample_size: Optional[int] = None,
        context_filter: str = "none",
        time_window_hours: float = 3.0,
        geo_radius_km: float = 10.0,
        conversation_window: int = 0,
    ) -> pd.DataFrame:
        """
        Run the evaluation grid across partial methods, generation methods,
        and metrics, with optional context/conversation conditioning.
        """
        # Sample test data if requested
        if sample_size and sample_size < len(self.test_df):
            eval_df = self.test_df.sample(sample_size, random_state=42).reset_index(
                drop=True
            )
        else:
            eval_df = self.test_df

        results = []

        for i, row in eval_df.iterrows():
            target_text = row["content"]

            # Skip very short utterances
            if len(target_text.split()) < 3:
                continue

            print(f"Processing example {i+1}/{len(eval_df)}: '{target_text[:30]}...'")

            # Extract context information
            timestamp = row["timestamp"]
            latitude = row["latitude"]
            longitude = row["longitude"]
            base_context = f"Time: {timestamp}, Location: {latitude}, {longitude}"

            conversation_context = None
            if conversation_window > 0 and i > 0:
                start_idx = max(0, i - conversation_window)
                recent = eval_df.iloc[start_idx:i]["content"].tolist()
                if recent:
                    conversation_context = "\n".join([f"- {utt}" for utt in recent])

            filtered_corpus = self._filter_corpus_by_context(
                timestamp=timestamp,
                latitude=latitude,
                longitude=longitude,
                context_filter=context_filter,
                time_window_hours=time_window_hours,
                geo_radius_km=geo_radius_km,
            )

            # Build combined context string for prompts
            context_parts = []
            if base_context:
                context_parts.append(base_context)
            if context_filter and context_filter != "none":
                context_parts.append(f"Context filter: {context_filter}")
            context = "\n".join(context_parts)

            # Test each partial utterance method
            for partial_method_name, partial_method in partial_methods.items():
                partial_text = partial_method(target_text)

                # Skip if partial text is the same as target
                if partial_text == target_text:
                    continue

                # Test each generation method
                for gen_method_name, gen_method in generation_methods.items():
                    try:
                        # Generate proposal
                        proposal = gen_method(
                            self,
                            partial_text,
                            context,
                            filtered_corpus,
                            conversation_context,
                        )

                        # Evaluate with each metric
                        result = {
                            "target": target_text,
                            "partial": partial_text,
                            "proposal": proposal,
                            "partial_method": partial_method_name,
                            "generation_method": gen_method_name,
                        }

                        # Some metric functions (e.g., embedding/LLM judges) need the evaluator instance.
                        metrics_needing_evaluator = {"embedding_similarity", "llm_judge_score", "judge_similarity"}

                        for metric_name, metric_func in evaluation_metrics.items():
                            if metric_name in metrics_needing_evaluator:
                                result[metric_name] = metric_func(
                                    self, target_text, proposal
                                )
                            else:
                                result[metric_name] = metric_func(target_text, proposal)

                        results.append(result)

                    except Exception as e:
                        print(
                            f"  Error with {partial_method_name} + {gen_method_name}: {e}"
                        )

        return pd.DataFrame(results)

    def visualize_results(self, results_df: pd.DataFrame):
        """
        Create visualizations for evaluation results.
        """
        if results_df.empty:
            print("No results to visualize.")
            return

        # Group by methods and calculate mean scores
        grouped_results = (
            results_df.groupby(["partial_method", "generation_method"])
            .agg(
                {
                    "embedding_similarity": "mean",
                    "llm_judge_score": "mean",
                    "character_accuracy": "mean",
                    "word_accuracy": "mean",
                    "target": "count",  # Count of samples
                }
            )
            .rename(columns={"target": "count"})
            .reset_index()
        )

        # Display results
        display(grouped_results)

        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Performance by Partial Method and Generation Method", fontsize=16)

        metrics = [
            "embedding_similarity",
            "llm_judge_score",
            "character_accuracy",
            "word_accuracy",
        ]
        titles = [
            "Embedding Similarity",
            "LLM Judge Score",
            "Character Accuracy",
            "Word Accuracy",
        ]

        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            if metric in grouped_results.columns:
                pivot = grouped_results.pivot(
                    index="partial_method", columns="generation_method", values=metric
                )
                sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
                ax.set_title(title)

        plt.tight_layout()
        plt.show()
