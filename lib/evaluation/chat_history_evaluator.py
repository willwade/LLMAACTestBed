"""
Chat History Evaluator

Specialized evaluator for chat history-driven LLM evaluation.
Migrated from Phase 1 to shared library.
"""

import json
import math
import warnings
from collections import Counter
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

# Import external libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from ..llm_clients import create_llm_client
from ..utils import load_env

warnings.filterwarnings("ignore")


class ChatHistoryEvaluator:
    """
    A class for evaluating LLM-based text completion systems using chat history.
    """

    def __init__(
        self,
        chat_data_path: str,
        api_key: str | None = None,
        corpus_ratio: float = 0.67,
        model_name: str = "gemini-2.0-flash-exp",
        llm_provider: str | None = None,
    ):
        """
        Initialize evaluator, models, and dataset split.

        Args:
            chat_data_path: Path to the chat history JSON file.
            api_key: API key for the LLM (if not set as environment variable).
            corpus_ratio: Fraction of data to use as corpus (default 0.67).
            model_name: Name of the generative model to use.
            llm_provider: LLM provider (gemini, openai, etc.)
        """
        # Load environment variables
        load_env()

        self.chat_data_path = chat_data_path
        self.corpus_ratio = corpus_ratio
        self.model_name = model_name
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize LLM client
        provider = llm_provider
        if provider is None:
            lower_name = model_name.lower()
            if "gemini" in lower_name:
                provider = "gemini"
            elif lower_name.startswith("gpt") or lower_name.startswith("o"):
                provider = "openai"

        self.llm_client = create_llm_client(
            provider=provider,
            model=model_name
        )

        # Load and preprocess data
        self._load_data()
        self._preprocess_data()
        self._split_data()

    def _load_data(self):
        """Load chat history JSON from disk into memory."""
        try:
            with open(self.chat_data_path) as f:
                self.chat_data = json.load(f)
            print(f"[info] Loaded {len(self.chat_data['sentences'])} utterances from chat history.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Chat history file not found: {self.chat_data_path}") from None
        except Exception as e:
            raise Exception(f"Error loading chat data: {e}") from e

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
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    except ValueError:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
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

        print(f"[info] Processed data: {len(self.chat_df)} utterances with timestamps.")
        print(f"Date range: {self.chat_df['timestamp'].min()} to {self.chat_df['timestamp'].max()}")
        print(f"Average utterance length: {self.chat_df['word_count'].mean():.1f} words")

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
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _filter_corpus_by_context(
        self,
        timestamp: datetime | None,
        latitude: float | None,
        longitude: float | None,
        context_filter: str = "none",
        time_window_hours: float = 3.0,
        geo_radius_km: float = 10.0,
    ) -> pd.DataFrame:
        """
        Filter the corpus using temporal and/or location signals.
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
        """Create a partial utterance by taking the first N words."""
        words = text.split()
        if len(words) <= n_words:
            return text
        return " ".join(words[:n_words])

    @staticmethod
    def create_keyword_partial(text: str, n_keywords: int = 2) -> str:
        """Create a partial utterance by extracting the most salient keywords."""
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
        """Create a partial utterance by selecting random words."""
        words = text.split()
        n_words = min(len(words), np.random.randint(min_words, max_words + 1))

        # Select random indices
        indices = np.random.choice(len(words), size=n_words, replace=False)
        indices = sorted(indices)  # Maintain original order

        return " ".join([words[i] for i in indices])

    def retrieve_lexical_examples(
        self, partial_text: str, top_k: int = 3, corpus_df: pd.DataFrame | None = None
    ) -> list[dict]:
        """Retrieve corpus examples containing exact word overlaps with the partial text."""
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
        self, partial_text: str, top_k: int = 3, corpus_df: pd.DataFrame | None = None
    ) -> list[dict]:
        """Retrieve corpus examples using TF-IDF similarity over the corpus."""
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
        self, partial_text: str, top_k: int = 3, corpus_df: pd.DataFrame | None = None
    ) -> list[dict]:
        """Retrieve corpus examples using embedding similarity."""
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

    @staticmethod
    def _format_context_text(
        timestamp: datetime | None, latitude: float | None, longitude: float | None
    ) -> str | None:
        """Build a simple context string from time and location metadata."""
        parts = []
        if timestamp:
            parts.append(f"Time: {timestamp.isoformat()}")
        if latitude is not None and longitude is not None:
            parts.append(f"Location: ({latitude:.4f}, {longitude:.4f})")
        return "; ".join(parts) if parts else None

    def lexical_generate(
        self, partial_text: str, context: dict[str, Any] | None = None, top_k: int = 3
    ) -> str:
        """Generate completion using lexical retrieval examples."""
        ctx = context or {}
        examples = self.retrieve_lexical_examples(
            partial_text, top_k=top_k, corpus_df=ctx.get("corpus")
        )
        return self.generate_completion(
            partial_text,
            examples=examples,
            context=self._format_context_text(
                ctx.get("timestamp"), ctx.get("latitude"), ctx.get("longitude")
            ),
            conversation_context=ctx.get("conversation_context"),
        )

    def tfidf_generate(
        self, partial_text: str, context: dict[str, Any] | None = None, top_k: int = 3
    ) -> str:
        """Generate completion using TF-IDF retrieved examples."""
        ctx = context or {}
        examples = self.retrieve_tfidf_examples(
            partial_text, top_k=top_k, corpus_df=ctx.get("corpus")
        )
        return self.generate_completion(
            partial_text,
            examples=examples,
            context=self._format_context_text(
                ctx.get("timestamp"), ctx.get("latitude"), ctx.get("longitude")
            ),
            conversation_context=ctx.get("conversation_context"),
        )

    def embedding_generate(
        self, partial_text: str, context: dict[str, Any] | None = None, top_k: int = 3
    ) -> str:
        """Generate completion using embedding-based retrieval."""
        ctx = context or {}
        examples = self.retrieve_embedding_examples(
            partial_text, top_k=top_k, corpus_df=ctx.get("corpus")
        )
        return self.generate_completion(
            partial_text,
            examples=examples,
            context=self._format_context_text(
                ctx.get("timestamp"), ctx.get("latitude"), ctx.get("longitude")
            ),
            conversation_context=ctx.get("conversation_context"),
        )

    def context_only_generate(self, partial_text: str, context: dict[str, Any] | None = None) -> str:
        """Generate completion using only contextual metadata (no retrieved examples)."""
        ctx = context or {}
        return self.generate_completion(
            partial_text,
            context=self._format_context_text(
                ctx.get("timestamp"), ctx.get("latitude"), ctx.get("longitude")
            ),
            conversation_context=ctx.get("conversation_context"),
        )

    def embedding_similarity(self, prediction: str, target: str) -> float:
        """Wrapper for semantic similarity metric used by Phase 1."""
        return self.calculate_embedding_similarity(prediction, target)

    def llm_judge_score(self, prediction: str, target: str) -> int:
        """Wrapper for LLM judge metric used by Phase 1."""
        return self.judge_similarity(target, prediction)

    @staticmethod
    def character_accuracy(prediction: str, target: str) -> float:
        """Wrapper for character-level accuracy metric used by Phase 1."""
        return ChatHistoryEvaluator.calculate_character_accuracy(target, prediction)

    @staticmethod
    def word_accuracy(prediction: str, target: str) -> float:
        """Wrapper for word-level accuracy metric used by Phase 1."""
        return ChatHistoryEvaluator.calculate_word_accuracy(target, prediction)

    def run_evaluation(
        self,
        partial_methods: dict[str, Callable[[str], str]],
        generation_methods: dict[str, Callable[[str, dict[str, Any]], str]],
        evaluation_metrics: dict[str, Callable[[str, str], Any]],
        sample_size: int | None = None,
        context_filter: str = "none",
        time_window_hours: float = 3.0,
        geo_radius_km: float = 10.0,
        conversation_window: int = 0,
    ) -> pd.DataFrame:
        """
        Run evaluation across test utterances using provided methods and metrics.

        Args:
            partial_methods: Mapping of partial method name to function(text) -> partial text
            generation_methods: Mapping of generation name to function(partial, context) -> proposal
            evaluation_metrics: Mapping of metric name to function(prediction, target) -> score
            sample_size: Optional number of test samples to evaluate
            context_filter: How to filter corpus (none, time, geo, time_geo)
            time_window_hours: Window for temporal filtering
            geo_radius_km: Radius for geographic filtering
            conversation_window: Number of previous utterances to include as context

        Returns:
            DataFrame of evaluation results.
        """
        test_df = self.test_df.copy()
        if sample_size:
            sample_size = min(sample_size, len(test_df))
            test_df = (
                test_df.sample(n=sample_size, random_state=42)
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

        results: list[dict[str, Any]] = []

        for idx, row in test_df.iterrows():
            target = row["content"]
            timestamp = row.get("timestamp")
            latitude = row.get("latitude")
            longitude = row.get("longitude")

            filtered_corpus = self._filter_corpus_by_context(
                timestamp, latitude, longitude, context_filter, time_window_hours, geo_radius_km
            )

            conversation_context = None
            if conversation_window > 0 and idx > 0:
                start_idx = max(0, idx - conversation_window)
                previous = test_df.iloc[start_idx:idx]["content"].tolist()
                if previous:
                    conversation_context = "\n".join(previous)

            context = {
                "corpus": filtered_corpus,
                "timestamp": timestamp,
                "latitude": latitude,
                "longitude": longitude,
                "conversation_context": conversation_context,
            }

            for partial_name, partial_fn in partial_methods.items():
                try:
                    partial_text = partial_fn(target)
                except Exception as exc:  # pragma: no cover - defensive
                    partial_text = target
                    print(f"[warn] Partial method '{partial_name}' failed: {exc}")

                for generation_name, generation_fn in generation_methods.items():
                    try:
                        proposal = generation_fn(partial_text, context)
                    except Exception as exc:  # pragma: no cover - defensive
                        proposal = f"Error: {exc}"

                    metric_values: dict[str, Any] = {}
                    for metric_name, metric_fn in evaluation_metrics.items():
                        try:
                            metric_values[metric_name] = metric_fn(proposal, target)
                        except Exception as exc:  # pragma: no cover - defensive
                            metric_values[metric_name] = None
                            print(f"[warn] Metric '{metric_name}' failed: {exc}")

                    results.append(
                        {
                            "target": target,
                            "partial": partial_text,
                            "proposal": proposal,
                            "partial_method": partial_name,
                            "generation_method": generation_name,
                            "timestamp": timestamp,
                            "latitude": latitude,
                            "longitude": longitude,
                            "context_filter": context_filter,
                            **metric_values,
                        }
                    )

        return pd.DataFrame(results)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_completion(
        self,
        partial_text: str,
        examples: list[dict] | None = None,
        context: str | None = None,
        conversation_context: str | None = None,
    ) -> str:
        """Generate a completion for a partial utterance using an LLM."""
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
            response = self.llm_client.generate(
                user_prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )
            return response.strip()
        except Exception as e:
            print(f"Error generating completion: {e}")
            return partial_text  # Fallback to returning partial text

    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using sentence embeddings."""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def judge_similarity(self, target: str, proposal: str) -> int:
        """Use an LLM to judge semantic similarity between a target and proposal."""
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
            score_text = self.llm_client.generate(
                user_prompt,
                system_prompt="You are an expert evaluator of AAC text completions.",
                temperature=0.0
            )
            # Extract integer score from the response
            import re
            match = re.search(r'\b(\d{1,2})\b', score_text)
            if match:
                score = int(match.group(1))
                return min(10, max(1, score))
            return 5  # Default middle score
        except Exception as e:
            print(f"Error judging similarity: {e}")
            return 5  # Fallback to middle score

    @staticmethod
    def calculate_character_accuracy(target: str, proposal: str) -> float:
        """Calculate character-level accuracy between target and proposal."""
        if not target or not proposal:
            return 0.0

        # Use Levenshtein distance approximated by sequence matcher
        import difflib

        similarity = difflib.SequenceMatcher(None, target, proposal).ratio()
        return similarity

    @staticmethod
    def calculate_word_accuracy(target: str, proposal: str) -> float:
        """Calculate word-level accuracy between target and proposal."""
        if not target or not proposal:
            return 0.0

        target_words = set(target.lower().split())
        proposal_words = set(proposal.lower().split())

        if not target_words:
            return 0.0

        intersection = target_words.intersection(proposal_words)
        return len(intersection) / len(target_words)
