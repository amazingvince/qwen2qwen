# Story 11: Evaluation

| Field | Value |
|-------|-------|
| **Story ID** | QWEN3-ENC-DEC-11 |
| **Title** | Comprehensive Evaluation and Benchmarking |
| **Epic** | Qwen3 Encoder-Decoder Implementation |
| **Priority** | High |
| **Estimated Effort** | 2-3 days |
| **Dependencies** | Story 10 (Extracted encoder model) |

---

## Objective

Evaluate the extracted Qwen3 encoder on standard embedding benchmarks to measure quality and compare against baselines. This includes MTEB benchmark suite evaluation, semantic similarity tasks, retrieval benchmarks, and comparison against reference models.

---

## Background

The success of the encoder-decoder training approach is ultimately measured by the quality of the extracted encoder's embeddings. We need to:

1. **Benchmark Performance**: Evaluate on MTEB (Massive Text Embedding Benchmark)
2. **Compare Baselines**: Test against Qwen3-0.6B mean pooling and other encoders
3. **Task-Specific Evaluation**: Measure performance on retrieval, similarity, and classification
4. **Ablation Studies**: Understand the contribution of different training components

### Target Comparisons

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| Qwen3-0.6B (mean pool) | Decoder-only | 600M | Baseline (no bidirectional training) |
| Qwen3 Encoder (ours) | Bidirectional Encoder | ~600M | Our trained model |
| EmbeddingGemma-300M | Encoder | 300M | Size reference |
| T5Gemma 2 Encoder | Encoder | 270M-2B | Architecture reference |
| GTE-base | Encoder | 110M | Strong baseline |
| E5-base | Encoder | 110M | Strong baseline |

---

## Technical Requirements

### 11.1 MTEB Evaluation Framework

```python
# src/evaluation/mteb_eval.py

"""
MTEB (Massive Text Embedding Benchmark) evaluation for Qwen3 Encoder.

MTEB evaluates embeddings across 8 task categories:
- Classification
- Clustering  
- Pair Classification
- Reranking
- Retrieval
- STS (Semantic Textual Similarity)
- Summarization
- BitextMining
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field, asdict

import torch
import numpy as np
from tqdm import tqdm

# MTEB library
from mteb import MTEB, get_tasks
from mteb.abstasks import AbsTask

# Sentence Transformers for model wrapping
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MTEBConfig:
    """Configuration for MTEB evaluation."""
    
    # Model settings
    model_path: str = "./extracted_encoder"
    model_name: str = "qwen3-encoder"
    
    # Evaluation settings
    task_categories: List[str] = field(default_factory=lambda: [
        "Classification",
        "Clustering", 
        "PairClassification",
        "Reranking",
        "Retrieval",
        "STS",
        "Summarization",
    ])
    
    # Specific tasks (if None, run all in categories)
    task_names: Optional[List[str]] = None
    
    # Language filter
    languages: List[str] = field(default_factory=lambda: ["eng"])
    
    # Evaluation parameters
    batch_size: int = 32
    max_seq_length: int = 512
    
    # Output settings
    output_dir: str = "./mteb_results"
    save_predictions: bool = False
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


class Qwen3EncoderWrapper:
    """
    Wrapper to make Qwen3Encoder compatible with MTEB evaluation.
    
    MTEB expects a model with an `encode` method that takes:
    - sentences: List[str]
    - batch_size: int
    - show_progress_bar: bool
    
    And returns: np.ndarray of shape (num_sentences, embedding_dim)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
        pooling_mode: str = "mean",
    ):
        self.device = device
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.pooling_mode = pooling_mode
        
        # Load model and tokenizer
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """Load the Qwen3 encoder model."""
        from transformers import AutoTokenizer
        
        # Try loading as sentence-transformers model first
        try:
            self.model = SentenceTransformer(model_path, device=self.device)
            self.use_sentence_transformers = True
            logger.info(f"Loaded model via SentenceTransformers from {model_path}")
        except Exception as e:
            logger.info(f"SentenceTransformers loading failed, trying direct load: {e}")
            
            # Fall back to direct loading
            from models.encoder_only import Qwen3EncoderForEmbedding
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = Qwen3EncoderForEmbedding.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.use_sentence_transformers = False
            logger.info(f"Loaded model directly from {model_path}")
    
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: Optional[bool] = None,
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            normalize_embeddings: Whether to L2 normalize embeddings
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        if normalize_embeddings is None:
            normalize_embeddings = self.normalize_embeddings
            
        if self.use_sentence_transformers:
            return self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
        
        # Manual encoding
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding", total=len(sentences) // batch_size + 1)
        
        with torch.no_grad():
            for start_idx in iterator:
                batch = sentences[start_idx:start_idx + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                embeddings = outputs.embeddings  # Shape: (batch_size, hidden_size)
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings


class MTEBEvaluator:
    """
    MTEB evaluation runner for Qwen3 Encoder.
    """
    
    def __init__(self, config: MTEBConfig):
        self.config = config
        self.results = {}
        
        # Initialize model wrapper
        self.model = Qwen3EncoderWrapper(
            model_path=config.model_path,
            device=config.device,
            max_seq_length=config.max_seq_length,
        )
        
    def get_tasks(self) -> List[AbsTask]:
        """Get MTEB tasks to evaluate on."""
        if self.config.task_names:
            # Specific tasks requested
            tasks = get_tasks(tasks=self.config.task_names)
        else:
            # Get all tasks in specified categories
            tasks = get_tasks(
                task_types=self.config.task_categories,
                languages=self.config.languages,
            )
        
        logger.info(f"Loaded {len(tasks)} tasks for evaluation")
        return tasks
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full MTEB evaluation.
        
        Returns:
            Dictionary of results per task and category
        """
        tasks = self.get_tasks()
        
        # Initialize MTEB
        evaluation = MTEB(tasks=tasks)
        
        # Run evaluation
        logger.info("Starting MTEB evaluation...")
        results = evaluation.run(
            self.model,
            output_folder=self.config.output_dir,
            eval_splits=["test"],
            batch_size=self.config.batch_size,
        )
        
        self.results = results
        return results
    
    def run_single_task(self, task_name: str) -> Dict[str, Any]:
        """
        Run evaluation on a single task.
        
        Args:
            task_name: Name of the MTEB task
            
        Returns:
            Task results dictionary
        """
        tasks = get_tasks(tasks=[task_name])
        
        evaluation = MTEB(tasks=tasks)
        results = evaluation.run(
            self.model,
            output_folder=self.config.output_dir,
            eval_splits=["test"],
            batch_size=self.config.batch_size,
        )
        
        return results
    
    def summarize_results(self) -> Dict[str, float]:
        """
        Summarize results by category.
        
        Returns:
            Dictionary with average scores per category
        """
        if not self.results:
            raise ValueError("No results to summarize. Run evaluation first.")
        
        category_scores = {}
        
        for task_name, task_results in self.results.items():
            # Get the main score (usually the first metric)
            if "test" in task_results:
                scores = task_results["test"]
                main_score = scores.get("main_score", list(scores.values())[0])
                
                # Determine category from task
                category = self._get_task_category(task_name)
                
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(main_score)
        
        # Average per category
        summary = {}
        for category, scores in category_scores.items():
            summary[category] = np.mean(scores)
        
        # Overall average
        all_scores = [s for scores in category_scores.values() for s in scores]
        summary["Overall"] = np.mean(all_scores)
        
        return summary
    
    def _get_task_category(self, task_name: str) -> str:
        """Determine task category from task name."""
        # This is a simplified mapping - MTEB tasks have this info
        category_keywords = {
            "Classification": ["class", "sentiment", "emotion", "toxic"],
            "Clustering": ["cluster"],
            "PairClassification": ["pair", "nli", "wnli"],
            "Reranking": ["rerank"],
            "Retrieval": ["retrieval", "msmarco", "nfcorpus", "scifact"],
            "STS": ["sts", "sick", "biosses"],
            "Summarization": ["summ"],
        }
        
        task_lower = task_name.lower()
        for category, keywords in category_keywords.items():
            if any(kw in task_lower for kw in keywords):
                return category
        return "Other"
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mteb_results_{self.config.model_name}_{timestamp}.json"
        
        output_path = Path(self.config.output_dir) / filename
        
        output = {
            "model": self.config.model_name,
            "model_path": self.config.model_path,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "results": self.results,
            "summary": self.summarize_results() if self.results else {},
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return output_path


def run_mteb_evaluation(
    model_path: str,
    output_dir: str = "./mteb_results",
    task_categories: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Convenience function to run MTEB evaluation.
    
    Args:
        model_path: Path to the extracted encoder model
        output_dir: Directory to save results
        task_categories: List of task categories to evaluate
        task_names: Specific task names (overrides categories)
        batch_size: Batch size for encoding
        
    Returns:
        Evaluation results dictionary
    """
    config = MTEBConfig(
        model_path=model_path,
        output_dir=output_dir,
        batch_size=batch_size,
    )
    
    if task_categories:
        config.task_categories = task_categories
    if task_names:
        config.task_names = task_names
    
    evaluator = MTEBEvaluator(config)
    results = evaluator.run_evaluation()
    evaluator.save_results()
    
    return results
```

---

### 11.2 Semantic Similarity Evaluation

```python
# src/evaluation/similarity_eval.py

"""
Semantic Textual Similarity (STS) evaluation.

Evaluates embedding quality on sentence similarity tasks:
- STS Benchmark (STS-B)
- SICK Relatedness
- Custom similarity datasets
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm

from datasets import load_dataset


@dataclass
class STSResult:
    """Results from STS evaluation."""
    spearman: float
    pearson: float
    dataset_name: str
    num_samples: int
    
    def __repr__(self):
        return (
            f"STSResult(dataset={self.dataset_name}, "
            f"spearman={self.spearman:.4f}, "
            f"pearson={self.pearson:.4f}, "
            f"n={self.num_samples})"
        )


class STSEvaluator:
    """
    Evaluator for Semantic Textual Similarity tasks.
    """
    
    # Standard STS datasets
    STS_DATASETS = {
        "stsb": {
            "path": "mteb/stsbenchmark-sts",
            "split": "test",
            "sent1_col": "sentence1",
            "sent2_col": "sentence2", 
            "score_col": "score",
            "score_range": (0, 5),
        },
        "sick": {
            "path": "sick",
            "split": "test",
            "sent1_col": "sentence_A",
            "sent2_col": "sentence_B",
            "score_col": "relatedness_score",
            "score_range": (1, 5),
        },
    }
    
    def __init__(self, model, tokenizer=None, device: str = "cuda"):
        """
        Initialize STS evaluator.
        
        Args:
            model: Encoder model or model wrapper with encode() method
            tokenizer: Tokenizer (optional if model has encode method)
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Check if model has encode method
        self.has_encode = hasattr(model, 'encode')
        
    def encode_sentences(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode sentences to embeddings."""
        if self.has_encode:
            return self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        
        # Manual encoding
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for start_idx in iterator:
                batch = sentences[start_idx:start_idx + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = outputs.embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        batch_size: int = 32,
    ) -> STSResult:
        """
        Evaluate on a standard STS dataset.
        
        Args:
            dataset_name: Name of dataset (stsb, sick)
            batch_size: Batch size for encoding
            
        Returns:
            STSResult with correlation metrics
        """
        if dataset_name not in self.STS_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.STS_DATASETS[dataset_name]
        
        # Load dataset
        dataset = load_dataset(config["path"], split=config["split"])
        
        sentences1 = dataset[config["sent1_col"]]
        sentences2 = dataset[config["sent2_col"]]
        gold_scores = np.array(dataset[config["score_col"]])
        
        # Normalize scores to [0, 1]
        min_score, max_score = config["score_range"]
        gold_scores = (gold_scores - min_score) / (max_score - min_score)
        
        # Encode sentences
        embeddings1 = self.encode_sentences(sentences1, batch_size)
        embeddings2 = self.encode_sentences(sentences2, batch_size)
        
        # Compute cosine similarities
        pred_scores = np.array([
            cosine_similarity([e1], [e2])[0, 0]
            for e1, e2 in zip(embeddings1, embeddings2)
        ])
        
        # Compute correlations
        spearman = spearmanr(gold_scores, pred_scores)[0]
        pearson = pearsonr(gold_scores, pred_scores)[0]
        
        return STSResult(
            spearman=spearman,
            pearson=pearson,
            dataset_name=dataset_name,
            num_samples=len(gold_scores),
        )
    
    def evaluate_all(self, batch_size: int = 32) -> Dict[str, STSResult]:
        """Evaluate on all standard STS datasets."""
        results = {}
        
        for dataset_name in self.STS_DATASETS:
            print(f"\nEvaluating on {dataset_name}...")
            results[dataset_name] = self.evaluate_dataset(dataset_name, batch_size)
            print(results[dataset_name])
        
        return results
    
    def evaluate_custom(
        self,
        sentences1: List[str],
        sentences2: List[str],
        gold_scores: List[float],
        batch_size: int = 32,
    ) -> STSResult:
        """
        Evaluate on custom sentence pairs.
        
        Args:
            sentences1: First sentences
            sentences2: Second sentences
            gold_scores: Gold similarity scores (0-1)
            batch_size: Batch size for encoding
            
        Returns:
            STSResult with correlation metrics
        """
        embeddings1 = self.encode_sentences(sentences1, batch_size)
        embeddings2 = self.encode_sentences(sentences2, batch_size)
        
        pred_scores = np.array([
            cosine_similarity([e1], [e2])[0, 0]
            for e1, e2 in zip(embeddings1, embeddings2)
        ])
        
        gold_scores = np.array(gold_scores)
        
        spearman = spearmanr(gold_scores, pred_scores)[0]
        pearson = pearsonr(gold_scores, pred_scores)[0]
        
        return STSResult(
            spearman=spearman,
            pearson=pearson,
            dataset_name="custom",
            num_samples=len(gold_scores),
        )
```

---

### 11.3 Retrieval Evaluation

```python
# src/evaluation/retrieval_eval.py

"""
Retrieval evaluation for embedding models.

Evaluates on:
- MS MARCO passage retrieval
- Natural Questions
- Custom retrieval datasets
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
from tqdm import tqdm
import faiss

from datasets import load_dataset


@dataclass  
class RetrievalResult:
    """Results from retrieval evaluation."""
    mrr_at_10: float
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    dataset_name: str
    num_queries: int
    num_docs: int
    
    def __repr__(self):
        return (
            f"RetrievalResult(dataset={self.dataset_name}, "
            f"MRR@10={self.mrr_at_10:.4f}, "
            f"R@1={self.recall_at_1:.4f}, "
            f"R@10={self.recall_at_10:.4f}, "
            f"NDCG@10={self.ndcg_at_10:.4f})"
        )


class RetrievalEvaluator:
    """
    Evaluator for retrieval tasks.
    """
    
    def __init__(
        self,
        model,
        tokenizer=None,
        device: str = "cuda",
        use_faiss: bool = True,
    ):
        """
        Initialize retrieval evaluator.
        
        Args:
            model: Encoder model or wrapper with encode() method
            tokenizer: Tokenizer (optional if model has encode)
            device: Device for computation
            use_faiss: Whether to use FAISS for efficient similarity search
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_faiss = use_faiss
        self.has_encode = hasattr(model, 'encode')
        
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        if prefix:
            texts = [prefix + t for t in texts]
            
        if self.has_encode:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        
        # Manual encoding with normalization
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for start_idx in iterator:
                batch = texts[start_idx:start_idx + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = outputs.embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for efficient similarity search."""
        dim = embeddings.shape[1]
        
        # Use inner product (equivalent to cosine for normalized vectors)
        index = faiss.IndexFlatIP(dim)
        
        # Add embeddings
        embeddings = embeddings.astype(np.float32)
        index.add(embeddings)
        
        return index
    
    def search(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        k: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k documents for each query.
        
        Args:
            query_embeddings: Query embeddings (num_queries, dim)
            doc_embeddings: Document embeddings (num_docs, dim)
            k: Number of results per query
            
        Returns:
            scores: (num_queries, k) similarity scores
            indices: (num_queries, k) document indices
        """
        if self.use_faiss:
            index = self.build_index(doc_embeddings)
            scores, indices = index.search(
                query_embeddings.astype(np.float32), k
            )
        else:
            # Brute force search
            similarities = query_embeddings @ doc_embeddings.T
            indices = np.argsort(-similarities, axis=1)[:, :k]
            scores = np.take_along_axis(similarities, indices, axis=1)
        
        return scores, indices
    
    def compute_metrics(
        self,
        retrieved_indices: np.ndarray,
        relevant_docs: List[List[int]],
        k_values: List[int] = [1, 10, 100],
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Args:
            retrieved_indices: (num_queries, max_k) retrieved document indices
            relevant_docs: List of relevant doc indices for each query
            k_values: K values for recall@k
            
        Returns:
            Dictionary of metrics
        """
        num_queries = len(relevant_docs)
        
        # MRR@10
        mrr = 0.0
        for i, rel in enumerate(relevant_docs):
            rel_set = set(rel)
            for rank, doc_idx in enumerate(retrieved_indices[i, :10]):
                if doc_idx in rel_set:
                    mrr += 1.0 / (rank + 1)
                    break
        mrr /= num_queries
        
        # Recall@k
        recalls = {}
        for k in k_values:
            recall = 0.0
            for i, rel in enumerate(relevant_docs):
                rel_set = set(rel)
                retrieved_set = set(retrieved_indices[i, :k])
                recall += len(rel_set & retrieved_set) / len(rel_set)
            recalls[f"recall@{k}"] = recall / num_queries
        
        # NDCG@10
        ndcg = 0.0
        for i, rel in enumerate(relevant_docs):
            rel_set = set(rel)
            dcg = 0.0
            for rank, doc_idx in enumerate(retrieved_indices[i, :10]):
                if doc_idx in rel_set:
                    dcg += 1.0 / np.log2(rank + 2)
            
            # Ideal DCG
            ideal_dcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), 10)))
            ndcg += dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg /= num_queries
        
        return {
            "mrr@10": mrr,
            "ndcg@10": ndcg,
            **recalls,
        }
    
    def evaluate_msmarco(
        self,
        num_docs: int = 100000,  # Subset for faster evaluation
        batch_size: int = 32,
    ) -> RetrievalResult:
        """
        Evaluate on MS MARCO passage retrieval.
        
        Args:
            num_docs: Number of documents to use (subset)
            batch_size: Batch size for encoding
            
        Returns:
            RetrievalResult with metrics
        """
        print("Loading MS MARCO dataset...")
        
        # Load corpus
        corpus = load_dataset(
            "mteb/msmarco-passage-corpus",
            split=f"train[:{num_docs}]"
        )
        doc_texts = corpus["text"]
        
        # Load queries and qrels
        queries = load_dataset("mteb/msmarco", split="queries")
        qrels = load_dataset("mteb/msmarco", split="test")
        
        # Filter queries that have relevant docs in our subset
        query_texts = []
        relevant_docs = []
        
        for q in tqdm(qrels, desc="Filtering queries"):
            query_id = q["query-id"]
            # Get relevant doc indices
            rel_indices = [
                i for i, doc_id in enumerate(corpus["_id"])
                if doc_id in q["corpus-id"]
            ]
            if rel_indices:
                query_text = next(
                    (qr["text"] for qr in queries if qr["_id"] == query_id),
                    None
                )
                if query_text:
                    query_texts.append(query_text)
                    relevant_docs.append(rel_indices)
        
        print(f"Evaluating {len(query_texts)} queries against {num_docs} docs")
        
        # Encode
        doc_embeddings = self.encode_texts(doc_texts, batch_size, prefix="passage: ")
        query_embeddings = self.encode_texts(query_texts, batch_size, prefix="query: ")
        
        # Search
        scores, indices = self.search(query_embeddings, doc_embeddings, k=100)
        
        # Compute metrics
        metrics = self.compute_metrics(indices, relevant_docs)
        
        return RetrievalResult(
            mrr_at_10=metrics["mrr@10"],
            recall_at_1=metrics["recall@1"],
            recall_at_10=metrics["recall@10"],
            recall_at_100=metrics["recall@100"],
            ndcg_at_10=metrics["ndcg@10"],
            dataset_name="msmarco",
            num_queries=len(query_texts),
            num_docs=num_docs,
        )
    
    def evaluate_custom(
        self,
        queries: List[str],
        documents: List[str],
        relevant_docs: List[List[int]],
        batch_size: int = 32,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> RetrievalResult:
        """
        Evaluate on custom retrieval dataset.
        
        Args:
            queries: Query texts
            documents: Document texts
            relevant_docs: List of relevant doc indices per query
            batch_size: Batch size for encoding
            query_prefix: Prefix for queries
            doc_prefix: Prefix for documents
            
        Returns:
            RetrievalResult with metrics
        """
        # Encode
        doc_embeddings = self.encode_texts(documents, batch_size, prefix=doc_prefix)
        query_embeddings = self.encode_texts(queries, batch_size, prefix=query_prefix)
        
        # Search
        scores, indices = self.search(query_embeddings, doc_embeddings, k=100)
        
        # Compute metrics
        metrics = self.compute_metrics(indices, relevant_docs)
        
        return RetrievalResult(
            mrr_at_10=metrics["mrr@10"],
            recall_at_1=metrics["recall@1"],
            recall_at_10=metrics["recall@10"],
            recall_at_100=metrics["recall@100"],
            ndcg_at_10=metrics["ndcg@10"],
            dataset_name="custom",
            num_queries=len(queries),
            num_docs=len(documents),
        )
```

---

### 11.4 Baseline Comparison

```python
# src/evaluation/baseline_comparison.py

"""
Compare Qwen3 Encoder against baseline models.

Baselines:
- Qwen3-0.6B with mean pooling (no bidirectional training)
- Other embedding models (E5, GTE, etc.)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from .similarity_eval import STSEvaluator, STSResult
from .retrieval_eval import RetrievalEvaluator, RetrievalResult


@dataclass
class ModelInfo:
    """Information about a model for comparison."""
    name: str
    path: str
    model_type: str  # "qwen3_encoder", "decoder_pool", "sentence_transformer"
    pooling: str = "mean"
    normalize: bool = True


class BaselineComparison:
    """
    Compare multiple embedding models on the same benchmarks.
    """
    
    # Pre-defined baseline models
    BASELINES = {
        "e5-base": ModelInfo(
            name="E5-base",
            path="intfloat/e5-base-v2",
            model_type="sentence_transformer",
        ),
        "gte-base": ModelInfo(
            name="GTE-base", 
            path="thenlper/gte-base",
            model_type="sentence_transformer",
        ),
        "bge-base": ModelInfo(
            name="BGE-base",
            path="BAAI/bge-base-en-v1.5",
            model_type="sentence_transformer",
        ),
    }
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = {}
        self.results = {}
        
    def load_model(self, model_info: ModelInfo):
        """Load a model for evaluation."""
        if model_info.model_type == "sentence_transformer":
            model = SentenceTransformer(model_info.path, device=self.device)
        elif model_info.model_type == "decoder_pool":
            model = DecoderPoolWrapper(
                model_info.path,
                pooling=model_info.pooling,
                device=self.device,
            )
        elif model_info.model_type == "qwen3_encoder":
            from .mteb_eval import Qwen3EncoderWrapper
            model = Qwen3EncoderWrapper(
                model_info.path,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown model type: {model_info.model_type}")
        
        self.models[model_info.name] = model
        return model
    
    def add_model(
        self,
        name: str,
        path: str,
        model_type: str = "sentence_transformer",
        **kwargs
    ):
        """Add a model for comparison."""
        model_info = ModelInfo(name=name, path=path, model_type=model_type, **kwargs)
        self.load_model(model_info)
        
    def add_qwen3_baseline(self, qwen3_path: str = "Qwen/Qwen3-0.6B"):
        """Add Qwen3-0.6B with mean pooling as baseline."""
        model_info = ModelInfo(
            name="Qwen3-0.6B (mean pool)",
            path=qwen3_path,
            model_type="decoder_pool",
            pooling="mean",
        )
        self.load_model(model_info)
        
    def add_trained_encoder(self, encoder_path: str, name: str = "Qwen3-Encoder (ours)"):
        """Add our trained encoder."""
        model_info = ModelInfo(
            name=name,
            path=encoder_path,
            model_type="qwen3_encoder",
        )
        self.load_model(model_info)
        
    def run_sts_comparison(self, batch_size: int = 32) -> Dict[str, Dict[str, STSResult]]:
        """Run STS evaluation on all models."""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print('='*60)
            
            evaluator = STSEvaluator(model, device=self.device)
            results[name] = evaluator.evaluate_all(batch_size)
        
        self.results["sts"] = results
        return results
    
    def run_retrieval_comparison(
        self,
        num_docs: int = 10000,
        batch_size: int = 32,
    ) -> Dict[str, RetrievalResult]:
        """Run retrieval evaluation on all models."""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print('='*60)
            
            evaluator = RetrievalEvaluator(model, device=self.device)
            results[name] = evaluator.evaluate_msmarco(num_docs, batch_size)
        
        self.results["retrieval"] = results
        return results
    
    def create_comparison_table(self) -> str:
        """Create a formatted comparison table."""
        lines = []
        
        # STS Results
        if "sts" in self.results:
            lines.append("\n## Semantic Textual Similarity (Spearman Correlation)")
            lines.append("")
            
            # Header
            sts_results = self.results["sts"]
            datasets = list(next(iter(sts_results.values())).keys())
            header = "| Model | " + " | ".join(datasets) + " | Avg |"
            lines.append(header)
            lines.append("|" + "|".join(["---"] * (len(datasets) + 2)) + "|")
            
            # Rows
            for model_name, model_results in sts_results.items():
                scores = [model_results[d].spearman for d in datasets]
                avg = np.mean(scores)
                row = f"| {model_name} | " + " | ".join(f"{s:.4f}" for s in scores) + f" | {avg:.4f} |"
                lines.append(row)
        
        # Retrieval Results
        if "retrieval" in self.results:
            lines.append("\n## Retrieval (MS MARCO)")
            lines.append("")
            
            retrieval_results = self.results["retrieval"]
            header = "| Model | MRR@10 | R@1 | R@10 | R@100 | NDCG@10 |"
            lines.append(header)
            lines.append("|---|---|---|---|---|---|")
            
            for model_name, result in retrieval_results.items():
                row = f"| {model_name} | {result.mrr_at_10:.4f} | {result.recall_at_1:.4f} | {result.recall_at_10:.4f} | {result.recall_at_100:.4f} | {result.ndcg_at_10:.4f} |"
                lines.append(row)
        
        return "\n".join(lines)
    
    def save_results(self, output_path: str):
        """Save all results to JSON."""
        output = {
            "models": list(self.models.keys()),
            "results": {},
        }
        
        if "sts" in self.results:
            output["results"]["sts"] = {
                model: {
                    dataset: asdict(result)
                    for dataset, result in model_results.items()
                }
                for model, model_results in self.results["sts"].items()
            }
        
        if "retrieval" in self.results:
            output["results"]["retrieval"] = {
                model: asdict(result)
                for model, result in self.results["retrieval"].items()
            }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_path}")


class DecoderPoolWrapper:
    """
    Wrapper for decoder-only models with mean pooling.
    Used as baseline comparison (Qwen3-0.6B without bidirectional training).
    """
    
    def __init__(
        self,
        model_path: str,
        pooling: str = "mean",
        device: str = "cuda",
        max_seq_length: int = 512,
    ):
        self.device = device
        self.pooling = pooling
        self.max_seq_length = max_seq_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode sentences using decoder with pooling."""
        from tqdm import tqdm
        
        all_embeddings = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for start_idx in iterator:
                batch = sentences[start_idx:start_idx + batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if self.pooling == "mean":
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
                elif self.pooling == "last":
                    # Get last non-padded token
                    seq_lengths = inputs["attention_mask"].sum(1) - 1
                    embeddings = hidden_states[
                        torch.arange(hidden_states.size(0)), seq_lengths
                    ]
                elif self.pooling == "cls":
                    embeddings = hidden_states[:, 0]
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings
```

---

### 11.5 Evaluation Runner Script

```python
# scripts/run_evaluation.py

"""
Main evaluation script for Qwen3 Encoder.

Usage:
    python scripts/run_evaluation.py \
        --encoder_path ./extracted_encoder \
        --output_dir ./eval_results \
        --run_mteb \
        --run_sts \
        --run_retrieval \
        --compare_baselines
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 Encoder")
    
    # Model paths
    parser.add_argument(
        "--encoder_path",
        type=str,
        required=True,
        help="Path to extracted encoder model",
    )
    parser.add_argument(
        "--qwen3_baseline_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Path to Qwen3-0.6B for baseline comparison",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save results",
    )
    
    # Evaluation options
    parser.add_argument(
        "--run_mteb",
        action="store_true",
        help="Run full MTEB evaluation",
    )
    parser.add_argument(
        "--mteb_tasks",
        nargs="+",
        default=None,
        help="Specific MTEB tasks to run (default: all)",
    )
    parser.add_argument(
        "--run_sts",
        action="store_true",
        help="Run STS evaluation",
    )
    parser.add_argument(
        "--run_retrieval",
        action="store_true", 
        help="Run retrieval evaluation",
    )
    parser.add_argument(
        "--compare_baselines",
        action="store_true",
        help="Compare against baseline models",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["e5-base", "gte-base"],
        help="Baseline models to compare against",
    )
    
    # Hardware
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )
    
    # Retrieval settings
    parser.add_argument(
        "--retrieval_num_docs",
        type=int,
        default=10000,
        help="Number of documents for retrieval evaluation",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {}
    
    # Run MTEB evaluation
    if args.run_mteb:
        print("\n" + "="*60)
        print("Running MTEB Evaluation")
        print("="*60)
        
        from evaluation.mteb_eval import run_mteb_evaluation
        
        mteb_results = run_mteb_evaluation(
            model_path=args.encoder_path,
            output_dir=os.path.join(args.output_dir, "mteb"),
            task_names=args.mteb_tasks,
            batch_size=args.batch_size,
        )
        results["mteb"] = mteb_results
    
    # Run baseline comparison
    if args.compare_baselines:
        print("\n" + "="*60)
        print("Running Baseline Comparison")
        print("="*60)
        
        from evaluation.baseline_comparison import BaselineComparison
        
        comparison = BaselineComparison(device=args.device)
        
        # Add our encoder
        comparison.add_trained_encoder(args.encoder_path)
        
        # Add Qwen3 baseline
        comparison.add_qwen3_baseline(args.qwen3_baseline_path)
        
        # Add other baselines
        for baseline in args.baselines:
            if baseline in comparison.BASELINES:
                comparison.load_model(comparison.BASELINES[baseline])
        
        # Run evaluations
        if args.run_sts:
            comparison.run_sts_comparison(batch_size=args.batch_size)
        
        if args.run_retrieval:
            comparison.run_retrieval_comparison(
                num_docs=args.retrieval_num_docs,
                batch_size=args.batch_size,
            )
        
        # Print comparison table
        print("\n" + comparison.create_comparison_table())
        
        # Save results
        comparison.save_results(
            os.path.join(args.output_dir, f"comparison_{timestamp}.json")
        )
        results["comparison"] = comparison.results
    
    # Run standalone STS (if not doing comparison)
    elif args.run_sts:
        print("\n" + "="*60)
        print("Running STS Evaluation")
        print("="*60)
        
        from evaluation.mteb_eval import Qwen3EncoderWrapper
        from evaluation.similarity_eval import STSEvaluator
        
        model = Qwen3EncoderWrapper(args.encoder_path, device=args.device)
        evaluator = STSEvaluator(model, device=args.device)
        
        sts_results = evaluator.evaluate_all(batch_size=args.batch_size)
        results["sts"] = {k: v.__dict__ for k, v in sts_results.items()}
        
        for dataset, result in sts_results.items():
            print(f"{dataset}: {result}")
    
    # Run standalone retrieval (if not doing comparison)
    elif args.run_retrieval:
        print("\n" + "="*60)
        print("Running Retrieval Evaluation")
        print("="*60)
        
        from evaluation.mteb_eval import Qwen3EncoderWrapper
        from evaluation.retrieval_eval import RetrievalEvaluator
        
        model = Qwen3EncoderWrapper(args.encoder_path, device=args.device)
        evaluator = RetrievalEvaluator(model, device=args.device)
        
        retrieval_result = evaluator.evaluate_msmarco(
            num_docs=args.retrieval_num_docs,
            batch_size=args.batch_size,
        )
        results["retrieval"] = retrieval_result.__dict__
        
        print(f"\nRetrieval Results: {retrieval_result}")
    
    # Save all results
    results_path = os.path.join(args.output_dir, f"all_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
```

---

### 11.6 Quick Evaluation Script

```python
# scripts/quick_eval.py

"""
Quick evaluation script for fast iteration during development.

Runs a minimal evaluation suite to verify model quality:
- STS-B (semantic similarity)
- Small retrieval subset
- Embedding quality checks
"""

import argparse
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Quick evaluation for Qwen3 Encoder")
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def run_quick_eval(encoder_path: str, batch_size: int = 32):
    """Run quick evaluation suite."""
    
    from evaluation.mteb_eval import Qwen3EncoderWrapper
    
    print("Loading model...")
    model = Qwen3EncoderWrapper(encoder_path)
    
    # Test 1: Basic encoding
    print("\n" + "="*50)
    print("Test 1: Basic Encoding")
    print("="*50)
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny and warm.",
    ]
    
    embeddings = model.encode(test_sentences, batch_size=batch_size)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.4f}")
    
    # Test 2: Semantic similarity
    print("\n" + "="*50)
    print("Test 2: Semantic Similarity Check")
    print("="*50)
    
    similarities = cosine_similarity(embeddings)
    print("Similarity matrix:")
    print(f"  Sent 0-1 (similar): {similarities[0,1]:.4f}")
    print(f"  Sent 0-2 (different): {similarities[0,2]:.4f}")
    print(f"  Sent 0-3 (different): {similarities[0,3]:.4f}")
    print(f"  Sent 2-3 (different): {similarities[2,3]:.4f}")
    
    # Sanity check: similar sentences should have higher similarity
    if similarities[0,1] > similarities[0,2] and similarities[0,1] > similarities[0,3]:
        print("✓ Similar sentences have higher similarity")
    else:
        print("✗ WARNING: Similar sentences don't have highest similarity")
    
    # Test 3: STS-B evaluation
    print("\n" + "="*50)
    print("Test 3: STS-B Evaluation")
    print("="*50)
    
    from evaluation.similarity_eval import STSEvaluator
    
    evaluator = STSEvaluator(model)
    stsb_result = evaluator.evaluate_dataset("stsb", batch_size=batch_size)
    print(f"STS-B Spearman: {stsb_result.spearman:.4f}")
    print(f"STS-B Pearson: {stsb_result.pearson:.4f}")
    
    # Test 4: Embedding diversity
    print("\n" + "="*50)
    print("Test 4: Embedding Diversity Check")
    print("="*50)
    
    # Generate embeddings for diverse texts
    diverse_texts = [
        "I love programming in Python.",
        "The stock market crashed yesterday.",
        "She cooked a delicious Italian dinner.",
        "The football team won the championship.",
        "Quantum physics is fascinating.",
    ]
    
    diverse_embeddings = model.encode(diverse_texts, batch_size=batch_size)
    
    # Check variance across dimensions
    variance = np.var(diverse_embeddings, axis=0)
    print(f"Mean variance across dimensions: {np.mean(variance):.6f}")
    print(f"Min variance: {np.min(variance):.6f}")
    print(f"Max variance: {np.max(variance):.6f}")
    
    # Check for collapsed dimensions (very low variance)
    low_var_dims = np.sum(variance < 1e-6)
    print(f"Near-zero variance dimensions: {low_var_dims} / {len(variance)}")
    
    if low_var_dims > len(variance) * 0.1:
        print("✗ WARNING: Many dimensions have near-zero variance (possible collapse)")
    else:
        print("✓ Embedding dimensions show healthy variance")
    
    # Summary
    print("\n" + "="*50)
    print("Quick Evaluation Summary")
    print("="*50)
    print(f"• Embedding dimension: {embeddings.shape[1]}")
    print(f"• STS-B Spearman correlation: {stsb_result.spearman:.4f}")
    print(f"• Semantic similarity check: {'PASSED' if similarities[0,1] > similarities[0,2] else 'FAILED'}")
    print(f"• Embedding diversity: {'HEALTHY' if low_var_dims < len(variance) * 0.1 else 'WARNING'}")
    
    return {
        "stsb_spearman": stsb_result.spearman,
        "stsb_pearson": stsb_result.pearson,
        "embedding_dim": embeddings.shape[1],
        "low_variance_dims": int(low_var_dims),
    }


if __name__ == "__main__":
    args = parse_args()
    run_quick_eval(args.encoder_path, args.batch_size)
```

---

### 11.7 Model Card Generator

```python
# scripts/generate_model_card.py

"""
Generate a model card for the trained Qwen3 Encoder.

The model card includes:
- Model description
- Training details
- Evaluation results
- Usage examples
- Limitations
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


MODEL_CARD_TEMPLATE = '''---
language:
- en
- zh
- multilingual
license: apache-2.0
library_name: transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- mteb
- qwen3
- encoder
datasets:
- HuggingFaceFW/fineweb-edu
pipeline_tag: sentence-similarity
---

# {model_name}

{model_description}

## Model Details

### Model Description

- **Model Type:** Bidirectional Encoder (extracted from encoder-decoder)
- **Base Model:** Qwen3-0.6B
- **Training Objective:** UL2 (Mixture of Denoisers)
- **Hidden Size:** {hidden_size}
- **Layers:** {num_layers}
- **Attention Heads:** {num_heads}
- **Parameters:** ~{params_millions}M

### Training Details

- **Training Data:** FineWeb-Edu ({training_tokens} tokens)
- **Training Objective:** UL2 with 1:1:1:1:4 task mixing
  - R-Denoiser (μ=3, r=0.15)
  - R-Denoiser (μ=12, r=0.50)
  - X-Denoiser (μ=32, r=0.15)
  - X-Denoiser (μ=32, r=0.50)
  - S-Denoiser (r=0.75) - 50% of training
- **Hardware:** {hardware}
- **Training Time:** {training_time}

## Evaluation Results

### MTEB Results

{mteb_table}

### Comparison with Baselines

{comparison_table}

## Usage

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('{hf_model_id}')

sentences = [
    "This is an example sentence",
    "Each sentence is converted to a vector"
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (2, {hidden_size})
```

### Using Transformers

```python
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('{hf_model_id}')
model = AutoModel.from_pretrained('{hf_model_id}')

sentences = ["This is an example sentence"]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
```

### Pooling Options

The model supports multiple pooling strategies:

- **Mean Pooling** (default): Average of all token embeddings
- **CLS Pooling**: First token embedding
- **Last Token Pooling**: Last non-padded token embedding

## Limitations

- **Context Length:** Maximum {max_length} tokens
- **Languages:** Best performance on English and Chinese; trained on multilingual data from Qwen3
- **Specialized Domains:** May require fine-tuning for domain-specific applications

## Citation

```bibtex
@misc{{qwen3encoder2025,
  title={{Qwen3 Encoder: Bidirectional Encoder from UL2 Training}},
  author={{{author}}},
  year={{2025}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{hf_model_id}}}
}}
```

## Acknowledgments

- Based on [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) by Alibaba
- Inspired by [T5Gemma 2](https://huggingface.co/google/t5gemma-2-270m-270m) architecture
- Training approach based on [UL2](https://arxiv.org/abs/2205.05131)
'''


def generate_model_card(
    model_name: str,
    evaluation_results: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a model card for the Qwen3 Encoder.
    
    Args:
        model_name: Name of the model
        evaluation_results: Dictionary of evaluation results
        training_info: Dictionary of training information
        output_path: Path to save the model card
        
    Returns:
        Model card as string
    """
    # Default values
    defaults = {
        "model_name": model_name,
        "model_description": "A bidirectional encoder extracted from a Qwen3-based encoder-decoder model trained with UL2 objective.",
        "hidden_size": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "params_millions": 600,
        "training_tokens": "500B",
        "hardware": "8x A100 80GB",
        "training_time": "~2 weeks",
        "max_length": 8192,
        "hf_model_id": f"your-username/{model_name}",
        "author": "Your Name",
    }
    
    # Merge with provided info
    if training_info:
        defaults.update(training_info)
    
    # Generate MTEB table
    if evaluation_results and "mteb" in evaluation_results:
        mteb_table = generate_mteb_table(evaluation_results["mteb"])
    else:
        mteb_table = "*Evaluation results pending*"
    
    # Generate comparison table
    if evaluation_results and "comparison" in evaluation_results:
        comparison_table = generate_comparison_table(evaluation_results["comparison"])
    else:
        comparison_table = "*Baseline comparison pending*"
    
    defaults["mteb_table"] = mteb_table
    defaults["comparison_table"] = comparison_table
    
    # Generate card
    card = MODEL_CARD_TEMPLATE.format(**defaults)
    
    # Save if path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(card)
        print(f"Model card saved to {output_path}")
    
    return card


def generate_mteb_table(mteb_results: Dict[str, Any]) -> str:
    """Generate MTEB results table."""
    lines = ["| Task Category | Score |", "|---|---|"]
    
    if "summary" in mteb_results:
        for category, score in mteb_results["summary"].items():
            lines.append(f"| {category} | {score:.4f} |")
    
    return "\n".join(lines)


def generate_comparison_table(comparison_results: Dict[str, Any]) -> str:
    """Generate baseline comparison table."""
    lines = ["| Model | STS-B | MS MARCO MRR@10 |", "|---|---|---|"]
    
    # This would be populated from actual results
    # Placeholder for now
    lines.append("| *Results pending* | - | - |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate model card")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_results", type=str, help="Path to evaluation results JSON")
    parser.add_argument("--output", type=str, default="MODEL_CARD.md")
    
    args = parser.parse_args()
    
    # Load evaluation results if provided
    eval_results = None
    if args.eval_results:
        with open(args.eval_results) as f:
            eval_results = json.load(f)
    
    generate_model_card(
        model_name=args.model_name,
        evaluation_results=eval_results,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
```

---

## Acceptance Criteria

### 11.1 MTEB Framework
- [ ] Qwen3EncoderWrapper properly implements MTEB interface
- [ ] Can run evaluation on any MTEB task
- [ ] Results are saved in standard MTEB format
- [ ] Category summaries are computed correctly

### 11.2 STS Evaluation
- [ ] STSEvaluator runs on STS-B and SICK datasets
- [ ] Spearman and Pearson correlations are computed correctly
- [ ] Custom dataset evaluation works

### 11.3 Retrieval Evaluation
- [ ] RetrievalEvaluator computes MRR, Recall@k, NDCG
- [ ] FAISS index is built correctly for efficient search
- [ ] MS MARCO evaluation runs successfully

### 11.4 Baseline Comparison
- [ ] Can load and compare multiple models
- [ ] Qwen3-0.6B with mean pooling baseline works
- [ ] Comparison tables are generated correctly

### 11.5 Evaluation Scripts
- [ ] `run_evaluation.py` runs full evaluation suite
- [ ] `quick_eval.py` provides fast iteration feedback
- [ ] Model card generator produces valid markdown

### 11.6 Quality Targets
- [ ] STS-B Spearman > 0.75 (competitive with baselines)
- [ ] MS MARCO MRR@10 > 0.30 (reasonable retrieval)
- [ ] Embeddings show healthy variance (no dimension collapse)

---

## Verification Steps

### Step 1: Quick Sanity Check

```bash
# Run quick evaluation
python scripts/quick_eval.py --encoder_path ./extracted_encoder

# Expected output:
# - Embedding shape: (N, 1024)
# - STS-B Spearman: > 0.70
# - Semantic similarity check: PASSED
# - Embedding diversity: HEALTHY
```

### Step 2: STS Evaluation

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_sts \
    --output_dir ./eval_results
```

### Step 3: Retrieval Evaluation

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_retrieval \
    --retrieval_num_docs 10000 \
    --output_dir ./eval_results
```

### Step 4: Full Baseline Comparison

```bash
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --compare_baselines \
    --baselines e5-base gte-base \
    --run_sts \
    --run_retrieval \
    --output_dir ./eval_results
```

### Step 5: Full MTEB Evaluation

```bash
# Run full MTEB (takes several hours)
python scripts/run_evaluation.py \
    --encoder_path ./extracted_encoder \
    --run_mteb \
    --output_dir ./eval_results
```

### Step 6: Generate Model Card

```bash
python scripts/generate_model_card.py \
    --model_name qwen3-encoder \
    --eval_results ./eval_results/all_results.json \
    --output ./MODEL_CARD.md
```

---

## Test Cases

### Test 1: MTEB Wrapper

```python
def test_mteb_wrapper():
    """Test MTEB wrapper encoding."""
    from evaluation.mteb_eval import Qwen3EncoderWrapper
    
    wrapper = Qwen3EncoderWrapper("./extracted_encoder")
    
    sentences = ["Hello world", "Test sentence"]
    embeddings = wrapper.encode(sentences)
    
    assert embeddings.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
```

### Test 2: STS Evaluation

```python
def test_sts_evaluator():
    """Test STS evaluation."""
    from evaluation.similarity_eval import STSEvaluator
    
    # Mock model
    class MockModel:
        def encode(self, sentences, **kwargs):
            return np.random.randn(len(sentences), 768)
    
    evaluator = STSEvaluator(MockModel())
    
    result = evaluator.evaluate_custom(
        sentences1=["Hello", "Hi"],
        sentences2=["World", "Earth"],
        gold_scores=[0.5, 0.8],
    )
    
    assert hasattr(result, 'spearman')
    assert hasattr(result, 'pearson')
```

### Test 3: Retrieval Metrics

```python
def test_retrieval_metrics():
    """Test retrieval metric computation."""
    from evaluation.retrieval_eval import RetrievalEvaluator
    
    # Mock setup
    class MockModel:
        def encode(self, texts, **kwargs):
            return np.random.randn(len(texts), 768)
    
    evaluator = RetrievalEvaluator(MockModel(), use_faiss=False)
    
    # Simulate perfect retrieval
    retrieved = np.array([[0, 1, 2], [1, 0, 2]])
    relevant = [[0], [1]]
    
    metrics = evaluator.compute_metrics(retrieved, relevant, k_values=[1, 3])
    
    assert metrics["mrr@10"] == 1.0
    assert metrics["recall@1"] == 1.0
```

### Test 4: Baseline Wrapper

```python
def test_decoder_pool_wrapper():
    """Test decoder pooling wrapper."""
    from evaluation.baseline_comparison import DecoderPoolWrapper
    
    wrapper = DecoderPoolWrapper(
        "Qwen/Qwen3-0.6B",
        pooling="mean",
    )
    
    embeddings = wrapper.encode(["Test sentence"])
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0  # Hidden size
```

---

## Expected Results

### Target Performance (vs Baselines)

| Model | STS-B Spearman | MS MARCO MRR@10 |
|-------|----------------|-----------------|
| Qwen3-0.6B (mean pool) | ~0.60 | ~0.20 |
| **Qwen3 Encoder (ours)** | **>0.75** | **>0.30** |
| E5-base | ~0.84 | ~0.35 |
| GTE-base | ~0.82 | ~0.33 |

### Interpretation

- **Significant improvement over baseline**: The trained bidirectional encoder should substantially outperform simple mean pooling of the decoder-only model
- **Competitive with dedicated encoders**: While we may not match models trained specifically for embeddings, we should be in a reasonable range
- **Multilingual capability**: Inherits Qwen3's multilingual support

---

## File Structure

```
src/
├── evaluation/
│   ├── __init__.py
│   ├── mteb_eval.py           # MTEB evaluation framework
│   ├── similarity_eval.py      # STS evaluation
│   ├── retrieval_eval.py       # Retrieval evaluation
│   └── baseline_comparison.py  # Baseline comparison utilities

scripts/
├── run_evaluation.py           # Main evaluation script
├── quick_eval.py               # Quick sanity check
└── generate_model_card.py      # Model card generator

eval_results/                   # Output directory
├── mteb/                       # MTEB results
├── comparison_*.json           # Comparison results
└── all_results_*.json          # Combined results
```

---

## Dependencies

```
# requirements-eval.txt
mteb>=1.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU acceleration
scipy>=1.10.0
scikit-learn>=1.2.0
datasets>=2.14.0
tqdm>=4.65.0
```

---

## Notes

1. **MTEB Versioning**: MTEB benchmark tasks evolve; pin versions for reproducibility
2. **FAISS GPU**: For large-scale retrieval, use `faiss-gpu` for faster indexing
3. **Batch Size**: Adjust based on GPU memory; smaller batches for longer texts
4. **Evaluation Time**: Full MTEB takes several hours; use subsets for iteration
5. **Model Card**: Update with actual results before publishing to HuggingFace

---

## References

1. [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
2. [Sentence Transformers](https://www.sbert.net/)
3. [FAISS Documentation](https://github.com/facebookresearch/faiss)
4. [MS MARCO](https://microsoft.github.io/msmarco/)
5. [HuggingFace Model Cards](https://huggingface.co/docs/hub/model-cards)
