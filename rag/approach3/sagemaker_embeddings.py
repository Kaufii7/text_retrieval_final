"""SageMaker embedding adapter for Approach 3.

This module provides a unified interface for embeddings that can use either:
1. Local SentenceTransformer (default)
2. AWS SageMaker endpoint (when configured)

The adapter mimics the SentenceTransformer.encode() interface for easy integration.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# In-process cache so we don't re-load (and re-check HuggingFace Hub) per query.
_MODEL_CACHE: dict[Tuple[str, str, str], object] = {}


def _resolve_torch_device(requested: str) -> str:
    """Resolve user-friendly device strings to a torch-compatible device."""
    req = (requested or "auto").strip().lower()
    if req in ("", "auto", "gpu"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass
        return "cpu"
    return requested


class TransformersLocalEmbeddingAdapter:
    """Minimal local embedding adapter using HuggingFace Transformers (no sentence-transformers).

    This is used as a fallback when importing sentence-transformers fails due to optional
    dependency issues (e.g., datasets/pyarrow version mismatches).
    """

    def __init__(self, model_name: str, device: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Local embeddings fallback requires optional deps: torch + transformers. "
                "Install them or fix your sentence-transformers installation."
            ) from e

        self._torch = torch
        self.model_name = str(model_name)
        self.device = _resolve_torch_device(str(device or "auto"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def get_sentence_embedding_dimension(self) -> int:
        try:
            return int(getattr(self.model.config, "hidden_size"))
        except Exception:
            return 768

    def _mean_pool(self, last_hidden_state, attention_mask):
        torch = self._torch
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        # show_progress_bar ignored (kept for API compatibility)
        if not sentences:
            return np.array([])

        torch = self._torch
        bs = max(1, int(batch_size))
        outs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(sentences), bs):
                batch = ["" if s is None else str(s) for s in sentences[i : i + bs]]
                enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                if bool(normalize_embeddings):
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                arr = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                outs.append(arr)

        emb = np.vstack(outs) if outs else np.array([], dtype=np.float32)
        if bool(convert_to_numpy):
            return emb
        return emb


class SageMakerEmbeddingAdapter:
    """Adapter that uses SageMaker endpoint for embeddings instead of local model."""

    def __init__(self, endpoint_name: str, region: str = "eu-north-1", max_batch_size: int = 8):
        """Initialize SageMaker embedding adapter.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            region: AWS region where the endpoint is deployed
            max_batch_size: Maximum batch size for SageMaker requests (default: 8, smaller is safer)
        """
        try:
            from sagemaker.huggingface import HuggingFacePredictor
            import sagemaker
        except ImportError as e:
            raise RuntimeError(
                "Missing dependency for SageMaker embeddings: sagemaker. "
                "Install it with: pip install sagemaker"
            ) from e

        self.endpoint_name = endpoint_name
        self.region = region
        self.max_batch_size = max_batch_size
        self.predictor = HuggingFacePredictor(endpoint_name=endpoint_name)
        self._embedding_dim_cache = None  # Cache embedding dimension
        
        # Test endpoint health on initialization
        try:
            test_result = self._predict_with_retry({"inputs": ["test"]}, max_retries=1)
            logger.info(f"Endpoint health check passed for {endpoint_name}")
        except Exception as e:
            logger.warning(
                f"Endpoint health check failed for {endpoint_name}. "
                f"The endpoint may be in a bad state (CUDA errors detected). Error: {e}. "
                f"If 'fallback_to_local' is enabled in config, local SentenceTransformer will be used automatically. "
                f"Otherwise, consider: 1) Restarting the SageMaker endpoint, 2) Checking CloudWatch logs, "
                f"3) Enabling 'fallback_to_local: true' in config."
            )
        
        logger.info(f"Initialized SageMaker embedding adapter for endpoint: {endpoint_name} (max_batch_size={max_batch_size})")

    def _clean_text(self, text: str) -> str:
        """Clean text to avoid CUDA errors on SageMaker endpoint.
        
        Removes/replaces problematic characters that might cause issues.
        """
        if not text:
            return ""
        
        # Remove null bytes and control characters (except newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # Replace problematic unicode characters that might cause issues
        # Keep common printable characters
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _predict_with_retry(self, inputs: dict, max_retries: int = 3, base_delay: float = 1.0):
        """Call predictor with retry logic and exponential backoff."""
        for attempt in range(max_retries):
            try:
                return self.predictor.predict(inputs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)

    def _process_hf_response(self, result, expected_count: int) -> np.ndarray:
        """Process HuggingFace feature-extraction response and apply mean pooling.
        
        The response format from HuggingFace feature-extraction is:
        - result = list of sentences
        - result[i] = list with one element containing token embeddings  
        - result[i][0] = list of token embeddings [tok1(768), tok2(768), ...]
        
        We apply mean pooling across tokens to get sentence embeddings.
        """
        if not isinstance(result, list):
            raise ValueError(f"Expected list response, got {type(result)}")
        
        if len(result) != expected_count:
            raise ValueError(f"Expected {expected_count} results, got {len(result)}")
        
        pooled_embeddings = []
        for sent_idx, sent_result in enumerate(result):
            # sent_result should be a list with token embeddings
            if isinstance(sent_result, list):
                if len(sent_result) > 0:
                    # Check if this is nested (token-level embeddings)
                    first_elem = sent_result[0]
                    if isinstance(first_elem, (list, np.ndarray)):
                        # Token-level embeddings: sent_result[0] = [tok1_emb, tok2_emb, ...]
                        # or sent_result = [tok1_emb, tok2_emb, ...]
                        token_embeddings = np.array(sent_result[0] if isinstance(sent_result[0][0], (list, float, int)) else sent_result, dtype=np.float32)
                        if token_embeddings.ndim == 2:
                            # Mean pooling across tokens
                            pooled = token_embeddings.mean(axis=0)
                        else:
                            pooled = token_embeddings
                    else:
                        # Already a flat embedding vector
                        pooled = np.array(sent_result, dtype=np.float32)
                else:
                    logger.warning(f"Empty result for sentence {sent_idx}, using zeros")
                    pooled = np.zeros(768, dtype=np.float32)
            else:
                # Try to convert directly
                pooled = np.array(sent_result, dtype=np.float32).flatten()
            
            pooled_embeddings.append(pooled)
        
        return np.vstack(pooled_embeddings)

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode sentences using SageMaker endpoint.

        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for processing (used for batching requests)
            show_progress_bar: Ignored (SageMaker doesn't support progress bars)
            convert_to_numpy: If True, return numpy array (always True for SageMaker)
            normalize_embeddings: If True, normalize embeddings (handled by endpoint if model supports it)

        Returns:
            numpy array of shape (len(sentences), embedding_dim)
        """
        if not sentences:
            return np.array([])

        # Validate and clean inputs before sending to SageMaker
        # Filter out None, empty strings, and very long texts that might cause worker crashes/OOM
        # 256 chars is very conservative (~64 tokens) to avoid memory issues
        MAX_TEXT_LENGTH = 256
        cleaned_sentences = []
        original_indices = []  # Track which indices had valid inputs
        
        for idx, text in enumerate(sentences):
            if text is None:
                logger.warning(f"Skipping None at index {idx}")
                continue
            
            # Clean and sanitize text
            text_str = self._clean_text(str(text))
            
            if not text_str:
                logger.warning(f"Skipping empty string at index {idx}")
                continue
            
            # Truncate very long texts to avoid token limit issues
            # if len(text_str) > MAX_TEXT_LENGTH:
            #     logger.warning(f"Truncating text at index {idx} from {len(text_str)} to {MAX_TEXT_LENGTH} chars")
            #     text_str = text_str[:MAX_TEXT_LENGTH]
            
            # Final check - if still empty after cleaning, skip
            if not text_str.strip():
                logger.warning(f"Skipping text at index {idx} (empty after cleaning)")
                continue
                
            cleaned_sentences.append(text_str)
            original_indices.append(idx)

        if not cleaned_sentences:
            logger.warning("All sentences were filtered out, returning empty array")
            return np.zeros((len(sentences), 768), dtype=np.float32)  # Default dimension for all-mpnet-base-v2

        # Process in batches (use smaller batches for SageMaker to avoid OOM/worker crashes)
        # Use batch size of 1 for maximum safety
        safe_batch_size = min(batch_size, self.max_batch_size, 1)
        all_embeddings = []
        for i in range(0, len(cleaned_sentences), safe_batch_size):
            batch = cleaned_sentences[i : i + safe_batch_size]
            # Prepare input in the format expected by HuggingFace feature-extraction endpoint
            inputs = {"inputs": batch}
            try:
                # Call SageMaker endpoint with retry logic
                result = self._predict_with_retry(inputs)
                
                # Handle HuggingFace feature-extraction response format
                # The response is: result[sentence_idx][0] = [token1_emb(768), token2_emb(768), ...]
                # We need to apply mean pooling to get sentence embeddings
                batch_embeddings = self._process_hf_response(result, len(batch))
                
                # Validate embedding shape
                if batch_embeddings.shape[0] != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} embeddings but got {batch_embeddings.shape[0]}"
                    )
                
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error calling SageMaker endpoint for batch {i}: {e}")
                logger.error(f"Batch size: {len(batch)}, first text length: {len(batch[0]) if batch else 0}")
                # Try processing one at a time as fallback
                logger.info("Attempting to process batch one item at a time...")
                batch_embeddings_list = []
                for single_text in batch:
                    try:
                        # Try with retry logic
                        single_result = self._predict_with_retry({"inputs": [single_text]})
                        # Use the same processing as batch
                        single_emb = self._process_hf_response(single_result, 1)[0]
                        batch_embeddings_list.append(single_emb)
                    except Exception as single_e:
                        logger.error(f"Failed to encode single text (length={len(single_text)}, first 100 chars: {single_text[:100]}): {single_e}")
                        # Use zero vector as fallback
                        dim = 768  # Default for all-mpnet-base-v2
                        if batch_embeddings_list:
                            dim = batch_embeddings_list[0].shape[0]
                        batch_embeddings_list.append(np.zeros(dim, dtype=np.float32))
                
                if batch_embeddings_list:
                    batch_embeddings = np.vstack(batch_embeddings_list)
                    all_embeddings.append(batch_embeddings)
                else:
                    raise RuntimeError(f"Failed to get embeddings from SageMaker endpoint: {e}") from e

        # Concatenate all batches
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.array([])

        # Normalize if requested (endpoint may or may not normalize, so we do it here to be safe)
        if normalize_embeddings and len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms

        # If we filtered out some inputs, pad with zero vectors to match original length
        if len(original_indices) < len(sentences):
            if len(embeddings) == 0:
                # All were filtered, return zero matrix
                dim = 768  # Default for all-mpnet-base-v2
                return np.zeros((len(sentences), dim), dtype=np.float32)
            
            dim = embeddings.shape[1]
            full_embeddings = np.zeros((len(sentences), dim), dtype=np.float32)
            # Map embeddings back to original positions
            for cleaned_idx, orig_idx in enumerate(original_indices):
                full_embeddings[orig_idx] = embeddings[cleaned_idx]
            return full_embeddings

        return embeddings

    def get_sentence_embedding_dimension(self) -> int:
        """Get the embedding dimension.

        Note: This requires a test call to the endpoint since we don't know the dimension a priori.
        For efficiency, caches this value.
        """
        if self._embedding_dim_cache is not None:
            return self._embedding_dim_cache
        
        # Make a test call with a simple sentence to determine dimension
        try:
            test_result = self._predict_with_retry({"inputs": ["test"]})
            # Use proper processing to get the pooled embedding
            pooled = self._process_hf_response(test_result, 1)
            dim = pooled.shape[1]
            self._embedding_dim_cache = dim
            logger.info(f"Detected embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.warning(f"Failed to get embedding dimension from endpoint, using default 768: {e}")
            return 768  # Default for all-mpnet-base-v2


class FallbackEmbeddingAdapter:
    """Wrapper that falls back to local model if SageMaker fails."""
    
    def __init__(self, sagemaker_adapter, local_model, fallback_enabled: bool = True):
        self.sagemaker_adapter = sagemaker_adapter
        self.local_model = local_model
        self.fallback_enabled = fallback_enabled
        self._using_fallback = False
        self._sagemaker_failed = False
    
    def encode(self, *args, **kwargs):
        """Encode with SageMaker, fallback to local if it fails."""
        if self._sagemaker_failed and self.fallback_enabled:
            # Already failed, use local directly
            return self.local_model.encode(*args, **kwargs)
        
        try:
            return self.sagemaker_adapter.encode(*args, **kwargs)
        except Exception as e:
            if self.fallback_enabled:
                logger.warning(
                    f"SageMaker endpoint failed ({e}), falling back to local SentenceTransformer. "
                    f"Consider restarting the SageMaker endpoint if this persists."
                )
                self._sagemaker_failed = True
                self._using_fallback = True
                return self.local_model.encode(*args, **kwargs)
            else:
                raise
    
    def get_sentence_embedding_dimension(self):
        """Get embedding dimension."""
        if self._sagemaker_failed and self.fallback_enabled:
            return self.local_model.get_sentence_embedding_dimension()
        try:
            return self.sagemaker_adapter.get_sentence_embedding_dimension()
        except Exception:
            if self.fallback_enabled:
                return self.local_model.get_sentence_embedding_dimension()
            raise


def get_embedding_model(model_name: str, device: str, sagemaker_config: Optional[dict] = None):
    """Get an embedding model (local or SageMaker) based on configuration.

    Args:
        model_name: Model name (used for local SentenceTransformer)
        device: Device for local model (ignored for SageMaker)
        sagemaker_config: Optional dict with SageMaker config:
            - enabled: bool
            - endpoint_name: str
            - region: str (optional, defaults to eu-north-1)
            - max_batch_size: int (optional, default: 8)
            - fallback_to_local: bool (optional, default: False)

    Returns:
        Either a SentenceTransformer instance, SageMakerEmbeddingAdapter instance, 
        or FallbackEmbeddingAdapter instance
    """
    cache_key = (
        str(model_name),
        _resolve_torch_device(str(device or "auto")),
        str(bool(sagemaker_config and sagemaker_config.get("enabled", False)))
        + "|"
        + str(sagemaker_config.get("endpoint_name") if isinstance(sagemaker_config, dict) else None),
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Always prepare local model if fallback is enabled or SageMaker is disabled
    fallback_enabled = sagemaker_config and sagemaker_config.get("fallback_to_local", False)
    use_sagemaker = sagemaker_config and sagemaker_config.get("enabled", False)
    
    local_model = None
    if fallback_enabled or not use_sagemaker:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            local_model = SentenceTransformer(model_name, device=_resolve_torch_device(device))
            if not use_sagemaker:
                logger.info(f"Using local SentenceTransformer: {model_name} (device: {_resolve_torch_device(device)})")
                _MODEL_CACHE[cache_key] = local_model
                return local_model
        except Exception as e:
            if not use_sagemaker:
                # sentence-transformers can fail to import due to optional deps (e.g., datasets/pyarrow mismatch).
                # Fall back to a minimal transformers-based embedder so indexing can proceed.
                logger.warning(
                    "Failed to import/use sentence-transformers (%s). Falling back to transformers-based embeddings.",
                    e,
                )
                m = TransformersLocalEmbeddingAdapter(model_name=model_name, device=device)
                _MODEL_CACHE[cache_key] = m
                return m
            # If we're here, we still want a local model for fallback-to-local under SageMaker.
            local_model = TransformersLocalEmbeddingAdapter(model_name=model_name, device=device)
    
    if use_sagemaker:
        endpoint_name = sagemaker_config.get("endpoint_name")
        if not endpoint_name:
            raise ValueError("SageMaker is enabled but endpoint_name is not specified")
        region = sagemaker_config.get("region", "eu-north-1")
        max_batch_size = sagemaker_config.get("max_batch_size", 8)
        
        sagemaker_adapter = SageMakerEmbeddingAdapter(
            endpoint_name=endpoint_name,
            region=region,
            max_batch_size=max_batch_size,
        )
        
        if fallback_enabled:
            if local_model is None:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore

                    local_model = SentenceTransformer(model_name, device=_resolve_torch_device(device))
                except Exception:
                    local_model = TransformersLocalEmbeddingAdapter(model_name=model_name, device=device)
            logger.info(
                f"Using SageMaker endpoint with local fallback: {endpoint_name} "
                f"(region: {region}, max_batch_size: {max_batch_size})"
            )
            m = FallbackEmbeddingAdapter(sagemaker_adapter, local_model, fallback_enabled=True)
            _MODEL_CACHE[cache_key] = m
            return m
        else:
            logger.info(f"Using SageMaker endpoint: {endpoint_name} (region: {region}, max_batch_size: {max_batch_size})")
            _MODEL_CACHE[cache_key] = sagemaker_adapter
            return sagemaker_adapter
    
    # Should not reach here, but return local model as fallback
    if local_model is not None:
        _MODEL_CACHE[cache_key] = local_model
    return local_model
