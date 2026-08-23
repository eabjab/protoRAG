"""PyTorch custom embedding backend (full ``protorag`` install).

Wraps any ``torch.nn.Module`` together with a tokenizer (or a Hugging Face
``model_name`` resolved via ``AutoModel`` / ``AutoTokenizer``). Supports mean
or CLS pooling, optional L2 normalization, and an optional linear projection
layer to a custom output dimension.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np

from protorag.core.exceptions import EmbeddingError
from protorag.embeddings.base import l2_normalize


class TorchCustomEmbedder:
    """Embedder backed by an arbitrary PyTorch module + tokenizer."""

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_name: Optional[str] = None,
        pooling: str = "mean",
        normalize: bool = True,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        if model is None and model_name is None:
            raise EmbeddingError(
                "TorchCustomEmbedder requires either a torch module (model=) or a "
                "Hugging Face identifier (model_name=)."
            )
        if pooling not in ("mean", "cls"):
            raise EmbeddingError(f"Unsupported pooling '{pooling}'. Use 'mean' or 'cls'.")
        if projection_dim is not None and projection_dim <= 0:
            raise EmbeddingError(f"projection_dim must be positive, got {projection_dim!r}.")

        try:
            import torch
            from torch import nn
        except ImportError as err:
            raise ImportError(
                "PyTorch is not installed. Run 'pip install protorag' or install "
                "PyTorch for your system."
            ) from err

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if tokenizer is None:
            if model_name is None:
                raise EmbeddingError(
                    "TorchCustomEmbedder requires a tokenizer= when a raw torch module "
                    "is supplied, or a model_name= for Hugging Face auto-loading."
                )
            try:
                from transformers import AutoTokenizer
            except ImportError as err:
                raise ImportError(
                    "transformers is not installed. Run 'pip install protorag' "
                    "or 'pip install transformers'."
                ) from err
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model is None:
            try:
                from transformers import AutoModel
            except ImportError as err:
                raise ImportError(
                    "transformers is not installed. Run 'pip install protorag' "
                    "or 'pip install transformers'."
                ) from err
            model = AutoModel.from_pretrained(model_name)

        try:
            model = model.to(device).eval()
            hidden_size = int(model.config.hidden_size)
        except Exception as err:
            raise EmbeddingError(f"Invalid torch model for embedding: {err}") from err

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._pooling = pooling
        self._normalize = normalize
        self._device = device
        self._max_length = int(max_length)
        self._kwargs: Dict[str, Any] = dict(kwargs)

        if projection_dim is not None:
            self._projection: Optional[Any] = nn.Linear(hidden_size, int(projection_dim))
            self._projection = self._projection.to(device).eval()
            self._dimension = int(projection_dim)
        else:
            self._projection = None
            self._dimension = hidden_size
        self._model_name = model_name or type(model).__name__

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend(self) -> str:
        return "torch"

    @property
    def init_kwargs(self) -> Dict[str, Any]:
        """Serializable constructor kwargs recorded in persistence manifests.

        A raw ``nn.Module`` is not serializable; indexes built with one must be
        reloaded with ``override_embedder=``.
        """
        out: Dict[str, Any] = {
            "pooling": self._pooling,
            "normalize": self._normalize,
            "device": self._device,
            "max_length": self._max_length,
        }
        if self._projection is not None:
            out["projection_dim"] = self._dimension
        if self._kwargs:
            out.update(self._kwargs)
        return out

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        batched = list(texts)
        if not batched:
            return np.zeros((0, self._dimension), dtype=np.float32)
        torch = self._torch
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(batched), batch_size):
                batch = batched[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                if self._pooling == "cls":
                    vectors = hidden[:, 0]
                else:
                    vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                if self._projection is not None:
                    vectors = self._projection(vectors)
                outputs.append(vectors.cpu().numpy().astype(np.float32))
        stacked = np.stack(outputs).astype(np.float32, copy=False)
        return l2_normalize(stacked) if self._normalize else stacked

    def embed_query(self, text: str) -> np.ndarray:
        return cast("np.ndarray", self.embed_documents([text], 32)[0])
