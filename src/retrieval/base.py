"""
BhojRAG Base Retriever
========================
Abstract base class defining the retriever interface.
All retriever implementations (sparse, dense, hybrid) inherit from this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.data.chunker import Chunk


@dataclass
class RetrievalResult:
    """
    A single retrieval result with score and metadata.

    Attributes:
        chunk_id: ID of the retrieved chunk.
        text: Text content of the chunk.
        score: Retrieval score (higher = more relevant).
        rank: Rank in the result list (1-indexed).
        source: Source file/URL.
        metadata: Additional metadata.
    """

    chunk_id: str
    text: str
    score: float
    rank: int
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.

    Subclasses must implement:
        - index(chunks): Build retrieval index from corpus chunks
        - retrieve(query, top_k): Retrieve top-k results for a query

    Optional overrides:
        - save_index(path): Persist index to disk
        - load_index(path): Load index from disk
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._indexed = False
        self._corpus: List[Chunk] = []

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        """
        Build retrieval index from corpus chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects sorted by score (descending).
        """
        ...

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve results for multiple queries.
        Default implementation: sequential calls to retrieve().
        Override in subclasses for batched efficiency.

        Args:
            queries: List of query strings.
            top_k: Number of results per query.

        Returns:
            List of result lists, one per query.
        """
        return [self.retrieve(q, top_k) for q in queries]

    def save_index(self, path: str) -> None:
        """Persist the retrieval index to disk. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support index persistence."
        )

    def load_index(self, path: str) -> None:
        """Load the retrieval index from disk. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support index persistence."
        )

    @property
    def is_indexed(self) -> bool:
        """Whether the index has been built."""
        return self._indexed

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"indexed={self._indexed}, corpus_size={len(self._corpus)})"
        )
