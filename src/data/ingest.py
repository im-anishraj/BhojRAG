"""
BhojRAG Document Ingestion
============================
Loads raw text from multiple formats: plain text, JSONL, and PDF.
Produces a unified list of Document objects for downstream processing.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Document:
    """
    A single ingested document.
    
    Attributes:
        doc_id: Unique identifier.
        text: Raw document text.
        source: Source file path or URL.
        metadata: Arbitrary metadata (e.g., page number, title).
    """
    doc_id: str
    text: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)


class DocumentLoader:
    """
    Multi-format document loader.
    
    Supports:
        - .txt files (one document per file or paragraph-split)
        - .jsonl files (expects {"text": ..., "metadata": ...} per line)
        - .pdf files (requires PyPDF2; optional dependency)
        - directories (recursively loads all supported files)
    
    Usage:
        loader = DocumentLoader()
        docs = loader.load("data/raw/")
    """

    SUPPORTED_EXTENSIONS = {".txt", ".jsonl", ".pdf"}

    def load(self, path: str | Path) -> List[Document]:
        """
        Load documents from a file or directory.
        
        Args:
            path: Path to a single file or directory.
            
        Returns:
            List of Document objects.
        """
        path = Path(path)

        if path.is_dir():
            return self._load_directory(path)
        elif path.is_file():
            return self._load_file(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

    def _load_directory(self, dir_path: Path) -> List[Document]:
        """Recursively load all supported files from a directory."""
        docs: List[Document] = []
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                docs.extend(self._load_file(file_path))
        logger.info(f"Loaded {len(docs)} documents from {dir_path}")
        return docs

    def _load_file(self, file_path: Path) -> List[Document]:
        """Dispatch to format-specific loader."""
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self._load_txt(file_path)
        elif suffix == ".jsonl":
            return self._load_jsonl(file_path)
        elif suffix == ".pdf":
            return self._load_pdf(file_path)
        else:
            logger.warning(f"Skipping unsupported file: {file_path}")
            return []

    def _load_txt(self, file_path: Path) -> List[Document]:
        """
        Load a text file.
        Splits on double newlines to separate logical documents/paragraphs.
        """
        text = file_path.read_text(encoding="utf-8")
        # Split on double newlines — each paragraph becomes a document
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        docs = []
        for i, para in enumerate(paragraphs):
            docs.append(Document(
                doc_id=f"{file_path.stem}_{i:04d}",
                text=para,
                source=str(file_path),
                metadata={"paragraph_index": str(i)},
            ))

        logger.info(f"Loaded {len(docs)} paragraphs from {file_path}")
        return docs

    def _load_jsonl(self, file_path: Path) -> List[Document]:
        """
        Load a JSONL file.
        Each line must have a "text" field; "metadata" is optional.
        """
        import json

        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                docs.append(Document(
                    doc_id=f"{file_path.stem}_{i:04d}",
                    text=record["text"],
                    source=str(file_path),
                    metadata=record.get("metadata", {}),
                ))

        logger.info(f"Loaded {len(docs)} records from {file_path}")
        return docs

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a PDF file (requires PyPDF2).
        Each page becomes a separate document.
        """
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            logger.error(
                "PyPDF2 not installed. Install with: pip install PyPDF2"
            )
            return []

        reader = PdfReader(str(file_path))
        docs = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    doc_id=f"{file_path.stem}_p{page_num:04d}",
                    text=text.strip(),
                    source=str(file_path),
                    metadata={"page_number": str(page_num)},
                ))

        logger.info(f"Loaded {len(docs)} pages from {file_path}")
        return docs
