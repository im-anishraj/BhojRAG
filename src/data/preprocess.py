"""
BhojRAG Text Preprocessing
============================
Handles Unicode normalization, Devanagari-specific cleaning,
noise removal, deduplication, and basic transliteration support
for Hinglish / Latin-script Bhojpuri.
"""

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional, Set

from src.data.ingest import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Transliteration mappings (Latin → Devanagari, basic phonetic)
# ---------------------------------------------------------------------------
# This covers common Hinglish / romanized Bhojpuri spellings.
# For production, use a learned or rule-based transliteration model.
LATIN_TO_DEVANAGARI: Dict[str, str] = {
    "a": "अ", "aa": "आ", "i": "इ", "ee": "ई", "u": "उ", "oo": "ऊ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ",
    "ka": "क", "kha": "ख", "ga": "ग", "gha": "घ",
    "cha": "च", "chha": "छ", "ja": "ज", "jha": "झ",
    "ta": "ट", "tha": "ठ", "da": "ड", "dha": "ढ",
    "na": "न", "pa": "प", "pha": "फ", "ba": "ब", "bha": "भ",
    "ma": "म", "ya": "य", "ra": "र", "la": "ल", "va": "व",
    "wa": "व", "sha": "श", "sa": "स", "ha": "ह",
    "kya": "क्या", "gya": "ज्ञ",
    "ri": "रि", "re": "रे", "ro": "रो", "ru": "रु",
    "se": "से", "ke": "के", "me": "में", "ko": "को",
    "ki": "कि", "hi": "हि", "ho": "हो", "he": "हे",
    "nahi": "नहीं", "hai": "है", "hain": "हैं", "tha": "था",
    "aur": "और", "par": "पर", "bhi": "भी", "koi": "कोई",
}


class TextPreprocessor:
    """
    Text preprocessing pipeline for Bhojpuri text.
    
    Steps (configurable):
        1. Unicode normalization (NFC/NFKC)
        2. Remove URLs, emails, control characters
        3. Devanagari-specific normalization (nukta, matra variants)
        4. Optional basic transliteration (Latin → Devanagari)
        5. Whitespace normalization
        6. Length filtering
        7. Deduplication (exact hash)
    
    Usage:
        preprocessor = TextPreprocessor(
            normalize_unicode=True,
            remove_urls=True,
            transliterate=True,
        )
        cleaned_docs = preprocessor.process(raw_docs)
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        lowercase: bool = False,
        min_doc_length: int = 20,
        deduplicate: bool = True,
        transliterate: bool = False,
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.lowercase = lowercase
        self.min_doc_length = min_doc_length
        self.deduplicate = deduplicate
        self.transliterate = transliterate
        self._seen_hashes: Set[str] = set()

    def process(self, documents: List[Document]) -> List[Document]:
        """
        Apply full preprocessing pipeline to a list of documents.
        
        Args:
            documents: Raw Document objects.
            
        Returns:
            Cleaned and filtered Document objects.
        """
        cleaned: List[Document] = []
        stats = {"total": len(documents), "kept": 0, "removed_short": 0,
                 "removed_dup": 0}

        for doc in documents:
            text = self.clean_text(doc.text)

            # Length filter
            if len(text) < self.min_doc_length:
                stats["removed_short"] += 1
                continue

            # Deduplication
            if self.deduplicate:
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if text_hash in self._seen_hashes:
                    stats["removed_dup"] += 1
                    continue
                self._seen_hashes.add(text_hash)

            cleaned.append(Document(
                doc_id=doc.doc_id,
                text=text,
                source=doc.source,
                metadata=doc.metadata,
            ))
            stats["kept"] += 1

        logger.info(
            f"Preprocessing: {stats['total']} → {stats['kept']} docs "
            f"(short={stats['removed_short']}, dup={stats['removed_dup']})"
        )
        return cleaned

    def clean_text(self, text: str) -> str:
        """
        Apply all text-level cleaning steps.
        
        Args:
            text: Raw text string.
            
        Returns:
            Cleaned text string.
        """
        # 1. Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # 2. Remove URLs
        if self.remove_urls:
            text = re.sub(
                r"https?://\S+|www\.\S+", "", text
            )

        # 3. Remove emails
        if self.remove_emails:
            text = re.sub(
                r"\S+@\S+\.\S+", "", text
            )

        # 4. Remove control characters (keep Devanagari, punctuation, digits)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 5. Devanagari-specific normalization
        text = self._normalize_devanagari(text)

        # 6. Basic transliteration
        if self.transliterate:
            text = self._transliterate_latin_to_devanagari(text)

        # 7. Lowercase (generally not useful for Devanagari)
        if self.lowercase:
            text = text.lower()

        # 8. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _normalize_devanagari(text: str) -> str:
        """
        Normalize Devanagari-specific variations:
        - Nukta normalization (e.g., क़ → क)
        - Anusvara/chandrabindu normalization
        - Remove zero-width joiners/non-joiners
        """
        # Remove zero-width characters
        text = text.replace("\u200b", "")  # ZWSP
        text = text.replace("\u200c", "")  # ZWNJ
        text = text.replace("\u200d", "")  # ZWJ
        text = text.replace("\ufeff", "")  # BOM

        # Normalize nukta: remove nukta (U+093C) to merge variant forms
        # e.g., क़ (क + ़) → क
        # This is aggressive but helps with spelling variation
        text = text.replace("\u093c", "")

        # Normalize chandrabindu to anusvara in some contexts
        # (conservative: only when followed by consonant)
        text = re.sub("\u0901([\u0915-\u0939])", "\u0902\\1", text)

        return text

    @staticmethod
    def _transliterate_latin_to_devanagari(text: str) -> str:
        """
        Basic rule-based transliteration from Latin/Hinglish to Devanagari.
        
        This is a coarse heuristic. For research, consider integrating
        IndicNLP or a learned transliteration model for higher accuracy.
        
        Only transliterates words that appear fully Latin-script.
        """
        words = text.split()
        result = []

        for word in words:
            # Check if word is primarily Latin characters
            if re.match(r"^[a-zA-Z]+$", word):
                lower_word = word.lower()
                # Try multi-char mappings first (longest match)
                transliterated = LATIN_TO_DEVANAGARI.get(lower_word)
                if transliterated:
                    result.append(transliterated)
                else:
                    # Keep original if no mapping found
                    result.append(word)
            else:
                result.append(word)

        return " ".join(result)

    def reset_dedup_cache(self) -> None:
        """Reset the deduplication hash cache."""
        self._seen_hashes.clear()
