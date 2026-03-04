
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import logger




@dataclass
class ExtractionResult:

    filename:       str
    raw_text:       str
    clean_text:     str
    quality_score:  float
    word_count:     int
    char_count:     int
    issues:         List[str]
    metadata:       Dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "unknown"

    @property
    def is_acceptable(self) -> bool:
        return self.quality_score >= 0.4 and self.word_count >= 50



BOILERPLATE_PATTERNS = [

    r'^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$',
    r'^\s*-\s*\d+\s*-\s*$',

    r'^\s*(?:confidential|proprietary|all rights reserved)\s*$',

    r'^\s*(?:www\.|http)[^\s]+\s*$',
    r'^\s*©\s*\d{4}.*$',

    r'^[A-Z\s]{3,40}$',

    r'\x0c',
]


GARBLED_PATTERNS = [
    r'[^\x00-\x7F]{5,}',
    r'[_|]{4,}',
    r'\.{5,}',
]




class TextCleaner:


    def __init__(
        self,
        remove_boilerplate: bool = True,
        fix_encoding:       bool = True,
        normalize_whitespace: bool = True,
        min_line_length:    int  = 5,
    ) -> None:
        self.remove_boilerplate    = remove_boilerplate
        self.fix_encoding          = fix_encoding
        self.normalize_whitespace  = normalize_whitespace
        self.min_line_length       = min_line_length

        self._boilerplate_re = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in BOILERPLATE_PATTERNS
        ]
        self._garbled_re = [re.compile(p) for p in GARBLED_PATTERNS]

    def clean(self, text: str) -> Tuple[str, List[str]]:

        issues: List[str] = []
        original_len = len(text)

        if not text.strip():
            return "", ["empty_document"]


        if self.fix_encoding:
            text = self._fix_encoding(text)


        garbled_count = sum(
            len(p.findall(text)) for p in self._garbled_re
        )
        if garbled_count > 5:
            issues.append(f"garbled_content ({garbled_count} patterns)")

            for p in self._garbled_re:
                text = p.sub(' ', text)


        if self.remove_boilerplate:
            text, removed_count = self._remove_boilerplate_lines(text)
            if removed_count > 0:
                removal_pct = removed_count / max(original_len, 1) * 100
                if removal_pct > 15:
                    issues.append(f"high_boilerplate ({removal_pct:.0f}% removed)")


        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)


        if len(text) < 100:
            issues.append("very_short_document")
        if text.count(' ') / max(len(text), 1) < 0.05:
            issues.append("low_word_density (possible binary/encoded content)")

        return text.strip(), issues

    def _fix_encoding(self, text: str) -> str:

        replacements = {
            '\u2019': "'",   '\u2018': "'",
            '\u201c': '"',   '\u201d': '"',
            '\u2013': '-',   '\u2014': '--',
            '\u00a0': ' ',   '\u00ad': '',
            '\ufeff': '',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_boilerplate_lines(self, text: str) -> Tuple[str, int]:

        lines = text.split('\n')
        clean_lines = []
        removed = 0

        for line in lines:
            is_boilerplate = False
            for pattern in self._boilerplate_re:
                if pattern.fullmatch(line.strip()):
                    is_boilerplate = True
                    removed += len(line)
                    break

            if not is_boilerplate:

                if len(line.strip()) < self.min_line_length and line.strip().isdigit():
                    removed += len(line)
                else:
                    clean_lines.append(line)

        return '\n'.join(clean_lines), removed

    def _normalize_whitespace(self, text: str) -> str:


        text = re.sub(r'\n{3,}', '\n\n', text)

        text = re.sub(r'[ \t]{2,}', ' ', text)

        text = re.sub(r'\s+([.,:;!?])', r'\1', text)
        return text




class ExtractionQualityChecker:


    def score(self, text: str) -> Tuple[float, List[str]]:

        issues = []

        if not text or not text.strip():
            return 0.0, ["empty_text"]

        words = text.split()
        word_count = len(words)
        char_count = len(text)


        if word_count < 20:
            length_score = 0.1
            issues.append("too_short")
        elif word_count < 100:
            length_score = 0.5
        elif word_count > 200:
            length_score = 1.0
        else:
            length_score = word_count / 200


        avg_word_len = char_count / max(word_count, 1)
        if 3 <= avg_word_len <= 9:
            density_score = 1.0
        elif avg_word_len < 2:
            density_score = 0.2
            issues.append("very_short_words (possible OCR garbage)")
        elif avg_word_len > 15:
            density_score = 0.3
            issues.append("very_long_words (possible concatenated text)")
        else:
            density_score = 0.6


        sentence_count = len(re.findall(r'[.!?]\s', text))
        if sentence_count == 0:
            struct_score = 0.3
            issues.append("no_sentence_boundaries")
        elif word_count / max(sentence_count, 1) < 3:
            struct_score = 0.5
            issues.append("very_short_sentences")
        else:
            struct_score = 1.0


        unique_words = len(set(w.lower() for w in words if len(w) > 3))
        richness_ratio = unique_words / max(word_count, 1)
        if richness_ratio >= 0.4:
            vocab_score = 1.0
        elif richness_ratio < 0.1:
            vocab_score = 0.2
            issues.append("low_vocabulary_richness (possible repetitive/garbled content)")
        else:
            vocab_score = richness_ratio / 0.4


        quality = round(
            0.20 * length_score +
            0.30 * density_score +
            0.25 * struct_score +
            0.25 * vocab_score,
            3
        )

        return quality, issues




class QualityExtractor:


    def __init__(
        self,
        min_quality_threshold: float = 0.35,
        warn_on_issues:        bool  = True,
    ) -> None:
        self.min_quality_threshold = min_quality_threshold
        self.warn_on_issues        = warn_on_issues
        self._cleaner  = TextCleaner()
        self._checker  = ExtractionQualityChecker()

    def extract(self, file_path: str | Path) -> ExtractionResult:

        path = Path(file_path)
        logger.info(f"[QualityExtractor] Processing: {path.name}")


        raw_text, metadata, method = self._load_raw(path)

        if not raw_text:
            return ExtractionResult(
                filename          = path.name,
                raw_text          = "",
                clean_text        = "",
                quality_score     = 0.0,
                word_count        = 0,
                char_count        = 0,
                issues            = ["extraction_failed"],
                metadata          = metadata,
                extraction_method = method,
            )


        clean_text, clean_issues = self._cleaner.clean(raw_text)

        quality_score, quality_issues = self._checker.score(clean_text)
        all_issues = clean_issues + quality_issues

        if all_issues and self.warn_on_issues:
            logger.warning(
                f"[QualityExtractor] {path.name} — quality={quality_score:.2f} "
                f"issues: {all_issues}"
            )
        elif quality_score >= 0.7:
            logger.info(
                f"[QualityExtractor] {path.name} — quality={quality_score:.2f} ✅"
            )

        words = clean_text.split()
        result = ExtractionResult(
            filename          = path.name,
            raw_text          = raw_text,
            clean_text        = clean_text,
            quality_score     = quality_score,
            word_count        = len(words),
            char_count        = len(clean_text),
            issues            = all_issues,
            metadata          = metadata,
            extraction_method = method,
        )

        return result

    def _load_raw(self, path: Path) -> Tuple[str, Dict, str]:
        """Load raw text with fallback strategies."""
        suffix  = path.suffix.lower()
        metadata: Dict = {"filename": path.name, "file_type": suffix.lstrip(".")}
        method  = "unknown"

        try:
            if suffix == ".pdf":
                text, meta, method = self._load_pdf(path)
                metadata.update(meta)
                return text, metadata, method

            elif suffix == ".docx":
                from src.utils.document_loader import DocumentLoader
                text, meta = DocumentLoader()._load_docx(path)
                metadata.update(meta)
                return text, metadata, "docx"

            elif suffix in (".txt", ".md"):
                from src.utils.document_loader import DocumentLoader
                text, meta = DocumentLoader()._load_text(path)
                metadata.update(meta)
                return text, metadata, "plaintext"

            else:
                return "", metadata, "unsupported"

        except Exception as e:
            logger.error(f"[QualityExtractor] Load failed for {path.name}: {e}")
            return "", metadata, f"failed ({e})"

    def _load_pdf(self, path: Path) -> Tuple[str, Dict, str]:

        # Try pdfplumber first (layout-aware)
        try:
            import pdfplumber
            pages = []
            metadata = {}
            with pdfplumber.open(path) as pdf:
                metadata["page_count"] = len(pdf.pages)
                if pdf.metadata:
                    metadata.update({
                        k: v for k, v in pdf.metadata.items()
                        if v and isinstance(v, str)
                    })
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text)

            text = "\n\n".join(pages)
            if len(text.strip()) > 100:
                return text, metadata, "pdfplumber"
        except Exception as e:
            logger.debug(f"[QualityExtractor] pdfplumber failed: {e}")

        try:
            import PyPDF2
            pages = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages), {}, "pypdf2"
        except Exception as e:
            logger.debug(f"[QualityExtractor] PyPDF2 failed: {e}")

        return "", {}, "all_methods_failed"

    def batch_extract(
        self,
        file_paths: List[str | Path],
    ) -> List[ExtractionResult]:

        results = []
        for fp in file_paths:
            result = self.extract(fp)
            results.append(result)

      
        passed = sum(1 for r in results if r.is_acceptable)
        logger.info(
            f"[QualityExtractor] Batch: {passed}/{len(results)} documents "
            f"passed quality threshold"
        )

        return sorted(results, key=lambda r: r.quality_score, reverse=True)
