
from pathlib import Path
from typing import Dict, Tuple

from src.utils.logger import logger


class DocumentLoader:


    SUPPORTED = {".pdf", ".docx", ".txt", ".md"}



    def load(self, file_path: str | Path) -> Tuple[str, Dict]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED))}"
            )

        logger.info(f"Loading document: {path.name} ({suffix})")

        loaders = {
            ".pdf":  self._load_pdf,
            ".docx": self._load_docx,
            ".txt":  self._load_text,
            ".md":   self._load_text,
        }

        text, metadata = loaders[suffix](path)
        metadata.update(
            {
                "filename":  path.name,
                "file_type": suffix.lstrip("."),
                "file_size": path.stat().st_size,
                "file_path": str(path.resolve()),
            }
        )

        logger.info(
            f"Loaded '{path.name}': {len(text)} characters extracted"
        )
        return text, metadata



    def _load_pdf(self, path: Path) -> Tuple[str, Dict]:

        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

        pages_text: list[str] = []
        metadata: Dict = {}

        with pdfplumber.open(path) as pdf:
            metadata["page_count"] = len(pdf.pages)
            if pdf.metadata:
                metadata["pdf_author"]  = pdf.metadata.get("Author", "")
                metadata["pdf_title"]   = pdf.metadata.get("Title", "")
                metadata["pdf_subject"] = pdf.metadata.get("Subject", "")

            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(f"[Page {i + 1}]\n{page_text}")

        full_text = "\n\n".join(pages_text)
        return full_text, metadata

    def _load_docx(self, path: Path) -> Tuple[str, Dict]:

        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)

        metadata = {
            "paragraph_count": len(paragraphs),
        }
        if doc.core_properties:
            props = doc.core_properties
            metadata["docx_author"]  = props.author or ""
            metadata["docx_title"]   = props.title  or ""
            metadata["docx_subject"] = props.subject or ""

        return full_text, metadata

    def _load_text(self, path: Path) -> Tuple[str, Dict]:
        """Load plain text or Markdown files."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                text = path.read_text(encoding=enc)
                return text, {"encoding": enc}
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Cannot decode file '{path.name}' with common encodings")
