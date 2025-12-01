#!/usr/bin/env python3
"""Extract key information from research papers."""

import pdfplumber
from pathlib import Path
import json

def extract_first_pages(pdf_path, max_pages=3):
    """Extract text from first few pages (abstract, intro)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages[:max_pages]):
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return f"Error extracting {pdf_path}: {e}"

def main():
    papers_dir = Path("papers")
    results = {}

    for pdf_file in sorted(papers_dir.glob("*.pdf")):
        print(f"Processing {pdf_file.name}...")
        text = extract_first_pages(pdf_file, max_pages=3)
        results[pdf_file.name] = {
            "text_preview": text[:2000] if text else "",  # First 2000 chars
            "full_length": len(text)
        }

    # Save to JSON
    with open("papers/extracted_info.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExtracted information from {len(results)} papers")
    print("Saved to papers/extracted_info.json")

if __name__ == "__main__":
    main()
