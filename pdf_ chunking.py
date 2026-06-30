import fitz  # PyMuPDF
import re


def detect_numbering_level(text):
    text = text.strip()
    # Level 3: e.g. "8.1.2 Title"
    if re.match(r"^\d+\.[1-9]\d*\.\d+\s+", text):
        return 3
    # Level 2: e.g. "8.1 Title", "8.3 Title"
    if re.match(r"^\d+\.[1-9]\d*\s+", text):
        return 2
    # Level 1: e.g. "5.0 Title", "4 Title"
    if re.match(r"^\d+\.0\s+", text) or re.match(r"^\d+\s+", text):
        return 1
    return None


def is_heading(text, is_bold):
    """A span is a heading if it is bold AND starts with a number."""
    return is_bold and detect_numbering_level(text) is not None


def parse_pdf_to_section_chunks(pdf_path):
    doc = fitz.open(pdf_path)

    all_chunks = []
    current_chunk = {
        "title": "Document_Header",
        "heading_level": 0,
        "text_content": []
    }

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    page_text = span["text"].strip()
                    if not page_text:
                        continue

                    is_bold = bool(span["flags"] & 2)
                    heading_level = detect_numbering_level(page_text)

                    if is_heading(page_text, is_bold):
                        # Save current chunk and start a new one
                        if current_chunk["text_content"]:
                            current_chunk["text_content"] = " ".join(current_chunk["text_content"])
                            all_chunks.append(current_chunk)

                        current_chunk = {
                            "title": page_text,
                            "heading_level": heading_level,
                            "text_content": []
                        }
                    else:
                        # Overflow protection: split at 1000 chars
                        current_text_len = len(" ".join(current_chunk["text_content"]))
                        if current_text_len + len(page_text) > 1000:
                            current_chunk["text_content"] = " ".join(current_chunk["text_content"])
                            all_chunks.append(current_chunk)
                            current_chunk = {
                                "title": current_chunk["title"] + " (Continued)",
                                "heading_level": current_chunk["heading_level"],
                                "text_content": [page_text]
                            }
                        else:
                            current_chunk["text_content"].append(page_text)

    # Save the last chunk
    if current_chunk["text_content"]:
        current_chunk["text_content"] = " ".join(current_chunk["text_content"])
        all_chunks.append(current_chunk)

    return all_chunks


# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    sample_pdf = "your_document.pdf"

    try:
        chunks = parse_pdf_to_section_chunks(sample_pdf)

        for i, chunk in enumerate(chunks[:3]):
            print(f"=== [Chunk {i+1}] ===")
            print(f"Heading: {chunk['title']} (Level: {chunk['heading_level']})")
            print(f"Content Sample: {chunk['text_content'][:150]}...")
            print("=" * 20, "\n")

    except Exception as e:
        print(f"File not found or an error occurred: {e}")
