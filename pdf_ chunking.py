import fitz  # PyMuPDF
import re

# 'text' (page_text) is the actual text content of a single span extracted one by one from the main loop using PyMuPDF.
# 1. Define common section keywords
COMMON_SECTIONS = {
    "abstract", "introduction", "background", "related work",
    "methods", "methodology", "experiments", "results",
    "discussion", "conclusion", "references"
}

# 2. Numbering level detection function
def detect_numbering_level(text):
    text = text.strip()
    if re.match(r"^\d+\.\d+\.\d+\s+", text):
        return 3
    if re.match(r"^\d+\.\d+\s+", text):
        return 2
    if re.match(r"^\d+\s+", text):
        return 1
    return None

# 3. Numbering removal function
def clean_heading_text(text):
    text = text.strip()
    text = re.sub(r"^\d+(\.\d+)*\s+", "", text)
    return text.strip()

# 4. Heading confidence score calculation function
def heading_confidence(text, is_bold, font_size, body_font_size):
    text_clean = clean_heading_text(text).lower()
    score = 0

    if detect_numbering_level(text) is not None:
        score += 3
    if text_clean in COMMON_SECTIONS:
        score += 3
    if is_bold:
        score += 1
    if font_size > body_font_size + 1:
        score += 1
    if len(text.split()) <= 10:
        score += 1

    return score

# 5. Function to dynamically determine the body font size (optimized using pages 3 & 4)
def get_body_font_size(doc):
    font_sizes = []
    # Safely target pages 3 and 4 (index 2, 3) where body text is most likely to appear.
    target_pages = [2, 3] if len(doc) >= 4 else [0]

    for page_idx in target_pages:
        if page_idx < len(doc):
            blocks = doc[page_idx].get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                font_sizes.append(round(span["size"], 1))

    # Dictionary count and value sorting logic to find the most common font size.
    size_dict = {}
    for size in font_sizes:
        size_dict[size] = size_dict.get(size, 0) + 1

    if not size_dict:
        return 10.0  # Default value fallback for empty/ghost documents with no text

    return max(size_dict.keys())  # Return the most frequently appearing font size


# 6. Main pipeline function: core logic for reading a PDF and splitting it into chunks
def parse_pdf_to_section_chunks(pdf_path):
    doc = fitz.open(pdf_path)

    # [Step 1] First, determine the base body font size of the document.
    body_font_size = get_body_font_size(doc)
    print(f"[*] Detected Body Font Size: {body_font_size}pt\n")

    all_chunks = []
    # Create a default chunk to temporarily hold content before the first heading appears.
    current_chunk = {
        "title": "Document_Header_or_Abstract",
        "heading_level": 0,
        "text_content": []
    }

    # [Step 2] Sequentially scan the entire document page by page.
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    for span in line["spans"]:
                        # page_text and related info are extracted here from each span.
                        page_text = span["text"].strip()
                        if not page_text:
                            continue  # Skip empty spans

                        font_size = span["size"]
                        # If the 2nd bit of PyMuPDF's flags is set, the text is Bold.
                        is_bold = bool(span["flags"] & 2)

                        # [Step 3] Calculate the heading confidence score.
                        score = heading_confidence(page_text, is_bold, font_size, body_font_size)
                        heading_level = detect_numbering_level(page_text)

                        # If the score is 4 or above, treat it as a new section heading.
                        if score >= 4:
                            if current_chunk["text_content"]:
                                current_chunk["text_content"] = " ".join(current_chunk["text_content"])
                                all_chunks.append(current_chunk)

                            current_chunk = {
                                "title": page_text,
                                "heading_level": heading_level if heading_level else 1,
                                "text_content": []
                            }
                        else:
                            # Character count overflow protection logic.
                            # Calculate the total length of accumulated text plus the new incoming text.
                            current_text_len = len(" ".join(current_chunk["text_content"]))

                            if current_text_len + len(page_text) > 1000:
                                # Step 1: Exceeds 1000 chars — safely finalize and store the current chunk.
                                current_chunk["text_content"] = " ".join(current_chunk["text_content"])
                                all_chunks.append(current_chunk)

                                # Step 2: Open a new continuation chunk with the same title but an empty content list.
                                current_chunk = {
                                    "title": current_chunk["title"] + " (Continued)",  # e.g., "1. Intro (Continued)"
                                    "heading_level": current_chunk["heading_level"],
                                    "text_content": [page_text]  # Add the new text as the first entry in the new chunk.
                                }
                            else:
                                # If under 1000 chars, append to the current chunk as usual.
                                current_chunk["text_content"].append(page_text)

    # [Step 4] After all loops finish, save any remaining data in the last chunk.
    if current_chunk["text_content"]:
        current_chunk["text_content"] = " ".join(current_chunk["text_content"])
        all_chunks.append(current_chunk)

    return all_chunks

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    # Enter the path to the PDF file you want to test.
    sample_pdf = "your_document.pdf"

    try:
        chunks = parse_pdf_to_section_chunks(sample_pdf)

        # Print the top 3 chunked results neatly.
        for i, chunk in enumerate(chunks[:3]):
            print(f"=== [Chunk {i+1}] ===")
            print(f"Heading: {chunk['title']} (Level: {chunk['heading_level']})")
            print(f"Content Sample: {chunk['text_content'][:150]}...")  # Print only the first 150 characters
            print("=" * 20, "\n")

    except Exception as e:
        print(f"File not found or an error occurred: {e}")
