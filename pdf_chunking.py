import os
import fitz  # PyMuPDF
import re

from config import OUTPUT_DIR
from utils import write_dicts_to_csv


def detect_numbering_level(text):
    """Depth of the numbering prefix, e.g. "3.1.2.1 Title" -> 4. "X.0 Title" counts as level 1.
    Trailing "." after the last digit group is optional, so "2. Title" (top-level) matches too.
    "Appendix D" / "Annex 1" style headings (no numeric prefix) also count as level 1 --
    these were previously missed entirely, silently absorbed into whatever section came
    before them (or Document_Header, if before the first real heading)."""
    text = text.strip()
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", text)
    if match:
        parts = match.group(1).split(".")
        if len(parts) == 2 and parts[1] == "0":
            return 1
        return len(parts)
    if APPENDIX_ANNEX_RE.match(text):
        return 1
    return None


APPENDIX_ANNEX_RE = re.compile(r"^(?:Appendix|Annex)\b", re.IGNORECASE)

TOC_DOT_LEADER_RE = re.compile(r"\.{5,}\s*\d+\s*$")  # e.g. "Introduction .......... 13"

# Real headings are short standalone lines that leave most of the page width blank
# to their right (observed 223-474pt). Table rows and revision-history entries run
# almost to the margin (observed <=72pt) or, for wrapped prose, land well below 200.
HEADING_MIN_TRAILING_SPACE = 200


def is_plausible_section_number(numbering_prefix):
    """Reject numbering prefixes that are real numbers but not real section numbers:
    revision cells ("01", "04") are zero-padded, and years ("2022") or doc IDs
    ("70002557") run past 2 digits -- no real section in this doc goes past
    double digits (e.g. "2.4.10")."""
    parts = numbering_prefix.split(".")
    return all(len(p) <= 2 and not (len(p) > 1 and p[0] == "0") for p in parts)


def is_heading(text, trailing_space):
    """A line is a heading if it starts with a plausible section-numbering prefix
    (or an "Appendix"/"Annex" label) and sits on a short standalone line (see
    HEADING_MIN_TRAILING_SPACE).

    Font size can't be used as the signal here: only the first few headings in this
    doc (docx export) got the Heading style applied and render large (18-24pt) --
    everything from section 2.4 onward is plain body-sized text (9.96pt), identical
    to surrounding paragraphs and table rows. Bold doesn't work either, since this
    PDF's span flags never set the bold bit (flags & 16) even on visibly bold text.
    TOC false-positives are filtered separately by skipping TOC pages entirely
    (see is_toc_page).
    """
    text = text.strip()
    match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", text)
    if match:
        if not is_plausible_section_number(match.group(1)):
            return False
    elif not APPENDIX_ANNEX_RE.match(text):
        return False
    return trailing_space is not None and trailing_space >= HEADING_MIN_TRAILING_SPACE


def _line_in_any_table(bbox, table_bboxes):
    """True if a line's bbox center falls inside any detected table region.

    Table cells (revision-history rows, wiring/jumper lists, etc.) are frequently
    short, numbered, and left-aligned -- geometrically indistinguishable from a
    real heading by is_heading()'s numbering + trailing-space checks alone, e.g.
    "1 W36 P1002734, 24V AND 48V" in a wiring table matches the same "N <text>"
    shape as a real top-level heading. Excluding anything PyMuPDF's table detector
    finds removes this whole class of false positives without needing table rows
    to look any particular way.
    """
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return any(tx0 <= cx <= tx1 and ty0 <= cy <= ty1 for tx0, ty0, tx1, ty1 in table_bboxes)


def is_toc_page(page):
    """A page is a Table of Contents page if any line ends in a dot-leader + page
    number (e.g. "...................... 13"). Real headings never look like this --
    only TOC entries connect a title to a page number this way -- so pages matching
    this are skipped wholesale rather than trying to filter individual lines, since
    TOC line-splitting is inconsistent (see parse_pdf_to_section_chunks)."""
    return bool(TOC_DOT_LEADER_RE.search(page.get_text()))


def merge_block_lines(block):
    """Group a block's PyMuPDF lines into visual lines by shared y-position.

    Some docx-export PDFs split what looks like one line (e.g. a heading number and
    its title) into separate `line` dicts positioned via tab stops, rather than
    separate spans within a single line -- they share the same bbox top (y0) but
    have no `line` object joining them. Grouping by y0 reunites them so the
    combined text can be matched against the numbering regex.
    """
    groups = {}
    for line in block["lines"]:
        y0 = round(line["bbox"][1], 1)
        groups.setdefault(y0, []).append(line)

    visual_lines = []
    for y0 in sorted(groups):
        lines_at_y = sorted(groups[y0], key=lambda l: l["bbox"][0])
        spans = [s for line in lines_at_y for s in line["spans"] if s["text"]]
        visual_lines.append(spans)
    return visual_lines


def format_page_range(page_start, page_end):
    """Render a chunk's page span as "3" (single page) or "3-5" (spans pages)."""
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _finalize_chunk(chunk: dict) -> dict:
    """Collapse a chunk's accumulated text_content list into one string, and its
    page_start/page_end into a single "page" range string. page_start/page_end
    are only needed as working state while parsing -- popping them here means
    callers never see them, instead of every caller having to filter them out."""
    chunk["text_content"] = " ".join(chunk["text_content"])
    chunk["page"] = format_page_range(chunk.pop("page_start"), chunk.pop("page_end"))
    return chunk


def parse_pdf_to_section_chunks(pdf_path):
    doc = fitz.open(pdf_path)

    all_chunks = []
    heading_stack = {}  # level -> heading text, e.g. {1: "3 GOPS...", 2: "3.1 Global..."}
    current_chunk = {
        "title": "Document_Header",
        "heading_level": 0,
        "text_content": [],
        "page_start": None,
        "page_end": None
    }

    for page_num, page in enumerate(doc, start=1):
        if is_toc_page(page):
            continue

        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        table_bboxes = [t.bbox for t in page.find_tables().tables]

        for b in blocks:
            if "lines" not in b:
                continue
            for spans in merge_block_lines(b):
                # Combine all spans on this visual line before checking for a heading --
                # PDF generators (esp. docx exports) routinely split one heading like
                # "2.1 Design and Development Planning" into separate spans ("2.1", " ",
                # "Design and Development Planning"), so no single span ever matches.
                if not spans:
                    continue

                page_text = "".join(s["text"] for s in spans).strip()
                if not page_text:
                    continue

                heading_level = detect_numbering_level(page_text)
                line_bbox = (
                    min(s["bbox"][0] for s in spans),
                    min(s["bbox"][1] for s in spans),
                    max(s["bbox"][2] for s in spans),
                    max(s["bbox"][3] for s in spans),
                )
                line_right = line_bbox[2]
                trailing_space = page_width - line_right

                if is_heading(page_text, trailing_space) and not _line_in_any_table(line_bbox, table_bboxes):
                    # Save current chunk and start a new one
                    if current_chunk["text_content"]:
                        all_chunks.append(_finalize_chunk(current_chunk))

                    # Drop stale deeper/equal-level headings, then record this one
                    heading_stack = {lvl: h for lvl, h in heading_stack.items() if lvl < heading_level}
                    heading_stack[heading_level] = page_text
                    heading_path = " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))

                    current_chunk = {
                        "title": heading_path,
                        "heading_level": heading_level,
                        "text_content": [],
                        "page_start": page_num,
                        "page_end": page_num
                    }
                else:
                    # Overflow protection: split at 1000 chars
                    current_text_len = len(" ".join(current_chunk["text_content"]))
                    if current_text_len + len(page_text) > 1000:
                        all_chunks.append(_finalize_chunk(current_chunk))
                        base_title = current_chunk["title"]
                        if not base_title.endswith(" (Continued)"):
                            base_title += " (Continued)"
                        current_chunk = {
                            "title": base_title,
                            "heading_level": current_chunk["heading_level"],
                            "text_content": [page_text],
                            "page_start": page_num,
                            "page_end": page_num
                        }
                    else:
                        if current_chunk["page_start"] is None:
                            current_chunk["page_start"] = page_num
                        current_chunk["page_end"] = page_num
                        current_chunk["text_content"].append(page_text)

    # Save the last chunk
    if current_chunk["text_content"]:
        all_chunks.append(_finalize_chunk(current_chunk))

    return all_chunks


def extract_bold_text(pdf_path):
    """
    Return every bold-flagged span in the PDF (merged by visual line), as
    [{"page": 1-based page number, "text": ..., "font": ..., "size": ...}, ...].
    Useful for checking whether "bold" is actually a meaningful signal in a
    given document before relying on it for heading detection.
    """
    doc = fitz.open(pdf_path)
    results = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                bold_spans = [s for s in line["spans"] if s["text"].strip() and (s["flags"] & 16)]
                if not bold_spans:
                    continue
                results.append({
                    "page": page_num,
                    "text": "".join(s["text"] for s in bold_spans).strip(),
                    "font": bold_spans[0]["font"],
                    "size": bold_spans[0]["size"],
                })

    return results


def dump_all_spans(pdf_path):
    """
    Return raw info for every span in the PDF (no bold filter), as
    [{"page": ..., "text": ..., "font": ..., "flags": ..., "is_bold": ..., "size": ...}, ...].
    Use this to see why a given heading line isn't being flagged bold --
    compare its font/flags against a line that IS detected correctly.
    """
    doc = fitz.open(pdf_path)
    results = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    if not s["text"].strip():
                        continue
                    results.append({
                        "page": page_num,
                        "text": s["text"],
                        "font": s["font"],
                        "flags": s["flags"],
                        "is_bold": bool(s["flags"] & 16),
                        "size": s["size"],
                    })

    return results


def save_all_spans_csv(pdf_path, output_path):
    """Dump every span's raw font/flags info to CSV for debugging heading detection."""
    spans = dump_all_spans(pdf_path)

    fieldnames = ["page", "text", "font", "flags", "is_bold", "size"]
    write_dicts_to_csv(output_path, spans, fieldnames)
    print(f"Saved {len(spans)} spans to {output_path}")


def save_bold_text_csv(pdf_path, output_path):
    """Extract all bold text from a PDF and write it to CSV for review."""
    bold_lines = extract_bold_text(pdf_path)

    fieldnames = ["page", "text", "font", "size"]
    write_dicts_to_csv(output_path, bold_lines, fieldnames)
    print(f"Saved {len(bold_lines)} bold lines to {output_path}")


def save_chunks_csv(pdf_path, output_path, sample_length=1000):
    """Parse a PDF into section chunks and write heading + content sample to CSV for review."""
    chunks = parse_pdf_to_section_chunks(pdf_path)

    fieldnames = ["chunk_number", "heading", "heading_level", "page", "content_sample"]
    rows = [
        {
            "chunk_number": i,
            "heading": chunk["title"],
            "heading_level": chunk["heading_level"],
            "page": chunk["page"],
            "content_sample": chunk["text_content"][:sample_length],
        }
        for i, chunk in enumerate(chunks, start=1)
    ]
    write_dicts_to_csv(output_path, rows, fieldnames)
    print(f"Saved {len(chunks)} chunks to {output_path}")


# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    sample_pdf = r"C:/Users/z005a2sy/AppData/Local/Temp/ce404d7b-f745-4351-b634-b6f60e365b51_Design Changes.zip.Design Changes.zip/Maintenance Releases/D69533 HAL 4.0 MR1 Maintenance Release Document Rev05.docx.pdf"
    output_csv = os.path.join(OUTPUT_DIR, "debug", "chunks_sample.csv")

    try:
        save_chunks_csv(sample_pdf, output_csv)
    except Exception as e:
        print(f"File not found or an error occurred: {e}")
