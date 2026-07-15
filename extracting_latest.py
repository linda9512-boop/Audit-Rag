import re
from pathlib import Path

from config import LATEST_REVISIONS_CSV
from utils import write_dicts_to_csv


MIN_ID_DIGITS = 3  # digit runs shorter than this (e.g. "CH2.2") aren't treated as a doc ID


def extract_document_id(filename):
    """
    Extract document ID: leading letters (if any) + a run of digits, anchored
    at the very start of the filename, e.g.:
    D72001_xyz_REV02.pdf       -> D72001
    CD0534-A HAL 4.0...pdf     -> CD0534
    70001791 Rev0.1...pdf      -> 70001791

    Any leading special characters (quotes, dashes, etc.) are stripped first,
    so the ID match starts at the first letter/digit. The digit run must be
    at least MIN_ID_DIGITS long, so short/incidental numbers (version
    numbers, change-order labels like "CH2.2") don't get mistaken for the
    ID. If no such pattern is found at the very start, there is no ID (no
    fallback search elsewhere in the filename) -- the file is left out of
    dedup and always included as-is.
    """
    name = re.sub(r"^[^A-Za-z0-9]+", "", filename)
    match = re.match(rf"([A-Za-z]*\d{{{MIN_ID_DIGITS},}})", name)
    return match.group(1).upper() if match else None


def extract_revision(filename):
    """
    Supports:
    Rev01
    REV01
    REV 1
    rev1
    revision01
    Revision 02
    REV_003
    Rev-4
    rev.05
    v02
    V2
    version02
    ver.2
    rev.0.2   (decimal "0.N" form -- treated the same as "N", e.g. rev.0.2 == v2)

    Returns None if no revision pattern is found at all (as opposed to an
    actual revision of 0) -- callers should treat "no revision info" as
    "can't safely compare against other files with the same doc ID."
    """
    stem = Path(filename).stem

    # (?<![A-Za-z]) / (?!\d) instead of \b: \b treats "_" as a word char, so
    # e.g. "MR1_Rev03" or "REV_003" would otherwise fail to match (no boundary
    # before "rev" / after the digits when adjacent to "_").
    #
    # The optional "(?:\.0*(\d+))?" tail captures a decimal suffix like the
    # ".2" in "rev.0.2". When the whole-number part is 0, that decimal part
    # is the real revision (rev.0.2 == v2), so it takes priority below.
    patterns = [
        r"(?<![A-Za-z])(?:rev(?:ision)?|version|ver|v)[\s_\-\.]*0*(\d+)(?:\.0*(\d+))?(?!\d)",
        r"(?<![A-Za-z])r[\s_\-\.]*0*(\d+)(?:\.0*(\d+))?(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, stem, re.IGNORECASE)
        if match:
            whole, decimal = match.group(1), match.group(2)
            if decimal is not None and int(whole) == 0:
                return int(decimal)
            return int(whole)

    return None


def build_metadata(device_name, folder_type, pdf_path):
    """
    pdf_path : Path object or string
    """

    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    return {
        "device_name": device_name,
        "folder_type": folder_type,
        "filename": filename,
        "document_id": extract_document_id(filename),
        "revision": extract_revision(filename),
        "local_path": str(pdf_path)
    }


    """
    Given a PDF path, return folder_type and subfolder relative to docs_folder.

    Example:
      docs/design changes/concept proposal/D72001_REV03.pdf
        → folder_type = "design changes"
        → subfolder   = "concept proposal"

      docs/design changes/D85002_REV02.pdf
        → folder_type = "design changes"
        → subfolder   = None  (file is directly inside folder_type)

      docs/D99001_REV01.pdf
        → folder_type = None  (file is directly inside docs root)
        → subfolder   = None
    """
    parts = pdf_path.relative_to(docs_folder).parts  # e.g. ("design changes", "concept proposal", "file.pdf")

    folder_type = parts[0] if len(parts) >= 2 else None
    subfolder   = parts[1] if len(parts) >= 3 else None

    return {"folder_type": folder_type, "subfolder": subfolder}


def get_latest_revisions(docs_folder: str) -> list[dict]:
    """
    Scan docs_folder recursively, group PDFs by document ID, and return
    only the highest-revision file per document ID.

    Each entry in the returned list is a metadata dict:
        {
          "local_path"   : str,
          "filename"     : str,
          "document_id"  : str | None,
          "revision"     : int,
          "folder_type"  : str | None,   # top-level folder under docs_folder
          "subfolder"    : str | None,   # one level deeper (if it exists)
        }
    """
    docs_folder = Path(docs_folder)
    pdf_files = list(docs_folder.rglob("*.pdf"))  # recursive — searches all subfolders

    # Group by document ID: { "D72001": (revision_int, Path), ... }
    latest: dict[str, tuple[int, Path]] = {}
    always_include: list[Path] = []  # no ID, or ID but no revision info -- can't safely dedup these

    for pdf_path in pdf_files:
        doc_id = extract_document_id(pdf_path.name)
        if doc_id is None:
            always_include.append(pdf_path)
            continue

        rev = extract_revision(pdf_path.name)
        if rev is None:
            # Has an ID but no revision marker -- nothing to compare against, keep it.
            always_include.append(pdf_path)
            continue

        if doc_id not in latest or rev > latest[doc_id][0]:
            latest[doc_id] = (rev, pdf_path)

    def _to_meta(pdf_path: Path) -> dict:
        folder_info = _extract_folder_info(pdf_path, docs_folder)
        meta = {
            "local_path":  str(pdf_path),
            "filename":    pdf_path.name,
            "document_id": extract_document_id(pdf_path.name),
            "revision":    extract_revision(pdf_path.name),
            "folder_type": folder_info["folder_type"],
        }
        if folder_info["subfolder"] is not None:
            meta["subfolder"] = folder_info["subfolder"]
        return meta

    return [_to_meta(path) for _, path in latest.values()] + [_to_meta(p) for p in always_include]


def save_latest_revisions_csv(docs_folder: str, output_path: str):
    """Write the latest-revision selection to a CSV file for review."""
    latest = get_latest_revisions(docs_folder)
    latest.sort(key=lambda m: m["filename"].lower())

    fieldnames = ["filename", "document_id", "revision", "folder_type", "subfolder", "local_path"]
    write_dicts_to_csv(output_path, latest, fieldnames)
    print(f"Saved {len(latest)} latest-revision file entries to {output_path}")


if __name__ == "__main__":
   

    _docs_folder = Path(__file__).parent / "docs"
    _output_path = Path(__file__).parent / LATEST_REVISIONS_CSV
    save_latest_revisions_csv(str(_docs_folder), str(_output_path))