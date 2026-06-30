import re
from pathlib import Path


def extract_document_id(filename):
    """
    Extract document ID such as:
    D72001_xyz_REV02.pdf -> D72001
    """
    match = re.search(r"\b(D\d+)\b", filename, re.IGNORECASE)
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
    """
    stem = Path(filename).stem

    patterns = [
        r"\brev(?:ision)?[\s_\-\.]*0*(\d+)\b",
        r"\br[\s_\-\.]*0*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, stem, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return 0


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


def _extract_folder_info(pdf_path: Path, docs_folder: Path) -> dict:
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
    no_id: list[Path] = []

    for pdf_path in pdf_files:
        doc_id = extract_document_id(pdf_path.name)
        if doc_id is None:
            no_id.append(pdf_path)
            continue

        rev = extract_revision(pdf_path.name)
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

    return [_to_meta(path) for _, path in latest.values()] + [_to_meta(p) for p in no_id]