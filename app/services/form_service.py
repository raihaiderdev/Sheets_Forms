import io
import re
import openpyxl

from app.models.db import db
from app.models.form import Form, FormField, FormResponse, FormFieldResponse


# ---------------------------------------------------------------------------
# Excel parsing — smart header detection
# ---------------------------------------------------------------------------

def _cell_str(cell) -> str:
    """Return trimmed string value of a cell, empty string if None/blank."""
    if cell is None:
        return ""
    return re.sub(r'\s+', ' ', str(cell)).strip()


def _row_score(row_values: list[str]) -> float:
    """
    Score a row to judge how likely it is to be a header row.
    Higher = more likely to be the column-header row.

    Heuristics:
    - Penalise rows with very few filled cells (merged title rows).
    - Penalise rows where a single cell spans most of the non-empty content
      (merged title cell scenario).
    - Reward rows where most cells are short labels (not long sentences).
    - Reward rows that look like typical column headers (short words,
      mixed case or title case, no trailing numbers).
    """
    filled = [v for v in row_values if v]
    n_filled = len(filled)
    n_total = len(row_values)

    if n_filled == 0:
        return -1.0

    score = 0.0

    # Reward density of filled cells
    density = n_filled / max(n_total, 1)
    score += density * 5

    # Penalise if only 1 or 2 cells filled in a wide row (likely a title)
    if n_filled <= 2 and n_total > 4:
        score -= 4

    # Reward short cell values (column headers are usually short)
    avg_len = sum(len(v) for v in filled) / n_filled
    if avg_len <= 30:
        score += 2
    elif avg_len > 60:
        score -= 2

    # Penalise cells that look like data values: pure numbers, dates, etc.
    data_like = sum(1 for v in filled if re.match(r'^\d[\d\s/\-\.]*$', v))
    score -= (data_like / n_filled) * 3

    return score


def parse_excel(file_bytes: bytes) -> dict:
    """
    Read the full sheet and return:
      {
        "headers": [...],        # best-guess header row values
        "header_row": int,       # 1-based row index
        "sheet_title": str,      # text from rows BEFORE the header row
        "preview": [[...], ...]  # first 5 data rows after the header
      }
    Raises ValueError if no usable header row is found.
    """
    wb = openpyxl.load_workbook(
        filename=io.BytesIO(file_bytes), read_only=True, data_only=True
    )
    ws = wb.worksheets[0]

    # Collect all rows (up to 50 to keep it fast)
    all_rows: list[list[str]] = []
    for row in ws.iter_rows(min_row=1, max_row=50, values_only=True):
        all_rows.append([_cell_str(c) for c in row])

    wb.close()

    if not all_rows:
        raise ValueError("The sheet appears to be empty.")

    # Find the best header row
    best_idx = 0
    best_score = -999.0
    for i, row in enumerate(all_rows):
        s = _row_score(row)
        if s > best_score:
            best_score = s
            best_idx = i

    header_row_values = [v for v in all_rows[best_idx] if v]
    if not header_row_values:
        raise ValueError("Could not detect a header row in the sheet.")

    # Collect title text from rows before the header row
    title_parts = []
    for row in all_rows[:best_idx]:
        line = " ".join(v for v in row if v)
        if line:
            title_parts.append(line)
    sheet_title = " | ".join(title_parts) if title_parts else ""

    # Collect up to 5 preview data rows after the header row
    preview = []
    for row in all_rows[best_idx + 1: best_idx + 6]:
        filled = [v for v in row if v]
        if filled:
            preview.append(filled)

    return {
        "headers": header_row_values,
        "header_row": best_idx + 1,   # 1-based
        "sheet_title": sheet_title,
        "preview": preview,
    }


# Keep old name as thin wrapper for backwards compat
def headers_from_excel(file_bytes: bytes) -> list[str]:
    return parse_excel(file_bytes)["headers"]


# ---------------------------------------------------------------------------
# Form CRUD
# ---------------------------------------------------------------------------

def create_form(title: str, description: str, user_id: int, fields: list[dict]) -> Form:
    """Create a Form with its FormField rows and commit."""
    form = Form(title=title, description=description, created_by=user_id)
    db.session.add(form)
    db.session.flush()   # get form.id before inserting fields

    for idx, f in enumerate(fields):
        field = FormField(
            form_id=form.id,
            label=f.get("label", f"Field {idx + 1}"),
            field_type=f.get("field_type", "text"),
            placeholder=f.get("placeholder", ""),
            is_required=bool(f.get("is_required", False)),
            options=f.get("options", ""),
            order=idx,
        )
        db.session.add(field)

    db.session.commit()
    return form


def get_form(form_id: int) -> Form | None:
    return db.session.get(Form, form_id)


def list_forms(user_id: int) -> list[Form]:
    return Form.query.filter_by(created_by=user_id).order_by(Form.created_at.desc()).all()


def delete_form(form_id: int, user_id: int) -> bool:
    form = Form.query.filter_by(id=form_id, created_by=user_id).first()
    if not form:
        return False
    db.session.delete(form)
    db.session.commit()
    return True


# ---------------------------------------------------------------------------
# Response submission
# ---------------------------------------------------------------------------

def submit_response(form_id: int, user_id: int | None, answers: dict) -> FormResponse:
    """answers = {field_id: value, ...}"""
    response = FormResponse(form_id=form_id, submitted_by=user_id)
    db.session.add(response)
    db.session.flush()

    for field_id, value in answers.items():
        answer = FormFieldResponse(
            response_id=response.id,
            field_id=int(field_id),
            value=str(value) if value is not None else "",
        )
        db.session.add(answer)

    db.session.commit()
    return response
