import json
from pathlib import Path
from xml.etree.ElementTree import ParseError as XMLParseError
from zipfile import BadZipFile, ZipFile

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException


MAX_ROWS = 500_000
MAX_COLUMNS = 200
MAX_XLSX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_XLSX_ENTRIES = 5_000


def read_dataset(path: Path, sheet: str | None = None) -> tuple[pd.DataFrame, list[str], str | None]:
    extension = path.suffix.lower()

    try:
        if extension == ".csv":
            if sheet is not None:
                raise ValueError("CSV không hỗ trợ chọn sheet.")
            with path.open("rb") as csv_file:
                prefix = csv_file.read(4)
            encodings = (
                ("utf-16", "utf-8-sig", "cp1258", "cp1252")
                if prefix.startswith((b"\xff\xfe", b"\xfe\xff"))
                else ("utf-8-sig", "cp1258", "cp1252")
            )
            last_error = None
            for encoding in encodings:
                try:
                    dataframe = pd.read_csv(path, encoding=encoding, nrows=MAX_ROWS + 1)
                    break
                except UnicodeDecodeError as exc:
                    last_error = exc
            else:
                raise ValueError("Không đọc được mã hóa CSV; hãy lưu file dưới dạng UTF-8.") from last_error
            sheets: list[str] = []
            selected_sheet = None
        elif extension == ".xlsx":
            with ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_XLSX_ENTRIES:
                    raise ValueError("Excel có quá nhiều thành phần.")
                if sum(entry.file_size for entry in entries) > MAX_XLSX_UNCOMPRESSED_BYTES:
                    raise ValueError("Dung lượng Excel sau giải nén vượt giới hạn.")
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                    raise ValueError("File không phải workbook XLSX hợp lệ.")

            with pd.ExcelFile(path, engine="openpyxl") as workbook:
                sheets = workbook.sheet_names
                if not sheets:
                    raise ValueError("Workbook không có sheet.")
                selected_sheet = sheet or sheets[0]
                if selected_sheet not in sheets:
                    raise ValueError(f"Không tìm thấy sheet '{selected_sheet}'.")
                dataframe = pd.read_excel(
                    workbook,
                    sheet_name=selected_sheet,
                    nrows=MAX_ROWS + 1,
                )
        else:
            raise ValueError("Chỉ hỗ trợ file CSV và XLSX.")
    except (
        BadZipFile,
        InvalidFileException,
        XMLParseError,
        KeyError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError("Không thể đọc file. Hãy kiểm tra định dạng và nội dung file.") from exc

    if len(dataframe) > MAX_ROWS:
        raise ValueError(f"File vượt giới hạn {MAX_ROWS:,} dòng.")
    if len(dataframe.columns) == 0:
        raise ValueError("File không có cột dữ liệu.")
    if len(dataframe.columns) > MAX_COLUMNS:
        raise ValueError(f"File vượt giới hạn {MAX_COLUMNS} cột.")
    column_names = [str(column) for column in dataframe.columns]
    if len(column_names) != len(set(column_names)):
        raise ValueError("Tên cột bị trùng sau khi chuyển thành văn bản.")
    dataframe.columns = column_names
    return dataframe, sheets, selected_sheet


def profile_dataset(path: Path, sheet: str | None = None) -> dict:
    dataframe, sheets, selected_sheet = read_dataset(path, sheet)
    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": [
            {
                "name": str(column),
                "dtype": str(dataframe[column].dtype),
                "missing_count": int(dataframe[column].isna().sum()),
            }
            for column in dataframe.columns
        ],
        "sheets": sheets,
        "active_sheet": selected_sheet,
    }


def preview_dataset(path: Path, sheet: str | None = None, limit: int = 100) -> dict:
    dataframe, sheets, selected_sheet = read_dataset(path, sheet)
    preview = dataframe.head(limit)
    return {
        "columns": [str(column) for column in preview.columns],
        "rows": json.loads(preview.to_json(orient="records", date_format="iso", force_ascii=False)),
        "total_rows": int(len(dataframe)),
        "sheets": sheets,
        "active_sheet": selected_sheet,
    }
