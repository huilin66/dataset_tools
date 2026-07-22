#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_bib.py

检查 LaTeX 项目中的 .bib 文件与引用是否一致。

主要检查：
1. LaTeX 中引用但 .bib 中不存在的 key
2. .bib 中重复的 entry key
3. .bib 中存在但 LaTeX 未引用的条目
4. BibTeX 条目括号是否闭合
5. 常见条目类型的必填字段是否缺失
6. 重复 DOI
7. 疑似重复标题
8. 空字段
9. 未找到 bibliography 配置或 bib 文件

仅使用 Python 标准库，无需安装额外依赖。

用法：
    python check_bib.py main.tex
    python check_bib.py main.tex references.bib
    python check_bib.py ./latex_project
    python check_bib.py main.tex --strict
    python check_bib.py main.tex --json report.json

退出码：
    0: 未发现错误
    1: 发现错误
    2: 参数或文件读取错误
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


# 常见 BibTeX / BibLaTeX 条目的最低字段要求。
# 这里采用较宽松规则，避免把合法条目误判为错误。
REQUIRED_FIELDS = {
    "article": [
        {"author", "title", "journal", "year"},
        {"author", "title", "journaltitle", "year"},
        {"author", "title", "journal", "date"},
        {"author", "title", "journaltitle", "date"},
    ],
    "book": [
        {"author", "title", "publisher", "year"},
        {"editor", "title", "publisher", "year"},
        {"author", "title", "publisher", "date"},
        {"editor", "title", "publisher", "date"},
    ],
    "inproceedings": [
        {"author", "title", "booktitle", "year"},
        {"author", "title", "booktitle", "date"},
    ],
    "conference": [
        {"author", "title", "booktitle", "year"},
        {"author", "title", "booktitle", "date"},
    ],
    "incollection": [
        {"author", "title", "booktitle", "publisher", "year"},
        {"author", "title", "booktitle", "publisher", "date"},
    ],
    "inbook": [
        {"author", "title", "publisher", "year"},
        {"editor", "title", "publisher", "year"},
        {"author", "title", "publisher", "date"},
        {"editor", "title", "publisher", "date"},
    ],
    "phdthesis": [
        {"author", "title", "school", "year"},
        {"author", "title", "institution", "year"},
        {"author", "title", "school", "date"},
        {"author", "title", "institution", "date"},
    ],
    "mastersthesis": [
        {"author", "title", "school", "year"},
        {"author", "title", "institution", "year"},
        {"author", "title", "school", "date"},
        {"author", "title", "institution", "date"},
    ],
    "techreport": [
        {"author", "title", "institution", "year"},
        {"author", "title", "institution", "date"},
    ],
    "report": [
        {"author", "title", "institution", "year"},
        {"author", "title", "institution", "date"},
    ],
    "proceedings": [
        {"title", "year"},
        {"title", "date"},
    ],
    "misc": [
        {"title"},
    ],
    "online": [
        {"title", "url"},
        {"title", "doi"},
    ],
    "dataset": [
        {"title", "year"},
        {"title", "date"},
    ],
    "software": [
        {"title", "year"},
        {"title", "date"},
    ],
}


CITE_COMMAND_PATTERN = re.compile(
    r"""
    \\                                     # backslash
    (?:
        cite
        |citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar
        |parencite|textcite|autocite|smartcite|supercite
        |footcite|footcitetext|fullcite|nocite
        |Cite|Citep|Citet|Parencite|Textcite|Autocite
    )
    \*?
    (?:\s*\[[^\]]*\]){0,2}
    \s*\{([^{}]*)\}
    """,
    re.VERBOSE | re.MULTILINE,
)

BIBLIOGRAPHY_PATTERN = re.compile(
    r"""
    \\bibliography\s*\{([^{}]+)\}
    |
    \\addbibresource
    (?:\s*\[[^\]]*\])?
    \s*\{([^{}]+)\}
    """,
    re.VERBOSE | re.MULTILINE,
)

INPUT_PATTERN = re.compile(
    r"""
    \\(?:input|include|subfile)\s*\{([^{}]+)\}
    """,
    re.VERBOSE,
)


@dataclass
class Issue:
    severity: str
    category: str
    message: str
    file: str | None = None
    line: int | None = None
    key: str | None = None


@dataclass
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]
    file: Path
    line: int
    raw: str


def strip_latex_comments(text: str) -> str:
    """移除未转义的 LaTeX 注释。"""
    output = []
    for line in text.splitlines():
        escaped = False
        kept = []
        for i, ch in enumerate(line):
            if ch == "%" and not escaped:
                break
            kept.append(ch)

            if ch == "\\":
                escaped = not escaped
            else:
                escaped = False

        output.append("".join(kept))
    return "\n".join(output)


def normalize_path(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def read_text(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
    last_error = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"无法读取文件 {path}: {last_error}")


def resolve_tex_include(base_file: Path, value: str) -> Path:
    value = value.strip()
    path = Path(value)
    if not path.suffix:
        path = path.with_suffix(".tex")
    if not path.is_absolute():
        path = base_file.parent / path
    return normalize_path(path)


def collect_tex_files(entry: Path) -> tuple[list[Path], list[Issue]]:
    """递归收集 main.tex 使用的 input/include/subfile 文件。"""
    issues: list[Issue] = []
    visited: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        path = normalize_path(path)
        if path in visited:
            return
        visited.add(path)

        if not path.exists():
            issues.append(
                Issue(
                    severity="error",
                    category="missing_tex_file",
                    message=f"找不到 LaTeX 文件：{path}",
                    file=str(path),
                )
            )
            return

        try:
            text = strip_latex_comments(read_text(path))
        except Exception as exc:
            issues.append(
                Issue(
                    severity="error",
                    category="read_error",
                    message=str(exc),
                    file=str(path),
                )
            )
            return

        ordered.append(path)

        for match in INPUT_PATTERN.finditer(text):
            child = resolve_tex_include(path, match.group(1))
            visit(child)

    visit(entry)
    return ordered, issues


def find_main_tex(project_dir: Path) -> Path | None:
    """从目录中猜测主 tex 文件。"""
    candidates = sorted(project_dir.glob("*.tex"))
    if not candidates:
        return None

    preferred_names = ("main.tex", "paper.tex", "manuscript.tex", "article.tex")
    by_name = {p.name.lower(): p for p in candidates}
    for name in preferred_names:
        if name in by_name:
            return by_name[name]

    for path in candidates:
        try:
            text = strip_latex_comments(read_text(path))
        except Exception:
            continue
        if r"\documentclass" in text and r"\begin{document}" in text:
            return path

    return candidates[0]


def extract_citations(tex_files: Iterable[Path]) -> tuple[dict[str, list[tuple[Path, int]]], bool]:
    citations: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    cite_all = False

    for path in tex_files:
        text = strip_latex_comments(read_text(path))
        for match in CITE_COMMAND_PATTERN.finditer(text):
            keys_text = match.group(1)
            line = text.count("\n", 0, match.start()) + 1

            for key in keys_text.split(","):
                key = key.strip()
                if not key:
                    continue
                if key == "*":
                    cite_all = True
                    continue
                citations[key].append((path, line))

    return citations, cite_all


def discover_bib_files(tex_files: Iterable[Path]) -> tuple[list[Path], list[Issue]]:
    bib_files: list[Path] = []
    issues: list[Issue] = []
    seen: set[Path] = set()

    for tex_path in tex_files:
        text = strip_latex_comments(read_text(tex_path))

        for match in BIBLIOGRAPHY_PATTERN.finditer(text):
            value = match.group(1) or match.group(2)
            if not value:
                continue

            # \bibliography{a,b}; \addbibresource 通常只有单个文件，但兼容逗号。
            for item in value.split(","):
                item = item.strip()
                if not item:
                    continue

                bib_path = Path(item)
                if not bib_path.suffix:
                    bib_path = bib_path.with_suffix(".bib")
                if not bib_path.is_absolute():
                    bib_path = tex_path.parent / bib_path
                bib_path = normalize_path(bib_path)

                if bib_path in seen:
                    continue
                seen.add(bib_path)

                if not bib_path.exists():
                    issues.append(
                        Issue(
                            severity="error",
                            category="missing_bib_file",
                            message=f"找不到 bibliography 文件：{bib_path}",
                            file=str(tex_path),
                        )
                    )
                else:
                    bib_files.append(bib_path)

    return bib_files, issues


def find_matching_delimiter(text: str, start: int, opener: str, closer: str) -> int | None:
    depth = 0
    in_quote = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == '"':
            in_quote = not in_quote
            continue

        if in_quote:
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i

    return None


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    brace_depth = 0
    paren_depth = 0
    in_quote = False
    escaped = False

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == '"':
            in_quote = not in_quote
            continue

        if in_quote:
            continue

        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == delimiter and brace_depth == 0 and paren_depth == 0:
            parts.append(text[start:i])
            start = i + 1

    parts.append(text[start:])
    return parts


def unwrap_value(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    changed = True

    while changed and len(value) >= 2:
        changed = False
        if value[0] == "{" and value[-1] == "}":
            end = find_matching_delimiter(value, 0, "{", "}")
            if end == len(value) - 1:
                value = value[1:-1].strip()
                changed = True
        elif value[0] == '"' and value[-1] == '"':
            value = value[1:-1].strip()
            changed = True

    return value


def parse_fields(body: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    malformed: list[str] = []

    parts = split_top_level(body)
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            malformed.append(part)
            continue

        name, value = part.split("=", 1)
        name = name.strip().lower()
        value = value.strip()

        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:\-]*", name):
            malformed.append(part)
            continue

        fields[name] = unwrap_value(value)

    return fields, malformed


def parse_bib_file(path: Path) -> tuple[list[BibEntry], list[Issue]]:
    entries: list[BibEntry] = []
    issues: list[Issue] = []

    text = read_text(path)
    text_no_comments = strip_latex_comments(text)

    i = 0
    while i < len(text_no_comments):
        at = text_no_comments.find("@", i)
        if at < 0:
            break

        type_match = re.match(r"@([A-Za-z]+)\s*([\{\(])", text_no_comments[at:])
        if not type_match:
            i = at + 1
            continue

        entry_type = type_match.group(1).lower()
        opener = type_match.group(2)
        closer = "}" if opener == "{" else ")"
        open_pos = at + type_match.end() - 1
        close_pos = find_matching_delimiter(text_no_comments, open_pos, opener, closer)
        line = text_no_comments.count("\n", 0, at) + 1

        if close_pos is None:
            issues.append(
                Issue(
                    severity="error",
                    category="unclosed_entry",
                    message=f"条目 @{entry_type} 从第 {line} 行开始，但未找到闭合符号 {closer}",
                    file=str(path),
                    line=line,
                )
            )
            break

        raw = text_no_comments[at : close_pos + 1]
        content = text_no_comments[open_pos + 1 : close_pos].strip()

        # @string、@preamble、@comment 不是普通文献条目。
        if entry_type in {"string", "preamble", "comment"}:
            i = close_pos + 1
            continue

        first_parts = split_top_level(content, delimiter=",")
        if not first_parts:
            issues.append(
                Issue(
                    severity="error",
                    category="missing_key",
                    message=f"第 {line} 行附近的 @{entry_type} 条目缺少 key",
                    file=str(path),
                    line=line,
                )
            )
            i = close_pos + 1
            continue

        key = first_parts[0].strip()
        if not key:
            issues.append(
                Issue(
                    severity="error",
                    category="missing_key",
                    message=f"第 {line} 行附近的 @{entry_type} 条目 key 为空",
                    file=str(path),
                    line=line,
                )
            )
            i = close_pos + 1
            continue

        comma_pos = content.find(",")
        fields_text = content[comma_pos + 1 :] if comma_pos >= 0 else ""
        fields, malformed = parse_fields(fields_text)

        entries.append(
            BibEntry(
                entry_type=entry_type,
                key=key,
                fields=fields,
                file=path,
                line=line,
                raw=raw,
            )
        )

        for item in malformed:
            preview = re.sub(r"\s+", " ", item).strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            issues.append(
                Issue(
                    severity="warning",
                    category="malformed_field",
                    message=f"无法解析字段：{preview}",
                    file=str(path),
                    line=line,
                    key=key,
                )
            )

        i = close_pos + 1

    return entries, issues


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.strip()


def normalize_title(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", value)
    value = re.sub(r"[{}\"']", "", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def validate_entries(entries: list[BibEntry], strict: bool) -> list[Issue]:
    issues: list[Issue] = []

    key_map: dict[str, list[BibEntry]] = defaultdict(list)
    doi_map: dict[str, list[BibEntry]] = defaultdict(list)
    title_map: dict[str, list[BibEntry]] = defaultdict(list)

    for entry in entries:
        key_map[entry.key].append(entry)

        if not re.fullmatch(r"[A-Za-z0-9_:\-./+]+", entry.key):
            issues.append(
                Issue(
                    severity="warning",
                    category="suspicious_key",
                    message="引用 key 含空格或特殊字符，可能导致 BibTeX/Biber 解析问题",
                    file=str(entry.file),
                    line=entry.line,
                    key=entry.key,
                )
            )

        alternatives = REQUIRED_FIELDS.get(entry.entry_type)
        if alternatives and not any(required <= set(entry.fields) for required in alternatives):
            readable = " 或 ".join(
                "{" + ", ".join(sorted(fields)) + "}" for fields in alternatives
            )
            severity = "error" if strict else "warning"
            issues.append(
                Issue(
                    severity=severity,
                    category="missing_required_fields",
                    message=f"@{entry.entry_type} 缺少一组必要字段，期望满足：{readable}",
                    file=str(entry.file),
                    line=entry.line,
                    key=entry.key,
                )
            )

        for field, value in entry.fields.items():
            if not value.strip():
                issues.append(
                    Issue(
                        severity="warning",
                        category="empty_field",
                        message=f"字段 {field} 为空",
                        file=str(entry.file),
                        line=entry.line,
                        key=entry.key,
                    )
                )

        if "doi" in entry.fields:
            doi = normalize_doi(entry.fields["doi"])
            if doi:
                doi_map[doi].append(entry)

        if "title" in entry.fields:
            title = normalize_title(entry.fields["title"])
            if len(title) >= 12:
                title_map[title].append(entry)

        year = entry.fields.get("year", "").strip()
        if year and not re.fullmatch(r"\d{4}[a-z]?", year, flags=re.IGNORECASE):
            issues.append(
                Issue(
                    severity="warning",
                    category="invalid_year",
                    message=f"year 字段格式可疑：{year}",
                    file=str(entry.file),
                    line=entry.line,
                    key=entry.key,
                )
            )

        doi = entry.fields.get("doi")
        if doi:
            normalized = normalize_doi(doi)
            if normalized and not normalized.startswith("10."):
                issues.append(
                    Issue(
                        severity="warning",
                        category="invalid_doi",
                        message=f"DOI 格式可疑：{doi}",
                        file=str(entry.file),
                        line=entry.line,
                        key=entry.key,
                    )
                )

    for key, duplicated in key_map.items():
        if len(duplicated) > 1:
            locations = ", ".join(f"{item.file}:{item.line}" for item in duplicated)
            issues.append(
                Issue(
                    severity="error",
                    category="duplicate_key",
                    message=f"重复的 BibTeX key，出现于：{locations}",
                    key=key,
                )
            )

    for doi, duplicated in doi_map.items():
        if len(duplicated) > 1:
            keys = ", ".join(entry.key for entry in duplicated)
            issues.append(
                Issue(
                    severity="warning",
                    category="duplicate_doi",
                    message=f"多个条目使用相同 DOI {doi}：{keys}",
                )
            )

    for title, duplicated in title_map.items():
        unique_keys = sorted({entry.key for entry in duplicated})
        if len(unique_keys) > 1:
            issues.append(
                Issue(
                    severity="warning",
                    category="duplicate_title",
                    message=f"疑似重复标题，涉及 key：{', '.join(unique_keys)}",
                )
            )

    return issues


def validate_citations(
    citations: dict[str, list[tuple[Path, int]]],
    entries: list[BibEntry],
    cite_all: bool,
    strict: bool,
) -> list[Issue]:
    issues: list[Issue] = []
    bib_keys = {entry.key for entry in entries}
    cited_keys = set(citations)

    for key in sorted(cited_keys - bib_keys):
        locations = citations[key]
        first_file, first_line = locations[0]
        issues.append(
            Issue(
                severity="error",
                category="missing_citation_key",
                message=f"LaTeX 引用了不存在的文献 key，共出现 {len(locations)} 次",
                file=str(first_file),
                line=first_line,
                key=key,
            )
        )

    if not cite_all:
        for key in sorted(bib_keys - cited_keys):
            issues.append(
                Issue(
                    severity="warning" if not strict else "error",
                    category="uncited_entry",
                    message=".bib 中存在但正文未引用",
                    key=key,
                )
            )

    return issues


def format_issue(issue: Issue) -> str:
    severity_label = {
        "error": "ERROR",
        "warning": "WARN ",
        "info": "INFO ",
    }.get(issue.severity, issue.severity.upper())

    location = ""
    if issue.file:
        location = issue.file
        if issue.line is not None:
            location += f":{issue.line}"
        location = f" [{location}]"

    key = f" <{issue.key}>" if issue.key else ""
    return f"{severity_label} [{issue.category}]{key}{location} {issue.message}"


def build_report(
    main_tex: Path,
    tex_files: list[Path],
    bib_files: list[Path],
    entries: list[BibEntry],
    citations: dict[str, list[tuple[Path, int]]],
    cite_all: bool,
    issues: list[Issue],
) -> dict:
    counter = Counter(issue.severity for issue in issues)
    category_counter = Counter(issue.category for issue in issues)

    return {
        "main_tex": str(main_tex),
        "tex_files": [str(path) for path in tex_files],
        "bib_files": [str(path) for path in bib_files],
        "statistics": {
            "tex_file_count": len(tex_files),
            "bib_file_count": len(bib_files),
            "bib_entry_count": len(entries),
            "unique_bib_key_count": len({entry.key for entry in entries}),
            "cited_key_count": len(citations),
            "nocite_all": cite_all,
            "error_count": counter["error"],
            "warning_count": counter["warning"],
            "issue_count_by_category": dict(sorted(category_counter.items())),
        },
        "issues": [asdict(issue) for issue in issues],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 LaTeX 项目中的引用与 BibTeX/BibLaTeX 文件。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        type=Path,
        help="主 .tex 文件或 LaTeX 项目目录",
    )
    parser.add_argument(
        "bib_files",
        nargs="*",
        type=Path,
        help="可选：手动指定一个或多个 .bib 文件",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把未引用条目和必填字段缺失视为错误",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="将检查结果写入 JSON 文件",
    )
    parser.add_argument(
        "--no-unused",
        action="store_true",
        help="不检查未引用的 bibliography 条目",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = normalize_path(args.target)

    if not target.exists():
        print(f"ERROR: 路径不存在：{target}", file=sys.stderr)
        return 2

    if target.is_dir():
        main_tex = find_main_tex(target)
        if main_tex is None:
            print(f"ERROR: 目录中未找到 .tex 文件：{target}", file=sys.stderr)
            return 2
        main_tex = normalize_path(main_tex)
    else:
        main_tex = target

    if main_tex.suffix.lower() != ".tex":
        print(f"ERROR: 主文件不是 .tex：{main_tex}", file=sys.stderr)
        return 2

    tex_files, issues = collect_tex_files(main_tex)

    try:
        citations, cite_all = extract_citations(tex_files)
    except Exception as exc:
        print(f"ERROR: 解析 LaTeX 引用失败：{exc}", file=sys.stderr)
        return 2

    if args.bib_files:
        bib_files = []
        for bib in args.bib_files:
            bib = normalize_path(bib if bib.is_absolute() else Path.cwd() / bib)
            if not bib.exists():
                issues.append(
                    Issue(
                        severity="error",
                        category="missing_bib_file",
                        message=f"手动指定的 .bib 文件不存在：{bib}",
                        file=str(bib),
                    )
                )
            else:
                bib_files.append(bib)
    else:
        discovered, bib_issues = discover_bib_files(tex_files)
        bib_files = discovered
        issues.extend(bib_issues)

    if not bib_files:
        issues.append(
            Issue(
                severity="error",
                category="no_bib_file",
                message="未发现可用的 .bib 文件。请检查 \\bibliography 或 \\addbibresource，"
                        "或在命令行手动指定 .bib 文件。",
                file=str(main_tex),
            )
        )

    entries: list[BibEntry] = []
    for bib_file in bib_files:
        try:
            parsed_entries, parse_issues = parse_bib_file(bib_file)
            entries.extend(parsed_entries)
            issues.extend(parse_issues)
        except Exception as exc:
            issues.append(
                Issue(
                    severity="error",
                    category="bib_read_error",
                    message=f"读取或解析 .bib 文件失败：{exc}",
                    file=str(bib_file),
                )
            )

    issues.extend(validate_entries(entries, strict=args.strict))

    citation_issues = validate_citations(
        citations=citations,
        entries=entries,
        cite_all=cite_all,
        strict=args.strict,
    )
    if args.no_unused:
        citation_issues = [
            issue for issue in citation_issues
            if issue.category != "uncited_entry"
        ]
    issues.extend(citation_issues)

    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(
        key=lambda issue: (
            severity_order.get(issue.severity, 99),
            issue.category,
            issue.file or "",
            issue.line or 0,
            issue.key or "",
        )
    )

    report = build_report(
        main_tex=main_tex,
        tex_files=tex_files,
        bib_files=bib_files,
        entries=entries,
        citations=citations,
        cite_all=cite_all,
        issues=issues,
    )

    print("=" * 72)
    print("LaTeX / BibTeX 检查结果")
    print("=" * 72)
    print(f"主文件       : {main_tex}")
    print(f"扫描 tex 数量: {len(tex_files)}")
    print(f"扫描 bib 数量: {len(bib_files)}")
    print(f"Bib 条目数量 : {len(entries)}")
    print(f"正文引用 key : {len(citations)}")
    print(f"错误数量     : {report['statistics']['error_count']}")
    print(f"警告数量     : {report['statistics']['warning_count']}")
    print("-" * 72)

    if issues:
        for issue in issues:
            print(format_issue(issue))
    else:
        print("OK: 未发现明显问题。")

    if args.json_path:
        json_path = normalize_path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("-" * 72)
        print(f"JSON 报告已写入：{json_path}")

    print("=" * 72)

    return 1 if report["statistics"]["error_count"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
