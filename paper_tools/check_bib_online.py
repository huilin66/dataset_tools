#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_bib_online.py

同时检查：
1. LaTeX 引用与 .bib 的本地一致性；
2. BibTeX 条目的结构与字段完整性；
3. 论文是否能在 Crossref / arXiv 在线核验；
4. 本地元数据与在线元数据是否一致。

在线匹配的两个硬条件：
    - title 相似度 >= --title-threshold
    - journal / publication venue 相似度 >= --journal-threshold

只有标题和期刊（或会议论文集名称）同时通过阈值，才会将在线结果判定为匹配。
DOI 只用于优先定位候选记录，不能绕过标题和期刊校验。

依赖：
    pip install requests

常用命令：
    python check_bib_online.py main.tex --email your_email@example.com
    python check_bib_online.py references.bib --email your_email@example.com
    python check_bib_online.py ./latex_project --all
    python check_bib_online.py main.tex --json bib_report.json --csv bib_report.csv
    python check_bib_online.py main.tex --title-threshold 0.92 --journal-threshold 0.82

期刊简称无法自动识别时，可提供别名：
    python check_bib_online.py main.tex \
        --journal-alias "IEEE J-STARS=IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing"

说明：
- Crossref 是正式出版元数据的主要来源。
- arXiv 仅作为兜底来源，而且必须存在 journal_ref 并通过期刊匹配；
  没有 journal_ref 的 arXiv 记录不会被当作“期刊论文已核验”。
- “在线未核验”不等于论文一定不存在，可能是数据库未收录或元数据不足。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    print(
        "缺少 requests。请先运行：pip install requests",
        file=sys.stderr,
    )
    raise SystemExit(2)


CROSSREF_API = "https://api.crossref.org"
ARXIV_API = "https://export.arxiv.org/api/query"

# 宽松的最低字段要求。期刊文章重点要求 title、journal、author、year/date。
REQUIRED_FIELD_ALTERNATIVES: dict[str, list[set[str]]] = {
    "article": [
        {"author", "title", "journal", "year"},
        {"author", "title", "journaltitle", "year"},
        {"author", "title", "journal", "date"},
        {"author", "title", "journaltitle", "date"},
    ],
    "inproceedings": [
        {"author", "title", "booktitle", "year"},
        {"author", "title", "booktitle", "date"},
    ],
    "conference": [
        {"author", "title", "booktitle", "year"},
        {"author", "title", "booktitle", "date"},
    ],
    "book": [
        {"author", "title", "publisher", "year"},
        {"editor", "title", "publisher", "year"},
        {"author", "title", "publisher", "date"},
        {"editor", "title", "publisher", "date"},
    ],
    "incollection": [
        {"author", "title", "booktitle", "publisher", "year"},
        {"author", "title", "booktitle", "publisher", "date"},
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
    "misc": [{"title"}],
    "online": [{"title", "url"}, {"title", "doi"}],
}

CITE_COMMAND_PATTERN = re.compile(
    r"""
    \\
    (?:
        cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar
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
    r"""\\(?:input|include|subfile)\s*\{([^{}]+)\}""",
    re.VERBOSE,
)

VENUE_FIELDS = ("journal", "journaltitle", "booktitle")
YEAR_FIELDS = ("year", "date")

STOP_WORDS = {
    "a", "an", "the", "of", "and", "or", "on", "in", "for", "to", "from",
    "with", "by", "at", "de", "la", "le", "der", "die", "das",
}

VENUE_ABBREVIATIONS = {
    "j": "journal",
    "jrnl": "journal",
    "trans": "transactions",
    "t": "transactions",
    "proc": "proceedings",
    "procs": "proceedings",
    "conf": "conference",
    "int": "international",
    "intl": "international",
    "natl": "national",
    "appl": "applied",
    "app": "applied",
    "sci": "science",
    "sc": "science",
    "technol": "technology",
    "tech": "technology",
    "eng": "engineering",
    "comput": "computer",
    "comp": "computer",
    "commun": "communications",
    "comm": "communications",
    "inform": "information",
    "inf": "information",
    "imag": "image",
    "imaging": "image",
    "observ": "observations",
    "obs": "observations",
    "sens": "sensing",
    "geosci": "geoscience",
    "geosc": "geoscience",
    "rem": "remote",
    "sel": "selected",
    "lett": "letters",
    "l": "letters",
    "mag": "magazine",
    "rev": "review",
    "res": "research",
    "syst": "systems",
    "sys": "systems",
    "robot": "robotics",
    "autom": "automation",
    "pattern": "pattern",
    "anal": "analysis",
    "mach": "machine",
    "learn": "learning",
    "electron": "electronics",
    "electr": "electrical",
}


@dataclass
class Issue:
    severity: str
    category: str
    message: str
    key: str | None = None
    file: str | None = None
    line: int | None = None


@dataclass
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str]
    file: Path
    line: int
    raw: str

    @property
    def title(self) -> str:
        return self.fields.get("title", "").strip()

    @property
    def venue(self) -> str:
        for name in VENUE_FIELDS:
            value = self.fields.get(name, "").strip()
            if value:
                return value
        return ""

    @property
    def doi(self) -> str:
        return normalize_doi(self.fields.get("doi", ""))

    @property
    def year(self) -> str:
        for name in YEAR_FIELDS:
            value = self.fields.get(name, "").strip()
            if value:
                match = re.search(r"\b(?:19|20)\d{2}\b", value)
                return match.group(0) if match else value
        return ""


@dataclass
class RemoteCandidate:
    source: str
    title: str
    venue: str
    doi: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publisher: str = ""
    item_type: str = ""
    url: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    candidate: RemoteCandidate
    title_score: float
    journal_score: float
    combined_score: float


@dataclass
class OnlineCheckResult:
    key: str
    status: str
    source: str = ""
    method: str = ""
    title_score: float = 0.0
    journal_score: float = 0.0
    combined_score: float = 0.0
    local_title: str = ""
    remote_title: str = ""
    local_journal: str = ""
    remote_journal: str = ""
    local_doi: str = ""
    remote_doi: str = ""
    local_year: str = ""
    remote_year: str = ""
    missing_local_fields: list[str] = field(default_factory=list)
    mismatched_fields: dict[str, dict[str, str]] = field(default_factory=dict)
    online_missing_fields: list[str] = field(default_factory=list)
    candidate_url: str = ""
    message: str = ""
    top_candidates: list[dict[str, Any]] = field(default_factory=list)


class JsonCache:
    def __init__(self, path: Path | None, enabled: bool = True):
        self.path = path
        self.enabled = enabled and path is not None
        self.data: dict[str, Any] = {}

        if self.enabled and self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    @staticmethod
    def make_key(namespace: str, payload: Any) -> str:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.data[key] = value
        self.save()

    def save(self) -> None:
        if not self.enabled or not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.path)


def normalize_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def read_text(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"无法读取 {path}: {last_error}")


def strip_latex_comments(text: str) -> str:
    output: list[str] = []

    for line in text.splitlines():
        cut = len(line)
        for index, char in enumerate(line):
            if char != "%":
                continue

            slash_count = 0
            pos = index - 1
            while pos >= 0 and line[pos] == "\\":
                slash_count += 1
                pos -= 1

            if slash_count % 2 == 0:
                cut = index
                break

        output.append(line[:cut])

    return "\n".join(output)


def clean_latex_text(value: str) -> str:
    value = value.replace(r"\&", " and ")
    value = value.replace("~", " ")

    # 尽量保留重音命令中的字母。
    value = re.sub(
        r"""\\["'`^~=.uvHckbdtr]\s*\{?\s*([A-Za-z])\s*\}?""",
        r"\1",
        value,
    )

    # 去掉常见格式命令，但保留命令参数。
    for _ in range(4):
        updated = re.sub(
            r"""\\(?:textit|textbf|emph|mathrm|mathbf|mathit|mathsf|textrm|textsc)\s*\{([^{}]*)\}""",
            r"\1",
            value,
        )
        if updated == value:
            break
        value = updated

    value = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", value)
    value = re.sub(r"\\.", " ", value)
    value = value.replace("{", " ").replace("}", " ")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value).strip()


def normalize_general_text(value: str) -> str:
    value = clean_latex_text(value).casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_doi(value: str) -> str:
    value = clean_latex_text(value).strip().casefold()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi\s*:\s*", "", value)
    return value.strip().rstrip(".,")


def normalize_pages(value: str) -> str:
    value = clean_latex_text(value).casefold()
    value = value.replace("--", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", "", value)


def title_similarity(left: str, right: str) -> float:
    a = normalize_general_text(left)
    b = normalize_general_text(right)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    seq = SequenceMatcher(None, a, b).ratio()

    tokens_a = a.split()
    tokens_b = b.split()
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = len(set_a & set_b)
    precision = intersection / len(set_b) if set_b else 0.0
    recall = intersection / len(set_a) if set_a else 0.0
    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    containment = min(len(set_a), len(set_b))
    containment_score = intersection / containment if containment else 0.0

    # 标题只允许有限的子标题差异，不让短标题轻易产生高分。
    if min(len(tokens_a), len(tokens_b)) < 4:
        containment_score *= 0.85

    return min(1.0, max(seq, token_f1, containment_score * 0.98))


def canonical_venue_token(token: str) -> str:
    token = token.casefold().strip(".")
    return VENUE_ABBREVIATIONS.get(token, token)


def venue_tokens(value: str) -> list[str]:
    normalized = normalize_general_text(value)
    tokens = []
    for token in normalized.split():
        if token in STOP_WORDS:
            continue
        token = canonical_venue_token(token)
        if token and token not in STOP_WORDS:
            tokens.append(token)
    return tokens


def token_compatible(left: str, right: str) -> bool:
    if left == right:
        return True

    left = canonical_venue_token(left)
    right = canonical_venue_token(right)

    if left == right:
        return True

    minimum = min(len(left), len(right))
    if minimum >= 3 and (left.startswith(right) or right.startswith(left)):
        return True

    return SequenceMatcher(None, left, right).ratio() >= 0.86


def greedy_token_matches(left: list[str], right: list[str]) -> int:
    used: set[int] = set()
    matches = 0

    # 先匹配更长、更有区分度的 token。
    for token_left in sorted(left, key=len, reverse=True):
        best_index = None
        best_score = -1.0

        for index, token_right in enumerate(right):
            if index in used or not token_compatible(token_left, token_right):
                continue

            score = SequenceMatcher(None, token_left, token_right).ratio()
            if token_left == token_right:
                score = 1.0

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None:
            used.add(best_index)
            matches += 1

    return matches


def journal_similarity(
    left: str,
    right: str,
    aliases: dict[str, str] | None = None,
) -> float:
    aliases = aliases or {}

    normalized_left = normalize_general_text(left)
    normalized_right = normalize_general_text(right)

    if normalized_left in aliases:
        left = aliases[normalized_left]
        normalized_left = normalize_general_text(left)
    if normalized_right in aliases:
        right = aliases[normalized_right]
        normalized_right = normalize_general_text(right)

    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    seq = SequenceMatcher(None, normalized_left, normalized_right).ratio()

    tokens_left = venue_tokens(left)
    tokens_right = venue_tokens(right)
    if not tokens_left or not tokens_right:
        return seq

    matches = greedy_token_matches(tokens_left, tokens_right)
    precision = matches / len(tokens_right)
    recall = matches / len(tokens_left)
    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    # journal_ref 可能包含卷期、页码等附加信息，因此保留本地期刊 token 覆盖率。
    local_coverage = matches / len(tokens_left)
    coverage_score = local_coverage * 0.97

    compact_left = "".join(tokens_left)
    compact_right = "".join(tokens_right)
    compact_seq = SequenceMatcher(None, compact_left, compact_right).ratio()

    return min(1.0, max(seq, token_f1, coverage_score, compact_seq))


def author_surnames_from_bib(value: str) -> set[str]:
    surnames: set[str] = set()

    for author in re.split(r"\s+and\s+", clean_latex_text(value), flags=re.I):
        author = author.strip()
        if not author:
            continue

        if "," in author:
            surname = author.split(",", 1)[0]
        else:
            parts = author.split()
            surname = parts[-1] if parts else ""

        surname = normalize_general_text(surname)
        if surname:
            surnames.add(surname)

    return surnames


def author_similarity(local_author: str, remote_authors: list[str]) -> float:
    local = author_surnames_from_bib(local_author)
    remote = {
        normalize_general_text(name.split(",")[0] if "," in name else name.split()[-1])
        for name in remote_authors
        if name.strip()
    }
    remote.discard("")

    if not local or not remote:
        return 0.0

    common = len(local & remote)
    precision = common / len(remote)
    recall = common / len(local)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def find_matching_delimiter(
    text: str,
    start: int,
    opener: str,
    closer: str,
) -> int | None:
    depth = 0
    in_quote = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue

        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index

    return None


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    brace_depth = 0
    paren_depth = 0
    in_quote = False
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue

        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == delimiter and brace_depth == 0 and paren_depth == 0:
            parts.append(text[start:index])
            start = index + 1

    parts.append(text[start:])
    return parts


def unwrap_value(value: str) -> str:
    value = value.strip().rstrip(",").strip()

    while len(value) >= 2:
        if value[0] == "{" and value[-1] == "}":
            end = find_matching_delimiter(value, 0, "{", "}")
            if end == len(value) - 1:
                value = value[1:-1].strip()
                continue
        if value[0] == '"' and value[-1] == '"':
            value = value[1:-1].strip()
            continue
        break

    return value


def parse_fields(body: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    malformed: list[str] = []

    for part in split_top_level(body):
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            malformed.append(part)
            continue

        name, value = part.split("=", 1)
        name = name.strip().casefold()
        value = value.strip()

        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:\-]*", name):
            malformed.append(part)
            continue

        fields[name] = unwrap_value(value)

    return fields, malformed


def parse_bib_file(path: Path) -> tuple[list[BibEntry], list[Issue]]:
    text = strip_latex_comments(read_text(path))
    entries: list[BibEntry] = []
    issues: list[Issue] = []

    position = 0
    while position < len(text):
        at = text.find("@", position)
        if at < 0:
            break

        header = re.match(r"@([A-Za-z]+)\s*([\{\(])", text[at:])
        if not header:
            position = at + 1
            continue

        entry_type = header.group(1).casefold()
        opener = header.group(2)
        closer = "}" if opener == "{" else ")"
        open_position = at + header.end() - 1
        close_position = find_matching_delimiter(
            text,
            open_position,
            opener,
            closer,
        )
        line = text.count("\n", 0, at) + 1

        if close_position is None:
            issues.append(
                Issue(
                    severity="error",
                    category="unclosed_entry",
                    message=f"@{entry_type} 条目缺少闭合符号 {closer}",
                    file=str(path),
                    line=line,
                )
            )
            break

        raw = text[at:close_position + 1]
        content = text[open_position + 1:close_position].strip()

        if entry_type in {"comment", "preamble", "string"}:
            position = close_position + 1
            continue

        comma_parts = split_top_level(content)
        key = comma_parts[0].strip() if comma_parts else ""

        if not key:
            issues.append(
                Issue(
                    severity="error",
                    category="missing_key",
                    message=f"@{entry_type} 条目的 key 为空",
                    file=str(path),
                    line=line,
                )
            )
            position = close_position + 1
            continue

        comma_position = content.find(",")
        fields_body = content[comma_position + 1:] if comma_position >= 0 else ""
        fields, malformed = parse_fields(fields_body)

        entry = BibEntry(
            entry_type=entry_type,
            key=key,
            fields=fields,
            file=path,
            line=line,
            raw=raw,
        )
        entries.append(entry)

        for malformed_field in malformed:
            preview = re.sub(r"\s+", " ", malformed_field).strip()
            if len(preview) > 120:
                preview = preview[:117] + "..."
            issues.append(
                Issue(
                    severity="warning",
                    category="malformed_field",
                    message=f"无法解析字段：{preview}",
                    key=key,
                    file=str(path),
                    line=line,
                )
            )

        position = close_position + 1

    return entries, issues


def resolve_tex_include(base_file: Path, value: str) -> Path:
    include_path = Path(value.strip())
    if not include_path.suffix:
        include_path = include_path.with_suffix(".tex")
    if not include_path.is_absolute():
        include_path = base_file.parent / include_path
    return normalize_path(include_path)


def collect_tex_files(main_tex: Path) -> tuple[list[Path], list[Issue]]:
    visited: set[Path] = set()
    ordered: list[Path] = []
    issues: list[Issue] = []

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
                    message="找不到通过 input/include 引入的 LaTeX 文件",
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
                    category="tex_read_error",
                    message=str(exc),
                    file=str(path),
                )
            )
            return

        ordered.append(path)
        for match in INPUT_PATTERN.finditer(text):
            visit(resolve_tex_include(path, match.group(1)))

    visit(main_tex)
    return ordered, issues


def find_main_tex(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("*.tex"))
    if not candidates:
        return None

    preferred = ("main.tex", "paper.tex", "manuscript.tex", "article.tex")
    name_map = {path.name.casefold(): path for path in candidates}
    for name in preferred:
        if name in name_map:
            return normalize_path(name_map[name])

    for path in candidates:
        try:
            text = strip_latex_comments(read_text(path))
        except Exception:
            continue
        if r"\documentclass" in text and r"\begin{document}" in text:
            return normalize_path(path)

    return normalize_path(candidates[0])


def extract_citations(
    tex_files: Iterable[Path],
) -> tuple[dict[str, list[tuple[Path, int]]], bool]:
    citations: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    cite_all = False

    for path in tex_files:
        text = strip_latex_comments(read_text(path))
        for match in CITE_COMMAND_PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            for key in match.group(1).split(","):
                key = key.strip()
                if not key:
                    continue
                if key == "*":
                    cite_all = True
                else:
                    citations[key].append((path, line))

    return citations, cite_all


def discover_bib_files(
    tex_files: Iterable[Path],
) -> tuple[list[Path], list[Issue]]:
    bib_files: list[Path] = []
    seen: set[Path] = set()
    issues: list[Issue] = []

    for tex_path in tex_files:
        text = strip_latex_comments(read_text(tex_path))
        for match in BIBLIOGRAPHY_PATTERN.finditer(text):
            value = match.group(1) or match.group(2) or ""
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

                if bib_path.exists():
                    bib_files.append(bib_path)
                else:
                    issues.append(
                        Issue(
                            severity="error",
                            category="missing_bib_file",
                            message=f"找不到 bibliography 文件：{bib_path}",
                            file=str(tex_path),
                        )
                    )

    return bib_files, issues


def validate_local_entries(entries: list[BibEntry]) -> list[Issue]:
    issues: list[Issue] = []
    keys: dict[str, list[BibEntry]] = defaultdict(list)
    dois: dict[str, list[BibEntry]] = defaultdict(list)
    titles: dict[str, list[BibEntry]] = defaultdict(list)

    for entry in entries:
        keys[entry.key].append(entry)

        alternatives = REQUIRED_FIELD_ALTERNATIVES.get(entry.entry_type)
        if alternatives and not any(
            required.issubset(entry.fields.keys())
            for required in alternatives
        ):
            required_text = " 或 ".join(
                "{" + ", ".join(sorted(fields)) + "}"
                for fields in alternatives
            )
            issues.append(
                Issue(
                    severity="error",
                    category="missing_required_fields",
                    message=f"缺少最低字段组合：{required_text}",
                    key=entry.key,
                    file=str(entry.file),
                    line=entry.line,
                )
            )

        for field_name, field_value in entry.fields.items():
            if not field_value.strip():
                issues.append(
                    Issue(
                        severity="warning",
                        category="empty_field",
                        message=f"字段 {field_name} 为空",
                        key=entry.key,
                        file=str(entry.file),
                        line=entry.line,
                    )
                )

        if entry.doi:
            dois[entry.doi].append(entry)

        normalized_title = normalize_general_text(entry.title)
        if normalized_title:
            titles[normalized_title].append(entry)

    for key, duplicated in keys.items():
        if len(duplicated) > 1:
            locations = ", ".join(
                f"{entry.file}:{entry.line}" for entry in duplicated
            )
            issues.append(
                Issue(
                    severity="error",
                    category="duplicate_key",
                    message=f"重复 key：{locations}",
                    key=key,
                )
            )

    for doi, duplicated in dois.items():
        if len(duplicated) > 1:
            issues.append(
                Issue(
                    severity="warning",
                    category="duplicate_doi",
                    message=(
                        f"相同 DOI {doi} 被多个条目使用："
                        + ", ".join(item.key for item in duplicated)
                    ),
                )
            )

    for normalized_title, duplicated in titles.items():
        unique_keys = sorted({item.key for item in duplicated})
        if len(unique_keys) > 1:
            issues.append(
                Issue(
                    severity="warning",
                    category="duplicate_title",
                    message="疑似重复标题：" + ", ".join(unique_keys),
                )
            )

    return issues


def validate_citations(
    citations: dict[str, list[tuple[Path, int]]],
    entries: list[BibEntry],
    cite_all: bool,
    check_unused: bool,
) -> list[Issue]:
    issues: list[Issue] = []
    bib_keys = {entry.key for entry in entries}
    cited_keys = set(citations)

    for key in sorted(cited_keys - bib_keys):
        path, line = citations[key][0]
        issues.append(
            Issue(
                severity="error",
                category="missing_citation_key",
                message=f"正文引用了不存在的 key，共出现 {len(citations[key])} 次",
                key=key,
                file=str(path),
                line=line,
            )
        )

    if check_unused and not cite_all:
        for key in sorted(bib_keys - cited_keys):
            issues.append(
                Issue(
                    severity="warning",
                    category="uncited_entry",
                    message=".bib 中存在但正文未引用",
                    key=key,
                )
            )

    return issues


def crossref_year(item: dict[str, Any]) -> str:
    for field_name in (
        "published-print",
        "published-online",
        "published",
        "issued",
        "created",
    ):
        value = item.get(field_name)
        if not isinstance(value, dict):
            continue

        date_parts = value.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
        ):
            return str(date_parts[0][0])

    return ""


def first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if item is not None and str(item).strip():
                return str(item).strip()
        return ""
    if value is None:
        return ""
    return str(value).strip()


def crossref_authors(item: dict[str, Any]) -> list[str]:
    names: list[str] = []
    authors = item.get("author", [])

    if not isinstance(authors, list):
        return names

    for author in authors:
        if not isinstance(author, dict):
            continue
        family = str(author.get("family", "")).strip()
        given = str(author.get("given", "")).strip()
        name = ", ".join(part for part in (family, given) if part)
        if name:
            names.append(name)

    return names


def candidate_from_crossref(item: dict[str, Any]) -> RemoteCandidate:
    venue = first_text(item.get("container-title"))
    if not venue:
        venue = first_text(item.get("short-container-title"))

    return RemoteCandidate(
        source="crossref",
        title=first_text(item.get("title")),
        venue=venue,
        doi=normalize_doi(str(item.get("DOI", ""))),
        year=crossref_year(item),
        authors=crossref_authors(item),
        volume=str(item.get("volume", "") or "").strip(),
        issue=str(item.get("issue", "") or "").strip(),
        pages=str(
            item.get("page", "")
            or item.get("article-number", "")
            or ""
        ).strip(),
        publisher=str(item.get("publisher", "") or "").strip(),
        item_type=str(item.get("type", "") or "").strip(),
        url=str(item.get("URL", "") or "").strip(),
        raw=item,
    )


def score_candidate(
    entry: BibEntry,
    candidate: RemoteCandidate,
    aliases: dict[str, str],
) -> ScoredCandidate:
    title_score = title_similarity(entry.title, candidate.title)
    journal_score = journal_similarity(entry.venue, candidate.venue, aliases)
    combined = 0.75 * title_score + 0.25 * journal_score

    return ScoredCandidate(
        candidate=candidate,
        title_score=title_score,
        journal_score=journal_score,
        combined_score=combined,
    )


class OnlineVerifier:
    def __init__(
        self,
        email: str,
        title_threshold: float,
        journal_threshold: float,
        rows: int,
        delay: float,
        arxiv_delay: float,
        retries: int,
        timeout: float,
        cache: JsonCache,
        aliases: dict[str, str],
        use_arxiv: bool,
    ):
        self.email = email.strip()
        self.title_threshold = title_threshold
        self.journal_threshold = journal_threshold
        self.rows = rows
        self.delay = delay
        self.arxiv_delay = arxiv_delay
        self.retries = retries
        self.timeout = timeout
        self.cache = cache
        self.aliases = aliases
        self.use_arxiv = use_arxiv
        self.session = requests.Session()

        user_agent = "latex-bib-online-checker/2.0"
        if self.email:
            user_agent += f" (mailto:{self.email})"

        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )
        self._last_crossref_request = 0.0
        self._last_arxiv_request = 0.0

    def _respect_delay(self, source: str) -> None:
        now = time.monotonic()

        if source == "crossref":
            elapsed = now - self._last_crossref_request
            wait = self.delay - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_crossref_request = time.monotonic()
        else:
            elapsed = now - self._last_arxiv_request
            wait = self.arxiv_delay - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_arxiv_request = time.monotonic()

    def _get_json(
        self,
        url: str,
        params: dict[str, Any] | None,
        cache_namespace: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        payload = {"url": url, "params": params or {}}
        cache_key = self.cache.make_key(cache_namespace, payload)
        cached = self.cache.get(cache_key)

        if cached is not None:
            if isinstance(cached, dict) and "_cached_error" in cached:
                return None, str(cached["_cached_error"])
            return cached, None

        last_error: str | None = None

        for attempt in range(self.retries + 1):
            try:
                self._respect_delay("crossref")
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 404:
                    self.cache.set(cache_key, {"_not_found": True})
                    return {"_not_found": True}, None

                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_error = (
                        f"HTTP {response.status_code}: "
                        f"{response.text[:150]}"
                    )
                    if attempt < self.retries:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            time.sleep(float(retry_after))
                        else:
                            time.sleep(2 ** attempt)
                        continue

                response.raise_for_status()
                data = response.json()
                self.cache.set(cache_key, data)
                return data, None

            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        return None, last_error or "未知 Crossref 请求错误"

    def _get_text(
        self,
        url: str,
        params: dict[str, Any],
        cache_namespace: str,
    ) -> tuple[str | None, str | None]:
        payload = {"url": url, "params": params}
        cache_key = self.cache.make_key(cache_namespace, payload)
        cached = self.cache.get(cache_key)

        if cached is not None:
            if isinstance(cached, dict) and "_cached_error" in cached:
                return None, str(cached["_cached_error"])
            if isinstance(cached, dict):
                return str(cached.get("text", "")), None

        last_error: str | None = None

        for attempt in range(self.retries + 1):
            try:
                self._respect_delay("arxiv")
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers={
                        "User-Agent": self.session.headers["User-Agent"],
                        "Accept": "application/atom+xml",
                    },
                )

                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < self.retries:
                        time.sleep(2 ** attempt)
                        continue

                response.raise_for_status()
                text = response.text
                self.cache.set(cache_key, {"text": text})
                return text, None

            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        return None, last_error or "未知 arXiv 请求错误"

    def crossref_by_doi(
        self,
        doi: str,
    ) -> tuple[RemoteCandidate | None, str | None]:
        if not doi:
            return None, None

        encoded_doi = requests.utils.quote(doi, safe="")
        url = f"{CROSSREF_API}/works/{encoded_doi}"
        params = {"mailto": self.email} if self.email else None
        data, error = self._get_json(url, params, "crossref_doi")

        if error:
            return None, error
        if not data or data.get("_not_found"):
            return None, None

        message = data.get("message")
        if not isinstance(message, dict):
            return None, "Crossref DOI 响应中缺少 message"

        return candidate_from_crossref(message), None

    def crossref_search(
        self,
        title: str,
        journal: str,
    ) -> tuple[list[RemoteCandidate], str | None]:
        params: dict[str, Any] = {
            # query.title 已弃用，因此使用 query.bibliographic。
            "query.bibliographic": title,
            "query.container-title": journal,
            "rows": self.rows,
            "sort": "relevance",
            "order": "desc",
        }
        if self.email:
            params["mailto"] = self.email

        data, error = self._get_json(
            f"{CROSSREF_API}/works",
            params,
            "crossref_search",
        )

        if error:
            return [], error
        if not data:
            return [], None

        items = data.get("message", {}).get("items", [])
        if not isinstance(items, list):
            return [], "Crossref 搜索响应中的 items 格式错误"

        return [
            candidate_from_crossref(item)
            for item in items
            if isinstance(item, dict)
        ], None

    def arxiv_search(
        self,
        title: str,
    ) -> tuple[list[RemoteCandidate], str | None]:
        params = {
            "search_query": f'ti:"{clean_latex_text(title)}"',
            "start": 0,
            "max_results": min(self.rows, 10),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        text, error = self._get_text(
            ARXIV_API,
            params,
            "arxiv_search",
        )

        if error:
            return [], error
        if not text:
            return [], None

        atom = "{http://www.w3.org/2005/Atom}"
        arxiv = "{http://arxiv.org/schemas/atom}"

        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            return [], f"arXiv XML 解析失败：{exc}"

        candidates: list[RemoteCandidate] = []

        for item in root.findall(f"{atom}entry"):
            title_element = item.find(f"{atom}title")
            id_element = item.find(f"{atom}id")
            published_element = item.find(f"{atom}published")
            journal_element = item.find(f"{arxiv}journal_ref")
            doi_element = item.find(f"{arxiv}doi")

            authors = []
            for author in item.findall(f"{atom}author"):
                name_element = author.find(f"{atom}name")
                if name_element is not None and name_element.text:
                    authors.append(name_element.text.strip())

            candidates.append(
                RemoteCandidate(
                    source="arxiv",
                    title=(
                        re.sub(r"\s+", " ", title_element.text).strip()
                        if title_element is not None and title_element.text
                        else ""
                    ),
                    venue=(
                        re.sub(r"\s+", " ", journal_element.text).strip()
                        if journal_element is not None and journal_element.text
                        else ""
                    ),
                    doi=(
                        normalize_doi(doi_element.text)
                        if doi_element is not None and doi_element.text
                        else ""
                    ),
                    year=(
                        published_element.text[:4]
                        if published_element is not None
                        and published_element.text
                        else ""
                    ),
                    authors=authors,
                    item_type="preprint",
                    url=(
                        id_element.text.strip()
                        if id_element is not None and id_element.text
                        else ""
                    ),
                    raw={},
                )
            )

        return candidates, None

    def eligible(self, scored: ScoredCandidate) -> bool:
        return (
            scored.title_score >= self.title_threshold
            and scored.journal_score >= self.journal_threshold
        )

    def score_candidates(
        self,
        entry: BibEntry,
        candidates: list[RemoteCandidate],
    ) -> list[ScoredCandidate]:
        scored = [
            score_candidate(entry, candidate, self.aliases)
            for candidate in candidates
        ]
        return sorted(
            scored,
            key=lambda item: item.combined_score,
            reverse=True,
        )

    @staticmethod
    def serialize_top_candidates(
        scored: list[ScoredCandidate],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        output = []
        for item in scored[:limit]:
            output.append(
                {
                    "source": item.candidate.source,
                    "title": item.candidate.title,
                    "journal": item.candidate.venue,
                    "doi": item.candidate.doi,
                    "title_score": round(item.title_score, 4),
                    "journal_score": round(item.journal_score, 4),
                    "combined_score": round(item.combined_score, 4),
                }
            )
        return output

    def verify(self, entry: BibEntry) -> OnlineCheckResult:
        result = OnlineCheckResult(
            key=entry.key,
            status="not_checked",
            local_title=clean_latex_text(entry.title),
            local_journal=clean_latex_text(entry.venue),
            local_doi=entry.doi,
            local_year=entry.year,
        )

        if not entry.title:
            result.status = "insufficient_local_metadata"
            result.message = "本地 BibTeX 缺少 title，无法在线匹配"
            return result

        if not entry.venue:
            result.status = "insufficient_local_metadata"
            result.message = (
                "本地 BibTeX 缺少 journal/journaltitle/booktitle；"
                "由于要求 title 与期刊同时匹配，因此无法核验"
            )
            return result

        operational_errors: list[str] = []
        direct_doi_mismatch: ScoredCandidate | None = None

        # 第一步：有 DOI 时优先按 DOI 定位，但仍必须验证 title + journal。
        if entry.doi:
            doi_candidate, error = self.crossref_by_doi(entry.doi)
            if error:
                operational_errors.append(f"Crossref DOI 查询失败：{error}")
            elif doi_candidate:
                scored = score_candidate(entry, doi_candidate, self.aliases)
                if self.eligible(scored):
                    return build_verified_result(
                        entry,
                        scored,
                        method="doi",
                    )
                direct_doi_mismatch = scored

        # 第二步：用 title 与 journal 同时检索 Crossref。
        crossref_candidates, error = self.crossref_search(
            entry.title,
            entry.venue,
        )
        if error:
            operational_errors.append(f"Crossref 搜索失败：{error}")

        scored_crossref = self.score_candidates(entry, crossref_candidates)
        eligible_crossref = [
            item for item in scored_crossref if self.eligible(item)
        ]

        if eligible_crossref:
            best = eligible_crossref[0]

            # 两个不同 DOI 的候选分数过近时，不自动确认。
            if (
                len(eligible_crossref) >= 2
                and eligible_crossref[0].candidate.doi
                != eligible_crossref[1].candidate.doi
                and (
                    eligible_crossref[0].combined_score
                    - eligible_crossref[1].combined_score
                ) < 0.015
            ):
                result.status = "ambiguous"
                result.source = "crossref"
                result.message = (
                    "存在多个标题和期刊都通过阈值、且综合分数接近的候选，"
                    "需要人工确认"
                )
                result.top_candidates = self.serialize_top_candidates(
                    eligible_crossref
                )
                return result

            verified = build_verified_result(
                entry,
                best,
                method="title+journal",
            )
            verified.top_candidates = self.serialize_top_candidates(
                scored_crossref
            )

            if direct_doi_mismatch:
                verified.message += (
                    "；注意：本地 DOI 指向的 Crossref 记录未通过"
                    " title+journal 校验，当前结果来自重新检索"
                )

            return verified

        # 第三步：arXiv 兜底。必须有 journal_ref 且同时满足两个阈值。
        scored_arxiv: list[ScoredCandidate] = []
        if self.use_arxiv:
            arxiv_candidates, arxiv_error = self.arxiv_search(entry.title)
            if arxiv_error:
                operational_errors.append(f"arXiv 查询失败：{arxiv_error}")

            # 没有 journal_ref 的记录不能满足期刊匹配要求。
            arxiv_candidates = [
                candidate
                for candidate in arxiv_candidates
                if candidate.venue.strip()
            ]
            scored_arxiv = self.score_candidates(entry, arxiv_candidates)
            eligible_arxiv = [
                item for item in scored_arxiv if self.eligible(item)
            ]

            if eligible_arxiv:
                verified = build_verified_result(
                    entry,
                    eligible_arxiv[0],
                    method="title+journal_ref",
                )
                verified.top_candidates = self.serialize_top_candidates(
                    eligible_arxiv
                )
                verified.message += (
                    "；该结果来自 arXiv journal_ref，建议再人工确认正式出版信息"
                )
                return verified

        # 失败原因分类。
        all_scored = scored_crossref + scored_arxiv
        all_scored.sort(
            key=lambda item: item.combined_score,
            reverse=True,
        )
        result.top_candidates = self.serialize_top_candidates(all_scored)

        if operational_errors and not all_scored:
            result.status = "network_or_api_error"
            result.message = "；".join(operational_errors)
            return result

        best = all_scored[0] if all_scored else direct_doi_mismatch
        if best:
            result.source = best.candidate.source
            result.title_score = best.title_score
            result.journal_score = best.journal_score
            result.combined_score = best.combined_score
            result.remote_title = best.candidate.title
            result.remote_journal = best.candidate.venue
            result.remote_doi = best.candidate.doi
            result.remote_year = best.candidate.year
            result.candidate_url = best.candidate.url

            title_ok = best.title_score >= self.title_threshold
            journal_ok = best.journal_score >= self.journal_threshold

            if title_ok and not journal_ok:
                result.status = "journal_mismatch"
                result.message = (
                    "找到了高相似标题，但期刊名称未达到匹配阈值"
                )
            elif journal_ok and not title_ok:
                result.status = "title_mismatch"
                result.message = (
                    "找到了相近期刊中的候选，但标题未达到匹配阈值"
                )
            else:
                result.status = "not_verified_online"
                result.message = (
                    "未找到同时满足 title 与 journal 阈值的在线记录"
                )
        else:
            result.status = "not_verified_online"
            result.message = (
                "Crossref/arXiv 未返回可用于 title+journal 核验的记录；"
                "这不能单独证明论文不存在"
            )

        if operational_errors:
            result.message += "；部分请求异常：" + "；".join(operational_errors)

        return result


def build_verified_result(
    entry: BibEntry,
    scored: ScoredCandidate,
    method: str,
) -> OnlineCheckResult:
    candidate = scored.candidate

    result = OnlineCheckResult(
        key=entry.key,
        status="verified",
        source=candidate.source,
        method=method,
        title_score=scored.title_score,
        journal_score=scored.journal_score,
        combined_score=scored.combined_score,
        local_title=clean_latex_text(entry.title),
        remote_title=candidate.title,
        local_journal=clean_latex_text(entry.venue),
        remote_journal=candidate.venue,
        local_doi=entry.doi,
        remote_doi=candidate.doi,
        local_year=entry.year,
        remote_year=candidate.year,
        candidate_url=candidate.url,
        message="title 与 journal 均通过匹配阈值",
    )

    local = entry.fields

    # 远程存在、本地缺失时，说明本地 BibTeX 信息不完整。
    remote_to_local = {
        "doi": candidate.doi,
        "year": candidate.year,
        "volume": candidate.volume,
        "number": candidate.issue,
        "pages": candidate.pages,
        "publisher": candidate.publisher,
    }

    for local_name, remote_value in remote_to_local.items():
        if not remote_value:
            continue

        if local_name == "year":
            local_value = entry.year
        elif local_name == "number":
            local_value = (
                local.get("number", "").strip()
                or local.get("issue", "").strip()
            )
        elif local_name == "pages":
            local_value = local.get("pages", "").strip()
        else:
            local_value = local.get(local_name, "").strip()

        if not local_value:
            result.missing_local_fields.append(local_name)

    # DOI 不一致属于强错误。
    if entry.doi and candidate.doi and entry.doi != candidate.doi:
        result.mismatched_fields["doi"] = {
            "local": entry.doi,
            "online": candidate.doi,
        }

    if entry.year and candidate.year and entry.year != candidate.year:
        result.mismatched_fields["year"] = {
            "local": entry.year,
            "online": candidate.year,
        }

    local_volume = local.get("volume", "").strip()
    if (
        local_volume
        and candidate.volume
        and normalize_general_text(local_volume)
        != normalize_general_text(candidate.volume)
    ):
        result.mismatched_fields["volume"] = {
            "local": local_volume,
            "online": candidate.volume,
        }

    local_issue = (
        local.get("number", "").strip()
        or local.get("issue", "").strip()
    )
    if (
        local_issue
        and candidate.issue
        and normalize_general_text(local_issue)
        != normalize_general_text(candidate.issue)
    ):
        result.mismatched_fields["number"] = {
            "local": local_issue,
            "online": candidate.issue,
        }

    local_pages = local.get("pages", "").strip()
    if (
        local_pages
        and candidate.pages
        and normalize_pages(local_pages) != normalize_pages(candidate.pages)
    ):
        result.mismatched_fields["pages"] = {
            "local": local_pages,
            "online": candidate.pages,
        }

    local_author = local.get("author", "")
    if local_author and candidate.authors:
        score = author_similarity(local_author, candidate.authors)
        if score < 0.5:
            result.mismatched_fields["author"] = {
                "local": clean_latex_text(local_author),
                "online": " and ".join(candidate.authors),
            }

    # 检查在线记录自身是否缺少关键元数据。
    online_required = {
        "title": candidate.title,
        "journal": candidate.venue,
        "author": candidate.authors,
        "year": candidate.year,
    }
    if candidate.source == "crossref":
        online_required["doi"] = candidate.doi

    for name, value in online_required.items():
        if not value:
            result.online_missing_fields.append(name)

    return result


def parse_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"期刊别名格式错误：{value!r}，应为 SHORT=FULL"
            )

        short, full = value.split("=", 1)
        short = short.strip()
        full = full.strip()

        if not short or not full:
            raise ValueError(
                f"期刊别名格式错误：{value!r}，左右两侧不能为空"
            )

        aliases[normalize_general_text(short)] = full

    return aliases


def issue_from_online_result(
    result: OnlineCheckResult,
    entry: BibEntry,
    strict: bool,
) -> list[Issue]:
    issues: list[Issue] = []

    if result.status == "verified":
        for field_name in result.missing_local_fields:
            issues.append(
                Issue(
                    severity="warning",
                    category="missing_online_available_field",
                    message=(
                        f"在线记录包含 {field_name}，但本地 BibTeX 缺少该字段"
                    ),
                    key=entry.key,
                    file=str(entry.file),
                    line=entry.line,
                )
            )

        for field_name, values in result.mismatched_fields.items():
            severity = "error" if field_name == "doi" else "warning"
            if strict:
                severity = "error"
            issues.append(
                Issue(
                    severity=severity,
                    category="metadata_mismatch",
                    message=(
                        f"{field_name} 不一致："
                        f"本地={values['local']!r}，在线={values['online']!r}"
                    ),
                    key=entry.key,
                    file=str(entry.file),
                    line=entry.line,
                )
            )

        for field_name in result.online_missing_fields:
            issues.append(
                Issue(
                    severity="warning",
                    category="online_metadata_incomplete",
                    message=f"在线数据库记录缺少 {field_name}",
                    key=entry.key,
                    file=str(entry.file),
                    line=entry.line,
                )
            )

        return issues

    severity = "error" if strict else "warning"
    if result.status == "insufficient_local_metadata":
        severity = "error"

    issues.append(
        Issue(
            severity=severity,
            category=f"online_{result.status}",
            message=result.message,
            key=entry.key,
            file=str(entry.file),
            line=entry.line,
        )
    )
    return issues


def format_issue(issue: Issue) -> str:
    label = {
        "error": "ERROR",
        "warning": "WARN ",
        "info": "INFO ",
    }.get(issue.severity, issue.severity.upper())

    key_text = f" <{issue.key}>" if issue.key else ""
    location = ""
    if issue.file:
        location = issue.file
        if issue.line:
            location += f":{issue.line}"
        location = f" [{location}]"

    return f"{label} [{issue.category}]{key_text}{location} {issue.message}"


def write_csv_report(
    path: Path,
    online_results: list[OnlineCheckResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "key",
        "status",
        "source",
        "method",
        "title_score",
        "journal_score",
        "combined_score",
        "local_title",
        "remote_title",
        "local_journal",
        "remote_journal",
        "local_doi",
        "remote_doi",
        "local_year",
        "remote_year",
        "missing_local_fields",
        "mismatched_fields",
        "online_missing_fields",
        "candidate_url",
        "message",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in online_results:
            row = asdict(result)
            row["title_score"] = f"{result.title_score:.4f}"
            row["journal_score"] = f"{result.journal_score:.4f}"
            row["combined_score"] = f"{result.combined_score:.4f}"
            row["missing_local_fields"] = "; ".join(
                result.missing_local_fields
            )
            row["mismatched_fields"] = json.dumps(
                result.mismatched_fields,
                ensure_ascii=False,
            )
            row["online_missing_fields"] = "; ".join(
                result.online_missing_fields
            )
            row.pop("top_candidates", None)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检查 LaTeX/BibTeX，并使用 title + journal 双条件在线核验。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        type=Path,
        help="主 .tex、单个 .bib 或 LaTeX 项目目录",
    )
    parser.add_argument(
        "--bib",
        action="append",
        type=Path,
        default=[],
        help="手动指定 .bib 文件；可重复使用",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="在线检查全部 BibTeX 条目，而不仅是正文已引用条目",
    )
    parser.add_argument(
        "--no-unused",
        action="store_true",
        help="不报告 .bib 中未被正文引用的条目",
    )
    parser.add_argument(
        "--no-arxiv",
        action="store_true",
        help="禁用 arXiv 兜底查询",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("CROSSREF_MAILTO", ""),
        help="Crossref polite-pool 联系邮箱，也可设置 CROSSREF_MAILTO",
    )
    parser.add_argument(
        "--title-threshold",
        type=float,
        default=0.92,
        help="标题匹配阈值，范围 0-1",
    )
    parser.add_argument(
        "--journal-threshold",
        type=float,
        default=0.82,
        help="期刊/出版物名称匹配阈值，范围 0-1",
    )
    parser.add_argument(
        "--journal-alias",
        action="append",
        default=[],
        metavar="SHORT=FULL",
        help="设置期刊简称映射；可重复使用",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="每次在线查询最多获取的候选数",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Crossref 请求之间的最小间隔秒数",
    )
    parser.add_argument(
        "--arxiv-delay",
        type=float,
        default=3.0,
        help="arXiv 请求之间的最小间隔秒数",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="单次 HTTP 请求超时秒数",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="网络错误或限流时的重试次数",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".bibcheck_online_cache.json"),
        help="在线响应缓存文件",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用在线响应缓存",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把在线未核验及普通元数据差异也视为错误",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("bib_online_report.json"),
        help="JSON 报告路径",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("bib_online_report.csv"),
        help="CSV 报告路径",
    )
    return parser.parse_args()


def resolve_inputs(
    args: argparse.Namespace,
) -> tuple[
    Path | None,
    list[Path],
    list[Path],
    dict[str, list[tuple[Path, int]]],
    bool,
    list[Issue],
]:
    target = normalize_path(args.target)
    issues: list[Issue] = []
    main_tex: Path | None = None
    tex_files: list[Path] = []
    citations: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    cite_all = False

    if not target.exists():
        raise FileNotFoundError(f"目标路径不存在：{target}")

    if target.is_dir():
        main_tex = find_main_tex(target)
        if main_tex is None:
            raise FileNotFoundError(f"目录中未找到 .tex：{target}")
    elif target.suffix.casefold() == ".tex":
        main_tex = target
    elif target.suffix.casefold() == ".bib":
        pass
    else:
        raise ValueError("target 必须是目录、.tex 或 .bib 文件")

    if main_tex:
        tex_files, tex_issues = collect_tex_files(main_tex)
        issues.extend(tex_issues)
        citations, cite_all = extract_citations(tex_files)

    if args.bib:
        bib_files = []
        for path in args.bib:
            resolved = normalize_path(
                path if path.is_absolute() else Path.cwd() / path
            )
            if not resolved.exists():
                issues.append(
                    Issue(
                        severity="error",
                        category="missing_bib_file",
                        message=f"手动指定的 .bib 不存在：{resolved}",
                        file=str(resolved),
                    )
                )
            else:
                bib_files.append(resolved)
    elif target.suffix.casefold() == ".bib":
        bib_files = [target]
    else:
        bib_files, bib_issues = discover_bib_files(tex_files)
        issues.extend(bib_issues)

    if not bib_files:
        issues.append(
            Issue(
                severity="error",
                category="no_bib_file",
                message=(
                    "未找到 .bib。请检查 \\bibliography / "
                    "\\addbibresource，或使用 --bib 指定"
                ),
                file=str(main_tex) if main_tex else None,
            )
        )

    return (
        main_tex,
        tex_files,
        bib_files,
        citations,
        cite_all,
        issues,
    )


def main() -> int:
    args = parse_args()

    for name, value in (
        ("title-threshold", args.title_threshold),
        ("journal-threshold", args.journal_threshold),
    ):
        if not 0 <= value <= 1:
            print(f"ERROR: --{name} 必须在 0 到 1 之间", file=sys.stderr)
            return 2

    if args.rows < 1:
        print("ERROR: --rows 必须大于 0", file=sys.stderr)
        return 2

    try:
        aliases = parse_aliases(args.journal_alias)
        (
            main_tex,
            tex_files,
            bib_files,
            citations,
            cite_all,
            issues,
        ) = resolve_inputs(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    entries: list[BibEntry] = []
    for bib_file in bib_files:
        try:
            parsed, parse_issues = parse_bib_file(bib_file)
            entries.extend(parsed)
            issues.extend(parse_issues)
        except Exception as exc:
            issues.append(
                Issue(
                    severity="error",
                    category="bib_read_error",
                    message=str(exc),
                    file=str(bib_file),
                )
            )

    issues.extend(validate_local_entries(entries))
    if main_tex:
        issues.extend(
            validate_citations(
                citations,
                entries,
                cite_all,
                check_unused=not args.no_unused,
            )
        )

    entry_by_key: dict[str, BibEntry] = {}
    for entry in entries:
        entry_by_key.setdefault(entry.key, entry)

    if args.all or cite_all or not main_tex:
        online_entries = list(entry_by_key.values())
    else:
        online_entries = [
            entry_by_key[key]
            for key in citations
            if key in entry_by_key
        ]

    cache_path = None if args.no_cache else normalize_path(args.cache)
    verifier = OnlineVerifier(
        email=args.email,
        title_threshold=args.title_threshold,
        journal_threshold=args.journal_threshold,
        rows=args.rows,
        delay=max(0.0, args.delay),
        arxiv_delay=max(0.0, args.arxiv_delay),
        retries=max(0, args.retries),
        timeout=max(1.0, args.timeout),
        cache=JsonCache(cache_path, enabled=not args.no_cache),
        aliases=aliases,
        use_arxiv=not args.no_arxiv,
    )

    online_results: list[OnlineCheckResult] = []

    print("=" * 78)
    print("LaTeX / BibTeX 本地检查 + title/journal 在线核验")
    print("=" * 78)
    print(f"主文件            : {main_tex or '(直接检查 .bib)'}")
    print(f"Tex 文件数        : {len(tex_files)}")
    print(f"Bib 文件数        : {len(bib_files)}")
    print(f"Bib 条目数        : {len(entries)}")
    print(f"在线检查条目数    : {len(online_entries)}")
    print(f"title 阈值        : {args.title_threshold:.2f}")
    print(f"journal 阈值      : {args.journal_threshold:.2f}")
    print("-" * 78)

    for index, entry in enumerate(online_entries, start=1):
        print(
            f"[{index:>4}/{len(online_entries)}] {entry.key}: ",
            end="",
            flush=True,
        )

        result = verifier.verify(entry)
        online_results.append(result)
        issues.extend(issue_from_online_result(result, entry, args.strict))

        if result.status == "verified":
            print(
                f"VERIFIED [{result.source}/{result.method}] "
                f"title={result.title_score:.3f}, "
                f"journal={result.journal_score:.3f}"
            )
        else:
            print(
                f"{result.status.upper()} "
                f"title={result.title_score:.3f}, "
                f"journal={result.journal_score:.3f}"
            )

    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(
        key=lambda item: (
            severity_order.get(item.severity, 99),
            item.category,
            item.file or "",
            item.line or 0,
            item.key or "",
        )
    )

    counts = Counter(issue.severity for issue in issues)
    status_counts = Counter(result.status for result in online_results)

    report = {
        "configuration": {
            "title_threshold": args.title_threshold,
            "journal_threshold": args.journal_threshold,
            "crossref_rows": args.rows,
            "use_arxiv": not args.no_arxiv,
            "strict": args.strict,
            "journal_aliases": aliases,
        },
        "inputs": {
            "main_tex": str(main_tex) if main_tex else None,
            "tex_files": [str(path) for path in tex_files],
            "bib_files": [str(path) for path in bib_files],
        },
        "statistics": {
            "bib_entry_count": len(entries),
            "online_checked_count": len(online_results),
            "online_status_counts": dict(sorted(status_counts.items())),
            "error_count": counts["error"],
            "warning_count": counts["warning"],
        },
        "issues": [asdict(issue) for issue in issues],
        "online_results": [asdict(result) for result in online_results],
    }

    json_path = normalize_path(args.json)
    csv_path = normalize_path(args.csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv_report(csv_path, online_results)

    print("-" * 78)
    print("问题明细")
    print("-" * 78)
    if issues:
        for issue in issues:
            print(format_issue(issue))
    else:
        print("OK: 未发现明显问题。")

    print("=" * 78)
    print(f"在线状态统计      : {dict(sorted(status_counts.items()))}")
    print(f"错误数量          : {counts['error']}")
    print(f"警告数量          : {counts['warning']}")
    print(f"JSON 报告         : {json_path}")
    print(f"CSV 报告          : {csv_path}")
    print("=" * 78)

    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
