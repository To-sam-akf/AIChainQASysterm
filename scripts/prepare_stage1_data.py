#!/usr/bin/env python3
"""Prepare stage-1 PDFs for the AI compute industry-chain KG project.

The script collects public annual reports and research reports, validates the
downloaded PDFs, and writes a reproducible manifest for later parsing stages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ANNUAL_DIR = DATA_DIR / "raw_pdfs" / "annual"
RESEARCH_DIR = DATA_DIR / "raw_pdfs" / "research"
METADATA_DIR = DATA_DIR / "metadata"
COMPANIES_CSV = METADATA_DIR / "companies.csv"
MANIFEST_CSV = METADATA_DIR / "reports_manifest.csv"

CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC_BASE = "https://static.cninfo.com.cn/"
EASTMONEY_REPORT_API = "https://reportapi.eastmoney.com/report/list"

TARGET_ANNUAL_YEARS = (2025, 2024)
DEFAULT_MAX_RESEARCH = 5
REQUEST_TIMEOUT = 30
MAX_TITLE_FILENAME_LEN = 36

MANIFEST_FIELDS = [
    "report_id",
    "kind",
    "company",
    "stock_code",
    "year",
    "title",
    "source_site",
    "source_url",
    "pdf_url",
    "local_path",
    "published_at",
    "downloaded_at",
    "sha256",
    "file_size",
    "pages",
    "status",
    "error",
]

DEFAULT_COMPANIES = [
    ("浪潮信息", "000977", "SZ", "AI服务器"),
    ("中科曙光", "603019", "SH", "AI服务器/高性能计算"),
    ("工业富联", "601138", "SH", "AI服务器/智能制造"),
    ("中际旭创", "300308", "SZ", "光模块"),
    ("新易盛", "300502", "SZ", "光模块"),
    ("天孚通信", "300394", "SZ", "光器件"),
    ("英维克", "002837", "SZ", "液冷/温控"),
    ("申菱环境", "301018", "SZ", "液冷/温控"),
    ("寒武纪", "688256", "SH", "AI芯片"),
    ("海光信息", "688041", "SH", "AI芯片/CPU"),
]

RESEARCH_KEYWORDS = [
    "AI算力产业链",
    "算力深度报告",
    "AIDC算力",
    "国产算力",
    "AI服务器 光模块 液冷",
]

RESEARCH_RELEVANCE_TERMS = {
    "算力": 3,
    "AI": 1,
    "人工智能": 1,
    "产业链": 3,
    "服务器": 2,
    "AI服务器": 3,
    "光模块": 3,
    "液冷": 3,
    "温控": 2,
    "国产算力": 4,
    "AIDC": 3,
    "数据中心": 2,
    "GPU": 1,
    "芯片": 1,
    "DeepSeek": 1,
}

ANNUAL_EXCLUDE_TERMS = (
    "摘要",
    "英文",
    "修订",
    "更正",
    "更新",
    "取消",
    "已取消",
    "问询",
    "回复",
    "意见",
    "内部控制",
    "社会责任",
    "可持续",
    "ESG",
    "独立董事",
    "审计报告",
)

# Public PDF seeds are used only if the live search returns too few direct PDFs.
RESEARCH_SEEDS = [
    {
        "title": "算力深度报告一：算力研究框架-产业链全梳理",
        "source_site": "东方财富研报",
        "source_url": "https://data.eastmoney.com/report/zw_industry.jshtml?infocode=AP202404021629415823",
        "pdf_url": "https://pdf.dfcfw.com/pdf/H3_AP202404021629415823_1.pdf",
        "org": "国金证券",
        "published_at": "2024-04-02",
        "year": "2024",
    },
    {
        "title": "国产算力系列专题报告：国产算力飞跃启新，AI引领产业链发展变革",
        "source_site": "长城证券",
        "source_url": "https://www.cgws.com/cczq/ggdt/ccyj/202406/t20240617_315123.html",
        "pdf_url": "https://www.cgws.com/cczq/ggdt/ccyj/202406/P020240617315123369639.pdf",
        "org": "长城证券",
        "published_at": "2024-06-17",
        "year": "2024",
    },
    {
        "title": "计算机行业专题研究：DeepSeek算力告急，利好AI应用+AI算力产业链",
        "source_site": "东方财富研报",
        "source_url": "https://data.eastmoney.com/report/zw_industry.jshtml?infocode=AP202502091642912986",
        "pdf_url": "https://pdf.dfcfw.com/pdf/H3_AP202502091642912986_1.pdf",
        "org": "公开研报",
        "published_at": "2025-02-09",
        "year": "2025",
    },
    {
        "title": "AI产业链业绩兑现，算力投资空间广阔",
        "source_site": "东方财富研报",
        "source_url": "https://data.eastmoney.com/report/zw_industry.jshtml?infocode=AP202507161709977534",
        "pdf_url": "https://pdf.dfcfw.com/pdf/H3_AP202507161709977534_1.pdf",
        "org": "公开研报",
        "published_at": "2025-07-16",
        "year": "2025",
    },
    {
        "title": "AI服务器与光模块产业链专题",
        "source_site": "东方财富研报",
        "source_url": "https://data.eastmoney.com/report/zw_industry.jshtml?infocode=AP202403141626724087",
        "pdf_url": "https://pdf.dfcfw.com/pdf/H3_AP202403141626724087_1.pdf",
        "org": "公开研报",
        "published_at": "2024-03-14",
        "year": "2024",
    },
]


@dataclass(frozen=True)
class Company:
    company: str
    stock_code: str
    market: str
    chain_segment: str


@dataclass
class ReportCandidate:
    report_id: str
    kind: str
    title: str
    source_site: str
    source_url: str
    pdf_url: str
    local_path: Path
    company: str = ""
    stock_code: str = ""
    year: str = ""
    published_at: str = ""
    alternate_pdf_urls: list[str] = field(default_factory=list)
    score: int = 0

    @property
    def all_pdf_urls(self) -> list[str]:
        urls = [self.pdf_url, *self.alternate_pdf_urls]
        return list(dict.fromkeys(url for url in urls if url))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_directories() -> None:
    for path in (ANNUAL_DIR, RESEARCH_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_default_companies(path: Path = COMPANIES_CSV) -> None:
    if path.exists():
        return
    ensure_directories()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["company", "stock_code", "market", "chain_segment"])
        writer.writerows(DEFAULT_COMPANIES)


def load_companies(path: Path = COMPANIES_CSV) -> list[Company]:
    write_default_companies(path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        companies = [
            Company(
                company=row["company"].strip(),
                stock_code=row["stock_code"].strip(),
                market=row["market"].strip().upper(),
                chain_segment=row["chain_segment"].strip(),
            )
            for row in reader
        ]
    return companies


def validate_companies(companies: Iterable[Company]) -> None:
    companies = list(companies)
    if len(companies) != 10:
        raise ValueError(f"Expected 10 target companies, got {len(companies)}")
    for company in companies:
        if not re.fullmatch(r"\d{6}", company.stock_code):
            raise ValueError(f"Invalid stock code for {company.company}: {company.stock_code}")
        if company.market not in {"SZ", "SH"}:
            raise ValueError(f"Invalid market for {company.company}: {company.market}")


def safe_filename(value: str, max_stem_len: int = 96) -> str:
    value = html.unescape(strip_tags(value)).strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._ ")
    if not value:
        value = "untitled"
    if len(value) > max_stem_len:
        value = value[:max_stem_len].rstrip("._ ")
    return value


def strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "")


def short_title(value: str, max_len: int = MAX_TITLE_FILENAME_LEN) -> str:
    compact = re.sub(r"\s+", "", strip_tags(value))
    return compact[:max_len]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pdf(path: Path) -> int:
    if not path.exists():
        raise ValueError(f"PDF does not exist: {path}")
    if path.stat().st_size < 5:
        raise ValueError(f"PDF is too small to be valid: {path}")
    with path.open("rb") as file:
        magic = file.read(5)
    if magic != b"%PDF-":
        raise ValueError(f"File is not a PDF: {path}")
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    if page_count <= 0:
        raise ValueError(f"PDF has no pages: {path}")
    return page_count


class ManifestStore:
    def __init__(self, path: Path = MANIFEST_CSV) -> None:
        self.path = path
        self.rows: list[dict[str, str]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as file:
                self.rows = list(csv.DictReader(file))

    def get_by_report_id(self, report_id: str) -> dict[str, str] | None:
        for row in self.rows:
            if row.get("report_id") == report_id:
                return row
        return None

    def upsert(self, row: dict[str, Any]) -> None:
        normalized = {field: str(row.get(field, "") or "") for field in MANIFEST_FIELDS}
        replaced = False
        for index, existing in enumerate(self.rows):
            same_id = existing.get("report_id") == normalized["report_id"]
            same_path = existing.get("local_path") and existing.get("local_path") == normalized["local_path"]
            if same_id or same_path:
                self.rows[index] = normalized
                replaced = True
                break
        if not replaced:
            self.rows.append(normalized)
        self.write()

    def write(self) -> None:
        ensure_directories()
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html,application/pdf,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    )
    return session


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def manifest_row(
    candidate: ReportCandidate,
    *,
    status: str,
    error: str = "",
    pdf_url: str | None = None,
    sha256: str = "",
    file_size: int | str = "",
    pages: int | str = "",
) -> dict[str, Any]:
    return {
        "report_id": candidate.report_id,
        "kind": candidate.kind,
        "company": candidate.company,
        "stock_code": candidate.stock_code,
        "year": candidate.year,
        "title": candidate.title,
        "source_site": candidate.source_site,
        "source_url": candidate.source_url,
        "pdf_url": pdf_url or candidate.pdf_url,
        "local_path": relative_path(candidate.local_path),
        "published_at": candidate.published_at,
        "downloaded_at": now_iso() if status == "downloaded" else "",
        "sha256": sha256,
        "file_size": file_size,
        "pages": pages,
        "status": status,
        "error": error,
    }


def download_report(
    session: requests.Session,
    candidate: ReportCandidate,
    manifest: ManifestStore,
    *,
    refresh: bool = False,
) -> str:
    target = candidate.local_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not refresh:
        try:
            pages = validate_pdf(target)
            digest = sha256_file(target)
            manifest.upsert(
                manifest_row(
                    candidate,
                    status="downloaded",
                    sha256=digest,
                    file_size=target.stat().st_size,
                    pages=pages,
                )
            )
            return f"SKIP existing {relative_path(target)}"
        except Exception:
            target.unlink(missing_ok=True)

    errors: list[str] = []
    for pdf_url in candidate.all_pdf_urls:
        temp_path = target.with_suffix(target.suffix + ".part")
        temp_path.unlink(missing_ok=True)
        try:
            with session.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                with temp_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            file.write(chunk)
            if "text/html" in content_type and temp_path.read_bytes()[:5] != b"%PDF-":
                raise ValueError("download returned HTML instead of PDF")
            pages = validate_pdf(temp_path)
            shutil.move(str(temp_path), target)
            digest = sha256_file(target)
            manifest.upsert(
                manifest_row(
                    candidate,
                    status="downloaded",
                    pdf_url=pdf_url,
                    sha256=digest,
                    file_size=target.stat().st_size,
                    pages=pages,
                )
            )
            return f"OK   {relative_path(target)} ({pages} pages)"
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            errors.append(f"{pdf_url}: {exc}")
    error = " | ".join(errors)
    manifest.upsert(manifest_row(candidate, status="failed", error=error))
    return f"FAIL {candidate.title}: {error}"


def cninfo_column(company: Company) -> str:
    return "szse" if company.market == "SZ" else "sse"


def cninfo_payload(company: Company, year: int, page_num: int, *, stock_scoped: bool) -> dict[str, str]:
    publish_year = year + 1
    search_key = f"{year}年年度报告" if stock_scoped else company.company
    return {
        "pageNum": str(page_num),
        "pageSize": "30",
        "column": cninfo_column(company),
        "tabName": "fulltext",
        "plate": "",
        "stock": company.stock_code if stock_scoped else "",
        "searchkey": search_key,
        "secid": "",
        "category": "category_ndbg_szsh",
        "trade": "",
        "seDate": f"{publish_year}-01-01~{publish_year}-12-31",
        "sortName": "time",
        "sortType": "desc",
        "isHLtitle": "true",
    }


def query_cninfo(company: Company, year: int, session: requests.Session) -> list[dict[str, Any]]:
    headers = {
        "Origin": "https://www.cninfo.com.cn",
        "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    announcements: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for stock_scoped in (False, True):
        for page_num in range(1, 4):
            payload = cninfo_payload(company, year, page_num, stock_scoped=stock_scoped)
            response = session.post(CNINFO_QUERY_URL, data=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            for item in data.get("announcements") or []:
                announcement_id = str(item.get("announcementId") or item.get("id") or "")
                if announcement_id and announcement_id in seen_ids:
                    continue
                if announcement_id:
                    seen_ids.add(announcement_id)
                announcements.append(item)
            if not data.get("hasMore", False):
                break
        if announcements:
            break
    return announcements


def clean_cninfo_title(title: str) -> str:
    return html.unescape(strip_tags(title)).replace("：", "_").strip()


def cninfo_pdf_url(announcement: dict[str, Any]) -> str:
    adjunct_url = str(announcement.get("adjunctUrl") or "").lstrip("/")
    return urljoin(CNINFO_STATIC_BASE, adjunct_url)


def cninfo_source_url(company: Company, announcement: dict[str, Any]) -> str:
    announcement_id = announcement.get("announcementId", "")
    org_id = announcement.get("orgId", "")
    announcement_time = announcement.get("announcementTime", "")
    return (
        "https://www.cninfo.com.cn/new/disclosure/detail?"
        f"stockCode={company.stock_code}&announcementId={announcement_id}"
        f"&orgId={org_id}&announcementTime={quote(str(announcement_time))}"
    )


def format_cninfo_time(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000).date().isoformat()
    if re.fullmatch(r"\d{10}", text):
        return datetime.fromtimestamp(int(text)).date().isoformat()
    return text[:10] if text else ""


def is_annual_report_title(title: str, company: Company, year: int) -> bool:
    plain = clean_cninfo_title(title)
    if str(year) not in plain or "年度报告" not in plain:
        return False
    if any(term.lower() in plain.lower() for term in ANNUAL_EXCLUDE_TERMS):
        return False
    if company.company not in plain and company.stock_code not in plain:
        # CNInfo titles often omit the stock code but include the company name.
        return True
    return True


def annual_score(announcement: dict[str, Any], company: Company, year: int) -> int:
    title = clean_cninfo_title(str(announcement.get("announcementTitle") or ""))
    score = 0
    if title == f"{company.company}_{year}年年度报告":
        score += 50
    if title.endswith(f"{year}年年度报告"):
        score += 30
    if company.company in title:
        score += 10
    if str(announcement.get("secCode") or "") == company.stock_code:
        score += 10
    if cninfo_pdf_url(announcement).lower().endswith(".pdf"):
        score += 5
    return score


def find_annual_candidate(company: Company, year: int, session: requests.Session) -> ReportCandidate | None:
    candidates = []
    for announcement in query_cninfo(company, year, session):
        sec_code = str(announcement.get("secCode") or "")
        sec_name = str(announcement.get("secName") or "")
        if sec_code and sec_code != company.stock_code:
            continue
        if not sec_code and company.company not in sec_name and company.company not in str(announcement.get("announcementTitle") or ""):
            continue
        title = str(announcement.get("announcementTitle") or "")
        if not is_annual_report_title(title, company, year):
            continue
        pdf_url = cninfo_pdf_url(announcement)
        if not pdf_url.lower().endswith(".pdf"):
            continue
        candidates.append((annual_score(announcement, company, year), announcement))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, announcement = candidates[0]
    title = clean_cninfo_title(str(announcement.get("announcementTitle") or f"{company.company}_{year}年年度报告"))
    local_path = ANNUAL_DIR / f"{safe_filename(company.company)}_{year}年报.pdf"
    return ReportCandidate(
        report_id=f"annual_{company.stock_code}_{year}",
        kind="annual",
        company=company.company,
        stock_code=company.stock_code,
        year=str(year),
        title=title,
        source_site="巨潮资讯",
        source_url=cninfo_source_url(company, announcement),
        pdf_url=cninfo_pdf_url(announcement),
        local_path=local_path,
        published_at=format_cninfo_time(announcement.get("announcementTime")),
    )


def find_latest_annual_candidate(company: Company, session: requests.Session) -> ReportCandidate | None:
    for year in TARGET_ANNUAL_YEARS:
        candidate = find_annual_candidate(company, year, session)
        if candidate is not None:
            return candidate
        time.sleep(0.2)
    return None


def get_any(mapping: dict[str, Any], *names: str) -> Any:
    lower_lookup = {key.lower(): key for key in mapping.keys()}
    for name in names:
        if name in mapping:
            return mapping[name]
        key = lower_lookup.get(name.lower())
        if key is not None:
            return mapping[key]
    return ""


def parse_json_or_jsonp(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\((\{.*\})\)\s*;?$", text, flags=re.S)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)


def research_score(title: str, summary: str = "") -> int:
    content = f"{title} {summary}"
    return sum(weight for term, weight in RESEARCH_RELEVANCE_TERMS.items() if term.lower() in content.lower())


def eastmoney_pdf_options(info_code: str, encode_url: str = "") -> list[str]:
    options = [
        f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf",
        f"https://pdf.dfcfw.com/pdf/H2_{info_code}_1.pdf",
    ]
    if encode_url:
        options.extend(
            [
                f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf?{encode_url}.pdf",
                f"https://pdf.dfcfw.com/pdf/H2_{info_code}_1.pdf?{encode_url}.pdf",
            ]
        )
    return options


def normalize_eastmoney_record(record: dict[str, Any]) -> ReportCandidate | None:
    title = strip_tags(str(get_any(record, "title", "TITLE", "reportName", "REPORT_NAME")))
    info_code = str(get_any(record, "infoCode", "INFO_CODE", "INFOCODE", "infocode"))
    if not title or not info_code.startswith("AP"):
        return None
    summary = strip_tags(str(get_any(record, "summary", "SUMMARY", "abstract", "ABSTRACT")))
    score = research_score(title, summary)
    if score <= 0:
        return None
    org = str(get_any(record, "orgSName", "ORG_S_NAME", "orgName", "ORG_NAME") or "公开研报").strip()
    published_at = str(get_any(record, "publishDate", "PUBLISH_DATE", "date", "DATE"))[:10]
    year_match = re.search(r"20\d{2}", published_at or title)
    year = year_match.group(0) if year_match else ""
    pdf_options = eastmoney_pdf_options(str(info_code), str(get_any(record, "encodeUrl", "ENCODE_URL")))
    file_name = research_filename(0, org, published_at, title)
    return ReportCandidate(
        report_id=f"research_{info_code}",
        kind="research",
        year=year,
        title=title,
        source_site="东方财富研报",
        source_url=f"https://data.eastmoney.com/report/zw_industry.jshtml?infocode={info_code}",
        pdf_url=pdf_options[0],
        alternate_pdf_urls=pdf_options[1:],
        local_path=RESEARCH_DIR / file_name,
        published_at=published_at,
        score=score,
    )


def query_eastmoney_research(session: requests.Session) -> list[ReportCandidate]:
    candidates: dict[str, ReportCandidate] = {}
    for keyword in RESEARCH_KEYWORDS:
        for q_type in ("1", "2", "0"):
            for page_no in range(1, 4):
                params = {
                    "industryCode": "*",
                    "pageSize": "100",
                    "industry": "*",
                    "rating": "*",
                    "ratingChange": "*",
                    "beginTime": "2024-01-01",
                    "endTime": "2026-12-31",
                    "pageNo": str(page_no),
                    "pageNumber": str(page_no),
                    "pageNum": str(page_no),
                    "qType": q_type,
                    "orgCode": "",
                    "code": "*",
                    "keyword": keyword,
                }
                try:
                    response = session.get(EASTMONEY_REPORT_API, params=params, timeout=REQUEST_TIMEOUT)
                    response.raise_for_status()
                    data = parse_json_or_jsonp(response.text)
                except Exception:
                    continue
                records = data.get("data") or data.get("Data") or []
                if isinstance(records, dict):
                    records = records.get("list") or records.get("List") or []
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    candidate = normalize_eastmoney_record(record)
                    if candidate is None:
                        continue
                    existing = candidates.get(candidate.report_id)
                    if existing is None or candidate.score > existing.score:
                        candidates[candidate.report_id] = candidate
                if len(records) < 100:
                    break
            time.sleep(0.2)
    return sorted(candidates.values(), key=lambda item: item.score, reverse=True)


def research_filename(index: int, org: str, published_at: str, title: str) -> str:
    index_part = f"{index:02d}" if index > 0 else "00"
    date_part = re.sub(r"[^0-9]", "", published_at or "")[:8] or "unknown"
    org_part = safe_filename(org or "公开研报", max_stem_len=16)
    title_part = safe_filename(short_title(title), max_stem_len=MAX_TITLE_FILENAME_LEN)
    return f"AI算力产业链研报_{index_part}_{org_part}_{date_part}_{title_part}.pdf"


def seed_research_candidates() -> list[ReportCandidate]:
    candidates = []
    for seed in RESEARCH_SEEDS:
        info_match = re.search(r"(AP\d+)", seed["pdf_url"])
        report_id = f"research_{info_match.group(1)}" if info_match else f"research_seed_{len(candidates) + 1}"
        candidates.append(
            ReportCandidate(
                report_id=report_id,
                kind="research",
                year=seed.get("year", ""),
                title=seed["title"],
                source_site=seed["source_site"],
                source_url=seed["source_url"],
                pdf_url=seed["pdf_url"],
                local_path=RESEARCH_DIR / research_filename(0, seed["org"], seed["published_at"], seed["title"]),
                published_at=seed["published_at"],
                score=research_score(seed["title"]) + 2,
            )
        )
    return candidates


def find_research_candidates(session: requests.Session, max_research: int) -> list[ReportCandidate]:
    by_id: dict[str, ReportCandidate] = {}
    for candidate in query_eastmoney_research(session) + seed_research_candidates():
        existing = by_id.get(candidate.report_id)
        if existing is None or candidate.score > existing.score:
            by_id[candidate.report_id] = candidate
    sorted_candidates = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
    selected = sorted_candidates[:max_research]
    for index, candidate in enumerate(selected, start=1):
        org = extract_org_from_research_candidate(candidate)
        candidate.local_path = RESEARCH_DIR / research_filename(index, org, candidate.published_at, candidate.title)
    return selected


def extract_org_from_research_candidate(candidate: ReportCandidate) -> str:
    match = re.search(r"AI算力产业链研报_\d+_([^_]+)_", candidate.local_path.name)
    if match:
        return match.group(1)
    for seed in RESEARCH_SEEDS:
        if seed["title"] == candidate.title:
            return seed.get("org", "公开研报")
    return "公开研报"


def try_extract_pdf_links_from_html(session: requests.Session, page_url: str) -> list[str]:
    try:
        response = session.get(page_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if ".pdf" in href.lower():
            links.append(urljoin(page_url, href))
    return list(dict.fromkeys(links))


def plan_candidate(candidate: ReportCandidate) -> str:
    return (
        f"PLAN {candidate.kind:<8} {candidate.title} -> "
        f"{relative_path(candidate.local_path)} [{candidate.pdf_url}]"
    )


def run_annual(args: argparse.Namespace, session: requests.Session, manifest: ManifestStore) -> int:
    companies = load_companies()
    validate_companies(companies)
    failures = 0
    for company in companies:
        print(f"Searching annual report: {company.company} ({company.stock_code})")
        try:
            candidate = find_latest_annual_candidate(company, session)
        except Exception as exc:
            candidate = None
            error = str(exc)
        else:
            error = ""
        if candidate is None:
            failures += 1
            missing = ReportCandidate(
                report_id=f"annual_{company.stock_code}_missing",
                kind="annual",
                company=company.company,
                stock_code=company.stock_code,
                title=f"{company.company} 最新可用年度报告",
                source_site="巨潮资讯",
                source_url="https://www.cninfo.com.cn/new/index?lang=zh",
                pdf_url="",
                local_path=ANNUAL_DIR / f"{safe_filename(company.company)}_missing.pdf",
            )
            if args.dry_run:
                print(f"FAIL no candidate found: {company.company} {error}")
            else:
                manifest.upsert(manifest_row(missing, status="failed", error=error or "no annual report candidate found"))
            continue
        if args.dry_run:
            print(plan_candidate(candidate))
        else:
            print(download_report(session, candidate, manifest, refresh=args.refresh))
        time.sleep(0.3)
    return failures


def run_research(args: argparse.Namespace, session: requests.Session, manifest: ManifestStore) -> int:
    try:
        candidates = find_research_candidates(session, args.max_research)
    except Exception as exc:
        print(f"Research search failed: {exc}")
        candidates = seed_research_candidates()[: args.max_research]
        for index, candidate in enumerate(candidates, start=1):
            candidate.local_path = RESEARCH_DIR / research_filename(index, extract_org_from_research_candidate(candidate), candidate.published_at, candidate.title)

    if len(candidates) < args.max_research:
        error = f"only found {len(candidates)} public research PDFs, expected {args.max_research}"
        print(f"FAIL {error}")
        if not args.dry_run:
            missing = ReportCandidate(
                report_id="research_missing",
                kind="research",
                title="AI算力产业链公开研报缺口",
                source_site="公开搜索",
                source_url="",
                pdf_url="",
                local_path=RESEARCH_DIR / "AI算力产业链研报_missing.pdf",
            )
            manifest.upsert(manifest_row(missing, status="failed", error=error))
        return args.max_research - len(candidates)

    for candidate in candidates:
        html_pdf_links = try_extract_pdf_links_from_html(session, candidate.source_url)
        candidate.alternate_pdf_urls.extend(url for url in html_pdf_links if url not in candidate.all_pdf_urls)
        if args.dry_run:
            print(plan_candidate(candidate))
        else:
            print(download_report(session, candidate, manifest, refresh=args.refresh))
        time.sleep(0.3)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare stage-1 annual and research PDFs.")
    parser.add_argument("--kind", choices=("annual", "research", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Query and print candidates without downloading or writing manifest rows.")
    parser.add_argument("--refresh", action="store_true", help="Re-download PDFs even when a valid local file exists.")
    parser.add_argument("--max-research", type=int, default=DEFAULT_MAX_RESEARCH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_directories()
    write_default_companies()
    manifest = ManifestStore()
    session = make_session()

    failures = 0
    if args.kind in {"annual", "all"}:
        failures += run_annual(args, session, manifest)
    if args.kind in {"research", "all"}:
        failures += run_research(args, session, manifest)

    if failures:
        print(f"Completed with {failures} missing or failed items.")
        return 1
    print("Stage-1 data preparation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
