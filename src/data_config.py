"""Configuration helpers for the AI compute industry data foundation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from src.extraction_schema import normalize_name


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"
COMPANIES_EXTENDED_CSV = METADATA_DIR / "companies_extended.csv"
RESEARCH_KEYWORDS_CSV = METADATA_DIR / "research_keywords.csv"
INDUSTRY_SOURCES_CSV = METADATA_DIR / "industry_sources.csv"

COMPANY_FIELDS = [
    "company",
    "stock_code",
    "market",
    "chain_segment",
    "aliases",
    "is_core_company",
    "notes",
]

RESEARCH_KEYWORD_FIELDS = ["keyword", "weight", "notes"]

INDUSTRY_SOURCE_FIELDS = [
    "report_id",
    "title",
    "source_site",
    "source_url",
    "pdf_url",
    "year",
    "published_at",
    "source_tier",
    "source_type",
    "topics",
]


DEFAULT_COMPANIES = [
    ("浪潮信息", "000977", "SZ", "AI服务器", "浪潮电子信息产业股份有限公司;Inspur", "服务器整机/AI基础设施"),
    ("中科曙光", "603019", "SH", "AI服务器/高性能计算", "曙光信息产业股份有限公司;曙光", "HPC/智算中心"),
    ("工业富联", "601138", "SH", "AI服务器/智能制造", "富士康工业互联网股份有限公司;FII", "云服务设备/AI服务器制造"),
    ("中际旭创", "300308", "SZ", "光模块", "中际旭创股份有限公司", "高速光模块"),
    ("新易盛", "300502", "SZ", "光模块", "成都新易盛通信技术股份有限公司", "高速光模块"),
    ("天孚通信", "300394", "SZ", "光器件", "苏州天孚光通信股份有限公司", "光器件/光引擎"),
    ("英维克", "002837", "SZ", "液冷/温控", "深圳市英维克科技股份有限公司", "数据中心温控"),
    ("申菱环境", "301018", "SZ", "液冷/温控", "广东申菱环境系统股份有限公司", "数据中心温控/液冷"),
    ("寒武纪", "688256", "SH", "AI芯片", "中科寒武纪科技股份有限公司;寒武纪-U", "AI训练/推理芯片"),
    ("海光信息", "688041", "SH", "AI芯片/CPU", "海光信息技术股份有限公司", "CPU/DCU"),
    ("紫光股份", "000938", "SZ", "交换机/网络设备", "紫光股份有限公司;新华三;H3C", "算力网络/交换机"),
    ("锐捷网络", "301165", "SZ", "交换机/网络设备", "锐捷网络股份有限公司", "数据中心交换机"),
    ("沪电股份", "002463", "SZ", "PCB", "沪士电子股份有限公司;沪电", "AI服务器PCB"),
    ("深南电路", "002916", "SZ", "PCB", "深南电路股份有限公司", "PCB/封装基板"),
    ("生益科技", "600183", "SH", "CCL/覆铜板", "广东生益科技股份有限公司", "覆铜板/电子材料"),
    ("胜宏科技", "300476", "SZ", "PCB", "胜宏科技(惠州)股份有限公司", "高多层PCB"),
    ("光迅科技", "002281", "SZ", "光模块/光芯片", "武汉光迅科技股份有限公司", "光模块/光芯片"),
    ("华工科技", "000988", "SZ", "光模块/激光设备", "华工科技产业股份有限公司", "光模块/激光设备"),
    ("太辰光", "300570", "SZ", "光器件", "深圳太辰光通信股份有限公司", "光连接器/光器件"),
    ("源杰科技", "688498", "SH", "光芯片", "陕西源杰半导体科技股份有限公司", "激光器芯片"),
    ("奥飞数据", "300738", "SZ", "IDC/数据中心", "广东奥飞数据科技股份有限公司", "IDC服务"),
    ("光环新网", "300383", "SZ", "IDC/云计算", "北京光环新网科技股份有限公司", "IDC/云服务"),
    ("数据港", "603881", "SH", "IDC/数据中心", "上海数据港股份有限公司", "批发型IDC"),
    ("宝信软件", "600845", "SH", "IDC/工业软件", "上海宝信软件股份有限公司", "IDC/工业软件"),
    ("科华数据", "002335", "SZ", "数据中心电源/UPS", "科华数据股份有限公司", "UPS/数据中心电源"),
    ("新雷能", "300593", "SZ", "电源", "北京新雷能科技股份有限公司", "高可靠电源"),
    ("欧陆通", "300870", "SZ", "服务器电源", "深圳欧陆通电子股份有限公司", "电源适配器/服务器电源"),
    ("飞荣达", "300602", "SZ", "散热/屏蔽材料", "深圳市飞荣达科技股份有限公司", "散热材料/电磁屏蔽"),
    ("高澜股份", "300499", "SZ", "液冷/热管理", "广州高澜节能技术股份有限公司", "液冷/热管理"),
    ("澜起科技", "688008", "SH", "内存接口芯片/AI芯片", "澜起科技股份有限公司", "内存接口/互连芯片"),
]

DEFAULT_RESEARCH_KEYWORDS = [
    ("AI算力产业链", 5, "总览"),
    ("算力深度报告", 4, "总览"),
    ("国产算力", 5, "国产替代"),
    ("AI服务器 光模块 液冷", 5, "跨环节"),
    ("AIDC 智算中心", 4, "数据中心"),
    ("数据中心液冷 温控", 4, "液冷温控"),
    ("光模块 800G 1.6T", 4, "高速光模块"),
    ("CPO 硅光 光模块", 4, "光互连"),
    ("GPU AI芯片 国产", 4, "芯片"),
    ("PCB CCL AI服务器", 3, "PCB/材料"),
    ("交换机 以太网 算力网络", 3, "算力网络"),
    ("服务器电源 算力", 3, "电源"),
    ("DeepSeek 算力产业链", 4, "应用催化"),
]

DEFAULT_INDUSTRY_SOURCES = [
    {
        "report_id": "industry_caict_green_compute_2025",
        "title": "绿色算力发展研究报告（2025年）",
        "source_site": "中国信通院",
        "source_url": "https://gma.caict.ac.cn/plat/news/full-collection-of-blue-books-and-report-published-by-caict-in-2025",
        "pdf_url": "https://www.caict.ac.cn/kxyj/qwfb/ztbg/202507/P020250724397149741659.pdf",
        "year": "2025",
        "published_at": "2025-07",
        "source_tier": "1",
        "source_type": "authority_whitepaper",
        "topics": "绿色算力;液冷;数据中心;算电协同",
    },
    {
        "report_id": "industry_caict_compute_service_provider_2025",
        "title": "算力中心服务商分析报告（2025年）",
        "source_site": "中国信通院",
        "source_url": "https://gma.caict.ac.cn/plat/news/full-collection-of-blue-books-and-report-published-by-caict-in-2025",
        "pdf_url": "https://www.caict.ac.cn/kxyj/qwfb/ztbg/202507/P020250707617798232823.pdf",
        "year": "2025",
        "published_at": "2025-07",
        "source_tier": "1",
        "source_type": "authority_whitepaper",
        "topics": "算力中心;服务商;智算中心",
    },
    {
        "report_id": "industry_caict_compute_power_coordination_2025",
        "title": "算力电力协同发展研究报告（2025年）",
        "source_site": "中国信通院",
        "source_url": "https://gma.caict.ac.cn/plat/news/full-collection-of-blue-books-and-report-published-by-caict-in-2025",
        "pdf_url": "https://www.caict.ac.cn/kxyj/qwfb/ztbg/202505/P020250509511369626787.pdf",
        "year": "2025",
        "published_at": "2025-05",
        "source_tier": "1",
        "source_type": "authority_whitepaper",
        "topics": "算电协同;绿色电力;数据中心",
    },
    {
        "report_id": "industry_caict_advanced_compute_index_2024",
        "title": "先进计算暨算力发展指数蓝皮书（2024年）",
        "source_site": "中国信通院",
        "source_url": "https://www.caict.ac.cn/kxyj/qwfb/bps/202501/P020250117503621662777.pdf",
        "pdf_url": "https://www.caict.ac.cn/kxyj/qwfb/bps/202501/P020250117503621662777.pdf",
        "year": "2024",
        "published_at": "2025-01",
        "source_tier": "1",
        "source_type": "authority_whitepaper",
        "topics": "算力指数;智能算力;先进计算",
    },
    {
        "report_id": "industry_caict_data_center_2022",
        "title": "数据中心白皮书（2022年）",
        "source_site": "中国信通院",
        "source_url": "https://www.caict.ac.cn/kxyj/qwfb/bps/202204/P020220422707354529853.pdf",
        "pdf_url": "https://www.caict.ac.cn/kxyj/qwfb/bps/202204/P020220422707354529853.pdf",
        "year": "2022",
        "published_at": "2022-04",
        "source_tier": "1",
        "source_type": "authority_whitepaper",
        "topics": "数据中心;液冷;PUE;基础设施",
    },
    {
        "report_id": "industry_caict_ai_core_technology_2021",
        "title": "人工智能核心技术产业白皮书",
        "source_site": "中国信通院",
        "source_url": "https://www.caict.ac.cn/kxyj/qwfb/bps/202104/P020210420614092578238.pdf",
        "pdf_url": "https://www.caict.ac.cn/kxyj/qwfb/bps/202104/P020210420614092578238.pdf",
        "year": "2021",
        "published_at": "2021-04",
        "source_tier": "1",
        "source_type": "authority_whitepaper",
        "topics": "人工智能;AI算力;云边端;智能计算",
    },
    {
        "report_id": "industry_tech_irds_more_moore_2024",
        "title": "IRDS 2024 More Moore",
        "source_site": "IEEE IRDS",
        "source_url": "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_MM.pdf",
        "pdf_url": "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_MM.pdf",
        "year": "2024",
        "published_at": "2024",
        "source_tier": "1",
        "source_type": "technical_roadmap",
        "topics": "semiconductor roadmap;logic scaling;memory;AI accelerators;PPAC;3D integration",
    },
    {
        "report_id": "industry_tech_irds_outside_system_connectivity_2024",
        "title": "IRDS 2024 Outside System Connectivity",
        "source_site": "IEEE IRDS",
        "source_url": "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_OSC.pdf",
        "pdf_url": "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_OSC.pdf",
        "year": "2024",
        "published_at": "2024",
        "source_tier": "1",
        "source_type": "technical_roadmap",
        "topics": "outside system connectivity;optical interconnect;SerDes;bandwidth;AI clusters",
    },
    {
        "report_id": "industry_tech_irds_packaging_tutorial_2024",
        "title": "IRDS 2024 Executive Packaging Tutorial Part 1",
        "source_site": "IEEE IRDS",
        "source_url": "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_EPT-Part1.pdf",
        "pdf_url": "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_EPT-Part1.pdf",
        "year": "2024",
        "published_at": "2024",
        "source_tier": "1",
        "source_type": "technical_roadmap",
        "topics": "advanced packaging;heterogeneous integration;chiplet;HBM;2.5D;3D integration",
    },
    {
        "report_id": "industry_tech_hir_photonics_2023",
        "title": "HIR 2023 Integrated Photonics",
        "source_site": "IEEE EPS HIR",
        "source_url": "https://eps.ieee.org/technology/heterogeneous-integration-roadmap/2023-edition.html",
        "pdf_url": "https://eps.ieee.org/wp-content/uploads/2025/11/ch09_photonics.pdf",
        "year": "2023",
        "published_at": "2023-03",
        "source_tier": "1",
        "source_type": "technical_roadmap",
        "topics": "integrated photonics;CPO;silicon photonics;optical engine;AI interconnect",
    },
    {
        "report_id": "industry_tech_hir_thermal_2023",
        "title": "HIR 2023 Thermal",
        "source_site": "IEEE EPS HIR",
        "source_url": "https://eps.ieee.org/technology/heterogeneous-integration-roadmap/2023-edition.html",
        "pdf_url": "https://eps.ieee.org/wp-content/uploads/2025/11/ch20_thermalfinal.pdf",
        "year": "2023",
        "published_at": "2023-03",
        "source_tier": "1",
        "source_type": "technical_roadmap",
        "topics": "thermal management;2.5D;3D chiplet;liquid cooling;TIM;power density",
    },
    {
        "report_id": "industry_tech_ocp_oai_ubb_2023",
        "title": "OCP OAI Universal Baseboard Base Specification r2.0",
        "source_site": "Open Compute Project",
        "source_url": "https://www.opencompute.org/documents/oai-ubb-base-specification-r2-0-v1-0-20230919-pdf",
        "pdf_url": "https://www.opencompute.org/documents/oai-ubb-base-specification-r2-0-v1-0-20230919-pdf",
        "year": "2023",
        "published_at": "2023-09",
        "source_tier": "1",
        "source_type": "manual_open_specification",
        "topics": "OAI;OAM;AI accelerator baseboard;scale-out;power;liquid cooling;112G PAM4",
    },
    {
        "report_id": "industry_tech_ocp_acs_cold_plate_2024",
        "title": "OCP ACS Liquid Cooling Cold Plate Requirements",
        "source_site": "Open Compute Project",
        "source_url": "https://www.opencompute.org/documents/ocp-acs-liquid-cooling-cold-plate-requirements-pdf",
        "pdf_url": "https://www.opencompute.org/documents/ocp-acs-liquid-cooling-cold-plate-requirements-pdf",
        "year": "2024",
        "published_at": "2024",
        "source_tier": "1",
        "source_type": "manual_open_specification",
        "topics": "liquid cooling;cold plate;CDU;rack manifold;quick disconnect;AI server thermal",
    },
    {
        "report_id": "industry_tech_oif_cei_224g_2022",
        "title": "OIF Next Generation CEI-224G Framework",
        "source_site": "OIF",
        "source_url": "https://www.oiforum.com/wp-content/uploads/OIF-FD-CEI-224G-01.0.pdf",
        "pdf_url": "https://www.oiforum.com/wp-content/uploads/OIF-FD-CEI-224G-01.0.pdf",
        "year": "2022",
        "published_at": "2022-02",
        "source_tier": "1",
        "source_type": "manual_open_specification",
        "topics": "CEI-224G;SerDes;PAM4;CPO;optical engine;switch bandwidth;800G;1.6T",
    },
    {
        "report_id": "industry_tech_ucie_2_whitepaper_2024",
        "title": "UCIe 2.0 Specification Continuing Innovation to Drive an Open Chiplet Ecosystem",
        "source_site": "UCIe Consortium",
        "source_url": "https://www.uciexpress.org/2-0-spec-download",
        "pdf_url": "https://www.uciexpress.org/_files/ugd/0c1418_b6481ec611e24c6e91f1beb743b0c860.pdf",
        "year": "2024",
        "published_at": "2024-08",
        "source_tier": "1",
        "source_type": "open_specification",
        "topics": "UCIe;chiplet;3D packaging;DFx;manageability;advanced packaging",
    },
    {
        "report_id": "industry_tech_ualink_200g_2025",
        "title": "UALink 200G 1.0 Specification",
        "source_site": "UALink Consortium",
        "source_url": "https://ualinkconsortium.org/specification/",
        "pdf_url": "https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink200_Specification_v1.0_Evaluation_Copy.pdf",
        "year": "2025",
        "published_at": "2025-04",
        "source_tier": "1",
        "source_type": "open_specification",
        "topics": "UALink;scale-up interconnect;AI accelerator pod;200G;load-store;switch fabric",
    },
    {
        "report_id": "industry_tech_ultra_ethernet_1_0_2_2026",
        "title": "Ultra Ethernet Specification v1.0.2",
        "source_site": "Ultra Ethernet Consortium",
        "source_url": "https://ultraethernet.org/ultra-ethernet-consortium-uec-launches-specification-1-0-transforming-ethernet-for-ai-and-hpc-at-scale/",
        "pdf_url": "https://ultraethernet.org/wp-content/uploads/sites/20/2026/01/UE-Specification-1.0.2-1.pdf",
        "year": "2026",
        "published_at": "2026-01",
        "source_tier": "1",
        "source_type": "open_specification",
        "topics": "Ultra Ethernet;UET;RDMA;congestion control;AI networking;HPC;scale-out",
    },
    {
        "report_id": "industry_tech_mlperf_training_2019",
        "title": "MLPerf Training Benchmark",
        "source_site": "arXiv",
        "source_url": "https://arxiv.org/abs/1910.01500",
        "pdf_url": "https://arxiv.org/pdf/1910.01500.pdf",
        "year": "2019",
        "published_at": "2019-10",
        "source_tier": "1",
        "source_type": "benchmark_methodology",
        "topics": "MLPerf;training benchmark;AI hardware;time-to-train;accelerator performance",
    },
    {
        "report_id": "industry_tech_mlperf_inference_2019",
        "title": "MLPerf Inference Benchmark",
        "source_site": "arXiv",
        "source_url": "https://arxiv.org/abs/1911.02549",
        "pdf_url": "https://arxiv.org/pdf/1911.02549.pdf",
        "year": "2019",
        "published_at": "2019-11",
        "source_tier": "1",
        "source_type": "benchmark_methodology",
        "topics": "MLPerf;inference benchmark;latency;throughput;power;AI accelerator",
    },
    {
        "report_id": "industry_tech_flashattention_2022",
        "title": "FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "source_site": "arXiv",
        "source_url": "https://arxiv.org/abs/2205.14135",
        "pdf_url": "https://arxiv.org/pdf/2205.14135.pdf",
        "year": "2022",
        "published_at": "2022-05",
        "source_tier": "2",
        "source_type": "technical_paper",
        "topics": "FlashAttention;HBM;SRAM;IO-aware;Transformer;training efficiency;memory bandwidth",
    },
    {
        "report_id": "industry_tech_pagedattention_vllm_2023",
        "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
        "source_site": "arXiv",
        "source_url": "https://arxiv.org/abs/2309.06180",
        "pdf_url": "https://arxiv.org/pdf/2309.06180.pdf",
        "year": "2023",
        "published_at": "2023-09",
        "source_tier": "2",
        "source_type": "technical_paper",
        "topics": "PagedAttention;vLLM;KV cache;LLM serving;inference throughput;memory management",
    },
    {
        "report_id": "industry_tech_fire_flyer_ai_hpc_2024",
        "title": "Fire-Flyer AI-HPC Cost-Effective Software-Hardware Co-Design for Deep Learning",
        "source_site": "arXiv",
        "source_url": "https://arxiv.org/abs/2408.14158",
        "pdf_url": "https://arxiv.org/pdf/2408.14158.pdf",
        "year": "2024",
        "published_at": "2024-08",
        "source_tier": "2",
        "source_type": "technical_paper",
        "topics": "AI-HPC;cluster architecture;PCIe A100;cost efficiency;allreduce;storage-compute network",
    },
    {
        "report_id": "industry_tech_deepseek_v3_2025",
        "title": "DeepSeek-V3 Technical Report",
        "source_site": "arXiv",
        "source_url": "https://arxiv.org/abs/2412.19437",
        "pdf_url": "https://arxiv.org/pdf/2412.19437.pdf",
        "year": "2025",
        "published_at": "2025-02",
        "source_tier": "2",
        "source_type": "model_technical_report",
        "topics": "DeepSeek-V3;MoE;MLA;H800;training efficiency;inference efficiency;multi-token prediction",
    },
    {
        "report_id": "industry_tech_heterogeneous_chiplets_ai_2024",
        "title": "Challenges and Opportunities to Enable Large-Scale Computing via Heterogeneous Chiplets",
        "source_site": "arXiv",
        "source_url": "https://arxiv.org/abs/2311.16417",
        "pdf_url": "https://arxiv.org/pdf/2311.16417.pdf",
        "year": "2024",
        "published_at": "2024-03",
        "source_tier": "2",
        "source_type": "technical_paper",
        "topics": "heterogeneous chiplets;AI workloads;chiplet interface;packaging;security;programming model",
    },
]


@dataclass(frozen=True)
class Company:
    company: str
    stock_code: str
    market: str
    chain_segment: str
    aliases: tuple[str, ...] = ()
    is_core_company: bool = True
    notes: str = ""


@dataclass(frozen=True)
class IndustrySource:
    report_id: str
    title: str
    source_site: str
    source_url: str
    pdf_url: str
    year: str
    published_at: str
    source_tier: str
    source_type: str
    topics: str


def ensure_metadata_dir() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def _split_aliases(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[;；|]", value or "") if item.strip())


def _boolish(value: str) -> bool:
    return str(value or "").strip().casefold() not in {"", "0", "false", "no", "否"}


def write_default_companies(path: Path = COMPANIES_EXTENDED_CSV) -> None:
    if path.exists():
        return
    ensure_metadata_dir()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COMPANY_FIELDS)
        writer.writeheader()
        for company, stock_code, market, segment, aliases, notes in DEFAULT_COMPANIES:
            writer.writerow(
                {
                    "company": company,
                    "stock_code": stock_code,
                    "market": market,
                    "chain_segment": segment,
                    "aliases": aliases,
                    "is_core_company": "true",
                    "notes": notes,
                }
            )


def load_companies(path: Path = COMPANIES_EXTENDED_CSV) -> list[Company]:
    write_default_companies(path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            Company(
                company=row["company"].strip(),
                stock_code=row["stock_code"].strip(),
                market=row["market"].strip().upper(),
                chain_segment=row["chain_segment"].strip(),
                aliases=_split_aliases(row.get("aliases", "")),
                is_core_company=_boolish(row.get("is_core_company", "true")),
                notes=row.get("notes", "").strip(),
            )
            for row in reader
        ]


def validate_companies(
    companies: list[Company],
    *,
    expected_count: int | None = 30,
    require_aliases: bool = True,
) -> None:
    if expected_count is not None and len(companies) != expected_count:
        raise ValueError(f"Expected {expected_count} target companies, got {len(companies)}")
    seen_codes: set[str] = set()
    seen_names: set[str] = set()
    for company in companies:
        if not company.company:
            raise ValueError("Company name is required")
        if not re.fullmatch(r"\d{6}", company.stock_code):
            raise ValueError(f"Invalid stock code for {company.company}: {company.stock_code}")
        if company.stock_code in seen_codes:
            raise ValueError(f"Duplicate stock code: {company.stock_code}")
        if company.company in seen_names:
            raise ValueError(f"Duplicate company name: {company.company}")
        if company.market not in {"SZ", "SH"}:
            raise ValueError(f"Invalid market for {company.company}: {company.market}")
        if not company.chain_segment:
            raise ValueError(f"Missing chain segment for {company.company}")
        if require_aliases and not company.aliases:
            raise ValueError(f"Missing aliases for {company.company}")
        seen_codes.add(company.stock_code)
        seen_names.add(company.company)


def core_company_alias_index(path: Path = COMPANIES_EXTENDED_CSV) -> dict[str, str]:
    companies = [company for company in load_companies(path) if company.is_core_company]
    index: dict[str, str] = {}
    for company in companies:
        names = (company.company, *company.aliases)
        for name in names:
            index[normalize_name(name, "Company")] = company.company
    return index


def write_default_research_keywords(path: Path = RESEARCH_KEYWORDS_CSV) -> None:
    if path.exists():
        return
    ensure_metadata_dir()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESEARCH_KEYWORD_FIELDS)
        writer.writeheader()
        for keyword, weight, notes in DEFAULT_RESEARCH_KEYWORDS:
            writer.writerow({"keyword": keyword, "weight": weight, "notes": notes})


def load_research_keywords(path: Path = RESEARCH_KEYWORDS_CSV) -> list[str]:
    write_default_research_keywords(path)
    with path.open(newline="", encoding="utf-8") as file:
        return [row["keyword"].strip() for row in csv.DictReader(file) if row.get("keyword", "").strip()]


def write_default_industry_sources(path: Path = INDUSTRY_SOURCES_CSV) -> None:
    if path.exists():
        return
    ensure_metadata_dir()
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INDUSTRY_SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(DEFAULT_INDUSTRY_SOURCES)


def load_industry_sources(path: Path = INDUSTRY_SOURCES_CSV) -> list[IndustrySource]:
    write_default_industry_sources(path)
    with path.open(newline="", encoding="utf-8") as file:
        return [
            IndustrySource(
                report_id=row["report_id"].strip(),
                title=row["title"].strip(),
                source_site=row["source_site"].strip(),
                source_url=row["source_url"].strip(),
                pdf_url=row["pdf_url"].strip(),
                year=row.get("year", "").strip(),
                published_at=row.get("published_at", "").strip(),
                source_tier=row.get("source_tier", "").strip(),
                source_type=row.get("source_type", "").strip(),
                topics=row.get("topics", "").strip(),
            )
            for row in csv.DictReader(file)
        ]


def validate_industry_sources(sources: list[IndustrySource]) -> None:
    if not sources:
        raise ValueError("At least one industry source is required")
    seen: set[str] = set()
    for source in sources:
        if not source.report_id or source.report_id in seen:
            raise ValueError(f"Invalid or duplicate industry report_id: {source.report_id}")
        if not source.title:
            raise ValueError(f"Industry source title is required: {source.report_id}")
        if not source.pdf_url.startswith(("http://", "https://")):
            raise ValueError(f"Industry source PDF URL is required: {source.report_id}")
        if source.source_tier not in {"1", "2", "3"}:
            raise ValueError(f"Invalid source_tier for {source.report_id}: {source.source_tier}")
        if not source.source_type:
            raise ValueError(f"Missing source_type for {source.report_id}")
        seen.add(source.report_id)
