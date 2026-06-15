-- ============================================================================
-- 检索系统数据库迁移脚本 (v001)
-- 功能: 创建检索系统所需的所有表、索引及全文搜索基础设施。
-- 依赖扩展: pg_search (BM25全文检索) + vector (HNSW向量索引)
-- ============================================================================

-- 启用 pg_search 扩展，提供 BM25 全文搜索能力（基于 ParadigmDB）
CREATE EXTENSION IF NOT EXISTS pg_search;
-- 启用 vector 扩展，提供向量嵌入存储和 HNSW 近似最近邻搜索
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 1. rag_chunks —— 知识库文档片段表
--    存储从原始文档中切分出的文本块，支持 BM25 全文搜索 + 向量语义搜索。
-- ============================================================================
CREATE TABLE IF NOT EXISTS rag_chunks (
    id bigserial PRIMARY KEY,                    -- 自增主键
    chunk_id text NOT NULL UNIQUE,               -- 外部唯一标识（如 UUID）
    report_id text NOT NULL DEFAULT '',           -- 所属报告 ID
    kind text NOT NULL DEFAULT '',                -- 块类别（如 summary / detail）
    company text NOT NULL DEFAULT '',             -- 关联公司名称
    source_title text NOT NULL DEFAULT '',        -- 来源文档标题
    source_url text NOT NULL DEFAULT '',          -- 来源文档 URL
    source_tier text NOT NULL DEFAULT '',         -- 来源层级（如 primary / secondary）
    source_type text NOT NULL DEFAULT '',         -- 来源类型（如 annual_report / news）
    page text NOT NULL DEFAULT '',                -- 页码
    section text NOT NULL DEFAULT '',             -- 章节
    content_type text NOT NULL DEFAULT 'text',    -- 内容类型（text / table / image）
    table_id text NOT NULL DEFAULT '',            -- 若为表格块，对应的表格 ID
    text text NOT NULL,                           -- 原始文本内容
    search_text text NOT NULL,                    -- 用于 BM25 检索的文本（经清洗/分词优化）
    semantic_text text NOT NULL,                  -- 用于向量嵌入的文本（语义保留，长度可控）
    token_counts jsonb NOT NULL DEFAULT '{}'::jsonb,  -- 各模型 token 计数记录
    token_count integer NOT NULL DEFAULT 0,       -- 总 token 数
    content_hash text NOT NULL,                   -- 内容哈希（用于去重和变更检测）
    embedding vector(2048),                       -- 语义嵌入向量（2048 维）
    embedding_status text NOT NULL DEFAULT 'missing'  -- 嵌入状态: missing/stale/ready/failed
        CHECK (embedding_status IN ('missing', 'stale', 'ready', 'failed')),
    embedding_model text NOT NULL DEFAULT '',      -- 生成嵌入的模型名称
    embedded_at timestamptz,                       -- 嵌入生成时间
    created_at timestamptz NOT NULL DEFAULT now(), -- 记录创建时间
    updated_at timestamptz NOT NULL DEFAULT now()  -- 记录更新时间
);

-- ============================================================================
-- 2. research_claims —— 研究断言/事实声明表
--    存储从文档中提取的结构化事实声明（公司指标、事件等），含多维度属性。
-- ============================================================================
CREATE TABLE IF NOT EXISTS research_claims (
    id bigserial PRIMARY KEY,                    -- 自增主键
    claim_id text NOT NULL UNIQUE,               -- 外部唯一标识（如 UUID）
    claim_type text NOT NULL DEFAULT '',          -- 声明类型（如 metric / event / status）
    topic text NOT NULL DEFAULT '',               -- 所属主题/细分领域
    claim_text text NOT NULL,                     -- 声明原文
    companies text[] NOT NULL DEFAULT ARRAY[]::text[],  -- 关联公司列表（数组）
    mechanism text NOT NULL DEFAULT '',           -- 影响机制/驱动因素
    direction text NOT NULL DEFAULT '',           -- 影响方向（up / down / neutral）
    horizon text NOT NULL DEFAULT '',             -- 时间跨度（如 short-term / long-term）
    metric text NOT NULL DEFAULT '',              -- 指标名称（如 revenue / market_share）
    value text NOT NULL DEFAULT '',               -- 指标数值
    unit text NOT NULL DEFAULT '',                -- 指标单位
    source_report_id text NOT NULL DEFAULT '',    -- 来源报告 ID
    source_title text NOT NULL DEFAULT '',        -- 来源文档标题
    page text NOT NULL DEFAULT '',                -- 页码
    section text NOT NULL DEFAULT '',             -- 章节
    source_tier text NOT NULL DEFAULT '',         -- 来源层级
    evidence_span text NOT NULL DEFAULT '',        -- 原文证据片段（引用位置）
    confidence text NOT NULL DEFAULT '',           -- 置信度
    as_of_date text NOT NULL DEFAULT '',           -- 数据截止日期
    exposure_level text NOT NULL DEFAULT '',       -- 风险暴露等级
    review_status text NOT NULL DEFAULT 'auto',    -- 审核状态: auto / reviewed / flagged
    reviewer_note text NOT NULL DEFAULT '',         -- 审核人备注
    quality_flags text NOT NULL DEFAULT '',         -- 质量标记（多个标记用逗号分隔）
    conflict_group_id text NOT NULL DEFAULT '',     -- 冲突分组 ID（互相矛盾的声明归入同组）
    reviewed_at timestamptz,                        -- 审核时间
    reviewer text NOT NULL DEFAULT '',              -- 审核人
    semantic_text text NOT NULL,                    -- 用于语义嵌入的文本
    content_hash text NOT NULL,                     -- 内容哈希
    embedding vector(2048),                         -- 语义嵌入向量
    embedding_status text NOT NULL DEFAULT 'missing'
        CHECK (embedding_status IN ('missing', 'stale', 'ready', 'failed')),
    embedding_model text NOT NULL DEFAULT '',
    embedded_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- ============================================================================
-- 3. claim_reviews —— 声明审核记录表
--    存储对 research_claims 中声明的每次人工审核或自动修正记录。
-- ============================================================================
CREATE TABLE IF NOT EXISTS claim_reviews (
    id bigserial PRIMARY KEY,                    -- 自增主键
    claim_id text NOT NULL,                       -- 被审声明的 ID
    updates jsonb NOT NULL,                       -- 审核变更内容（JSON 格式记录字段级变更）
    review_hash text NOT NULL UNIQUE,             -- 审核记录哈希（防重复提交）
    reviewer text NOT NULL DEFAULT 'frontend',    -- 审核来源（默认前端）
    created_at timestamptz NOT NULL DEFAULT now() -- 审核时间
);

-- ============================================================================
-- 4. segment_dossiers —— 领域/主题档案表
--    存储按主题聚合的知识档案，每个主题一个综合档案记录。
-- ============================================================================
CREATE TABLE IF NOT EXISTS segment_dossiers (
    id bigserial PRIMARY KEY,                    -- 自增主键
    topic text NOT NULL UNIQUE,                   -- 主题名称（唯一）
    summary text NOT NULL DEFAULT '',             -- 主题摘要
    payload jsonb NOT NULL,                       -- 完整档案数据（JSON）
    semantic_text text NOT NULL,                  -- 用于语义嵌入的文本
    content_hash text NOT NULL,                   -- 内容哈希
    embedding vector(2048),                       -- 语义嵌入向量
    embedding_status text NOT NULL DEFAULT 'missing'
        CHECK (embedding_status IN ('missing', 'stale', 'ready', 'failed')),
    embedding_model text NOT NULL DEFAULT '',
    embedded_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- ============================================================================
-- 5. retrieval_builds —— 检索构建记录表
--    记录每次检索索引构建/重建的任务元信息，用于追踪和审计。
-- ============================================================================
CREATE TABLE IF NOT EXISTS retrieval_builds (
    id bigserial PRIMARY KEY,                    -- 自增主键
    build_kind text NOT NULL,                     -- 构建类型（如 full / incremental）
    corpus_hash text NOT NULL DEFAULT '',         -- 语料库哈希（判断语料是否变化）
    record_count integer NOT NULL DEFAULT 0,      -- 处理的记录数
    embedding_model text NOT NULL DEFAULT '',     -- 所用嵌入模型
    embedding_dimensions integer,                 -- 向量维度
    status text NOT NULL,                         -- 构建状态（running / completed / failed）
    details jsonb NOT NULL DEFAULT '{}'::jsonb,   -- 额外详情（耗时、错误等）
    schema_version text NOT NULL DEFAULT '001_postgres_retrieval',  -- 表结构版本
    created_at timestamptz NOT NULL DEFAULT now(), -- 构建开始时间
    completed_at timestamptz                       -- 构建完成时间
);

-- ============================================================================
-- 6. BTREE / GIN 索引 —— 用于标量字段的精确匹配与排序
-- ============================================================================

-- rag_chunks: 按公司、来源类型、类别快速过滤
CREATE INDEX IF NOT EXISTS rag_chunks_company_idx ON rag_chunks(company);
CREATE INDEX IF NOT EXISTS rag_chunks_source_type_idx ON rag_chunks(source_type);
CREATE INDEX IF NOT EXISTS rag_chunks_kind_idx ON rag_chunks(kind);

-- research_claims: 按主题、类型、审核状态过滤，以及公司数组的 GIN 索引
CREATE INDEX IF NOT EXISTS research_claims_topic_idx ON research_claims(topic);
CREATE INDEX IF NOT EXISTS research_claims_type_idx ON research_claims(claim_type);
CREATE INDEX IF NOT EXISTS research_claims_review_status_idx ON research_claims(review_status);
CREATE INDEX IF NOT EXISTS research_claims_companies_idx ON research_claims USING gin(companies);

-- claim_reviews: 按 claim_id 检索最新审核记录
CREATE INDEX IF NOT EXISTS claim_reviews_claim_id_idx ON claim_reviews(claim_id, created_at DESC);

-- segment_dossiers: 按主题快速检索
CREATE INDEX IF NOT EXISTS segment_dossiers_topic_idx ON segment_dossiers(topic);

-- ============================================================================
-- 7. HNSW 向量索引 —— 用于语义相似度近似搜索（余弦距离）
--    需要 pg_vector 的 HNSW 索引支持。仅索引状态为 'ready' 的记录。
-- ============================================================================

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
ON rag_chunks
USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
WHERE embedding_status = 'ready';

CREATE INDEX IF NOT EXISTS research_claims_embedding_hnsw_idx
ON research_claims
USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
WHERE embedding_status = 'ready';

CREATE INDEX IF NOT EXISTS segment_dossiers_embedding_hnsw_idx
ON segment_dossiers
USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops)
WHERE embedding_status = 'ready';

-- ============================================================================
-- 8. BM25 全文搜索索引（pg_search）—— 基于关键词的文本检索
--    使用 jieba 分词器支持中文分词，覆盖 search_text 及主要过滤字段。
-- ============================================================================

CREATE INDEX IF NOT EXISTS rag_chunks_bm25_idx
ON rag_chunks
USING bm25 (
    id,
    (search_text::pdb.jieba),   -- 中文分词全文检索字段
    company,                     -- 过滤字段
    source_type,                 -- 过滤字段
    source_tier,                 -- 过滤字段
    kind,                        -- 过滤字段
    content_type                 -- 过滤字段
)
WITH (key_field = 'id');