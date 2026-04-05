# PRD: qmd 混合检索引擎 — 为 kb-compiler 引入 BM25 + Vector + RRF 检索层

## Problem Statement

当前 `kb-compiler` 的查询层（`QueryEngine`）基于正则解析 `INDEX.md` 和简单的关键词重叠评分。当 wiki 中的概念数量增长到 50 个以上时，这种检索方式暴露出三个核心问题：

1. **语义召回缺失**：用户查询"全局变暖"无法召回包含"气候变化"的概念文章，因为系统只匹配字面关键词。
2. **排序质量差**：没有 BM25 相关性评分，也没有向量语义相似度，导致返回结果的相关性不可控。
3. **扩展瓶颈**：随着 wiki 规模增长，正则扫描 `INDEX.md` + 全文文件 IO 的性能和精度都会急剧下降。

用户需要一个能够支撑"中大规模知识库"的检索引擎，同时保持 kb-compiler 一贯的**本地优先、轻量、零外部服务**的设计理念。

## Solution

引入 **qmd**（query-markdown）混合检索引擎，作为 `kb-compiler` 查询层的默认实现。qmd 将基于以下技术栈：

- **BM25**：SQLite 原生 FTS5 全文检索
- **Vector**：`sqlite-vec` 本地向量检索（单文件、零进程）
- **RRF**：Reciprocal Rank Fusion 融合排序
- **可选 Rerank**：`flashrank` 或 LLM pointwise 精排（默认关闭）

同时提供独立的 `qmd` CLI 入口和 MCP Server 的预留接口（MCP Server 本身为后续独立 PRD）。

## User Stories

1. 作为一位研究者，我想要通过语义搜索查询知识库，使得即使查询词和文章用词不同，也能召回相关概念。
2. 作为一位开发者，我想要 kb-compiler 的查询在概念数量超过 50 时仍然快速准确，使得我不必手动维护复杂的目录结构。
3. 作为一位 Obsidian 用户，我想要编译后的 wiki 自动生成可搜索的向量索引，使得我可以在终端通过 `qmd` CLI 快速定位知识点。
4. 作为一位本地 LLM 使用者，我想要整个检索流程不依赖任何云端向量数据库，使得我的知识库可以完全离线运行。
5. 作为一位 CLI 用户，我想要 `kb-compiler compile` 自动维护检索索引，使得我不需要手动执行额外的索引命令。
6. 作为一位高级用户，我想要在必要时启用重排序来提升检索精度，使得关键研究查询能拿到最相关的结果。
7. 作为一位 MCP 用户，我想要 qmd  exposing 一个标准化的检索接口，使得未来 Claude Desktop 等 Agent 可以直接调用我的知识库。
8. 作为一位测试驱动开发者，我想要 qmd 的核心模块都有单元测试覆盖，使得后续迭代不会破坏检索行为。
9. 作为一位 kb-compiler 现有用户，我想要升级后旧的 `kb-compiler query` 命令仍然可用但默认走新引擎，使得迁移成本最小化。
10. 作为一位低配置设备用户，我想要 embedding 模型可以回退到 sentence-transformers 本地运行，使得即使没有 Ollama/MLX 也能使用 qmd。

## Implementation Decisions

### 模块划分

本次 PRD 涉及以下新增/修改模块：

- **`EmbeddingProvider`（新，`embeddings.py`）**
  - 协议接口：`embed(texts: list[str]) -> list[list[float]]`
  - 实现：`OllamaEmbeddingProvider`、`SentenceTransformerProvider`
  - 不与现有 `KimiClient` 耦合，保持 LLM 生成与嵌入生成的独立性

- **`Chunker`（新，`chunker.py`）**
  - 接口：`chunk(markdown, source_path, concept_name) -> list[Chunk]`
  - 策略：先按 `##` / `###` 标题拆分，超大块再用滑动窗口（512 tokens / 128 overlap）二次切分

- **`QmdIndexStore`（新，`qmd_store.py`）**
  - 职责：管理 SQLite + FTS5 + sqlite-vec 的 schema、重建、RRF 混合查询
  - 数据库文件位置：`_meta/qmd.db`
  - 核心 SQL：RRF CTE 融合 FTS5 和 vec0 的结果

- **`QmdSearchEngine`（新，`qmd_search.py`）**
  - 职责：编排检索流程 — 嵌入 query → 混合检索 → 可选重排 → 返回上下文
  - 将**替换**现有 `QueryEngine` 成为 `kb-compiler query` 的默认实现

- **`Reranker`（新，`reranker.py`）**
  - 协议接口：`rerank(query, chunks, top_n) -> list[ScoredChunk]`
  - 默认实现：`NullReranker`（透传）
  - 可选实现：`FlashRankReranker`、`LLMReranker`

- **CLI 入口（修改 `main.py` + 新 `qmd_cli.py`）**
  - `kb-compiler query` 默认使用 `QmdSearchEngine`
  - 新增 `qmd` 独立 CLI 入口（`pyproject.toml` console script）
  - `qmd` 子命令：`search`、`index-rebuild`、`stats`

### 架构决策

- **统一查询路径**：取消"小规模用 index.md / 中大规模用 qmd"的双轨设计。`QmdSearchEngine` 成为唯一查询引擎。对于少量概念，索引构建可在数秒内完成，查询开销可忽略。
- **自动索引**：每次 `kb-compiler compile` 完成后，自动触发 qmd 索引的增量/全量重建。提供 `--skip-qmd` 开关跳过。
- **sqlite-vec 为 optional dependency**：放入 `pyproject.toml` 的 `[project.optional-dependencies] qmd`。核心安装不受影响。
- **Rerank 默认关闭**：CLI 查询默认只用 BM25 + Vector + RRF。重排需显式 `--rerank=flashrank` 或 `--rerank=llm`。
- **MCP Server 延后**：本 PRD 只为 qmd 设计内聚的 Python API，MCP Server 作为后续独立 PRD 实现。

### 依赖变更

```toml
[project.optional-dependencies]
qmd = [
    "sqlite-vec>=0.1.0",
    "sentence-transformers>=2.5.0",
    "flashrank>=0.2.0",
]
```

### API 契约

- `QmdSearchEngine.retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]` 为对外统一接口。
- `RetrievedChunk` 包含：`chunk_id`、`concept_name`、`section_header`、`content`、`source_path`、`score`。

## Testing Decisions

- **测试原则**：只测模块的外部行为，不测内部实现细节（如具体 SQL 语句）。
- **必测模块**：`EmbeddingProvider`、`Chunker`、`QmdIndexStore`、`QmdSearchEngine`、`Reranker` 全部需要测试覆盖。
- **测试策略**：
  - `QmdIndexStore`：使用临时目录中的内存/文件 SQLite，创建真实虚拟表进行集成测试。
  - `EmbeddingProvider`：mock HTTP server 测试 Ollama 路径；测试 SentenceTransformerProvider 输出维度。
  - `Chunker`：提供多样 markdown 输入，断言 chunk 边界和元数据正确性。
  - `QmdSearchEngine`：用 mock `EmbeddingProvider` + 内存 `QmdIndexStore` 做端到端检索断言。
  - `Reranker`：验证 `NullReranker` 透传顺序，以及 `FlashRankReranker` 的得分排序行为（如已安装）。

## Out of Scope

- MCP Server 的实现与部署（独立后续 PRD）。
- 多模态检索（图片、PDF 内容向量化）。
- 实时增量更新（当前 `kb-compiler compile` 结束后全量重建索引，增量优化可后续迭代）。
- 分布式/多用户检索场景。
- Web UI 或 Obsidian 插件形态的检索界面。

## Further Notes

- **向后兼容**：旧的 `QueryEngine` 代码在 PR 中保留一个版本后于下个 minor 版本移除。过渡期内若 qmd 索引不存在，提示用户运行 `qmd index-rebuild` 或 `kb-compiler compile`。
- **性能目标**：1000 个 concept（约 5000 个 chunks）的本地查询端到端延迟 < 2s（含 embedding + RRF，不含 rerank）。
- ** Embedding 默认模型**：若有 Ollama，推荐使用 `nomic-embed-text`；否则 fallback 到 `all-MiniLM-L6-v2`（384 维）。
