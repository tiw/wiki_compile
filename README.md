# KB-Compiler: LLM 知识编译器

基于 Karpathy 模式的知识管理系统——不是 RAG，不是摘要，而是**编译**。

```
传统 RAG:  切片 → 向量 → 检索 → 生成
本系统:    raw输入 → LLM编译 → 结构化wiki → 完整上下文查询
```

## 核心特性

- **增量编译** - 只处理变更的文件，保留已有知识
- **概念提取** - 自动识别关键概念并建立连接
- **矛盾检测** - 发现不同来源之间的分歧
- **查询反馈** - 深度查询结果回流到知识库
- **Obsidian 集成** - 完整的双向链接和图谱视图

## 安装

```bash
# 克隆仓库
cd kb-compiler

# 安装依赖
pip install -e ".[dev]"

# 或使用 uv
uv pip install -e "."
```

## 前置要求

### 方式一：Kimi 开放平台（默认）

1. **获取 API Key**: https://platform.moonshot.cn/console/api-keys
2. **设置环境变量**:
   ```bash
   export KIMI_API_KEY="sk-kimi-..."
   export KB_KIMI_BASE_URL="https://api.moonshot.cn/v1"
   ```

### 方式二：Kimi Code 订阅（推荐）

如果你订阅了 Kimi Code，使用 Anthropic 兼容格式：

1. **安装 anthropic SDK**:
   ```bash
   pip install -e ".[kimi-code]"
   # 或
   pip install anthropic>=0.20.0
   ```

2. **设置环境变量**:
   ```bash
   export KIMI_API_KEY="你的-kimi-code-key"
   export KB_KIMI_BASE_URL="https://api.kimi.com/coding/"
   export KB_KIMI_CODE_MODE="true"
   export KB_KIMI_MODEL="kimi-k2-5"  # 或其他可用模型
   ```

3. **Obsidian CLI** (可选但推荐):
   ```bash
   brew install yuangziweigithub/tap/obsidian-cli
   ```

## 快速开始

### 1. 初始化知识库

```bash
kb-compiler init ~/KnowledgeBase --vault ~/Documents/ObsidianVault
```

目录结构:
```
~/KnowledgeBase/
├── raw/              # 原始文档（你只负责丢进去）
│   ├── articles/     # 网页文章
│   ├── papers/       # PDF论文
│   └── inbox/        # 快速捕获
├── wiki/             # LLM编译产物（不要手动编辑）
│   ├── INDEX.md      # 全局索引
│   └── concepts/     # 概念文章
├── output/           # 查询输出/回溯
└── _meta/            # 编译状态
```

### 2. 摄入文档

```bash
# 摄入网页
kb-compiler ingest https://example.com/article --subdir articles

# 摄入文件
kb-compiler ingest ~/Downloads/paper.pdf --subdir papers

# 摄入整个目录
kb-compiler ingest ~/Documents/notes/

# 快速捕获想法
kb-compiler capture "今天学到的关于Rust所有权的新理解..."
```

### 3. 编译知识

```bash
# 增量编译（只处理变更）
kb-compiler compile

# 强制全量编译
kb-compiler compile --full

# 预览将要编译的文件
kb-compiler compile --dry-run
```

编译过程:
1. 读取 raw/ 目录的所有文档
2. 提取关键概念
3. 为每个概念生成 wiki 文章
4. 建立概念间的 [[wiki-links]] 连接
5. 更新 INDEX.md 索引

### 4. 查询知识

```bash
# 基本查询
kb-compiler query "Rust的所有权系统如何工作？"

# 探索概念
kb-compiler query --explore "Ownership"

# 对比概念
kb-compiler query --compare "Ownership" "Borrowing"

# 保存查询结果
kb-compiler query "内存安全" --save memory_safety
```

### 5. 维护知识库

```bash
# 检查健康状况
kb-compiler lint

# 检测矛盾
kb-compiler lint --contradictions

# 建议新概念
kb-compiler lint --suggest
```

## 概念文章格式

编译后生成的概念文章示例:

```markdown
---
title: 内存安全
sources: ["rust_ownership.md", "cpp_memory.md"]
related: [[Ownership]] [[Borrowing]] [[Lifetime]]
last_compiled: 2026-04-04T10:30:00
---

## 摘要
内存安全是编程语言防止内存错误（如 use-after-free、buffer overflow）的属性。

## 关键事实
- Rust 在编译期保证内存安全，零运行时开销
- 70% 的安全漏洞与内存安全相关（Microsoft, 2019）
- C/C++ 需要手动管理或使用智能指针

## 来源详情
### Rust Book
Rust 通过所有权系统实现内存安全...

### C++ Core Guidelines
C++ 依赖 RAII 和智能指针...

## 矛盾点
- Rust 认为 GC 有运行时开销 unacceptable
- Go/Java 认为 GC 的复杂度 trade-off 可接受

## 开放问题
- 如何在系统编程中完全消除 unsafe 代码？

## 相关概念
See also: [[Ownership]], [[Borrowing]], [[Garbage Collection]]
```

## 配置

创建配置文件 `~/.config/kb-compiler/config.yaml`:

```yaml
# Kimi API 配置
kimi_api_key: "your-api-key"
kimi_model: "moonshot-v1-128k"  # 或 moonshot-v1-32k/8k

# 知识库路径
kb_root: "~/KnowledgeBase"

# Obsidian 配置
obsidian_vault_path: "~/Documents/ObsidianVault"

# 编译设置
incremental_compile: true
max_concurrent_requests: 3
```

## 命令参考

| 命令 | 说明 |
|------|------|
| `init` | 初始化知识库目录 |
| `ingest` | 摄入文档（URL/文件/目录） |
| `capture` | 快速捕获想法 |
| `compile` | 编译 raw → wiki |
| `query` | 查询知识库 |
| `lint` | 检查和维护 |
| `stats` | 统计信息 |

## 工作流建议

### 日常收集
```bash
# 看到好文章
kb-compiler ingest https://... --subdir articles

# 突然有想法
kb-compiler capture

# 每周整理
kb-compiler compile
```

### 深度研究
```bash
# 1. 收集资料
kb-compiler ingest paper1.pdf paper2.pdf

# 2. 编译
kb-compiler compile

# 3. 探索查询
kb-compiler query --explore "Neural Networks"

# 4. 保存有价值的查询为新概念
# (手动将 output/query_*.md 移动到 wiki/concepts/)

# 5. 定期维护
kb-compiler lint --contradictions --suggest
```

## 与其他工具集成

### Obsidian
编译后的 wiki 目录可直接用 Obsidian 打开，享受:
- 图谱视图 (Graph View)
- 反向链接 (Backlinks)
- 快速跳转

### 浏览器插件
使用 Obsidian Web Clipper 配合 `kb-compiler ingest` 实现一键保存。

## License

MIT
