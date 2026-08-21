# 办公自动化 Agent

一个能**读 Excel / 写 Excel / 查数据库 / 发邮件**的 LangGraph Agent，用自然语言指挥工具完成办公任务，带命令行多轮对话。

## 功能亮点

| 能力 | 工具 | 说明 |
|---|---|---|
| 📊 读 Excel | `read_excel` | 读取 xlsx 返回文本表格 |
| 📈 精确统计 | `aggregate_excel` | pandas 分组聚合（sum/mean/count/max/min），**杜绝 LLM 心算出错** |
| 📝 写报表 | `write_excel` | 自动保存到 `output/`，返回路径 |
| 🗄️ 查数据库 | `query_database` | SQLite 查询，**仅允许 SELECT**（防注入/防破坏） |
| 📧 发邮件 | `send_email` | 未配置 SMTP 时模拟保存到 `outbox/`，如实说明 |
| 📂 感知资源 | `list_office_files` | 列出可用文件，让 Agent 知道能操作什么 |

## 目录结构

```
06-office-agent/
├── create_data.py      # 生成示例数据（可复现）
├── office_agent.py     # 核心：LangGraph Agent + 5 个工具
├── cli.py              # 命令行多轮对话入口
├── requirements.txt
├── data/
│   ├── sales_data.xlsx # 销售流水 Excel（36 行）
│   └── company.db      # SQLite：员工表（含邮箱）+ 产品表
├── outbox/             # 模拟发件箱（Agent 生成）
└── output/             # Agent 生成的报表（Agent 生成）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成示例数据

```bash
python create_data.py
```

### 3. 启动命令行对话

```bash
python cli.py
```

示例指令：
- 华东地区的总销售额是多少？
- 技术部的平均薪资是多少？
- 把每个地区的销售额汇总成 Excel 报表
- 给张伟发一封邮件，提醒他提交市场周报

支持多轮对话（携带上文记忆），输入 `exit` 退出。

### 4. 快速演示

```bash
python office_agent.py
```

跑一遍内置的 4 个演示场景。

## LLM 配置

默认在 `office_agent.py` 中通过环境变量配置：

```python
api_key = os.getenv("OPENAI_API_KEY", "<默认 key>")
base_url = os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1")
model    = os.getenv("MODEL", "deepseek-v4-pro")
```

如需真实发邮件，设置环境变量 `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_PORT`（默认 465 SSL）。

## 架构

```
用户指令
   │
   ▼
┌──────────── LangGraph Agent ────────────┐
│  LLM(bind_tools) ── 决定调用哪个工具      │
│   ┌───────────┐  tool_calls  ┌────────┐  │
│   │   agent   │ ───────────► │  tools │  │
│   └─────┬─────┘              └───┬────┘  │
│         │◄────────────────────────┘      │
└─────────┴───────────────────────────────┘
   │
   ▼
┌────── 工具层（确定性代码）──────┐
│  pandas 读/写/聚合 Excel        │
│  SQLite SELECT 查询（只读）      │
│  SMTP / 模拟发件箱               │
└───────────────────────────────┘
```

**关键设计：计算交给工具，不让 LLM 心算。**

初版让 LLM 读表后自己加总 36 行销售额，结果整张汇总报表数字全错（华北真实 560000 写成 526400）。
加了一个 `aggregate_excel` 工具，由 pandas 精确聚合，Agent 只负责编排和原样引用数字，
结果与 ground truth 完全一致。**「把计算从 LLM 迁移到确定性代码」是办公自动化 Agent 可靠性的关键。**

## 示例输出

```
你：把每个地区的销售额汇总成 Excel 报表
Agent：我按「地区」对销售额进行了求和汇总，结果如下（数值为工具精确计算）：

| 地区 | 销售额 |
|------|--------|
| 华东 | 692600 |
| 华北 | 560000 |
| 华南 | 739600 |
| 西南 | 559000 |

报表已保存到：...\output\sales_summary_by_region.xlsx
```

## 安全设计

- **`query_database` 只允许 SELECT**：拒绝任何以非 `SELECT` 开头的 SQL，防止注入/删库
- **`write_excel` 只写 `output/`**：工具内部拼接路径，不接受任意路径，防止越权写文件
- **邮件不编造信息**：员工表无邮箱字段时，Agent 会如实说明并询问用户，而不是瞎编一个地址

## 已知局限 & 优化方向

1. **LLM 对结果的解读偶尔夸张**：数字本身可靠（工具计算），但措辞可能添油加醋 → 回答模板化或加校验
2. **Excel 聚合维度单一**：`aggregate_excel` 支持单列分组 → 可支持多列分组、多指标、透视表
3. **无评测**：办公场景更适合「指令 → 期望结果」的验收式评测（如检查生成的 xlsx 数值）→ 可仿项目 1 做 QA 集
4. **无 Web 界面**：目前 CLI → 可仿项目 1 加 Gradio，或做内部工具页

## 免责声明

- 数据与场景均为教学演示用虚构数据
- `send_email` 默认模拟发送，真实 SMTP 请自行配置并遵守相关法规
