# 多 Agent 调研报告生成器（Supervisor 模式）

第 3 个项目：用 **LangGraph 手写 Supervisor（监督者）多 Agent 协作**。一个主管 Agent 把「调研报告」任务拆解，派发给两个 worker Agent 协作完成。

## 功能亮点

| 能力 | Agent | 说明 |
|---|---|---|
| 🧠 任务编排 | Supervisor | 主管 Agent，决定「派哪个 worker、传什么任务」，自己不干活 |
| 🔍 调研 | 研究员 Agent | 自动检索公司知识库（`search_company_kb`）+ 查销售数据库（`query_sales_data` / `query_top_products`） |
| ✍️ 写报告 | 报告撰写 Agent | 把调研结论整理成正式 Markdown 报告，保存到 `output/` |
| 📋 完成 | finish 工具 | Supervisor 觉得信息够了，调用 `finish` 结束任务 |

## 架构（Supervisor 模式）

```
用户任务
   │
   ▼
Supervisor（主管 Agent）── 决定派给谁
   │  researcher_tool      writer_tool       finish_tool
   ▼                      ▼                 ▼
研究员 Agent            报告撰写 Agent      完成
(search_kb + SQL)      (write_report)
   └──────────► Supervisor ◄──────────┘
```

**核心思想（面试重点）：**

- 每个 worker 是**独立的 ReAct Agent**（`create_react_agent`），各带自己的工具，分工明确
- Supervisor 也是 LLM，但它**不带业务工具**，而是把「每个 worker」声明成自己的**工具**
- Supervisor 通过 `tool_calls` 决定下一步派谁，worker 执行完把结果喂回，Supervisor 再决策 —— 这就是**编排循环**
- 直到 Supervisor 觉得信息足够，调用 `finish_tool` 结束

> 这正是 LangGraph 官方 supervisor 教程的做法：**worker 作为 supervisor 的「工具」，supervisor 通过 bind_tools 感知 worker 职责，用 tool_calls 编排多 Agent。**

## 目录结构

```
07-report-agent/
├── create_data.py      # 生成示例数据（公司资料 + 销售数据库，可复现）
├── report_agent.py     # 核心：手写 Supervisor + 研究员 + 报告撰写 Agent
├── cli.py              # 命令行多轮对话入口
├── requirements.txt
├── data/
│   ├── company_kb.txt  # 公司资料知识库（研究员检索）
│   └── company.db      # SQLite 销售数据库（144 条记录）
└── output/             # 生成的调研报告（Agent 生成）
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
python cli.py --key sk-xxx [--base-url http://... --model deepseek-v4-pro]
```

或先设置环境变量再运行：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-xxx"
$env:OPENAI_BASE_URL = "http://192.168.0.13:3000/v1"
$env:MODEL = "deepseek-v4-pro"
python cli.py
```

示例指令：
- 写一份启明星科技的市场调研报告
- 分析一下华东地区的销售情况，出个报告
- 调研公司融资和团队，写一份背景报告

输入 `exit` 退出，支持多轮对话（携带上文记忆）。

### 4. 快速演示

```bash
python report_agent.py
```

跑一遍内置的「市场调研报告」演示场景。

## LLM 配置

默认在 `report_agent.py` 中通过环境变量配置：

```python
api_key   = os.getenv("OPENAI_API_KEY", "<默认 key>")
base_url  = os.getenv("OPENAI_BASE_URL", "http://192.168.0.13:3000/v1")
model     = os.getenv("MODEL", "deepseek-v4-pro")
```

> 代码里**不写死 key**，必须通过环境变量或 `cli.py --key` 传入，防止 key 泄露到仓库。

## 端到端验证

跑「调研启明星科技的市场表现，写一份 2024 年市场调研报告」时的真实消息流：

```
Supervisor → 派 researcher（调研 Agent）
  ├─ 研究员检索知识库：公司简介 / 产品 / 定价 / 融资 / 团队 / 市场
  └─ 研究员查销售库：各地区销售额、TOP 产品
Supervisor → 派 writer（报告撰写 Agent）→ 生成正式 Markdown 报告到 output/
Supervisor → finish → 返回报告路径
```

生成的报告包含：公司概况、产品与商业模式、市场环境与竞争格局、市场表现与销售数据、结论与建议五大板块，数据均来自真实检索（未编造）。

## 与之前项目的区别

| | 项目 1 / 2：单 Agent | 项目 3：多 Agent 协作 |
|---|---|---|
| 模型 | 1 个 LLM + 多个工具 | 多个 LLM（各自独立的 ReAct Agent） |
| 编排 | Agent 自己决定调什么工具 | Supervisor 决定派哪个 Agent |
| 分工 | 工具粒度 | Agent 粒度（每个 Agent 带自己的工具） |
| 亮点 | 工具链 + 确定性计算 | 多 Agent 协作 + 任务拆解 |

## 已知局限 & 优化方向

1. **知识库检索用简单关键词**（教学演示）→ 可换 embedding 语义检索（复用项目 1 的 BGE + Chroma）
2. **无评测** → 可做「报告是否包含期望信息点」的验收式评测
3. **无 Web 界面** → 可加 Gradio（项目 1 已展示该技能）
4. **Supervisor 一次可能返回多个 tool_calls**（已处理，逐个执行）→ 可在提示词里约束一次只调一个

## 免责声明

- 数据与场景均为教学演示用虚构数据
- 生成的报告仅用于学习演示，不代表真实企业情况
