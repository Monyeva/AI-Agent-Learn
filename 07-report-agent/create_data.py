"""
生成多 Agent 调研报告生成器的示例数据（可复现）：
- data/company_kb.txt  公司资料知识库（供研究员检索）
- data/company.db      SQLite 业务库（销售数据，供研究员查询）
"""

import os
import random
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ============ 1. 公司资料知识库 ============

COMPANY_KB = """启明星科技公司简介
启明星科技是一家成立于 2019 年的人工智能企业，总部位于上海张江。公司聚焦企业级 AI 应用开发平台，核心产品「星图」Agent 平台帮助客户低代码搭建智能助手。公司已服务超过 300 家企业客户，覆盖金融、零售、制造、教育四个行业。

星图 Agent 平台
星图是启明星科技的主打产品，是一个可视化 Agent 开发平台。支持拖拽式编排工作流、内置 50+ 常用工具（数据库、邮件、Excel、Web API）、一键部署到私有云或公有云。企业客户可用它快速搭建客服助手、数据分析助手、办公自动化机器人。

定价策略
星图平台采用阶梯式定价：基础版 5 万元/年，专业版 20 万元/年，旗舰版 50 万元/年。专业版以上支持私有化部署。按年订阅，首年可享 8 折优惠。定价定位中高端，主要面向有一定预算的中大型企业。

融资历程
公司于 2021 年完成 A 轮融资 1 亿元，投资方为红杉资本。2023 年完成 B 轮融资 2 亿元，投资方为高瓴资本与深创投。目前估值约 20 亿元。融资主要用于产品研发与市场扩张。

核心团队
创始人兼 CEO 李明，前百度 AI 部门技术总监，拥有 15 年 AI 从业经验。CTO 王芳，前微软研究院研究员，专注自然语言处理。团队共 120 人，其中研发占比 70%，销售与市场 20%，行政 10%。

市场表现
2024 年公司营收 1.2 亿元，同比增长 80%。客户续约率 92%，新签客户 80 家。主要营收来源：星图平台订阅（70%）、定制开发服务（20%）、培训与咨询（10%）。

竞争优势
相比国内同类 Agent 平台，启明星的优势在于：1）私有化部署能力强，满足金融、政务的安全要求；2）生态工具丰富，开箱即用；3）服务响应快，7×24 小时技术支持。主要竞争对手为字节旗下「扣子」和阿里「百炼」。

客户案例
某头部银行用星图平台搭建了智能客服助手，年处理咨询 1000 万次，人工客服成本降低 30%。某零售集团用星图搭建了门店经营数据分析助手，管理层每日 8 点自动收到前一日经营报表。

行业趋势
据 IDC 报告，2025 年中国企业级 AI Agent 市场规模预计达 200 亿元，年复合增长率 45%。金融、政务、医疗是增速最快的三个行业。
"""

KB_PATH = os.path.join(DATA_DIR, "company_kb.txt")
with open(KB_PATH, "w", encoding="utf-8") as f:
    f.write(COMPANY_KB)
print(f"已生成知识库：{KB_PATH}（{len(COMPANY_KB)} 字符）")


# ============ 2. SQLite 业务库（销售数据） ============

DB_PATH = os.path.join(DATA_DIR, "company.db")
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,          -- 日期 YYYY-MM
    region TEXT NOT NULL,        -- 地区：华东/华北/华南/西南
    product TEXT NOT NULL,       -- 产品：星图平台/定制开发/培训服务
    amount INTEGER NOT NULL,     -- 销售额（元）
    quantity INTEGER NOT NULL    -- 销量（个/单）
)
""")

random.seed(42)
REGIONS = ["华东", "华北", "华南", "西南"]
PRODUCTS = ["星图平台", "定制开发", "培训服务"]

rows = []
for month in range(1, 13):            # 2024 年 12 个月
    for region in REGIONS:
        for product in PRODUCTS:
            date = f"2024-{month:02d}"
            # 不同产品/地区给不同量级，模拟真实分布
            if product == "星图平台":
                amount = random.randint(8, 25) * 10000
                quantity = random.randint(1, 5)
            elif product == "定制开发":
                amount = random.randint(3, 12) * 10000
                quantity = random.randint(1, 3)
            else:  # 培训服务
                amount = random.randint(1, 5) * 10000
                quantity = random.randint(2, 10)
            rows.append((date, region, product, amount, quantity))

cur.executemany("INSERT INTO sales (date, region, product, amount, quantity) VALUES (?, ?, ?, ?, ?)", rows)
conn.commit()

# 打印汇总，便于后续人工核对
cur.execute("SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY SUM(amount) DESC")
print("\n各地区销售额（供核对）：")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]:,} 元")

conn.close()
print(f"已生成业务库：{DB_PATH}（{len(rows)} 条销售记录）")
