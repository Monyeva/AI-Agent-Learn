"""
生成示例数据（可复现，固定随机种子）：
1. data/sales_data.xlsx —— 销售流水（Excel）
2. data/company.db     —— SQLite 业务库（员工表 + 产品表）

用法：python create_data.py
"""

import os
import random
import sqlite3
from datetime import date, timedelta

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
XLSX_PATH = os.path.join(DATA_DIR, "sales_data.xlsx")
DB_PATH = os.path.join(DATA_DIR, "company.db")

random.seed(42)

# ============ 1. 销售流水 Excel ============

REGIONS = ["华东", "华北", "华南", "西南"]
SALESPEOPLE = {  # 每个地区的销售员
    "华东": ["王强", "李娜"],
    "华北": ["张伟", "刘芳"],
    "华南": ["陈晨"],
    "西南": ["赵磊"],
}
PRODUCTS = {  # 产品 -> (单价, 用途)
    "星图平台": (9800, "SaaS 平台"),
    "数据分析服务": (5000, "咨询服务"),
    "培训服务": (2000, "教育培训"),
    "定制开发": (15000, "软件开发"),
}


def gen_sales_rows(n=36):
    """生成 n 条销售记录，日期分布在 2026-01-01 ~ 2026-06-30。"""
    start = date(2026, 1, 1)
    rows = []
    for _ in range(n):
        region = random.choice(REGIONS)
        product = random.choice(list(PRODUCTS))
        unit_price = PRODUCTS[product][0]
        quantity = random.randint(1, 20)
        day_offset = random.randint(0, 180)
        d = start + timedelta(days=day_offset)
        rows.append({
            "日期": d.strftime("%Y-%m-%d"),
            "地区": region,
            "产品": product,
            "销售员": random.choice(SALESPEOPLE[region]),
            "数量": quantity,
            "单价": unit_price,
            "销售额": quantity * unit_price,
        })
    # 按日期排序
    rows.sort(key=lambda r: r["日期"])
    return rows


# ============ 2. SQLite 业务库 ============

EMPLOYEES = [
    # (姓名, 部门, 职位, 月薪, 邮箱)
    ("陈宇", "技术部", "架构师", 38000, "chenyu@qimingxing.com"),
    ("王磊", "技术部", "后端工程师", 26000, "wanglei@qimingxing.com"),
    ("刘洋", "技术部", "算法工程师", 30000, "liuyang@qimingxing.com"),
    ("王强", "销售部", "销售经理", 22000, "wangqiang@qimingxing.com"),
    ("李娜", "销售部", "销售专员", 14000, "lina@qimingxing.com"),
    ("赵磊", "销售部", "销售专员", 12000, "zhaolei@qimingxing.com"),
    ("张伟", "市场部", "市场经理", 20000, "zhangwei@qimingxing.com"),
    ("孙丽", "市场部", "内容运营", 11000, "sunli@qimingxing.com"),
    ("周敏", "财务部", "财务经理", 24000, "zhoumin@qimingxing.com"),
    ("吴静", "财务部", "会计", 13000, "wujing@qimingxing.com"),
]

PRODUCT_ROWS = [
    # (名称, 分类, 价格)
    ("星图平台", "SaaS 平台", 9800),
    ("数据分析服务", "咨询服务", 5000),
    ("培训服务", "教育培训", 2000),
    ("定制开发", "软件开发", 15000),
]


def build_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS employees")
    cur.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dept TEXT NOT NULL,
            position TEXT NOT NULL,
            salary INTEGER NOT NULL,
            email TEXT NOT NULL
        )
    """)
    cur.executemany(
        "INSERT INTO employees (name, dept, position, salary, email) VALUES (?, ?, ?, ?, ?)",
        EMPLOYEES,
    )

    cur.execute("DROP TABLE IF EXISTS products")
    cur.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price INTEGER NOT NULL
        )
    """)
    cur.executemany(
        "INSERT INTO products (name, category, price) VALUES (?, ?, ?)",
        PRODUCT_ROWS,
    )

    conn.commit()
    conn.close()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Excel
    rows = gen_sales_rows()
    df = pd.DataFrame(rows)
    df.to_excel(XLSX_PATH, index=False, sheet_name="销售记录")
    print(f"✅ 已生成销售流水：{XLSX_PATH}（{len(df)} 行）")

    # SQLite
    build_db()
    print(f"✅ 已生成业务库：{DB_PATH}")

    # 预览
    print("\n--- 销售数据预览 ---")
    print(df.head(3).to_string(index=False))
    print("\n--- employees 预览 ---")
    conn = sqlite3.connect(DB_PATH)
    print(pd.read_sql("SELECT * FROM employees LIMIT 3", conn).to_string(index=False))
    print("\n--- products 预览 ---")
    print(pd.read_sql("SELECT * FROM products", conn).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
