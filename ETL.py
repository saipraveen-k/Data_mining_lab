"""
ETL (Extract - Transform - Load) pipeline.

Extract : read raw sales records (generated as CSV so the script is self-contained)
Transform: clean, standardise, derive new columns and aggregate
Load    : write the clean fact table and a summary into a SQLite data warehouse
"""
import io
import sqlite3

import pandas as pd

RAW_CSV = """order_id,order_date,customer,region,product,quantity,unit_price
1001,2024-01-05,Ravi,north,Laptop,1,55000
1002,2024-01-07,anita,South,Mouse,3,450
1003,2024-01-07,Kiran,EAST,Keyboard,2,1200
1004,2024-02-11,Ravi,north,Monitor,,12000
1005,2024-02-15,Meena,West,Laptop,2,54000
1005,2024-02-15,Meena,West,Laptop,2,54000
1006,2024-03-02,Suresh,south,Mouse,5,
1007,2024-03-09,anita,South,Monitor,1,11500
"""


def extract():
    df = pd.read_csv(io.StringIO(RAW_CSV))
    print(f"[Extract] {len(df)} raw rows")
    return df


def transform(df):
    df = df.drop_duplicates()
    df["customer"] = df["customer"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["quantity"] = df["quantity"].fillna(1).astype(int)
    df["unit_price"] = df.groupby("product")["unit_price"].transform(lambda s: s.fillna(s.mean()))
    df["total_amount"] = df["quantity"] * df["unit_price"]
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    print(f"[Transform] {len(df)} clean rows")
    summary = (df.groupby(["month", "region"], as_index=False)
                 .agg(orders=("order_id", "count"), revenue=("total_amount", "sum")))
    return df, summary


def load(fact, summary, db_path="sales_warehouse.db"):
    with sqlite3.connect(db_path) as conn:
        fact.assign(order_date=fact["order_date"].dt.date.astype(str)).to_sql(
            "fact_sales", conn, if_exists="replace", index=False)
        summary.to_sql("monthly_region_summary", conn, if_exists="replace", index=False)
        check = pd.read_sql("SELECT * FROM monthly_region_summary ORDER BY month, region", conn)
    print(f"[Load] wrote fact_sales and monthly_region_summary to {db_path}")
    return check


if __name__ == "__main__":
    raw = extract()
    fact, summary = transform(raw)
    print("\nClean fact table:\n", fact, "\n")
    print("Loaded summary (read back from warehouse):\n", load(fact, summary))
