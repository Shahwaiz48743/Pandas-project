import random
from datetime import datetime
import calendar
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("product_reports", exist_ok=True)

random.seed(42)
products = [f"P{i:03d}" for i in range(1, 31)]
categories = ["Electronics", "Home", "Beauty", "Grocery", "Sports"]
regions = ["North", "South", "East", "West"]
months = pd.period_range("2024-01", "2025-09", freq="M").astype(str)

rows = []
for pid in products:
    category = random.choice(categories)
    base_cost = random.uniform(3.0, 60.0)
    margin_mult = random.uniform(1.2, 1.9)
    for m in months:
        year, mon = map(int, m.split("-"))
        month_name = calendar.month_abbr[mon]
        region = random.choice(regions)
        unit_cost = round(base_cost * random.uniform(0.95, 1.10), 2)
        unit_price = round(unit_cost * margin_mult * random.uniform(0.95, 1.10), 2)
        base_demand = random.randint(20, 120)
        if mon in [11, 12]:
            base_demand = int(base_demand * 1.25)
        elif mon in [2]:
            base_demand = int(base_demand * 0.85)
        month_index = (year - 2024) * 12 + mon
        trend = 1.0 + (month_index * 0.005)
        units_sold = int(base_demand * trend * random.uniform(0.8, 1.2))
        rows.append({
            "Product_ID": pid,
            "Category": category,
            "Region": region,
            "Unit_Cost": unit_cost,
            "Unit_Price": unit_price,
            "Units_Sold": units_sold,
            "Month": m,
            "Year": year,
            "Month_Num": mon,
            "Month_Name": month_name
        })

df = pd.DataFrame(rows)
df["Unit_Margin"] = df["Unit_Price"] - df["Unit_Cost"]
df["Revenue"] = df["Unit_Price"] * df["Units_Sold"]
df["Cost"] = df["Unit_Cost"] * df["Units_Sold"]
df["Profit"] = df["Revenue"] - df["Cost"]
df["ROI_%"] = (df["Profit"] / df["Cost"]).replace([float("inf")], 0) * 100

product_profit = df.groupby(["Product_ID", "Category"], as_index=False)["Profit"].sum()
top5 = product_profit.sort_values("Profit", ascending=False).head(5)
bottom5 = product_profit.sort_values("Profit", ascending=True).head(5)

print("=== TOP 5 PERFORMERS (by total Profit) ===")
print(top5.to_string(index=False))
print("\n=== BOTTOM 5 PERFORMERS (by total Profit) ===")
print(bottom5.to_string(index=False))

region_month = df.groupby(["Region", "Month"], as_index=False).agg(
    Revenue=("Revenue", "sum"),
    Profit=("Profit", "sum"),
    Units=("Units_Sold", "sum")
)
print("\n=== REGION x MONTH (Revenue/Profit/Units) ===")
print(region_month.head(12).to_string(index=False))

df_sorted = df.sort_values(["Product_ID", "Year", "Month_Num"])
df_sorted["Units_Rolling_3M"] = (
    df_sorted.groupby("Product_ID")["Units_Sold"]
             .rolling(window=3, min_periods=1)
             .mean()
             .reset_index(level=0, drop=True)
             .round(0)
             .astype(int)
)

last_records = (
    df_sorted.groupby("Product_ID", as_index=False)
             .apply(lambda g: g.sort_values(["Year", "Month_Num"]).iloc[-1])
             .reset_index(drop=True)
)

def next_period(period_str):
    y, m = map(int, period_str.split("-"))
    m += 1
    if m == 13:
        y += 1
        m = 1
    return f"{y:04d}-{m:02d}"

forecast = last_records[[
    "Product_ID", "Category", "Region", "Unit_Cost", "Unit_Price",
    "Units_Sold", "Units_Rolling_3M", "Month"
]].copy()
forecast["Forecast_Month"] = forecast["Month"].apply(next_period)
forecast["Forecast_Units_Next_Month"] = forecast["Units_Rolling_3M"]
forecast["Forecast_Revenue"] = (forecast["Unit_Price"] * forecast["Forecast_Units_Next_Month"]).round(2)
forecast["Forecast_Profit"] = ((forecast["Unit_Price"] - forecast["Unit_Cost"]) * forecast["Forecast_Units_Next_Month"]).round(2)

print("\n=== NEXT-MONTH FORECAST (3M Rolling) ===")
print(forecast[[
    "Product_ID","Category","Region","Month","Forecast_Month","Units_Sold",
    "Units_Rolling_3M","Forecast_Units_Next_Month","Forecast_Revenue","Forecast_Profit"
]].head(10).to_string(index=False))

cat_profit = df.groupby("Category", as_index=False)["Profit"].sum()
plt.figure()
plt.bar(cat_profit["Category"], cat_profit["Profit"])
plt.title("Total Profit by Category")
plt.ylabel("Profit")
plt.xlabel("Category")
plt.tight_layout()
plt.show()

sample_pid = top5.iloc[0]["Product_ID"]
sample = df[df["Product_ID"] == sample_pid].sort_values(["Year","Month_Num"])
plt.figure()
plt.plot(sample["Month"], sample["Units_Sold"])
plt.title(f"Units Sold Trend — {sample_pid}")
plt.ylabel("Units")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

last6 = sample.tail(6)
plt.figure()
plt.plot(last6["Month"], last6["Units_Sold"], label="Actual (Last 6M)")
plt.plot([forecast.loc[forecast["Product_ID"] == sample_pid, "Forecast_Month"].iloc[0]],
         [forecast.loc[forecast["Product_ID"] == sample_pid, "Forecast_Units_Next_Month"].iloc[0]],
         marker="o", linestyle="None", label="Forecast (Next M)")
plt.title(f"Last 6M Actual vs Next Month Forecast — {sample_pid}")
plt.ylabel("Units")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

df.to_csv("product_reports/product_monthly_data.csv", index=False)
product_profit.to_csv("product_reports/product_total_profit.csv", index=False)
top5.to_csv("product_reports/top5_products.csv", index=False)
bottom5.to_csv("product_reports/bottom5_products.csv", index=False)
region_month.to_csv("product_reports/region_month_summary.csv", index=False)
forecast.to_csv("product_reports/next_month_forecast.csv", index=False)

print("\nFiles saved in 'product_reports' folder:")
print("- product_monthly_data.csv")
print("- product_total_profit.csv")
print("- top5_products.csv")
print("- bottom5_products.csv")
print("- region_month_summary.csv")
print("- next_month_forecast.csv")
