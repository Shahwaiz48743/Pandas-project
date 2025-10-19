import random
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("financial_reports", exist_ok=True)

random.seed(42)
months = pd.period_range("2024-01", "2025-12", freq="M").astype(str)
rows = []
active_users = 1500
for m in months:
    rev = random.uniform(45000, 120000)
    cost = random.uniform(30000, 90000)
    mkt = random.uniform(8000, 30000)
    active_users = int(active_users * random.uniform(1.01, 1.05))
    new_customers = int(random.uniform(180, 950))
    rows.append({
        "Month": m,
        "Revenue": round(rev, 2),
        "Cost": round(cost, 2),
        "Marketing_Spend": round(mkt, 2),
        "Active_Users": active_users,
        "New_Customers": new_customers
    })

df = pd.DataFrame(rows)
df["Profit"] = df["Revenue"] - df["Cost"] - df["Marketing_Spend"]
df["Profit_Margin_%"] = (df["Profit"] / df["Revenue"]).replace([float("inf"), -float("inf")], 0) * 100
df["CAC"] = df.apply(lambda r: (r["Marketing_Spend"] / r["New_Customers"]) if r["New_Customers"] else 0, axis=1)
df["Burn"] = (df["Cost"] + df["Marketing_Spend"]) - df["Revenue"]
df["Net_Margin_%"] = ((df["Revenue"] - (df["Cost"] + df["Marketing_Spend"])) / df["Revenue"]).replace([float("inf"), -float("inf")], 0) * 100

df["Month_Ord"] = pd.to_datetime(df["Month"])
trend_cols = ["Revenue", "Profit", "Profit_Margin_%", "CAC", "Active_Users", "New_Customers"]
trend_view = df.sort_values("Month_Ord")[["Month"] + trend_cols]

best_profit = df.loc[df["Profit"].idxmax(), ["Month", "Profit"]]
worst_profit = df.loc[df["Profit"].idxmin(), ["Month", "Profit"]]
best_margin = df.loc[df["Profit_Margin_%"].idxmax(), ["Month", "Profit_Margin_%"]]
worst_margin = df.loc[df["Profit_Margin_%"].idxmin(), ["Month", "Profit_Margin_%"]]
best_cac = df.loc[df["CAC"].idxmin(), ["Month", "CAC"]]
worst_cac = df.loc[df["CAC"].idxmax(), ["Month", "CAC"]]

print("=== KPI Snapshot (first 6 rows) ===")
print(df.head(6).to_string(index=False))
print("\n=== Best/Worst Months ===")
print(f"Best Profit Month: {best_profit['Month']}  Profit: {best_profit['Profit']:.2f}")
print(f"Worst Profit Month: {worst_profit['Month']}  Profit: {worst_profit['Profit']:.2f}")
print(f"Best Profit Margin: {best_margin['Month']}  Margin: {best_margin['Profit_Margin_%']:.2f}%")
print(f"Worst Profit Margin: {worst_margin['Month']}  Margin: {worst_margin['Profit_Margin_%']:.2f}%")
print(f"Best CAC (lowest): {best_cac['Month']}  CAC: {best_cac['CAC']:.2f}")
print(f"Worst CAC (highest): {worst_cac['Month']}  CAC: {worst_cac['CAC']:.2f}")

sim = df.copy()
sim["Marketing_Spend_WhatIf"] = (sim["Marketing_Spend"] * 0.9).round(2)
apply_elasticity = True
elasticity = 0.25
if apply_elasticity:
    pct_change_mkt = -0.10
    revenue_multiplier = 1.0 + (elasticity * pct_change_mkt)
    sim["Revenue_WhatIf"] = (sim["Revenue"] * revenue_multiplier).round(2)
else:
    sim["Revenue_WhatIf"] = sim["Revenue"]

sim["Profit_WhatIf"] = sim["Revenue_WhatIf"] - sim["Cost"] - sim["Marketing_Spend_WhatIf"]
sim["Profit_Margin_%_WhatIf"] = (sim["Profit_WhatIf"] / sim["Revenue_WhatIf"]).replace([float("inf"), -float("inf")], 0) * 100
sim["CAC_WhatIf"] = sim.apply(lambda r: (r["Marketing_Spend_WhatIf"] / r["New_Customers"]) if r["New_Customers"] else 0, axis=1)
sim["Burn_WhatIf"] = (sim["Cost"] + sim["Marketing_Spend_WhatIf"]) - sim["Revenue_WhatIf"]
sim["Net_Margin_%_WhatIf"] = ((sim["Revenue_WhatIf"] - (sim["Cost"] + sim["Marketing_Spend_WhatIf"])) / sim["Revenue_WhatIf"]).replace([float("inf"), -float("inf")], 0) * 100

whatif_view = sim[[
    "Month", "Revenue", "Marketing_Spend", "Profit", "Profit_Margin_%", "CAC",
    "Revenue_WhatIf", "Marketing_Spend_WhatIf", "Profit_WhatIf", "Profit_Margin_%_WhatIf", "CAC_WhatIf"
]]

print("\n=== What-if: Marketing Spend -10% (with elasticity) ===")
print(whatif_view.head(6).to_string(index=False))

plt.figure()
plt.plot(df["Month"], df["Revenue"], label="Revenue")
plt.plot(df["Month"], df["Cost"] + df["Marketing_Spend"], label="Total Cost (Ops + Mkt)")
plt.title("Revenue vs Total Cost over Time")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df["Month"], df["Profit"], label="Profit")
plt.plot(sim["Month"], sim["Profit_WhatIf"], label="Profit (What-if -10% Mkt)")
plt.title("Profit Trend (Actual vs What-if)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df["Month"], df["CAC"], label="CAC")
plt.plot(sim["Month"], sim["CAC_WhatIf"], label="CAC (What-if -10% Mkt)")
plt.title("CAC Trend (Actual vs What-if)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

df.drop(columns=["Month_Ord"]).to_csv("financial_reports/financial_kpis_actual.csv", index=False)
whatif_view.to_csv("financial_reports/financial_kpis_whatif.csv", index=False)
trend_view.to_csv("financial_reports/financial_kpis_trend_view.csv", index=False)
print("\nFiles saved in 'financial_reports' folder:")
print("- financial_kpis_actual.csv")
print("- financial_kpis_whatif.csv")
print("- financial_kpis_trend_view.csv")
