import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("pricing_reports", exist_ok=True)
random.seed(42)

products = [f"P{i:03d}" for i in range(1, 21)]
categories = ["Electronics", "Home", "Grocery", "Beauty"]
weeks = [f"2025-W{w:02d}" for w in range(1, 21)]

rows = []
for pid in products:
    cat = random.choice(categories)
    base_price = random.uniform(8, 120)
    price_volatility = random.uniform(0.85, 1.15)
    base_demand = random.uniform(80, 1200)
    price_sensitivity = random.uniform(0.6, 1.8)

    for wk in weeks:
        price = round(base_price * random.uniform(0.85, 1.15) * price_volatility, 2)
        noise = random.uniform(0.8, 1.2)
        units = int(max(0, (base_demand * (base_price / price) ** price_sensitivity) * noise))
        rows.append({
            "Product_ID": pid,
            "Category": cat,
            "Week": wk,
            "Price": price,
            "Units_Sold": units
        })

df = pd.DataFrame(rows)

corr_df = (
    df.groupby("Product_ID")
      .apply(lambda g: g["Price"].corr(g["Units_Sold"]))
      .reset_index(name="Corr_Price_Units")
)
print("=== Correlation Price vs Units (per product) ===")
print(corr_df.head(10).to_string(index=False))

def fit_linear_demand(group):
    p = group["Price"]
    q = group["Units_Sold"]
    var_p = p.var()
    if var_p == 0 or pd.isna(var_p):
        return pd.Series({"a": pd.NA, "b": pd.NA, "P_opt": pd.NA, "Rev_opt": pd.NA})
    cov_pq = p.cov(q)
    b = cov_pq / var_p
    a = q.mean() - b * p.mean()
    if b == 0 or pd.isna(b):
        possible_prices = [p.min(), p.max()]
        revs = [float(possible_prices[0] * (a + b * possible_prices[0] if pd.notna(a) and pd.notna(b) else q.mean())),
                float(possible_prices[1] * (a + b * possible_prices[1] if pd.notna(a) and pd.notna(b) else q.mean()))]
        idx = 0 if revs[0] >= revs[1] else 1
        return pd.Series({"a": a, "b": b, "P_opt": possible_prices[idx], "Rev_opt": revs[idx]})
    p_opt = -a / (2 * b)
    p_min = p.min() * 0.7
    p_max = p.max() * 1.3
    p_opt = max(p_min, min(p_opt, p_max))
    q_opt = max(0.0, a + b * p_opt)
    rev_opt = float(p_opt * q_opt)
    return pd.Series({"a": a, "b": b, "P_opt": round(float(p_opt), 2), "Rev_opt": round(rev_opt, 2)})

model_df = df.groupby(["Product_ID", "Category"], as_index=False).apply(fit_linear_demand)
print("\n=== Fitted linear demand & optimal price (per product) ===")
print(model_df.head(10).to_string(index=False))

def simulate_curve(group, model_row):
    p_obs = group["Price"]
    p_min = float(p_obs.min() * 0.7)
    p_max = float(p_obs.max() * 1.3)
    grid = [round(p_min + i * (p_max - p_min) / 40, 2) for i in range(41)]
    a = model_row["a"]
    b = model_row["b"]
    preds = []
    for price in grid:
        q = a + b * price if pd.notna(a) and pd.notna(b) else group["Units_Sold"].mean()
        q = max(0.0, float(q))
        preds.append({"Product_ID": group["Product_ID"].iloc[0],
                      "Category": group["Category"].iloc[0],
                      "Price": price,
                      "Pred_Units": q,
                      "Pred_Revenue": round(price * q, 2)})
    return pd.DataFrame(preds)

curve_list = []
for _, row in model_df.iterrows():
    g = df[df["Product_ID"] == row["Product_ID"]]
    curve_list.append(simulate_curve(g, row))
revenue_curves = pd.concat(curve_list, ignore_index=True)

print("\n=== Revenue curve sample (first 12 rows) ===")
print(revenue_curves.head(12).to_string(index=False))

def simulate_strategy(group, pct_change):
    base_avg_price = group["Price"].mean()
    new_price = round(base_avg_price * (1 + pct_change), 2)
    model = model_df[model_df["Product_ID"] == group["Product_ID"].iloc[0]].iloc[0]
    a, b = model["a"], model["b"]
    if pd.isna(a) or pd.isna(b):
        q = group["Units_Sold"].mean()
    else:
        q = max(0.0, a + b * new_price)
    rev = new_price * q
    return pd.Series({
        "Product_ID": group["Product_ID"].iloc[0],
        "Category": group["Category"].iloc[0],
        "Strategy": f"{int(pct_change*100)}%",
        "New_Price": new_price,
        "Expected_Units": round(q, 2),
        "Expected_Revenue": round(rev, 2)
    })

strategies = [-0.10, 0.00, 0.10]
sim_results = []
for pid, g in df.groupby("Product_ID"):
    for s in strategies:
        sim_results.append(simulate_strategy(g, s))
strategy_df = pd.DataFrame(sim_results)

print("\n=== Simulated strategies (per product) ===")
print(strategy_df.head(12).to_string(index=False))

best_strategy = (
    strategy_df.sort_values(["Product_ID", "Expected_Revenue"], ascending=[True, False])
               .groupby("Product_ID", as_index=False)
               .head(1)
               .rename(columns={"Strategy": "Best_Strategy"})
               [["Product_ID", "Category", "Best_Strategy", "New_Price", "Expected_Units", "Expected_Revenue"]]
)

summary = (
    model_df.merge(corr_df, on="Product_ID", how="left")
            .merge(best_strategy, on=["Product_ID", "Category"], how="left")
)
print("\n=== Optimal price & best simulated strategy (per product) ===")
print(summary.head(10).to_string(index=False))

df.to_csv("pricing_reports/pricing_weekly_raw.csv", index=False)
corr_df.to_csv("pricing_reports/price_units_correlation.csv", index=False)
model_df.to_csv("pricing_reports/linear_demand_and_optimal_price.csv", index=False)
revenue_curves.to_csv("pricing_reports/revenue_curves.csv", index=False)
strategy_df.to_csv("pricing_reports/strategy_simulations.csv", index=False)
summary.to_csv("pricing_reports/optimal_and_best_strategy_summary.csv", index=False)

print("\nFiles saved in 'pricing_reports' folder:")
print("- pricing_weekly_raw.csv")
print("- price_units_correlation.csv")
print("- linear_demand_and_optimal_price.csv")
print("- revenue_curves.csv")
print("- strategy_simulations.csv")
print("- optimal_and_best_strategy_summary.csv")

sample_ids = summary["Product_ID"].unique()[:3]
for pid in sample_ids:
    curve = revenue_curves[revenue_curves["Product_ID"] == pid].sort_values("Price")
    plt.figure()
    plt.plot(curve["Price"], curve["Pred_Revenue"])
    plt.title(f"Revenue Curve — {pid}")
    plt.xlabel("Price")
    plt.ylabel("Predicted Revenue")
    plt.tight_layout()
    plt.show()
