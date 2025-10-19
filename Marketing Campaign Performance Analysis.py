import random
import os
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("marketing_reports", exist_ok=True)

random.seed(42)
channels = ["Facebook", "Email", "Google"]
regions = ["North", "South", "East", "West"]
age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
start_day = date(2025, 6, 1)
campaign_rows = []
campaign_counter = 1000

for _ in range(120):
    campaign_id = f"CAM{campaign_counter}"
    campaign_counter += 1
    ch = random.choice(channels)
    rg = random.choice(regions)
    ag = random.choice(age_groups)
    base_spend = random.uniform(300, 5000)
    if ch == "Google":
        base_spend *= random.uniform(1.1, 1.5)
    elif ch == "Email":
        base_spend *= random.uniform(0.6, 0.9)
    spend = round(base_spend, 2)
    leads = int(spend / random.uniform(10, 60))
    conv_rate = random.uniform(0.03, 0.18)
    conversions = int(leads * conv_rate)
    price_per_conv = random.uniform(40, 300)
    revenue = round(conversions * price_per_conv, 2)
    day = start_day + timedelta(days=random.randint(0, 120))
    campaign_rows.append({
        "Campaign_ID": campaign_id,
        "Channel": ch,
        "Region": rg,
        "Age_Group": ag,
        "Date": day.isoformat(),
        "Spend": round(spend, 2),
        "Leads": leads,
        "Conversions": conversions,
        "Revenue": revenue
    })

df = pd.DataFrame(campaign_rows)
df["ROI"] = (df["Revenue"] - df["Spend"]) / df["Spend"]
df["CPL"] = df.apply(lambda r: (r["Spend"] / r["Leads"]) if r["Leads"] else 0, axis=1)
df["CPA"] = df.apply(lambda r: (r["Spend"] / r["Conversions"]) if r["Conversions"] else 0, axis=1)

print("=== Sample campaigns (first 10) ===")
print(df.head(10).to_string(index=False))

channel_perf = df.groupby("Channel", as_index=False).agg(
    Total_Spend=("Spend", "sum"),
    Total_Revenue=("Revenue", "sum"),
    Total_Leads=("Leads", "sum"),
    Total_Conversions=("Conversions", "sum"),
    Avg_ROI=("ROI", "mean"),
    Avg_CPL=("CPL", "mean"),
    Avg_CPA=("CPA", "mean")
)
channel_perf["Channel_ROI"] = (channel_perf["Total_Revenue"] - channel_perf["Total_Spend"]) / channel_perf["Total_Spend"]

print("\n=== Channel performance summary ===")
print(channel_perf.to_string(index=False))

region_channel = df.groupby(["Region", "Channel"], as_index=False).agg(
    Spend=("Spend", "sum"),
    Revenue=("Revenue", "sum"),
    Leads=("Leads", "sum"),
    Conversions=("Conversions", "sum")
)
region_channel["ROI"] = (region_channel["Revenue"] - region_channel["Spend"]) / region_channel["Spend"]
region_channel["CPL"] = region_channel.apply(lambda r: (r["Spend"] / r["Leads"]) if r["Leads"] else 0, axis=1)
region_channel["CPA"] = region_channel.apply(lambda r: (r["Spend"] / r["Conversions"]) if r["Conversions"] else 0, axis=1)

print("\n=== Region x Channel summary (first 12 rows) ===")
print(region_channel.head(12).to_string(index=False))

demo_perf = df.groupby(["Region", "Age_Group", "Channel"], as_index=False).agg(
    Spend=("Spend", "sum"),
    Revenue=("Revenue", "sum"),
    Leads=("Leads", "sum"),
    Conversions=("Conversions", "sum")
)
demo_perf["ROI"] = (demo_perf["Revenue"] - demo_perf["Spend"]) / demo_perf["Spend"]
demo_perf["CPL"] = demo_perf.apply(lambda r: (r["Spend"] / r["Leads"]) if r["Leads"] else 0, axis=1)
demo_perf["CPA"] = demo_perf.apply(lambda r: (r["Spend"] / r["Conversions"]) if r["Conversions"] else 0, axis=1)

print("\n=== Demographic performance sample (Region x Age x Channel, first 12) ===")
print(demo_perf.head(12).to_string(index=False))

plt.figure()
plt.bar(channel_perf["Channel"], channel_perf["Channel_ROI"])
plt.title("Channel ROI")
plt.ylabel("ROI")
plt.xlabel("Channel")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(channel_perf["Channel"], channel_perf["Avg_CPL"])
plt.title("Average Cost per Lead by Channel")
plt.ylabel("CPL")
plt.xlabel("Channel")
plt.tight_layout()
plt.show()

region_order = ["North", "South", "East", "West"]
tmp = region_channel.copy()
tmp["Region"] = pd.Categorical(tmp["Region"], categories=region_order, ordered=True)
tmp = tmp.sort_values(["Region", "ROI"])
plt.figure()
for ch in channels:
    sub = tmp[tmp["Channel"] == ch]
    plt.plot(sub["Region"], sub["ROI"], marker="o", label=ch)
plt.title("ROI by Region and Channel")
plt.ylabel("ROI")
plt.xlabel("Region")
plt.legend()
plt.tight_layout()
plt.show()

roi_rank = channel_perf.sort_values("Channel_ROI", ascending=False).reset_index(drop=True)
cpl_rank = channel_perf.sort_values("Avg_CPL", ascending=True).reset_index(drop=True)
cpa_rank = channel_perf.sort_values("Avg_CPA", ascending=True).reset_index(drop=True)

def rank_score(df_rank, channel_col="Channel", top_weight=3):
    scores = {}
    for i, row in df_rank.iterrows():
        ch = row[channel_col]
        scores[ch] = scores.get(ch, 0) + max(top_weight - i, 0)
    return scores

scores = {}
for sc in (rank_score(roi_rank), rank_score(cpl_rank), rank_score(cpa_rank)):
    for k, v in sc.items():
        scores[k] = scores.get(k, 0) + v

score_df = pd.DataFrame([{"Channel": k, "Score": v} for k, v in scores.items()]).sort_values("Score", ascending=False)

print("\n=== Overall channel prioritization score (higher is better) ===")
print(score_df.to_string(index=False))

best_channel = score_df.iloc[0]["Channel"]
print(f"\nRecommendation: Invest more in {best_channel} next quarter, based on combined ROI, CPL, and CPA ranking.")

best_segments = demo_perf.sort_values("ROI", ascending=False).groupby("Channel", as_index=False).head(3)
print("\nTop segments per channel (by ROI):")
print(best_segments[["Channel","Region","Age_Group","ROI","CPL","CPA"]].to_string(index=False))

df.to_csv("marketing_reports/marketing_campaigns_raw.csv", index=False)
channel_perf.to_csv("marketing_reports/channel_performance.csv", index=False)
region_channel.to_csv("marketing_reports/region_channel_summary.csv", index=False)
demo_perf.to_csv("marketing_reports/demographic_performance.csv", index=False)
score_df.to_csv("marketing_reports/channel_prioritization_score.csv", index=False)

print("\nFiles saved in 'marketing_reports' folder:")
print("- marketing_campaigns_raw.csv")
print("- channel_performance.csv")
print("- region_channel_summary.csv")
print("- demographic_performance.csv")
print("- channel_prioritization_score.csv")
