import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("rfm_reports", exist_ok=True)

random.seed(42)

customers = [f"CUST{i:04d}" for i in range(1, 501)]

rows = []
for cid in customers:
    recency = random.randint(1, 365)
    frequency = random.randint(1, 50)
    avg_spend = random.uniform(10, 300)
    monetary = round(frequency * avg_spend, 2)
    rows.append({
        "Customer_ID": cid,
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    })

df = pd.DataFrame(rows)

df["R_Score"] = pd.qcut(df["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
df["F_Score"] = pd.qcut(df["Frequency"], 4, labels=[1, 2, 3, 4]).astype(int)
df["M_Score"] = pd.qcut(df["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

def segment_label(score):
    if score >= 10:
        return "Gold"
    elif score >= 7:
        return "Silver"
    else:
        return "Bronze"

df["Segment"] = df["RFM_Score"].apply(segment_label)

seg_summary = df.groupby("Segment", as_index=False).agg(
    Customers=("Customer_ID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum")
).sort_values("Total_Revenue", ascending=False)

seg_summary["Revenue_%"] = (seg_summary["Total_Revenue"] / seg_summary["Total_Revenue"].sum() * 100).round(2)

print("=== RFM Segmentation Summary ===")
print(seg_summary.to_string(index=False))

plt.figure()
plt.bar(seg_summary["Segment"], seg_summary["Total_Revenue"])
plt.title("Total Revenue by Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(seg_summary["Segment"], seg_summary["Avg_Monetary"])
plt.title("Average Monetary Value by Segment")
plt.xlabel("Segment")
plt.ylabel("Avg Monetary Value")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(seg_summary["Segment"], seg_summary["Customers"])
plt.title("Number of Customers by Segment")
plt.xlabel("Segment")
plt.ylabel("Customers")
plt.tight_layout()
plt.show()

priority_segment = seg_summary.loc[seg_summary["Revenue_%"].idxmax(), "Segment"]
recommendation = f"🎯 Recommend targeting **{priority_segment} customers** for loyalty rewards and premium offers."
secondary_segment = seg_summary.loc[seg_summary["Revenue_%"].idxmin(), "Segment"]
discount_rec = f"💡 Suggest offering discounts to **{secondary_segment} customers** to re-engage them."

print("\n=== Marketing Recommendations ===")
print(recommendation)
print(discount_rec)

df.to_csv("rfm_reports/customer_rfm_data.csv", index=False)
seg_summary.to_csv("rfm_reports/rfm_segment_summary.csv", index=False)

print("\nFiles saved in 'rfm_reports' folder:")
print("- customer_rfm_data.csv")
print("- rfm_segment_summary.csv")
