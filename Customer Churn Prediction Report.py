# --------------------------------------------
# 📊 PROJECT 1: CUSTOMER CHURN ANALYSIS DASHBOARD
# --------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# 1) DATASET
company = {
    'Customer_ID': [
        'CUST1001','CUST1002','CUST1003','CUST1004','CUST1005','CUST1006','CUST1007','CUST1008','CUST1009','CUST1010',
        'CUST1011','CUST1012','CUST1013','CUST1014','CUST1015','CUST1016','CUST1017','CUST1018','CUST1019','CUST1020',
        'CUST1021','CUST1022','CUST1023','CUST1024','CUST1025','CUST1026','CUST1027','CUST1028','CUST1029','CUST1030'
    ],
    'Age': [
        22, 35, 29, 41, 56, 33, 27, 48, 39, 31,
        44, 52, 24, 37, 43, 60, 28, 34, 50, 25,
        46, 38, 42, 30, 49, 36, 32, 40, 58, 26
    ],
    'Country': [
        'Finland','Spain','Sweden','Germany','France','Italy','Netherlands','Spain','Finland','France',
        'Italy','Sweden','Germany','Finland','Spain','France','Netherlands','Italy','Germany','Spain',
        'Finland','Sweden','France','Italy','Germany','Finland','Spain','France','Netherlands','Sweden'
    ],
    'Monthly_Fee': [
        12.99, 19.99, 34.99, 19.99, 12.99, 34.99, 12.99, 19.99, 34.99, 19.99,
        12.99, 34.99, 12.99, 19.99, 34.99, 12.99, 34.99, 19.99, 12.99, 19.99,
        34.99, 12.99, 19.99, 12.99, 34.99, 19.99, 12.99, 19.99, 34.99, 12.99
    ],
    'Calls': [
        15, 9, 22, 5, 18, 13, 6, 20, 17, 8,
        25, 12, 7, 10, 21, 4, 14, 9, 18, 6,
        16, 11, 19, 8, 23, 10, 5, 17, 12, 9
    ],
    'Data_Usage': [
        8.5, 3.2, 12.7, 2.8, 9.6, 7.4, 1.9, 11.3, 10.6, 4.7,
        13.2, 9.1, 2.3, 6.8, 12.1, 1.7, 10.4, 5.6, 8.8, 3.9,
        11.8, 7.5, 9.2, 6.3, 13.9, 4.4, 2.6, 10.1, 8.9, 5.2
    ],
    'Complaints': [
        0, 1, 0, 2, 0, 1, 3, 0, 0, 2,
        0, 1, 2, 0, 1, 3, 0, 0, 1, 2,
        0, 1, 0, 2, 0, 1, 2, 0, 0, 1
    ],
    'Churned': [
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
        0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1
    ]
}

myvar = pd.DataFrame(company)

# 2) ARPU (Average Monthly Revenue)
print("\n=== DATA (head) ===")
print(myvar.head())
arpu = myvar["Monthly_Fee"].mean()
print("\n=== Average Monthly Revenue (ARPU) ===")
print(round(arpu, 2))

# 3) CORRELATION ANALYSIS (Top churn reasons)
corr_matrix = myvar.corr(numeric_only=True)
print("\n=== Correlation Matrix (numeric) ===")
print(corr_matrix)

# remove self-correlation before ranking features
churn_corr = corr_matrix["Churned"].drop("Churned").sort_values(ascending=False)
print("\n=== Correlation with Churned (no self-correlation) ===")
print(churn_corr)

print("\n=== Top 3 Positive Churn Reasons ===")
print(churn_corr[churn_corr > 0].head(3))

# (Optional) visualize correlation with churn
plt.figure(figsize=(6,4))
churn_corr.plot(kind='bar')
plt.title("Correlation of Each Factor with Churn")
plt.xlabel("Feature")
plt.ylabel("Correlation Strength")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 4) USAGE SEGMENTATION (Low / Medium / High)
myvar["Usage_Segment"] = pd.cut(
    myvar["Data_Usage"],
    bins=[0, 5, 10, float("inf")],   # <5 Low, 5–10 Medium, >10 High
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

print("\n=== Added Usage_Segment Column (sample) ===")
print(myvar[["Customer_ID", "Data_Usage", "Usage_Segment"]].head(10))

# churn rate by segment
segment_churn = (
    myvar.groupby("Usage_Segment", observed=True)["Churned"]
         .mean()
         .rename("Churn_Rate")
         .sort_values(ascending=False)
)
print("\n=== Churn Rate by Usage Segment ===")
print((segment_churn * 100).round(2).astype(str) + "%")

# visualize segment churn
plt.figure(figsize=(6,4))
(segment_churn * 100).plot(kind='bar')
plt.title("Churn Rate by Usage Segment (%)")
plt.xlabel("Usage Segment")
plt.ylabel("Churn Rate (%)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 5) TOP 5 CHURN SIGNALS DASHBOARD
# (use absolute correlation strength to rank signals)
churn_corr_abs_ranked = corr_matrix["Churned"].drop("Churned").sort_values(key=abs, ascending=False)
top5 = churn_corr_abs_ranked.head(5)

print("\n=== Top 5 Churn Signals (by |correlation|) ===")
print(top5)

print("\n=== Interpretation of Churn Signals ===")
for feature, value in top5.items():
    meaning = "→ Positive (higher value increases churn)" if value > 0 else "→ Negative (higher value reduces churn)"
    print(f"{feature}: {value:.2f} {meaning}")

# visualize top 5 signals with sign-based coloring
plt.figure(figsize=(6,4))
colors = ['tomato' if v > 0 else 'seagreen' for v in top5.values]
top5.plot(kind='bar', color=colors)
plt.title("Top 5 Churn Signals (Correlation with Churned)")
plt.xlabel("Feature")
plt.ylabel("Correlation Strength")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n=== Dashboard Insight Summary ===")
if (top5 > 0).any():
    print(f"• Strongest churn driver: {top5[top5 > 0].idxmax()} (positive correlation).")
if (top5 < 0).any():
    print(f"• Strongest loyalty factor: {top5[top5 < 0].idxmin()} (negative correlation).")
print("• Use these factors to build customer-retention strategies (support, pricing, engagement).")
