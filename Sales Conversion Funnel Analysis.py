import sys
sys.stdout.reconfigure(encoding='utf-8')

import random
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("funnel_reports", exist_ok=True)

random.seed(42)
N = 4000
start = datetime(2025, 1, 1, 8, 0, 0)

def random_dt(start_dt, days=180):
    return start_dt + timedelta(
        days=random.randint(0, days),
        hours=random.randint(8, 23),
        minutes=random.randint(0, 59)
    )

rows = []
for i in range(N):
    ts = random_dt(start)
    weekday = ts.strftime("%A")
    hour = ts.hour
    base_view_p = 0.92
    base_add_p = 0.38
    base_checkout_p = 0.65
    base_purchase_p = 0.78
    wd_mult_map = {
        "Monday": 0.95, "Tuesday": 0.98, "Wednesday": 1.00, "Thursday": 1.02,
        "Friday": 1.08, "Saturday": 1.10, "Sunday": 1.05
    }
    wd_mult = wd_mult_map.get(weekday, 1.0)
    if 18 <= hour <= 22:
        hr_mult = 1.12
    elif 12 <= hour <= 14:
        hr_mult = 1.05
    else:
        hr_mult = 0.95

    def happens(p):
        return 1 if random.random() < p else 0

    product_view = happens(base_view_p * wd_mult * hr_mult)
    added_to_cart = 0
    checkout = 0
    purchase = 0
    if product_view:
        added_to_cart = happens(base_add_p * wd_mult * hr_mult)
    if added_to_cart:
        checkout = happens(base_checkout_p * wd_mult * hr_mult)
    if checkout:
        purchase = happens(base_purchase_p * wd_mult * hr_mult)
    rows.append({
        "Session_ID": f"SESS{100000 + i}",
        "Timestamp": ts,
        "Weekday": weekday,
        "Hour": hour,
        "Product_View": product_view,
        "Added_to_Cart": added_to_cart,
        "Checkout": checkout,
        "Purchase": purchase
    })

df = pd.DataFrame(rows)
stages = ["Product_View", "Added_to_Cart", "Checkout", "Purchase"]
totals = df[stages].sum()

def safe_rate(n, d):
    return (n / d) if d and d != 0 else 0

conversion = pd.Series({
    "View->Add": safe_rate(totals["Added_to_Cart"], totals["Product_View"]),
    "Add->Checkout": safe_rate(totals["Checkout"], totals["Added_to_Cart"]),
    "Checkout->Purchase": safe_rate(totals["Purchase"], totals["Checkout"]),
    "View->Purchase (Overall)": safe_rate(totals["Purchase"], totals["Product_View"])
})

dropoffs = pd.Series({
    "After_View": totals["Product_View"] - totals["Added_to_Cart"],
    "After_Add": totals["Added_to_Cart"] - totals["Checkout"],
    "After_Checkout": totals["Checkout"] - totals["Purchase"]
})

funnel_df = pd.DataFrame({"Stage": stages, "Users": [totals[s] for s in stages]})
print("=== FUNNEL TOTALS ===")
print(funnel_df.to_string(index=False))
print("\n=== CONVERSION RATES ===")
print((conversion * 100).round(2).astype(str) + "%")
print("\n=== DROPOFFS ===")
print(dropoffs)

weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_funnel = df.groupby("Weekday")[stages].sum().reindex(weekday_order)
weekday_conv = pd.DataFrame({
    "View->Add": weekday_funnel.apply(lambda r: safe_rate(r["Added_to_Cart"], r["Product_View"]), axis=1),
    "Add->Checkout": weekday_funnel.apply(lambda r: safe_rate(r["Checkout"], r["Added_to_Cart"]), axis=1),
    "Checkout->Purchase": weekday_funnel.apply(lambda r: safe_rate(r["Purchase"], r["Checkout"]), axis=1),
    "View->Purchase": weekday_funnel.apply(lambda r: safe_rate(r["Purchase"], r["Product_View"]), axis=1),
})

hourly_funnel = df.groupby("Hour")[stages].sum()
hourly_conv = pd.DataFrame({
    "View->Add": hourly_funnel.apply(lambda r: safe_rate(r["Added_to_Cart"], r["Product_View"]), axis=1),
    "Add->Checkout": hourly_funnel.apply(lambda r: safe_rate(r["Checkout"], r["Added_to_Cart"]), axis=1),
    "Checkout->Purchase": hourly_funnel.apply(lambda r: safe_rate(r["Purchase"], r["Checkout"]), axis=1),
    "View->Purchase": hourly_funnel.apply(lambda r: safe_rate(r["Purchase"], r["Product_View"]), axis=1),
})

print("\n=== BEST WEEKDAYS (Overall View->Purchase) ===")
print((weekday_conv["View->Purchase"].sort_values(ascending=False).head(3) * 100).round(2).astype(str) + "%")
print("\n=== BEST HOURS (Overall View->Purchase) ===")
print((hourly_conv["View->Purchase"].sort_values(ascending=False).head(5) * 100).round(2).astype(str) + "%")

plt.figure()
plt.barh(list(funnel_df["Stage"])[::-1], list(funnel_df["Users"])[::-1])
plt.title("Sales Conversion Funnel (Sessions)")
plt.xlabel("Users")
plt.tight_layout()
plt.show()

plt.figure()
weekday_conv["View->Purchase"].plot()
plt.title("Overall Conversion by Weekday (View->Purchase)")
plt.ylabel("Conversion Rate")
plt.xlabel("Weekday")
plt.tight_layout()
plt.show()

plt.figure()
hourly_conv["View->Purchase"].plot()
plt.title("Overall Conversion by Hour (View->Purchase)")
plt.ylabel("Conversion Rate")
plt.xlabel("Hour of Day")
plt.tight_layout()
plt.show()

funnel_df.to_csv("funnel_reports/funnel_totals.csv", index=False)
weekday_conv.to_csv("funnel_reports/weekday_conversion_rates.csv")
hourly_conv.to_csv("funnel_reports/hourly_conversion_rates.csv")
df.to_csv("funnel_reports/sessions_raw.csv", index=False)
print("\nFiles saved in 'funnel_reports' folder:")
print("- sessions_raw.csv")
print("- funnel_totals.csv")
print("- weekday_conversion_rates.csv")
print("- hourly_conversion_rates.csv")
