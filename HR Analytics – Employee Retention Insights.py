import random
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("hr_reports", exist_ok=True)

random.seed(42)
departments = ["IT", "HR", "Finance", "Marketing", "Sales", "Operations"]
n = 200
rows = []

for i in range(1, n + 1):
    emp_id = f"EMP{i:03d}"
    dept = random.choice(departments)
    age = random.randint(22, 58)
    experience = random.randint(1, age - 21)
    base_salary = random.randint(28000, 100000)
    salary = base_salary + experience * random.randint(500, 2000)
    work_hours = random.randint(30, 60)
    promotion_years = random.randint(0, 10)
    attrition = random.choices(["Yes", "No"], weights=[0.25, 0.75])[0]
    rows.append({
        "Employee_ID": emp_id,
        "Age": age,
        "Salary": salary,
        "Department": dept,
        "Experience": experience,
        "Work_Hours": work_hours,
        "Promotion_Years": promotion_years,
        "Attrition": attrition
    })

df = pd.DataFrame(rows)
df["Attrition_Flag"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

print("=== Sample Employee Data ===")
print(df.head(10).to_string(index=False))

avg_salary = df.groupby("Department", as_index=False)["Salary"].mean().rename(columns={"Salary": "Avg_Salary"})
print("\n=== Average Salary by Department ===")
print(avg_salary.to_string(index=False))

attrition_rate = df["Attrition_Flag"].mean()
print(f"\nOverall Attrition Rate: {attrition_rate:.2%}")

salary_attrition = df.groupby("Attrition")["Salary"].mean()
promotion_attrition = df.groupby("Attrition")["Promotion_Years"].mean()
print("\n=== Salary vs Attrition ===")
print(salary_attrition)
print("\n=== Promotion Years vs Attrition ===")
print(promotion_attrition)

corr_cols = ["Age", "Salary", "Experience", "Work_Hours", "Promotion_Years", "Attrition_Flag"]
corr_matrix = df[corr_cols].corr()
print("\n=== Correlation Matrix ===")
print(corr_matrix.round(2))

plt.figure()
plt.bar(avg_salary["Department"], avg_salary["Avg_Salary"])
plt.title("Average Salary by Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(["Attrition=Yes", "Attrition=No"], salary_attrition)
plt.title("Average Salary by Attrition Status")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(["Attrition=Yes", "Attrition=No"], promotion_attrition)
plt.title("Average Promotion Years by Attrition Status")
plt.tight_layout()
plt.show()

salary_q75 = df["Salary"].quantile(0.75)
promotion_q25 = df["Promotion_Years"].quantile(0.25)
high_risk = df[
    (df["Attrition"] == "No") &
    ((df["Salary"] < salary_q75) & (df["Promotion_Years"] > promotion_q25) & (df["Work_Hours"] > 50))
]
high_risk["Risk_Level"] = "High"

print("\n=== Employees at High Attrition Risk (sample 10) ===")
print(high_risk.head(10).to_string(index=False))

risk_summary = high_risk.groupby("Department", as_index=False)["Employee_ID"].count().rename(columns={"Employee_ID": "High_Risk_Count"})
print("\n=== High Attrition Risk by Department ===")
print(risk_summary.to_string(index=False))

df.to_csv("hr_reports/employee_data.csv", index=False)
avg_salary.to_csv("hr_reports/avg_salary_by_dept.csv", index=False)
corr_matrix.to_csv("hr_reports/correlation_matrix.csv")
high_risk.to_csv("hr_reports/high_attrition_risk.csv", index=False)

print("\nFiles saved in 'hr_reports' folder:")
print("- employee_data.csv")
print("- avg_salary_by_dept.csv")
print("- correlation_matrix.csv")
print("- high_attrition_risk.csv")
