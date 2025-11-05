# save as clean_task1.py and run: python clean_task1.py
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- SETTINGS ----------
RAW_FILE = "raw_data.csv"          # <-- change to your original file name
CLEAN_FILE = "cleaned_data.csv"
SUMMARY_FILE = "cleaning_summary.txt"
# ------------------------------

def load_data(path):
    df = pd.read_csv(path)
    return df

def initial_report(df):
    report = {}
    report['shape'] = df.shape
    report['columns'] = df.columns.tolist()
    report['dtypes'] = df.dtypes.astype(str).to_dict()
    report['null_counts'] = df.isnull().sum().to_dict()
    report['duplicate_count'] = df.duplicated().sum()
    # small head/tail preview
    report['head'] = df.head(3).to_dict(orient='list')
    return report

def save_summary_text(summary_dict, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("DATA CLEANING SUMMARY\n")
        f.write("=====================\n\n")
        for k, v in summary_dict.items():
            f.write(f"{k}:\n{v}\n\n")
    print(f"Summary written to {filename}")

def standardize_column_names(df):
    df = df.rename(columns=lambda x: str(x).strip().lower().replace(' ', '_'))
    return df

def remove_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    return df, before - after

def handle_missing_values(df, threshold_drop_col=0.6):
    """
    - Drop columns with > threshold_drop_col fraction missing.
    - For numeric columns: fill with median.
    - For object/text columns: fill with 'Unknown' or mode.
    - For small % missing in key columns, you might drop rows.
    """
    summary = {'dropped_columns': [], 'filled_columns': []}

    # drop columns with too many nulls
    col_null_frac = df.isnull().mean()
    cols_to_drop = col_null_frac[col_null_frac > threshold_drop_col].index.tolist()
    df = df.drop(columns=cols_to_drop)
    summary['dropped_columns'] = cols_to_drop

    # numeric fills
    for col in df.select_dtypes(include=[np.number]).columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            summary['filled_columns'].append((col, 'median', int(median) if not pd.isnull(median) else None))

    # object/text fills
    for col in df.select_dtypes(include=['object', 'string']).columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            mode = df[col].mode()
            if len(mode) > 0:
                fill = mode.iloc[0]
            else:
                fill = "Unknown"
            df[col] = df[col].fillna(fill)
            summary['filled_columns'].append((col, 'mode_or_unknown', str(fill)))

    return df, summary

def standardize_text_columns(df, text_columns=None, mapping_dicts=None):
    """
    - Lowercase and strip whitespace.
    - Optionally apply mapping dicts to unify values (e.g., gender).
    mapping_dicts is a dict: {'gender': {'m': 'male', 'female ': 'female', ...}, ...}
    """
    if text_columns is None:
        text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip().str.lower().replace({'nan': np.nan})
    if mapping_dicts:
        for col, mp in mapping_dicts.items():
            if col in df.columns:
                df[col] = df[col].replace(mp).astype('string')
    return df

def convert_dates(df, date_cols):
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)  # set dayfirst=True if dates like dd-mm-yyyy
    return df

def fix_dtypes(df, dtype_map=None):
    """
    dtype_map example: {'age': 'Int64', 'salary': 'float'}
    """
    if dtype_map:
        for col, dt in dtype_map.items():
            if col in df.columns:
                try:
                    if 'int' in dt.lower():
                        df[col] = pd.to_numeric(df[col], errors='coerce').round(0).astype('Int64')
                    elif 'float' in dt.lower():
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float')
                    elif dt.lower() in ('datetime', 'datetime64'):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(dt)
                except Exception as e:
                    print(f"Warning: could not convert {col} to {dt}: {e}")
    return df

def detect_treat_outliers_iqr(df, numeric_cols=None, factor=1.5):
    """
    Simple IQR outlier detection and cap using winsorization.
    Returns modified df and a summary of changes.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out_summary = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty: 
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        before_extremes = ((df[col] < lower) | (df[col] > upper)).sum()
        # cap values
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
        out_summary[col] = {'lower_cap': float(lower), 'upper_cap': float(upper), 'capped_count': int(before_extremes)}
    return df, out_summary

def main():
    # 1. Load
    p = Path(RAW_FILE)
    if not p.exists():
        print(f"File {RAW_FILE} not found. Put your dataset in the same folder and rename it {RAW_FILE} or change RAW_FILE variable.")
        return
    df = load_data(RAW_FILE)
    init = initial_report(df)
    print("Initial shape:", init['shape'])
    print("Null counts (top 10):", dict(sorted(init['null_counts'].items(), key=lambda x: -x[1])[:10]))
    print("Duplicate rows:", init['duplicate_count'])

    # 2. Column renaming standardization
    df = standardize_column_names(df)

    # 3. Remove duplicates
    df, removed_dup = remove_duplicates(df)
    print(f"Removed {removed_dup} duplicate rows.")

    # 4. Missing values handling
    df, missing_summary = handle_missing_values(df, threshold_drop_col=0.6)
    print("Missing handling summary:", missing_summary)

    # 5. Standardize text columns (example mapping for gender/country)
    mappings = {}
    # Example mapping; modify to match your dataset values
    mappings['gender'] = {'m': 'male', 'male': 'male', 'f': 'female', 'female': 'female', 'nan': np.nan}
    df = standardize_text_columns(df, mapping_dicts=mappings)

    # 6. Date conversion: detect columns that look like dates by name
    date_like_cols = [c for c in df.columns if 'date' in c or 'dob' in c or 'joined' in c]
    df = convert_dates(df, date_like_cols)
    print("Converted date columns:", date_like_cols)

    # 7. Fix dtypes: example - try to coerce 'age' to integer if exists
    dtype_map = {}
    if 'age' in df.columns:
        dtype_map['age'] = 'Int64'
    if 'salary' in df.columns:
        dtype_map['salary'] = 'float'
    df = fix_dtypes(df, dtype_map=dtype_map)

    # 8. Outlier detection and capping (numeric)
    df, outlier_summary = detect_treat_outliers_iqr(df)
    print("Outlier summary:", outlier_summary)

    # 9. Final checks
    final_report = initial_report(df)
    print("Final shape:", final_report['shape'])
    print("Final null counts (top 10):", dict(sorted(final_report['null_counts'].items(), key=lambda x: -x[1])[:10]))

    # 10. Save cleaned dataset and summary
    df.to_csv(CLEAN_FILE, index=False)
    summary_dict = {
        'initial_shape': init['shape'],
        'initial_null_counts': init['null_counts'],
        'removed_duplicates': removed_dup,
        'dropped_columns_due_to_missing': missing_summary.get('dropped_columns', []),
        'filled_columns': missing_summary.get('filled_columns', []),
        'outlier_summary': outlier_summary,
        'final_shape': final_report['shape'],
        'final_null_counts': final_report['null_counts']
    }
    save_summary_text(summary_dict, SUMMARY_FILE)
    print(f"Cleaned dataset saved to {CLEAN_FILE}")

if __name__ == "__main__":
    main()
