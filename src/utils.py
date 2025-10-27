import os
import csv
import yaml
from typing import List, Optional

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_scalar_local(value, out_path: Optional[str], fmt: str):
    if not out_path or not fmt:
        return
    _ensure_dir(out_path)
    if fmt.lower() == 'csv':
        with open(os.path.join(out_path, 'part-00000.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['value'])
            w.writerow([value])

def write_list_local(values: List[str], out_path: Optional[str], fmt: str, col_name: str = 'value'):
    if not out_path or not fmt:
        return
    _ensure_dir(out_path)
    if fmt.lower() == 'csv':
        with open(os.path.join(out_path, 'part-00000.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([col_name])
            for v in values:
                w.writerow([v])

def write_df_local(df, out_path: Optional[str], fmt: str):
    if not out_path or not fmt:
        return
    _ensure_dir(out_path)
    if fmt.lower() == 'csv':
        # Small resultframes only: collect to driver and write
        pdf = df.toPandas()
        pdf.to_csv(os.path.join(out_path, 'part-00000.csv'), index=False)
