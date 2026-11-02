"""Print a reproducible JSON report: python evaluate_models.py [sales.csv]."""
import argparse
import json
from pathlib import Path

from app import DEMO, parse
from forecasting import model_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', help='Optional UTF-8 sales CSV; otherwise use synthetic demo')
    args = parser.parse_args()
    rows = parse(Path(args.csv).read_text(encoding='utf-8')) if args.csv else DEMO
    result = model_evaluation(rows)
    result['dataSource'] = 'uploaded_csv_unverified' if args.csv else 'synthetic_demo'
    result['synthetic'] = not bool(args.csv)
    print(json.dumps(result, indent=2, allow_nan=False))
