"""Explicit JSONL field converter. Example mapping is schematic, not a claimed raw schema."""
import argparse
import csv
import json
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--raw', required=True); p.add_argument('--out', required=True)
p.add_argument('--schema', required=True)
a = p.parse_args(); schema = json.loads(Path(a.schema).read_text())
out = Path(a.out)
if out.exists() and any(out.iterdir()):
    raise ValueError('Output must be empty')
out.mkdir(parents=True, exist_ok=True)
def get(record, field):
    value = record
    for component in field.split('.'):
        value = value[component]
    return value
for table in ('courses', 'users', 'interactions'):
    spec = schema[table]
    with open(Path(a.raw) / spec['file'], encoding='utf-8') as source, open(
            out / f'{table}.csv', 'w', encoding='utf-8', newline='') as dest:
        writer = csv.DictWriter(dest, fieldnames=list(spec['fields'])); writer.writeheader()
        for line in source:
            if not line.strip():
                continue
            obj = json.loads(line)
            row = {name: get(obj, path) for name, path in spec['fields'].items()}
            # Optional: expand parallel course_id/timestamp lists from a user history row.
            if spec.get('expand_history', False):
                courses, times = row['course_id'], row['timestamp']
                if len(courses) != len(times):
                    raise ValueError('History course/time lengths differ')
                for item, time in zip(courses, times):
                    writer.writerow(row | {'course_id': item, 'timestamp': time})
            else:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                                 else v for k, v in row.items()})
