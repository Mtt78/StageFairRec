# Data preparation contract

Obtain the original data under its access conditions:
[MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX) and
[User-Activity](http://moocdata.cn/data/user-activity), as listed in the manuscript.
This repository does not download, redistribute or invent demographic records.

Use UTF-8 CSV with headers. IDs are opaque strings. Courses include all items to
be ranked, including items without retained training interactions. Sensitive labels
are accessed only by fairness losses and audits, never by input feature encoders.

`courses.csv`:

```csv
course_id,name,about,field
c001,Linear Algebra,An introduction to matrices,Mathematics
c002,Python Basics,Introductory programming,Computer Science
```

For User-Activity supply `course_id,type,category` instead. Every non-ID field is
serialized deterministically in sorted field-name order. Do not put sensitive
user information into this table.

`users.csv`:

```csv
user_id,gender,age
u001,0,22
u002,1,35
```

The 0/1 assignment must follow the data provider's documented labels. It must not
be guessed from usernames or course selections. Blank/invalid gender records are
excluded for gender experiments. Age is numeric in years; invalid, missing, or
outside (0,120) is excluded for age experiments. Run one protected attribute per
prepared dataset/model; this code does not assert simultaneous intersectional fairness.

`interactions.csv`:

```csv
user_id,course_id,timestamp
u001,c001,1600000000
u001,c002,1600086400
```

Use numeric timestamps in one consistent unit; ISO strings must be converted first.
Stable input order breaks equal timestamps. Repeated events are retained. Unknown
course references fail loudly. Missing user labels are excluded.

```bash
stagefairrec prepare --raw data/raw --out data/gender --attribute gender
stagefairrec prepare --raw data/raw --out data/age --attribute age --age-bins 20 30 40
```

The age boundaries above are **only an example**, not values recovered from the
paper. Values equal to a boundary fall in the higher group. Empty groups are
removed and labels re-indexed; `metadata.json` records retained group values.

If source exports are JSONL, use `scripts/convert_jsonl.py` with an explicit schema
mapping. It supports dotted fields and optional expansion of course/time arrays;
it does not assume a specific version of the original datasets. Example schemas
are in `configs/jsonl_schema.example.json`. Verify actual raw fields first.

Outputs:

- `metadata.json`: stable mappings, text, chronological sequences, split cutoffs,
  retained labels, group mapping, and protocol details.
- `rows.npy`: int64 columns `[user_index, prefix_length, target_item, split, candidate_index]`.
  Splits are 0=train, 1=valid, 2=test; candidate index -1 for train.
- `candidates.npy`: target in first column followed by 99 distinct unseen items.
- `manifest.json`: dataset hash and synthetic-data marker.

Item 0 is padding. The backbone reserves `num_items+1` as BERT MASK, and neither
padding nor MASK is ranked. Sequential training skips empty prefixes; collaborative
training uses every training positive. All methods reuse the same candidate file.
Known future positives are excluded from the negative pool only, following the
paper's unobserved-course candidate definition; future history is not encoded.

The canonical converter cannot guarantee the exact published processed counts
without the author's original raw snapshots and filtering rules. If eligible users
have fewer than 99 unseen courses, preparation stops. Reduce `--eval-negatives`
only for a clearly labeled non-paper test.
