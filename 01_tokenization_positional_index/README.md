# Module 01 — Tokenization & Positional Index (PySpark)

**Owner:** Part A (Assigned Role)
**Domain:** Education (`d:\3. Education\d1.txt` – `d10.txt`)

## Responsibilities
1. Read all `.txt` documents from the Education dataset into PySpark.
2. Apply preprocessing (`lowercase`, whitespace trimming) and whitespace tokenization (all tokens preserved, no stop-word removal).
3. Track 1-based left-to-right token positions inside each document.
4. Build the Positional Index (`term -> {doc_id: [positions]}`) ordered naturally from `d1` to `d10`.
5. Export integration-ready files to `output/`:
   - `output/positional_index.json`
   - `output/positional_index_spark.jsonl`
   - `output/term_positions.csv`
   - `output/positional_index_report.txt`

## Run
```bash
python main.py
# or directly:
python 01_tokenization_positional_index/positional_index.py
```
