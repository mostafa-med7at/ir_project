"""
IR Project — Education Domain (Part A / Module 01)
Tokenization & Positional Index Construction using PySpark

Implements the Education version specifications from:
  IR_Project_Requirements_and_Expected_Outputs_Fall2026.pdf
    - Lowercase and whitespace tokenization (all tokens preserved, no stop-word removal)
    - 1-based left-to-right token positions per document
    - Positional Index mapping each term to document IDs and sorted position lists
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

# Ensure PySpark workers (if ever invoked) match the current Python executable
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = r"d:\3. Education"
OUTPUT_DIR = str(PROJECT_ROOT / "output")


def create_spark_session(app_name: str = "IR_Education_PositionalIndex") -> SparkSession:
    """Initialize and configure a minimal-footprint local PySpark session."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[1]")
        .config("spark.driver.memory", "512m")
        .config("spark.driver.extraJavaOptions", "-XX:+UseSerialGC -XX:ReservedCodeCacheSize=64m")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _doc_sort_key(filename: str) -> tuple[int, str]:
    """Sort document filenames numerically by ID (e.g. d1, d2, ..., d10)."""
    match = re.search(r"(\d+)", filename)
    return (int(match.group(1)) if match else 0, filename)


def load_documents(spark: SparkSession, data_dir: str) -> DataFrame:
    """
    Load all .txt documents into a PySpark DataFrame with
    (doc_id, doc_num, raw_text, word_count).

    Uses Spark's native wholetext reader when Hadoop native I/O is available;
    on Windows without HADOOP_HOME/winutils.exe (where Hadoop NativeIO.Windows.access0
    raises UnsatisfiedLinkError), reads the .txt files on the driver and builds the
    Spark DataFrame directly.
    """
    use_hadoop_reader = os.name != "nt" or bool(os.environ.get("HADOOP_HOME"))

    if use_hadoop_reader:
        glob_path = os.path.join(data_dir, "*.txt").replace("\\", "/")
        df = (
            spark.read.option("wholetext", "true")
            .text(glob_path)
            .withColumn("doc_id", F.regexp_extract(F.input_file_name(), r"([^/\\]+)\.txt$", 1))
            .withColumn("raw_text", F.trim(F.col("value")))
            .drop("value")
        )
    else:
        txt_files = sorted(
            (f for f in os.listdir(data_dir) if f.lower().endswith(".txt")),
            key=_doc_sort_key,
        )
        records = []
        for fname in txt_files:
            doc_id = os.path.splitext(fname)[0]
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                records.append((doc_id, fh.read().strip()))

        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("raw_text", StringType(), False),
        ])
        df = spark.createDataFrame(records, schema=schema)

    return (
        df
        .withColumn(
            "doc_num",
            F.regexp_extract(F.col("doc_id"), r"(\d+)", 1).cast(IntegerType()),
        )
        .withColumn(
            "word_count",
            F.when(F.trim(F.col("raw_text")) == "", F.lit(0)).otherwise(
                F.size(F.split(F.trim(F.col("raw_text")), r"\s+"))
            ),
        )
    )


def preprocess_and_tokenize(docs_df: DataFrame) -> DataFrame:
    """
    Apply project preprocessing & tokenization in Spark SQL (JVM-native, no Python UDF):
      1. Lowercase and trim whitespace
      2. Split on whitespace
      3. Assign 1-based positions from left to right using posexplode
      4. Keep every token (matching the PDF reference conventions; no stop-word removal)
    """
    normalized_df = docs_df.withColumn("clean_text", F.trim(F.lower(F.col("raw_text"))))

    return (
        normalized_df
        .select(
            "doc_id",
            "doc_num",
            F.posexplode(F.split(F.col("clean_text"), r"\s+")).alias("pos0", "term"),
        )
        .withColumn("position", F.col("pos0") + F.lit(1))
        .drop("pos0")
        .filter(F.col("term") != "")
    )


def build_positional_index(exploded_df: DataFrame) -> DataFrame:
    """
    Group term occurrences by (term, doc_id) to collect sorted position lists,
    then aggregate per term with postings sorted naturally by doc_num (d1..d10).
    """
    positions_per_doc = (
        exploded_df
        .groupBy("term", "doc_num", "doc_id")
        .agg(F.sort_array(F.collect_list("position")).alias("positions"))
    )

    return (
        positions_per_doc
        .groupBy("term")
        .agg(
            F.count("doc_id").alias("df"),
            F.expr(
                "transform("
                "  sort_array(collect_list(struct(doc_num, doc_id, positions))),"
                "  x -> named_struct('doc_id', x.doc_id, 'positions', x.positions)"
                ")"
            ).alias("postings"),
        )
        .orderBy("term")
    )


def format_postings_line(postings) -> str:
    """Format a term's postings list as 'd1: 1; d2: 1, 5' matching the reference PDF."""
    return "; ".join(
        f"{p['doc_id']}: {', '.join(str(pos) for pos in p['positions'])}"
        for p in postings
    )


def save_outputs(
    dataset_rows: list,
    index_rows: list,
    term_position_rows: list,
    output_dir: str,
) -> dict[str, str]:
    """
    Save the Positional Index in formats suitable for downstream modules and inspection:
      - positional_index.json         : {term: {doc_id: [positions]}}
      - positional_index_spark.jsonl  : line-delimited JSON records with term, df, postings
      - term_positions.csv            : flat (doc_id, position, term) table
      - positional_index_report.txt   : formatted text report matching PDF Part 2
    """
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "positional_index.json")
    index_dict = {
        row["term"]: {p["doc_id"]: list(p["positions"]) for p in row["postings"]}
        for row in index_rows
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(index_dict, jf, indent=2, ensure_ascii=False)

    jsonl_path = os.path.join(output_dir, "positional_index_spark.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jlf:
        for row in index_rows:
            record = {
                "term": row["term"],
                "df": int(row["df"]),
                "postings": [
                    {"doc_id": p["doc_id"], "positions": list(p["positions"])}
                    for p in row["postings"]
                ],
            }
            jlf.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = os.path.join(output_dir, "term_positions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["doc_id", "position", "term"])
        for row in term_position_rows:
            writer.writerow([row["doc_id"], row["position"], row["term"]])

    txt_path = os.path.join(output_dir, "positional_index_report.txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write("ASSIGNED DATASET - Education Domain\n")
        tf.write("=" * 116 + "\n")
        tf.write(f"{'File':<6} | {'Exact document text':<94} | {'Word count'}\n")
        tf.write("-" * 116 + "\n")
        for row in dataset_rows:
            tf.write(f"{row['doc_id']:<6} | {row['raw_text']:<94} | {row['word_count']}\n")
        tf.write("\n")

        tf.write("POSITIONAL INDEX - Education Domain\n")
        tf.write("=" * 72 + "\n")
        tf.write(f"{'Term':<15} | {'DF':<3} | Document IDs and 1-based positions\n")
        tf.write("-" * 72 + "\n")
        for row in index_rows:
            postings_str = format_postings_line(row["postings"])
            tf.write(f"{row['term']:<15} | {row['df']:<3} | {postings_str}\n")

    return {
        "json": json_path,
        "jsonl": jsonl_path,
        "csv": csv_path,
        "txt": txt_path,
    }


def run_pipeline(data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR) -> dict[str, str]:
    """Execute Part A (Tokenization & Positional Index) end-to-end using PySpark."""
    spark = create_spark_session()
    try:
        print("=" * 72)
        print("  IR Project - Tokenization & Positional Index (PySpark)")
        print("  Domain: Education")
        print("=" * 72)

        docs_df = load_documents(spark, data_dir).cache()
        doc_count = docs_df.count()
        assigned_dataset_df = docs_df.orderBy("doc_num").select("doc_id", "raw_text", "word_count")

        print("\n[1] Assigned Dataset (doc_id | raw_text | word_count):")
        assigned_dataset_df.show(doc_count, truncate=False)

        exploded_df = preprocess_and_tokenize(docs_df).cache()
        total_tokens = exploded_df.count()
        ordered_tokens_df = exploded_df.orderBy("doc_num", "position").select("doc_id", "position", "term")

        print("[2] Tokenization & Position Tracking (doc_id | position | term):")
        ordered_tokens_df.show(total_tokens, truncate=False)

        positional_index_df = build_positional_index(exploded_df).cache()
        vocab_size = positional_index_df.count()

        dataset_rows = assigned_dataset_df.collect()
        index_rows = positional_index_df.collect()
        term_position_rows = ordered_tokens_df.collect()

        print("[3] POSITIONAL INDEX -- Education Domain:")
        print("=" * 72)
        print(f"  {'Term':<15} | {'DF':<3} | Document IDs and 1-based positions")
        print("  " + "-" * 68)
        for row in index_rows:
            print(f"  {row['term']:<15} | {row['df']:<3} | {format_postings_line(row['postings'])}")
        print("=" * 72)

        saved_paths = save_outputs(dataset_rows, index_rows, term_position_rows, output_dir)
        print("\n[4] Saved Output Files:")
        for fmt, path in saved_paths.items():
            print(f"  {fmt.upper():<5} -> {path}")

        print("\n" + "=" * 72)
        print("  SUMMARY STATISTICS")
        print("=" * 72)
        print(f"  Documents indexed     : {doc_count}")
        print(f"  Vocabulary size       : {vocab_size} unique terms")
        print(f"  Total token occur.    : {total_tokens}")
        print("=" * 72)
        print("\n[Done] Positional Index construction complete.")
        return saved_paths
    finally:
        spark.stop()


if __name__ == "__main__":
    run_pipeline()
