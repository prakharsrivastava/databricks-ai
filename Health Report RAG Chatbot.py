# Databricks notebook source
# DBTITLE 1,e


# COMMAND ----------

# DBTITLE 1,Health Report RAG Chatbot
# MAGIC %md
# MAGIC # Health Report RAG Chatbot
# MAGIC
# MAGIC A complete RAG pipeline for health report documents: PDF parsing with `ai_parse_document`, structure-aware chunking, Vector Search retrieval, interactive Q&A, LLM-as-judge evaluation, and deployment as a **ChatAgent** to AI Playground with Review App and real-time tracing.
# MAGIC
# MAGIC **Pipeline stages:** PDF parsing → UC tables → chunking → Vector Search index → RAG chatbot → evaluation (RAGAS + NDCG + MMR) → MLflow tracing → ChatAgent deployment

# COMMAND ----------

# DBTITLE 1,Install + Imports
# MAGIC %pip install -q databricks-sdk mlflow databricks-agents pymupdf
# MAGIC import mlflow
# MAGIC import mlflow.deployments
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC deploy_client = mlflow.deployments.get_deploy_client("databricks")
# MAGIC
# MAGIC print(f"MLflow version: {mlflow.__version__}")
# MAGIC import databricks.agents
# MAGIC print(f"databricks-agents version: {databricks.agents.__version__}")

# COMMAND ----------

# DBTITLE 1,Config
# --- Config: all health_report prefixed ---
CATALOG = "workspace"
SCHEMA = "default"

# Uploaded health report PDF (IDBFS path from file upload)
PDF_SOURCE = "idbfs:/2026-09-29/05/_66e4facc-5c17-4691-8587-dffddd8e65a5"

# Vector Search (dedicated endpoint for fast search)
VS_ENDPOINT_NAME = "prakhar_vs_endpoint"
EMBEDDING_MODEL = "databricks-gte-large-en"

# Hybrid retrieval pipeline: vector search + keyword search → reranker → LLM
CANDIDATE_K = 20           # initial retrieval candidates (vector + keyword)
FINAL_K = 8                # chunks after reranking
MAX_CONTEXT_CHARS = 500    # truncate each chunk in context (more context for richer answers)
MAX_TOKENS = 512           # richer LLM responses with citations
CHUNK_MAX_CHARS = 300      # max chars per chunk during chunking
USEFUL_TYPES = {"text", "table", "section_header", "caption"}  # skip figure/footer/page_number
WAREHOUSE_ID = "807f80ae0b4f84a8"  # serverless SQL warehouse for keyword search
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
RERANKER_LLM = LLM_ENDPOINT  # use LLM-as-reranker (no dedicated reranker endpoint yet)
TOP_K = 3                # chunks to retrieve for RAG chat

# Derived names (all health_report prefixed)
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"
CHUNKS_TABLE = f"{FULL_SCHEMA}.health_report_pdf_chunks"
VS_INDEX_NAME = f"{FULL_SCHEMA}.health_report_chunks_index"

print(f"PDF source   : {PDF_SOURCE}")
print(f"Chunks table : {CHUNKS_TABLE}")
print(f"VS endpoint  : {VS_ENDPOINT_NAME}")
print(f"VS index     : {VS_INDEX_NAME}")
print(f"LLM          : {LLM_ENDPOINT}")
print(f"Top-K        : {TOP_K}")
print(f"Max ctx chars: {MAX_CONTEXT_CHARS}")
print(f"Max tokens   : {MAX_TOKENS}")

# COMMAND ----------

# DBTITLE 1,Verify PDF exists
# Verify health report PDF exists
try:
    items = dbutils.fs.ls(PDF_SOURCE)
    for item in items:
        print(f"Found: {item.path}  (size={item.size} bytes)")
except Exception as e:
    # Try as a single file
    try:
        file_info = dbutils.fs.head(PDF_SOURCE, 100)
        print(f"PDF found at: {PDF_SOURCE}")
        print(f"First 100 bytes readable: yes")
    except:
        print(f"Could not verify PDF at: {PDF_SOURCE}")
        print(f"Error: {e}")

# COMMAND ----------

# DBTITLE 1,Parse PDF
# Parse health report PDF with ai_parse_document (structure-aware)
pdf_parsed_sql = f"""
WITH raw_pdf AS (
  SELECT
    _metadata.file_name AS file_name,
    ai_parse_document(content, MAP('version', '2.0')) AS parsed
  FROM READ_FILES('{PDF_SOURCE}', format => 'binaryFile')
),
valid_pdf AS (
  SELECT * FROM raw_pdf
  WHERE is_variant_null(parsed:error_status)
),
element_rows AS (
  SELECT
    file_name,
    idx AS element_idx,
    try_cast(elem:type AS STRING) AS element_type,
    try_cast(elem:content AS STRING) AS element_text,
    try_cast(elem:bbox[0]:page_id AS INT) + 1 AS page_number
  FROM valid_pdf,
    LATERAL posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (idx, elem)
)
SELECT * FROM element_rows ORDER BY file_name, element_idx
"""

pdf_elements_df = spark.sql(pdf_parsed_sql)
print(f"PDF elements parsed: {pdf_elements_df.count()}")
display(pdf_elements_df.limit(20))

# COMMAND ----------

# DBTITLE 1,Create UC tables
# Create permanent Unity Catalog tables from the parsed PDF
spark.sql(f"""
CREATE OR REPLACE TABLE {FULL_SCHEMA}.health_report_pdf_parsed AS
SELECT
  _metadata.file_name AS file_name,
  content,
  ai_parse_document(content, MAP('version', '2.0')) AS parsed_content
FROM READ_FILES('{PDF_SOURCE}', format => 'binaryFile')
WHERE is_variant_null(ai_parse_document(content, MAP('version', '2.0')):error_status)
""")
print(f"Created table: {FULL_SCHEMA}.health_report_pdf_parsed")

spark.sql(f"""
CREATE OR REPLACE TABLE {FULL_SCHEMA}.health_report_pdf_elements AS
WITH raw_pdf AS (
  SELECT
    _metadata.file_name AS file_name,
    ai_parse_document(content, MAP('version', '2.0')) AS parsed
  FROM READ_FILES('{PDF_SOURCE}', format => 'binaryFile')
),
valid_pdf AS (
  SELECT * FROM raw_pdf
  WHERE is_variant_null(parsed:error_status)
),
element_rows AS (
  SELECT
    file_name,
    idx AS element_idx,
    try_cast(elem:type AS STRING) AS element_type,
    try_cast(elem:content AS STRING) AS element_text,
    try_cast(elem:bbox[0]:page_id AS INT) + 1 AS page_number
  FROM valid_pdf,
    LATERAL posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (idx, elem)
)
SELECT * FROM element_rows ORDER BY file_name, element_idx
""")
print(f"Created table: {FULL_SCHEMA}.health_report_pdf_elements")

count = spark.sql(f"SELECT COUNT(*) FROM {FULL_SCHEMA}.health_report_pdf_elements").collect()[0][0]
print(f"health_report_pdf_elements row count: {count}")
display(spark.sql(f"SELECT * FROM {FULL_SCHEMA}.health_report_pdf_elements LIMIT 20"))

# COMMAND ----------

# DBTITLE 1,Structure-aware chunking
# Structure-aware chunking: filter noise + strip HTML + split long text for minimal context
import hashlib
import re

def strip_html(text):
    """Convert HTML table markup to readable plain text."""
    # Replace table cells with pipe-separated text
    text = re.sub(r'<tr>', ' | ', text)
    text = re.sub(r'</tr>', '', text)
    text = re.sub(r'<td[^>]*>', ' ', text)
    text = re.sub(r'</td>', '', text)
    text = re.sub(r'<th[^>]*>', ' ', text)
    text = re.sub(r'</th>', '', text)
    text = re.sub(r'<[^>]+>', '', text)  # strip remaining tags
    text = re.sub(r'\s+', ' ', text).strip()  # collapse whitespace
    return text

pdf_elements_df = spark.sql("SELECT * FROM workspace.default.health_report_pdf_elements")
pdf_pandas = pdf_elements_df.toPandas()

section_counter = 0
current_section_id = None
chunk_rows = []

def make_chunk(file_name, element_idx, page, section_id, chunk_text, chunk_type):
    chunk_id = hashlib.sha256(f"{file_name}|{element_idx}|{chunk_text}".encode()).hexdigest()[:16]
    content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
    chunk_rows.append({
        "chunk_id": chunk_id, "doc_id": file_name, "page": int(page),
        "section_id": section_id, "chunk_text": chunk_text[:CHUNK_MAX_CHARS],
        "chunk_type": chunk_type, "content_hash": content_hash,
        "file_path": PDF_SOURCE, "file_name": file_name,
        "file_version": 1, "acl": "public", "source_type": "pdf",
    })

for _, row in pdf_pandas.iterrows():
    el_type = row["element_type"] or "text"
    el_text = row["element_text"] or ""
    page = row["page_number"] or 1

    if el_type not in USEFUL_TYPES:
        continue
    if not el_text or not el_text.strip():
        continue

    if el_type == "section_header":
        section_counter += 1
        current_section_id = f"sec_{section_counter:03d}"
        make_chunk(row["file_name"], row["element_idx"], page, current_section_id, el_text.strip(), el_type)
    elif el_type == "table":
        clean_text = strip_html(el_text.strip())
        make_chunk(row["file_name"], row["element_idx"], page, current_section_id or "sec_root", f"[Table p{page}] {clean_text}", el_type)
    else:
        text = el_text.strip()
        section_prefix = f"[Section: {current_section_id}] " if current_section_id else ""
        if len(text) > CHUNK_MAX_CHARS:
            words = text.split()
            current_chunk = section_prefix
            for word in words:
                if len(current_chunk) + len(word) + 1 > CHUNK_MAX_CHARS and len(current_chunk) > len(section_prefix):
                    make_chunk(row["file_name"], row["element_idx"], page, current_section_id or "sec_root", current_chunk, "text")
                    current_chunk = ""
                current_chunk += (" " + word) if current_chunk else word
            if current_chunk and len(current_chunk) > len(section_prefix):
                make_chunk(row["file_name"], row["element_idx"], page, current_section_id or "sec_root", current_chunk, "text")
        else:
            make_chunk(row["file_name"], row["element_idx"], page, current_section_id or "sec_root", section_prefix + text, el_type)

health_chunks_df = spark.createDataFrame(chunk_rows)
print(f"Health report chunks created: {health_chunks_df.count()}")
display(health_chunks_df.limit(10))

# COMMAND ----------

# DBTITLE 1,Save + optimize chunks
# Save chunks to UC table + liquid clustering for fast RAG retrieval
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_SCHEMA}")
health_chunks_df.write.mode("overwrite").saveAsTable(CHUNKS_TABLE)
print(f"Saved {health_chunks_df.count()} chunks to {CHUNKS_TABLE}")

# Apply liquid clustering on retrieval-relevant columns
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} CLUSTER BY (page, section_id, chunk_type)")
spark.sql(f"OPTIMIZE {CHUNKS_TABLE}")
print(f"Liquid clustering applied on (page, section_id, chunk_type)")


display(spark.sql(f"""
SELECT chunk_type, COUNT(*) AS cnt, MIN(page) AS min_pg, MAX(page) AS max_pg
FROM {CHUNKS_TABLE}
GROUP BY chunk_type
ORDER BY chunk_type
"""))

# COMMAND ----------

# DBTITLE 1,Extract Structured Lab Data
# Extract structured lab data from PDF table chunks using LLM
# Creates: workspace.default.health_report_lab_data (test_name, value, unit, reference_range, status, page)
import json
import re

LAB_DATA_TABLE = f"{FULL_SCHEMA}.health_report_lab_data"

# Get all table chunks that look like lab test results
table_chunks = spark.sql(f"""
    SELECT chunk_text, page, section_id 
    FROM {CHUNKS_TABLE} 
    WHERE chunk_type = 'table'
    AND (chunk_text ILIKE '%Test Name%' OR chunk_text ILIKE '%Bio. Ref%' OR chunk_text ILIKE '%Value%Unit%')
    ORDER BY page, section_id
""").collect()

print(f"Found {len(table_chunks)} lab data table chunks")

lab_records = []
for chunk in table_chunks:
    text = chunk['chunk_text']
    page = int(chunk['page'])
    section = chunk['section_id']
    
    prompt = f"""Extract lab test results from this health report table text. 
Return a JSON array of objects with fields: test_name (string), value (string), unit (string), reference_range (string), status (string: "normal"/"abnormal"/"borderline").
Only include actual test results with numeric values, not reference guidelines or patient info.

Table text: {text[:1500]}

Return ONLY the JSON array, no other text."""

    try:
        response = deploy_client.predict(
            endpoint=LLM_ENDPOINT,
            inputs={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.0,
            },
        )
        raw = response["choices"][0]["message"]["content"].strip()
        json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if json_match:
            records = json.loads(json_match.group())
            for r in records:
                r['page'] = page
                r['section_id'] = section
                lab_records.append(r)
    except Exception as e:
        print(f"  Error parsing p{page} {section}: {e}")

print(f"\nExtracted {len(lab_records)} lab test records")

# Create Delta table via SQL (safe: only creates if not exists)
if lab_records:
    lab_df = spark.createDataFrame(lab_records)
    lab_df.createOrReplaceTempView("lab_data_temp")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {LAB_DATA_TABLE} AS SELECT * FROM lab_data_temp")
    print(f"Saved to {LAB_DATA_TABLE}")
    display(spark.sql(f"SELECT * FROM {LAB_DATA_TABLE} ORDER BY page LIMIT 20"))
else:
    print("No lab records extracted")

# COMMAND ----------

# DBTITLE 1,Create VS index
# Create Vector Search index on dedicated endpoint (skip if already ready)
import time

vs_client = w.vector_search_indexes

# Check if index already exists and is ready
try:
    idx = vs_client.get_index(index_name=VS_INDEX_NAME)
    ready = idx.status.ready if idx.status else False
    rows = idx.status.indexed_row_count if idx.status else 0
    if ready and rows > 0:
        print(f"VS index already READY with {rows} rows — skipping creation.")
    else:
        raise Exception("Index not ready, will recreate.")
except:
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn, PipelineType, VectorIndexType
    )
    try:
        vs_client.delete_index(index_name=VS_INDEX_NAME)
        print(f"Deleted old index: {VS_INDEX_NAME}")
    except:
        pass
    print(f"Creating VS index on {VS_ENDPOINT_NAME}...")
    vs_client.create_index(
        name=VS_INDEX_NAME,
        endpoint_name=VS_ENDPOINT_NAME,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=CHUNKS_TABLE,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="chunk_text",
                    embedding_model_endpoint_name=EMBEDDING_MODEL,
                )
            ],
        ),
    )
    print(f"Created VS index: {VS_INDEX_NAME}")
    print("Waiting for index to be ready...")
    for attempt in range(20):
        time.sleep(30)
        try:
            idx = vs_client.get_index(index_name=VS_INDEX_NAME)
            ready = idx.status.ready if idx.status else False
            rows = idx.status.indexed_row_count if idx.status else 0
            msg = (idx.status.message or "")[:80] if idx.status else ""
            print(f"  [{(attempt+1)*30}s] ready={ready}, rows={rows}, msg={msg}")
            if ready and rows > 0:
                print(f"Index READY with {rows} rows!")
                break
            if "failed" in msg.lower():
                print(f"Index FAILED: {msg}")
                break
        except Exception as e:
            print(f"  [{(attempt+1)*30}s] Error: {e}")

# COMMAND ----------

# DBTITLE 1,RAG Chatbot with Source Visualization
# RAG Chatbot: Ask questions about your health report
# Shows: retrieved chunks with metadata (scores, type, page, section) + PDF source page snapshots

import pymupdf
from IPython.display import Image, display

VS_INDEX = VS_INDEX_NAME
LLM_EP = LLM_ENDPOINT
TOP_K = 3  # minimal retrieval for fast search

# Load PDF for page image extraction
_pdf_row = spark.sql("SELECT content FROM workspace.default.health_report_pdf_parsed LIMIT 1").collect()[0]
with open("/tmp/health_report.pdf", "wb") as f:
    f.write(_pdf_row["content"])
_pdf_doc = pymupdf.open("/tmp/health_report.pdf")
print(f"PDF loaded: {len(_pdf_doc)} pages for source snapshots")

def show_source_page(page_num):
    """Extract and display a PDF page as an image snapshot."""
    try:
        page_idx = int(page_num) - 1
        if 0 <= page_idx < len(_pdf_doc):
            page = _pdf_doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img_path = f"/tmp/health_report_p{int(page_num)}.png"
            pix.save(img_path)
            display(Image(filename=img_path, width=500))
            return True
    except Exception as e:
        print(f"  (Could not extract page image: {e})")
    return False

def retrieve_chunks(query, top_k=TOP_K):
    """Retrieve relevant content chunks from the Vector Search index."""
    results = w.vector_search_indexes.query_index(
        index_name=VS_INDEX,
        columns=["chunk_id", "chunk_text", "chunk_type", "page", "section_id", "file_name"],
        query_text=query,
        num_results=top_k,
        filters_json='{"chunk_type": ["text", "table", "section_header"]}',
    )
    return results.result.data_array

def retrieve_lab_data(query):
    """Retrieve structured lab test data matching the query keywords."""
    LAB_TABLE = f"{FULL_SCHEMA}.health_report_lab_data"
    stop = {"what", "is", "the", "are", "of", "in", "to", "a", "an", "and",
            "or", "level", "test", "my", "report", "serum", "about", "tell",
            "me", "show", "explain", "meaning", "mean", "does", "do", "why"}
    words = [w.lower().strip(".,!?;:()[]{}\"'\'") for w in query.split()]
    keywords = [w for w in words if w not in stop and len(w) > 2]
    if not keywords:
        return []
    like_clauses = " OR ".join(f"test_name ILIKE '%{k.replace("'", "''")}%'" for k in keywords[:5])
    try:
        results = spark.sql(f"""
            SELECT test_name, value, unit, reference_range, status, page
            FROM {LAB_TABLE}
            WHERE {like_clauses}
            ORDER BY page
        """).collect()
        return results
    except Exception as e:
        print(f"  (Lab data lookup error: {e})")
        return []

def health_rag_chat(question, top_k=TOP_K):
    """Ask a question and get an answer grounded in your health report."""
    chunks = retrieve_chunks(question, top_k)
    lab_data = retrieve_lab_data(question)
    if not chunks and not lab_data:
        print("No relevant chunks or lab data found.\n")
        return None

    # === SHOW RETRIEVED CHUNKS WITH FULL METADATA ===
    print("=" * 70)
    print(f"RETRIEVED CHUNKS (top-{top_k} from Vector Search)")
    print("=" * 70)

    context_parts = []
    sources = []
    source_pages = set()

    for i, row in enumerate(chunks):
        chunk_id, chunk_text, chunk_type, page, section_id, file_name = row[:6]
        score = row[-1]

        context_parts.append(f"[S{i+1}] p{page} {section_id}\n{chunk_text[:MAX_CONTEXT_CHARS]}")
        sources.append(f"  {i+1}. Page {page}, Section {section_id} ({chunk_type}, score: {score:.4f})")
        source_pages.add(int(page))

        print(f"\n[Chunk {i+1}]")
        print(f"  Chunk ID    : {chunk_id}")
        print(f"  Type        : {chunk_type}")
        print(f"  Page        : {page}")
        print(f"  Section     : {section_id}")
        print(f"  Score       : {score:.4f}")
        print(f"  Full text   : {chunk_text[:500]}")
        print(f"  Context used: {chunk_text[:MAX_CONTEXT_CHARS]}")

    # === ADD STRUCTURED LAB DATA TO CONTEXT ===
    if lab_data:
        print("\n" + "=" * 70)
        print("STRUCTURED LAB DATA (from health_report_lab_data table)")
        print("=" * 70)
        lab_parts = []
        for r in lab_data:
            test_name, value, unit, ref_range, status, page = r[0], r[1], r[2], r[3], r[4], r[5]
            lab_parts.append(f"[LAB] p{page} {test_name}: {value} {unit} (ref: {ref_range}, status: {status})")
            print(f"  {test_name}: {value} {unit} | ref: {ref_range} | {status} | p{page}")
        context_parts.append("Structured Lab Data:\n" + "\n".join(lab_parts))

    context = "\n\n".join(context_parts)

    # === SHOW PDF SOURCE PAGE IMAGES ===
    print("\n" + "=" * 70)
    print(f"PDF SOURCE PAGES (snapshots from original document)")
    print("=" * 70)
    for pg in sorted(source_pages):
        print(f"\n--- Page {pg} ---")
        show_source_page(pg)

    # === GENERATE ANSWER ===
    system_prompt = f"""You are a medical report assistant helping a patient understand their health report.

Answer using the provided health report context (document chunks + structured lab data). When the question is about a lab test or biomarker, structure your answer as:

1. **What it is**: A brief, plain-language explanation of what the test/substance is and why it's measured (use your general medical knowledge for this part).
2. **Your report shows**: List each matching test value with its unit, like:
   - Test name: value unit
   Include the reference range and whether the value is normal/abnormal/borderline.
3. **In simple terms**: A one-sentence plain-language summary of what the test is for and whether the patient's values are normal.

Always cite page numbers using (p<page>) format. If the report context is insufficient to answer, say: "I don't have enough information to answer this question based on the report." Be thorough yet concise.

Context:
{context}"""

    response = deploy_client.predict(
        endpoint=LLM_EP,
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.1,
        },
    )
    answer = response["choices"][0]["message"]["content"]

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)
    print(f"\nANSWER:\n{answer}")
    print("\n" + "\u2500" * 70)
    print(f"SOURCES (top-{top_k} retrieved chunks):")
    print("\n".join(sources))
    print("\u2500" * 70 + "\n")
    return answer


# Demo questions
health_rag_chat("What is serum bilirubin?")
health_rag_chat("What is the HbA1c level?")
health_rag_chat("What is the patient's name and age?")

# === BROWSE ALL CHUNKS STORED IN VECTOR DB ===

def parse_vector(vec):
    """Parse a VS scan_index vector into a readable dict."""
    d = vec.as_shallow_dict()
    result = {}
    if 'fields' in d:
        for entry in d['fields']:
            key = entry.key
            val = entry.value
            if val.string_value is not None:
                result[key] = val.string_value
            elif val.number_value is not None:
                result[key] = val.number_value
    return result

def browse_vector_db(show_images=True):
    """Scan and display all chunks stored in the Vector Search index (compact: 2 rows per chunk)."""
    all_vectors = []
    last_pk = None
    for _ in range(25):
        scan_result = w.vector_search_indexes.scan_index(
            index_name=VS_INDEX, num_results=10,
            last_primary_key=last_pk if last_pk else None,
        )
        if not scan_result.data:
            break
        all_vectors.extend(scan_result.data)
        last_pk = scan_result.last_primary_key
        if len(scan_result.data) < 10:
            break
    parsed = [parse_vector(v) for v in all_vectors]
    types = {}
    pages = set()
    for p in parsed:
        ct = p.get('chunk_type', 'unknown')
        types[ct] = types.get(ct, 0) + 1
        pages.add(int(p.get('page', 0)))
    print(f"\nVECTOR DB: {len(all_vectors)} vectors | {dict(sorted(types.items()))} | pages {sorted(pages)}")
    for i, p in enumerate(parsed):
        print(f"[{i+1}] {p.get('chunk_type','?')} p{int(p.get('page',0))} {p.get('section_id','?')} | {p.get('chunk_text','')[:80]}")
    if show_images:
        shown = set()
        print("\nPDF SOURCE SNAPSHOTS:")
        for p in parsed:
            pg = int(p.get('page', 0))
            if pg not in shown and 1 <= pg <= len(_pdf_doc):
                shown.add(pg)
                print(f"--- Page {pg} ---")
                show_source_page(pg)
                if len(shown) >= 3:
                    break
    return parsed

print("\n\n")
all_stored = browse_vector_db()

# COMMAND ----------

# DBTITLE 1,Evaluation
# RAG Chatbot Evaluation with LLM-as-judge
import pandas as pd
import re
import json

def llm_judge(question, answer, context, metric):
    """Score answer quality using LLM-as-judge. Returns (score 1-5, rationale)."""
    if metric == "faithfulness":
        prompt = f"""You are an evaluation judge. Rate the faithfulness of the answer to the retrieved context.

Question: {question}

Retrieved Context:
{context}

Answer: {answer}

Is the answer fully supported by the retrieved context (no hallucination)?
Respond with ONLY a JSON object: {{"score": <1-5>, "rationale": "<one sentence>"}}
5 = fully grounded in context, 1 = complete hallucination."""
    elif metric == "relevancy":
        prompt = f"""You are an evaluation judge. Rate the relevancy of the answer to the question.

Question: {question}

Answer: {answer}

Is the answer relevant and directly responsive to the question?
Respond with ONLY a JSON object: {{"score": <1-5>, "rationale": "<one sentence>"}}
5 = perfectly relevant, 1 = completely irrelevant."""

    try:
        response = deploy_client.predict(
            endpoint=LLM_ENDPOINT,
            inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0},
        )
        raw = response["choices"][0]["message"]["content"].strip()
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            parsed = json.loads(json_match.group())
            return int(parsed.get("score", 0)), parsed.get("rationale", "")
        score_match = re.search(r'(\d)', raw)
        return (int(score_match.group(1)), raw[:80]) if score_match else (0, raw[:80])
    except Exception as e:
        return 0, f"Judge error: {e}"


# Test cases for health report
test_cases = [
    {"question": "What is the HbA1c level?", "expected_facts": ["HbA1c", "target", "level"], "description": "HbA1c test question"},
    {"question": "What is the patient name and age?", "expected_facts": ["Prakhar", "37"], "description": "Patient info question"},
    {"question": "What is the capital of France?", "expected_facts": [], "description": "Out-of-scope (should decline)"},
]

print("=" * 80)
print("HEALTH REPORT RAG EVALUATION")
print("=" * 80)

results = []
for i, tc in enumerate(test_cases, 1):
    q = tc["question"]
    expected = tc["expected_facts"]
    print(f"\n[{i}/{len(test_cases)}] {tc['description']}: \"{q}\"")

    chunks = retrieve_chunks(q)
    context = "\n\n".join(row[1] for row in chunks) if chunks else ""
    avg_sim = sum(row[-1] for row in chunks) / len(chunks) if chunks else 0.0
    chunk_types = list(set(row[2] for row in chunks)) if chunks else []

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        answer = health_rag_chat(q)
    answer = answer or ""

    if expected:
        answer_lower = answer.lower()
        found = [f for f in expected if f.lower() in answer_lower]
        fact_cov = len(found) / len(expected)
    else:
        fact_cov = 1.0 if "don't have enough information" in answer.lower() else 0.0
        found = []

    faith_score, faith_rationale = llm_judge(q, answer, context, "faithfulness")
    rel_score, rel_rationale = llm_judge(q, answer, context, "relevancy")

    results.append({
        "#": i, "description": tc["description"],
        "retrieval_score": round(avg_sim, 4),
        "chunk_types": ", ".join(chunk_types),
        "fact_coverage": f"{fact_cov:.0%}",
        "faithfulness": f"{faith_score}/5", "relevancy": f"{rel_score}/5",
    })
    print(f"  Retrieval: avg_score={avg_sim:.4f}, types={chunk_types}")
    print(f"  Fact coverage: {fact_cov:.0%}")
    print(f"  Faithfulness: {faith_score}/5 -- {faith_rationale}")
    print(f"  Relevancy: {rel_score}/5 -- {rel_rationale}")

display(pd.DataFrame(results))
print(f"\nAvg faithfulness: {sum(int(r['faithfulness'][0]) for r in results)/len(results):.1f}/5")
print(f"Avg relevancy: {sum(int(r['relevancy'][0]) for r in results)/len(results):.1f}/5")

# COMMAND ----------

# DBTITLE 1,Traces + RAGAS + NDCG + MMR Evaluation
# Comprehensive RAG Evaluation: MLflow Traces + RAGAS + NDCG + MMR
# All metrics + traces logged to MLflow experiment

import mlflow
import numpy as np
import pandas as pd
import re
import json
from mlflow.entities import SpanType

EXPERIMENT = "/Users/prakhar1207srivastava@gmail.com/Drafts/Health Report RAG Chatbot"
mlflow.set_experiment(EXPERIMENT)

# === TRACED RAG PIPELINE (each prompt traced: retrieval -> generation) ===

@mlflow.trace(span_type=SpanType.RETRIEVER)
def traced_retrieve(query, top_k=3):
    results = w.vector_search_indexes.query_index(
        index_name=VS_INDEX_NAME,
        columns=["chunk_id", "chunk_text", "chunk_type", "page", "section_id", "file_name"],
        query_text=query, num_results=top_k,
        filters_json='{"chunk_type": ["text", "table", "section_header"]}',
    )
    return results.result.data_array

@mlflow.trace(span_type=SpanType.LLM)
def traced_generate(question, context):
    system_prompt = f"Answer using ONLY this health report context. If insufficient, say 'I don't have enough information.' Cite page numbers. Be concise.\n\nContext:\n{context}"
    response = deploy_client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}], "max_tokens": MAX_TOKENS, "temperature": 0.1},
    )
    return response["choices"][0]["message"]["content"]

@mlflow.trace(span_type=SpanType.CHAIN)
def traced_rag_chat(question, session_id="eval-session"):
    mlflow.update_current_trace(metadata={"mlflow.trace.user": "prakhar1207srivastava@gmail.com", "mlflow.trace.session": session_id})
    chunks = traced_retrieve(question)
    context = "\n\n".join(f"[S{i+1}] p{r[3]} {r[4]}\n{r[1][:MAX_CONTEXT_CHARS]}" for i, r in enumerate(chunks)) if chunks else ""
    answer = traced_generate(question, context)
    return {"answer": answer, "chunks": chunks, "context": context}

# === RAGAS METRICS via LLM-as-judge ===

def llm_score(prompt):
    resp = deploy_client.predict(endpoint=LLM_ENDPOINT, inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 100, "temperature": 0})
    raw = resp["choices"][0]["message"]["content"].strip()
    m = re.search(r'\{[^}]+\}', raw)
    if m:
        d = json.loads(m.group())
        return int(d.get("score", 0)), d.get("rationale", "")
    s = re.search(r'(\d)', raw)
    return (int(s.group(1)), raw[:80]) if s else (0, raw[:80])

def ragas_faithfulness(answer, context):
    return llm_score(f"Rate faithfulness (1-5). Is every claim in the answer supported by the context?\n\nContext: {context[:800]}\nAnswer: {answer}\n\nJSON: {{\"score\": <1-5>, \"rationale\": \"<one sentence>\"}}")

def ragas_answer_relevancy(question, answer):
    return llm_score(f"Rate answer relevancy (1-5). Does the answer directly address the question?\n\nQuestion: {question}\nAnswer: {answer}\n\nJSON: {{\"score\": <1-5>, \"rationale\": \"<one sentence>\"}}")

def ragas_context_precision(question, context, ground_truth):
    return llm_score(f"Rate context precision (1-5). Do the retrieved chunks contain information needed to answer?\n\nQuestion: {question}\nGround Truth: {ground_truth}\nRetrieved Context: {context[:500]}\n\nJSON: {{\"score\": <1-5>, \"rationale\": \"<one sentence>\"}}")

def ragas_context_recall(context, ground_truth):
    return llm_score(f"Rate context recall (1-5). Does the retrieved context contain ALL the ground truth facts?\n\nGround Truth: {ground_truth}\nRetrieved Context: {context[:500]}\n\nJSON: {{\"score\": <1-5>, \"rationale\": \"<one sentence>\"}}")

# === NDCG: Normalized Discounted Cumulative Gain ===

def compute_ndcg(retrieved_scores):
    def dcg(scores):
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))
    dcg_val = dcg(retrieved_scores)
    idcg_val = dcg(sorted(retrieved_scores, reverse=True))
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

# === MMR: Maximal Marginal Relevance ===

def compute_mmr(chunks, lambda_param=0.7):
    if len(chunks) <= 1:
        return 1.0
    texts = [r[1][:200] for r in chunks]
    scores = [r[-1] for r in chunks]
    def text_sim(t1, t2):
        w1, w2 = set(t1.lower().split()), set(t2.lower().split())
        return len(w1 & w2) / max(len(w1 | w2), 1)
    selected = [0]
    remaining = list(range(1, len(texts)))
    mmr_scores = [scores[0]]
    while remaining:
        best_mmr, best_idx = -float('inf'), None
        for idx in remaining:
            max_sim = max(text_sim(texts[idx], texts[s]) for s in selected)
            mmr = lambda_param * scores[idx] - (1 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr, best_idx = mmr, idx
        selected.append(best_idx)
        mmr_scores.append(best_mmr)
        remaining.remove(best_idx)
    return float(np.mean(mmr_scores))

# === RUN EVALUATION WITH TRACING ===

eval_cases = [
    {"question": "What is the HbA1c level?", "ground_truth": "HbA1c target for pregnancy <6%, paediatric <7.5%"},
    {"question": "What is the patient's name and age?", "ground_truth": "Mr Prakhar Srivastava, 37Y 1M 29D"},
    {"question": "What is the fasting blood sugar?", "ground_truth": "Fasting blood sugar level in mg/dl"},
    {"question": "What is the capital of France?", "ground_truth": "Not in health report"},
]

all_results = []

with mlflow.start_run(run_name="health-rag-eval-ragas-ndcg-mmr") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print("=" * 80)
    print("RAG EVALUATION: Traces + RAGAS + NDCG + MMR")
    print("=" * 80)

    for i, tc in enumerate(eval_cases, 1):
        q = tc["question"]
        gt = tc["ground_truth"]

        # Run traced RAG pipeline (creates MLflow trace for each prompt)
        result = traced_rag_chat(q, session_id=f"eval-q{i}")
        answer = result["answer"]
        chunks = result["chunks"]
        context = result["context"]

        # NDCG
        retrieved_scores = [r[-1] for r in chunks] if chunks else [0]
        ndcg = compute_ndcg(retrieved_scores)

        # MMR
        mmr = compute_mmr(chunks) if chunks else 0

        # RAGAS metrics
        faith, faith_r = ragas_faithfulness(answer, context)
        rel, rel_r = ragas_answer_relevancy(q, answer)
        ctx_prec, ctx_prec_r = ragas_context_precision(q, context, gt)
        ctx_recall, ctx_recall_r = ragas_context_recall(context, gt)

        # Log per-question metrics to MLflow
        mlflow.log_metric(f"q{i}_faithfulness", faith)
        mlflow.log_metric(f"q{i}_answer_relevancy", rel)
        mlflow.log_metric(f"q{i}_context_precision", ctx_prec)
        mlflow.log_metric(f"q{i}_context_recall", ctx_recall)
        mlflow.log_metric(f"q{i}_ndcg", ndcg)
        mlflow.log_metric(f"q{i}_mmr", mmr)
        mlflow.log_metric(f"q{i}_avg_similarity", float(np.mean(retrieved_scores)))

        all_results.append({
            "#": i, "question": q, "answer": answer[:80],
            "faithfulness": f"{faith}/5", "answer_relevancy": f"{rel}/5",
            "context_precision": f"{ctx_prec}/5", "context_recall": f"{ctx_recall}/5",
            "ndcg": f"{ndcg:.4f}", "mmr": f"{mmr:.4f}",
            "avg_sim": f"{np.mean(retrieved_scores):.4f}",
        })

        print(f"\n[Q{i}] {q}")
        print(f"  Answer: {answer[:100]}")
        print(f"  RAGAS: faith={faith}/5 | rel={rel}/5 | ctx_prec={ctx_prec}/5 | ctx_recall={ctx_recall}/5")
        print(f"  NDCG={ndcg:.4f} | MMR={mmr:.4f} | avg_sim={np.mean(retrieved_scores):.4f}")

    # Log averages
    avg_faith = np.mean([int(r["faithfulness"][0]) for r in all_results])
    avg_rel = np.mean([int(r["answer_relevancy"][0]) for r in all_results])
    avg_cp = np.mean([int(r["context_precision"][0]) for r in all_results])
    avg_cr = np.mean([int(r["context_recall"][0]) for r in all_results])
    avg_ndcg = np.mean([float(r["ndcg"]) for r in all_results])
    avg_mmr = np.mean([float(r["mmr"]) for r in all_results])

    mlflow.log_metric("avg_faithfulness", float(avg_faith))
    mlflow.log_metric("avg_answer_relevancy", float(avg_rel))
    mlflow.log_metric("avg_context_precision", float(avg_cp))
    mlflow.log_metric("avg_context_recall", float(avg_cr))
    mlflow.log_metric("avg_ndcg", float(avg_ndcg))
    mlflow.log_metric("avg_mmr", float(avg_mmr))

    print(f"\n{'=' * 80}")
    print(f"AVERAGES: faith={avg_faith:.1f} | rel={avg_rel:.1f} | ctx_prec={avg_cp:.1f} | ctx_recall={avg_cr:.1f} | ndcg={avg_ndcg:.4f} | mmr={avg_mmr:.4f}")
    print(f"\nTraces + metrics logged to MLflow experiment")
    print(f"View traces: Experiments tab -> Run {run_id} -> Traces")

display(pd.DataFrame(all_results))

# COMMAND ----------

# DBTITLE 1,Deploy -- Log + Register
# Deploy: Step 1 -- Log + Register ChatAgent (code-based logging)
# ChatAgent auto-sets task='agent/v2/chat' and appears in AI Playground / Review App

import mlflow
from mlflow import MlflowClient

MODEL_FILE_PATH = "/Workspace/Users/prakhar1207srivastava@gmail.com/Drafts/health_report_agent.py"

mlflow.set_experiment("/Users/prakhar1207srivastava@gmail.com/Drafts/Health Report RAG Chatbot")

input_example = {"messages": [{"role": "user", "content": "What is this health report about?"}]}

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex, DatabricksSQLWarehouse, DatabricksTable
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME),
    DatabricksSQLWarehouse(warehouse_id=WAREHOUSE_ID),
    DatabricksTable(table_name=CHUNKS_TABLE),
    DatabricksTable(table_name=f"{FULL_SCHEMA}.health_report_lab_data"),
]

with mlflow.start_run(run_name="health-report-agent-deployment") as run:
    model_info = mlflow.pyfunc.log_model(
        name="health_report_agent",
        python_model=MODEL_FILE_PATH,
        input_example=input_example,
        pip_requirements=["databricks-sdk>=0.30.0", "mlflow"],
        resources=resources,
    )
    print(f"Model logged: {model_info.model_uri}")
    print(f"Run ID: {run.info.run_id}")

mlflow.set_registry_uri("databricks-uc")
REGISTERED_MODEL_NAME = "workspace.default.health_report_rag_chatbot"

client = MlflowClient(registry_uri="databricks-uc")
try:
    client.create_registered_model(name=REGISTERED_MODEL_NAME)
    print(f"Created registered model: {REGISTERED_MODEL_NAME}")
except Exception:
    print(f"Registered model exists: {REGISTERED_MODEL_NAME}")

registered_version = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=REGISTERED_MODEL_NAME,
    await_registration_for=300,
)
model_version = registered_version.version
print(f"Registered version {model_version}")
print(f"Model URI: models:/{REGISTERED_MODEL_NAME}/{model_version}")
print(f"Task: agent/v2/chat (appears in AI Playground & Review App)")

# COMMAND ----------

# DBTITLE 1,Deploy -- agents.deploy
# Deploy: Step 2 -- Deploy with databricks.agents.deploy()
# Auto-creates: serving endpoint + Review App (chat UI) + tracing + inference tables

from databricks.agents import deploy
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateConfigUpdate, EndpointStateReady
import time

ENDPOINT_NAME = "health-report-agent-v2"
w = WorkspaceClient()

# deploy() will create the endpoint if it doesn't exist, or update it if it does
print(f"\nDeploying: {REGISTERED_MODEL_NAME} v{model_version}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"This typically takes 5-10 minutes...")

deployment = deploy(
    model_name=REGISTERED_MODEL_NAME,
    model_version=int(model_version),
    endpoint_name=ENDPOINT_NAME,
    scale_to_zero=True,
)

print(f"\nDeployment initiated!")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Review App: enabled")
print(f"Tracing: enabled")

def wait_for_endpoint_ready(name, timeout_s=900, poll_s=15):
    deadline = time.time() + timeout_s
    failure_states = {EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED}
    while time.time() < deadline:
        try:
            state = w.serving_endpoints.get(name).state
        except Exception as e:
            print(f"  [poll] Error fetching state: {e}")
            time.sleep(poll_s)
            continue
        if state.ready == EndpointStateReady.READY and state.config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            return True
        if state.config_update in failure_states:
            raise RuntimeError(f"{name} deployment failed: {state.config_update.value}")
        elapsed = int(time.time() - (deadline - timeout_s))
        print(f"  [{elapsed}s] State: ready={state.ready.value}, config_update={state.config_update.value}")
        time.sleep(poll_s)
    raise TimeoutError(f"{name} not ready after {timeout_s}s")

print("\nWaiting for endpoint to be ready...")
try:
    wait_for_endpoint_ready(ENDPOINT_NAME)
    print(f"\n✅ Endpoint '{ENDPOINT_NAME}' is READY!")
    print(f"To chat: Open endpoint page > Review App")
    print(f"Or use AI Playground: select {ENDPOINT_NAME} from model dropdown")
except Exception as e:
    print(f"\n❌ Endpoint not ready yet: {e}")
    print(f"Check: Serving > Endpoints > {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Deploy -- Test endpoint
# Deploy: Step 3 -- Test the agent endpoint
import json
import mlflow.deployments
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
ENDPOINT_NAME = "health-report-agent-v2"
deploy_client = mlflow.deployments.get_deploy_client("databricks")

test_questions = [
    "What is this health report about?",
    "What are the key findings in this report?",
    "What recommendations are given?",
    "summarize report"
]

print("=" * 60)
print("TESTING HEALTH REPORT AGENT ENDPOINT")
print("=" * 60)

for q in test_questions:
    try:
        response = deploy_client.predict(
            endpoint=ENDPOINT_NAME,
            inputs={"messages": [{"role": "user", "content": q}]},
        )
        if isinstance(response, dict):
            msgs = response.get("messages", [])
            answer = msgs[-1].get("content", str(response)) if msgs else str(response)
        else:
            answer = str(response)
        print(f"\nQUESTION: {q}")
        print(f"ANSWER: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print("\u2500" * 60)
    except Exception as e:
        print(f"\nQUESTION: {q}")
        print(f"Error: {e}")
        print("\u2500" * 60)

# Curl command
host = w.config.host.rstrip("/")
endpoint_url = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
print(f"\nEndpoint URL: {endpoint_url}")
print(f"\nCurl example:")
print(f"curl -X POST '{endpoint_url}' -H 'Authorization: Bearer $DATABRICKS_TOKEN' -H 'Content-Type: application/json' -d '{{\"messages\": [{{\"role\": \"user\", \"content\": \"What findings are in this report?\"}}]}}'")
