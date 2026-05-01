
import os
import re
import pandas as pd
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# --- ENVIRONMENT CONFIGURATION ---
# Allows RENCI to override paths via environment variables
PDF_PATH = os.environ.get("CDC_PDF_PATH", "data/cdc_mec_tables_only.pdf")
DB_URL = os.environ.get("DATABASE_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "cdc_mec_rules")

print(f"📄 Parsing CDC Guidelines from {PDF_PATH} with Docling...")
converter = DocumentConverter()
try:
    result = converter.convert(PDF_PATH)
    doc = result.document
except Exception as e:
    print(f"🚨 ERROR: Could not read {PDF_PATH}. Ensure the file exists in the correct directory. Details: {e}")
    exit()

unrolled_documents = []
print(f"🔍 Found {len(doc.tables)} tables. Unrolling into semantic statements with Hierarchy Tracking...")

current_parent_condition = ""

for table_ix, table in enumerate(doc.tables):
    df = table.export_to_dataframe()
    
    if df.empty or len(df.columns) < 2:
        continue
        
    condition_col = df.columns[0]
    methods = [col for col in df.columns if "clarification" not in col.lower() and col.strip() not in [condition_col, "None", ""]]
    
    for index, row in df.iterrows():
        condition = str(row[condition_col]).strip()
        clarification = str(row.get("Clarification", "")).strip()
        
        if not condition or condition.lower() in ["nan", "none", "condition"]:
            continue
            
        # 🛠️ THE FIX: If it starts with a bullet point (a., b., i., ii.), attach it to the parent!
        if re.match(r'^([a-z]\.|[ivx]+\.)', condition.lower()):
            full_condition = f"{current_parent_condition} : {condition}"
        else:
            # If it doesn't start with a bullet, it IS the new parent condition.
            current_parent_condition = condition
            full_condition = condition
            
        for method in methods:
            category_raw = str(row[method]).strip()
            category = "".join([c for c in category_raw if c.isdigit()])
            
            if category:
                statement = (
                    f"According to the 2024 CDC Medical Eligibility Criteria (MEC), "
                    f"for a patient with the condition '{full_condition}', "
                    f"the safety category for using '{method}' is Category {category[0]}."
                )
                
                if clarification and clarification.lower() not in ["nan", "none"]:
                    statement += f" Clinical clarification: {clarification}"
                
                unrolled_documents.append(Document(
                    page_content=statement,
                    metadata={
                        "condition": full_condition.lower(), # Save the FULL condition in metadata
                        "method": method.lower(),
                        "category_score": int(category[0]),
                        "source": "cdc_mec_docling"
                    }
                ))

print(f"✅ Generated {len(unrolled_documents)} highly structured chunks!")

# --- DATABASE UPLOAD PHASE ---
if not DB_URL:
    print("🚨 ERROR: DATABASE_URL not found. Cannot push to Postgres. Please configure your .env file.")
    exit()

print(f"🧠 Loading embedding model ({EMBEDDING_MODEL})...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("🗑️ Wiping old fragmented RAG tables...")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_URL,
    use_jsonb=True, 
)
vector_store.drop_tables() 

print("🏗️ Rebuilding fresh tables and collection...")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_URL,
    use_jsonb=True, 
)

print(f"💾 Uploading {len(unrolled_documents)} structured chunks to the database...")
vector_store.add_documents(unrolled_documents)
print("✅ SUCCESS: The fully optimized, hierarchical RAG data is loaded into CloudNativePG!")