import logging
import time
import faiss
import numpy as np
import pickle
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def row_to_text(row: pd.Series) -> str:
    """Build a compact clinical summary string for embedding.
    
    Includes all columns except:
    - patient_id, primary_site_code (identifiers)
    - is_alive, survived_1year, survived_2year, survived_5year (binary 0/1)
    """
    # Convert binary died_from_cancer to Yes/No
    died_from_cancer = row.get('died_from_cancer', '')
    died_from_cancer_text = "Yes" if died_from_cancer == 1 else "No" if died_from_cancer == 0 else ''
    
    fields = [
        f"Primary site: {row.get('primary_site_labeled', '')}",
        f"Age group: {row.get('age_group', '')}",
        f"Age numeric: {row.get('age_numeric', '')}",
        f"Age category: {row.get('age_category', '')}",
        f"Race: {row.get('race', '')}",
        f"Sex: {row.get('sex', '')}",
        f"Year of diagnosis: {row.get('year_diagnosis', '')}",
        f"Treatment era: {row.get('treatment_era', '')}",
        f"Site recode: {row.get('site_recode', '')}",
        f"Cancer system: {row.get('cancer_system', '')}",
        f"Histology code: {row.get('histology_code', '')}",
        f"Behavior: {row.get('behavior', '')}",
        f"Survival months raw: {row.get('survival_months_raw', '')}",
        f"Survival months: {row.get('survival_months', '')}",
        f"Survival years: {row.get('survival_years', '')}",
        f"Survival category: {row.get('survival_category', '')}",
        f"Censored status: {row.get('censored_status', '')}",
        f"Vital status: {row.get('vital_status', '')}",
        #f"Cause of death: {row.get('cause_death', '')}",
        f"Died from cancer: {died_from_cancer_text}",
    ]

    parts = [str(val) for val in fields if val and pd.notna(val) and str(val).strip()]
    return " | ".join(parts)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

df = pd.read_csv("./rag/seer_data_train.csv")
logging.info(f"Loaded dataframe with {len(df)} rows from seer_data_train.csv")

load_start = time.time()
model = SentenceTransformer('NeuML/pubmedbert-base-embeddings', device=device)
logging.info(f"Loaded model in {time.time() - load_start:.2f}s")

# Convert rows to text and embed
logging.info("Converting rows to text for embedding...")
texts = [row_to_text(row) for _, row in df.iterrows()]
logging.info("Encoding texts to embeddings...")
encode_start = time.time()
embeddings = model.encode(
    texts,
    batch_size=96,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
logging.info(f"Encoded {len(texts)} texts in {time.time() - encode_start:.2f}s")
embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)



# Build index on CPU (avoid GPU -> CPU copy OOM)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.omp_set_num_threads(12)

batch_add = 8192  # keep batching; adjust 4k–16k if you like
N = embeddings.shape[0]
add_start = time.time()
for i in range(0, N, batch_add):
    j = min(N, i + batch_add)
    t0 = time.time()
    index.add(embeddings[i:j])  # type: ignore
    logging.info(f"Added vectors {i}:{j} in {time.time() - t0:.2f}s")
logging.info(f"Total index.add time: {time.time() - add_start:.2f}s")
logging.info(f"Built FAISS index with {index.ntotal} vectors of dim {dimension}")

faiss.write_index(index, "medical_index.faiss")
logging.info("Saved FAISS index to medical_index.faiss")

# Save metadata separately
metadata = {
    'texts': texts,
    'original_data': df.to_dict('records')
}
pickle.dump(metadata, open("medical_metadata.pkl", "wb"))
logging.info("Saved metadata to medical_metadata.pkl")