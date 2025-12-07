import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("seer_data_cleaned.csv")
model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

# Convert rows to text and embed
texts = [row_to_text(row) for _, row in df.iterrows()]
embeddings = model.encode(texts)

# Build index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Save index
faiss.write_index(index, "medical_index.faiss")

# Save metadata separately
metadata = {
    'texts': texts,
    'original_data': df.to_dict('records')
}
pickle.dump(metadata, open("medical_metadata.pkl", "wb"))