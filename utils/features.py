import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# ==============================================================================
# TF-IDF
# ==============================================================================
class TFIDFExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.75,
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)


# ==============================================================================
# BERT EMBEDDINGS
# ==============================================================================
class BERTEmbeddingExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def _split_into_chunks(self, text, chunk_size=200):
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def get_embedding(self, text):
        if pd.isna(text) or text.strip() == "":
            return np.zeros(384)
        chunks = self._split_into_chunks(text)
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        return np.mean(embeddings, axis=0)

    def extract_dataframe(self, df, text_column='LYRICS_EN'):
        print(f"Extracting BERT embeddings for {len(df)} texts...")
        all_embeddings = []
        for i, text in enumerate(df[text_column].fillna("").tolist()):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(df)}...")
            all_embeddings.append(self.get_embedding(text))
        embeddings = np.array(all_embeddings)
        print(f"BERT embeddings shape: {embeddings.shape}")
        return embeddings
