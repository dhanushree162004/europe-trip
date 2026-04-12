import json
import os
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    CHROMA_AVAILABLE = False


class VectorStore:
    def __init__(
        self,
        dataset_path=None,
        model_name="all-MiniLM-L6-v2",
        chroma_path=None,
        collection_name=None,
    ):
        self.model = SentenceTransformer(model_name)
        self.dataset_path = dataset_path or os.getenv("DATASET_PATH", "data/dataset.json")
        self.chroma_path = chroma_path or os.getenv(
            "CHROMA_PATH", os.path.join(os.getcwd(), "chroma_store")
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", "trip_planner_docs"
        )
        self.documents: List[Dict[str, Any]] = []
        self.embeddings = None
        self.use_chroma = CHROMA_AVAILABLE
        self.client = None
        self.collection = None
        self.doc_lookup: Dict[str, Dict[str, Any]] = {}
        self.load_dataset()
        self.build_index()

    def load_dataset(self):
        # Resolve path relative to project root
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        else:
            abs_path = os.path.join(os.getcwd(), self.dataset_path)
            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            else:
                print(f"Warning: Dataset not found at {self.dataset_path}")

    def _document_to_text(self, doc):
        metadata = doc.get("metadata", {})
        parts = [
            doc.get("country", ""),
            doc.get("category", ""),
            doc.get("content", ""),
        ]

        for key in [
            "name",
            "city",
            "type",
            "price_range",
            "route",
            "from",
            "to",
            "cost",
            "hours",
            "duration",
            "rating",
            "avg_cost",
            "avg_price",
        ]:
            value = metadata.get(key)
            if value is not None:
                parts.append(f"{key}: {value}")

        return " | ".join(str(part) for part in parts if part not in [None, ""])

    def _document_metadata(self, doc):
        metadata = doc.get("metadata", {})
        return {
            "country": str(doc.get("country", "")),
            "category": str(doc.get("category", "")),
            "name": str(metadata.get("name", "")),
            "city": str(metadata.get("city", "")),
            "type": str(metadata.get("type", "")),
            "price_range": str(metadata.get("price_range", "")),
            "route": str(metadata.get("route", "")),
            "from": str(metadata.get("from", "")),
            "to": str(metadata.get("to", "")),
            "cost": float(metadata.get("cost", 0) or 0),
            "hours": float(metadata.get("hours", 0) or 0),
            "duration": float(metadata.get("duration", 0) or 0),
            "rating": float(metadata.get("rating", 0) or 0),
            "avg_cost": float(metadata.get("avg_cost", 0) or 0),
            "avg_price": float(metadata.get("avg_price", 0) or 0),
        }

    def build_index(self):
        if not self.documents:
            return

        texts = [self._document_to_text(doc) for doc in self.documents]
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        self.embeddings = np.array(self.embeddings).astype("float32")
        self.doc_lookup = {
            str(doc.get("id", idx)): doc for idx, doc in enumerate(self.documents)
        }

        if self.use_chroma:
            self._build_chroma_index(texts)
            print(f"Index built with {len(self.documents)} documents in Chroma.")
        else:
            print(f"Index built with {len(self.documents)} documents using Numpy.")

    def _build_chroma_index(self, texts):
        if not CHROMA_AVAILABLE:
            return

        os.makedirs(self.chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = []
        metadatas = []
        for idx, doc in enumerate(self.documents):
            doc_id = str(doc.get("id", idx))
            ids.append(doc_id)
            metadatas.append(self._document_metadata(doc))

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=self.embeddings.tolist(),
        )

    def search(self, query, top_k=5, filter_country=None, filter_category=None):
        if self.use_chroma and self.collection is not None:
            return self._search_chroma(query, top_k, filter_country, filter_category)
        return self._search_numpy(query, top_k, filter_country, filter_category)

    def _search_chroma(self, query, top_k=5, filter_country=None, filter_category=None):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")[0]

        query_count = max(top_k * 5, top_k)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=query_count,
            include=["distances", "metadatas", "documents"],
        )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        packed = []
        for doc_id, distance, metadata in zip(ids, distances, metadatas):
            doc = self.doc_lookup.get(str(doc_id))
            if not doc:
                continue
            if filter_country and filter_country.lower() != "multi":
                if doc.get("country", "").lower() != filter_country.lower() and doc.get("country", "").lower() != "multi":
                    continue
            if filter_category and doc.get("category", "").lower() != filter_category.lower():
                continue
            packed.append(
                {
                    "document": doc,
                    "score": float(1.0 - distance),
                }
            )
            if len(packed) >= top_k:
                break
        return packed

    def _search_numpy(self, query, top_k=5, filter_country=None, filter_category=None):
        if self.embeddings is None:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        indices = np.argsort(similarities)[::-1][: top_k * 5]

        results = []
        for idx in indices:
            doc = self.documents[idx]
            if filter_country and filter_country.lower() != "multi" and doc["country"].lower() != filter_country.lower() and doc["country"].lower() != "multi":
                continue
            if filter_category and doc["category"].lower() != filter_category.lower():
                continue
            results.append({"document": doc, "score": float(similarities[idx])})
            if len(results) >= top_k:
                break

        return results

    def ingest(self, dataset_path=None):
        if dataset_path:
            self.dataset_path = dataset_path
        self.documents = []
        self.embeddings = None
        self.collection = None
        self.doc_lookup = {}
        self.load_dataset()
        self.build_index()
        return len(self.documents)

    def rebuild_index(self):
        self.build_index()


if __name__ == "__main__":
    vs = VectorStore()
    res = vs.search("Where to go in Spain for cheap food?", top_k=3)
    for r in res:
        print(f"Score: {r['score']:.4f} | Content: {r['document']['content']}")
