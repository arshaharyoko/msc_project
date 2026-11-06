"""FAISS based nearest-neighbour store mapping vectors back to component names."""

import numpy as np
import faiss
import os
import pickle

__all__ = ["VectorDatabase"]


class VectorDatabase:
    def __init__(self):
        self.pubchem_index = None
        self.uniprot_index = None
        self.pubchem_data = []
        self.uniprot_data = []
        self.pubchem_ids = set()
        self.uniprot_ids = set()

    def add_pubchem_vectors(self, vectors, dim, names=None):
        if names is None:
            names = list(range(len(vectors)))

        if self.pubchem_index is None:
            d = dim
            self.pubchem_index = faiss.IndexFlatL2(d)

        assert len(vectors) == len(names), "Vectors and names lengths must match."

        new_entries = []
        for idx, name in enumerate(names):
            if name not in self.pubchem_ids:
                self.pubchem_ids.add(name)
                new_entries.append(idx)

        if new_entries:
            dedup_vectors = vectors[new_entries]
            self.pubchem_index.add(dedup_vectors.astype(np.float32))
            for i in new_entries:
                self.pubchem_data.append(names[i])
        else:
            print("No new pubchem entries to add.")

    def add_uniprot_vectors(self, vectors, dim, names=None):
        if names is None:
            names = list(range(len(vectors)))

        if self.uniprot_index is None:
            d = dim
            self.uniprot_index = faiss.IndexFlatL2(d)

        assert len(vectors) == len(names), "Vectors and names lengths must match."

        new_entries = []
        for idx, name in enumerate(names):
            if name not in self.uniprot_ids:
                self.uniprot_ids.add(name)
                new_entries.append(idx)

        if new_entries:
            dedup_vectors = vectors[new_entries]
            self.uniprot_index.add(dedup_vectors.astype(np.float32))
            for i in new_entries:
                self.uniprot_data.append(names[i])
        else:
            print("No new uniprot entries to add.")

    def vec_lookup_pubchem(self, vector, k=1):
        # Reshape the query vector to (1, -1) because FAISS expects a 2D array.
        D, I = self.pubchem_index.search(vector.reshape(1, -1).astype(np.float32), k)
        return I.flatten().tolist(), D.flatten().tolist()

    def vec_lookup_uniprot(self, vector, k=1):
        D, I = self.uniprot_index.search(vector.reshape(1, -1).astype(np.float32), k)
        return I.flatten().tolist(), D.flatten().tolist()

    def save(self, pubchem_filename="pubchem_index.bin", uniprot_filename="uniprot_index.bin"):
        if self.pubchem_index is not None:
            faiss.write_index(self.pubchem_index, pubchem_filename)
            with open(pubchem_filename + ".meta.pkl", "wb") as f:
                pickle.dump({"data": self.pubchem_data, "ids": self.pubchem_ids}, f)
        if self.uniprot_index is not None:
            faiss.write_index(self.uniprot_index, uniprot_filename)
            with open(uniprot_filename + ".meta.pkl", "wb") as f:
                pickle.dump({"data": self.uniprot_data, "ids": self.uniprot_ids}, f)

    def load(self, pubchem_filename="pubchem_index.bin", uniprot_filename="uniprot_index.bin"):
        if os.path.exists(pubchem_filename):
            self.pubchem_index = faiss.read_index(pubchem_filename)
            with open(pubchem_filename + ".meta.pkl", "rb") as f:
                meta = pickle.load(f)
                self.pubchem_data = meta["data"]
                self.pubchem_ids = meta["ids"]
        if os.path.exists(uniprot_filename):
            self.uniprot_index = faiss.read_index(uniprot_filename)
            with open(uniprot_filename + ".meta.pkl", "rb") as f:
                meta = pickle.load(f)
                self.uniprot_data = meta["data"]
                self.uniprot_ids = meta["ids"]
