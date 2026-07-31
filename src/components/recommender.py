from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib
from src.components.DataPreprocessing import DataPreprocessing
import os

class Recommender:

    def __init__(self,data):
        self.data = data

        preprocessing = DataPreprocessing(data)

        self.data = preprocessing.preprocess_data()

        self.tfidf = TfidfVectorizer(
            stop_words="english"
        )

        self.vectors = self.tfidf.fit_transform(
            self.data["tags"]
        )

        # knn model
        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute"
        )

        self.model.fit(self.vectors)

    def save_model(self):
        os.makedirs("models",exist_ok=True)
        joblib.dump(
            self.model,
            "models/knn_model.pkl"
        )

        joblib.dump(
            self.tfidf,
            "models/tfidf.pkl"
        )

        joblib.dump(
            self.vectors,
            "models/vectors.pkl"
        )

        joblib.dump(
            self.data,
            "models/data.pkl"
        )

        print("Model Saved Successfully")

    def recommend(self, product_name):

        matched = self.data[
            self.data["name"].str.contains(
                product_name,
                case=False,
                na=False
            )
        ]

        if matched.empty:

            return "Product not found"

        idx = matched.index[0]

        distances, indices = self.model.kneighbors(
            self.vectors[idx],
            n_neighbors=6
        )

        recommend_products = []

        for i in indices[0][1:]:

            item = self.data.iloc[i]

            recommend_products.append({

                "name": item['name'],
                "image": item['image'],
                "price": item['discount_price'],
                "ratings": item['ratings']

            })

        return recommend_products