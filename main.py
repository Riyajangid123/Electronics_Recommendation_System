import os
import joblib
from fastapi import FastAPI
import pandas as pd
from schema.schema import Product, Recommendation

app = FastAPI()

model = joblib.load("models/knn_model.pkl")
product_data = joblib.load("models/data.pkl")
vectors=joblib.load("models/vectors.pkl")


@app.get("/")
def home():
    return {"message": "Recommendation System API Running"}


@app.post("/recommend")
def GetRecommendation(product: Product):
    matched = product_data[
        product_data["name"].str.contains(
            product.name,
            case=False,
            na=False)]
    if matched.empty:
        return {
            "error": "Product not found"}
    idx = matched.index[0]

   
    distances, indices = model.kneighbors(
        vectors[idx],
        n_neighbors=6
    )

    recommendations = []

    for i in indices[0][1:]:

        item = product_data.iloc[i]

        recommendations.append({

                "name": str(item["name"]),

                "image": str(item["image"]) if pd.notna(item["image"]) else None,

                "price": float(item["discount_price"]) if pd.notna(item["discount_price"]) else None,

                "ratings": float(item["ratings"]) if pd.notna(item["ratings"]) else None,

                "link": str(item["link"]) if pd.notna(item["link"]) else None
            })

    return {
        "query": product.name,
        "recommendations": recommendations
    }