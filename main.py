import os
import joblib
from fastapi import FastAPI
import pandas as pd
from schema.schema import Product, Recommendation

app = FastAPI()

# Load the model and product dataset
model = joblib.load("models/model.pkl")
product_data = joblib.load("models/product_data.pkl")


@app.get("/")
def home():
    return {"message": "Recommendation System API Running"}


@app.post("/recommend")
def GetRecommendation(data: Product):

    # Find the product in the dataset (case-insensitive search)
    input_product = product_data[
        product_data['name'].str.contains(data.name, case=False, na=False)
    ]
    if input_product.empty:
        return {"error": "Product not found in database."}

    # Columns used in training
    transformer_cols = ['actual_price', 'no_of_ratings', 'discount_price', 'sub_category', 'main_category']
    input_data = input_product[transformer_cols].copy()

    # Clean numeric columns
    for col in ['actual_price', 'discount_price', 'no_of_ratings']:
        input_data[col] = (
            input_data[col].astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

    # Fill missing numeric values
    input_data['discount_price'] = input_data['discount_price'].fillna(input_data['discount_price'].median())
    input_data['actual_price'] = input_data['actual_price'].fillna(input_data['actual_price'].median())
    input_data['no_of_ratings'] = input_data['no_of_ratings'].fillna(0)

    # Transform features using the preprocessor from the pipeline
    transformed = model.named_steps["preprocessor"].transform(input_data)

    # Get top 5 similar products
    distance, indices = model.named_steps["model"].kneighbors(transformed, n_neighbors=5)
    results = product_data.iloc[indices[0]].copy()

    # Clean numeric columns in results before returning
    for col in ['actual_price', 'discount_price', 'no_of_ratings', 'ratings']:
        if col in results.columns:
            results[col] = (
                results[col].astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            results[col] = pd.to_numeric(results[col], errors='coerce')

    # Return recommendations
    recommendations = []
    for _, row in results.iterrows():
        recommendations.append(Product(
            name=row.get('name'),
            sub_category=row.get('sub_category'),
            main_category=row.get('main_category'),
            image=row.get('image'),
            link=row.get('link'),
            ratings=row.get('ratings'),
            no_of_ratings=row.get('no_of_ratings'),
            discount_price=row.get('discount_price'),
            actual_price=row.get('actual_price')
        ))

    return Recommendation(
        query=data.name,
        recommendations=recommendations
    )