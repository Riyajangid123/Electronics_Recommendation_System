import streamlit as st
import requests

st.title("Electronics Recommendation System")

# User input
query_name = st.text_input("Enter Product Name")

if st.button("Get Recommendations") and query_name.strip() != "":
    # Prepare JSON payload
    input_data = {"name": query_name.strip()}

    # Call FastAPI endpoint
    try:
        response = requests.post("http://127.0.0.1:8000/recommend", json=input_data)
        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"Top 5 recommendations for: {result['query']}")
            for idx, product in enumerate(result["recommendations"], start=1):
                st.markdown("---")
                st.markdown(f"### {product['name']}")
                if product.get("image"):
                    st.image(product["image"], width=200)
                st.markdown(f"**Sub Category:** {product.get('sub_category', 'N/A')}")
                st.markdown(f"**Main Category:** {product.get('main_category', 'N/A')}")
                st.markdown(f"**Ratings:** {product.get('ratings', 'N/A')} ⭐ ({product.get('no_of_ratings', 'N/A')} reviews)")
                st.markdown(f"**Discount Price:** ₹{product.get('discount_price', 'N/A')}")
                st.markdown(f"**Actual Price:** ₹{product.get('actual_price', 'N/A')}")
                if product.get("link"):
                    st.markdown(f"[View Product]({product['link']})")
    except Exception as e:
        st.error(f"API request failed: {e}")