import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Electronics Recommender",
    page_icon="🛒",
    layout="wide"
)


st.title("🛒 Electronics Recommendation System")

st.markdown(
    "Get similar electronic product recommendations instantly."
)


query_name = st.text_input(
    "Enter Product Name",
    placeholder="Example: iPhone, Samsung, OnePlus..."
)


if st.button("Get Recommendations"):

    if query_name.strip() == "":

        st.warning("Please enter a product name.")

    else:

        input_data = {
            "name": query_name.strip()
        }

        try:

            with st.spinner("Finding similar products..."):

                response = requests.post(
                    API_URL,
                    json=input_data,
                    timeout=10
                )

                result=response.json()
            if "error" in result:

                st.error(result["error"])

            else:

                st.success(
                    f"Top recommendations for: {result['query']}"
                )

                recommendations = result["recommendations"]

                # display products
                for product in recommendations:

                    st.markdown("---")

                    col1, col2 = st.columns([1, 2])

                    # image column
                    with col1:

                        if product.get("image"):

                            st.image(
                                product["image"],
                                width=220
                            )

                    # details column
                    with col2:

                        st.subheader(product["name"])

                        st.markdown(
                            f"⭐ Rating: "
                            f"{product.get('ratings', 'N/A')}"
                        )

                        st.markdown(
                            f"💰 Price: ₹"
                            f"{product.get('price', 'N/A')}"
                        )

                        if product.get("link"):

                            st.markdown(
                                f"[🔗 View Product]"
                                f"({product['link']})"
                            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI server."
            )

        except Exception as e:

            st.error(f"Error: {e}")