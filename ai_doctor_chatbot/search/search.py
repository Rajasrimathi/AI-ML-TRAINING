import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import streamlit as st
import json

# Simple in-memory search without external dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure page
st.set_page_config(
    page_title="Simple Product Search",
    page_icon="🔍",
    layout="wide"
)

# Simple Dataset Generator
class SimpleProductGenerator:
    def __init__(self):
        self.categories = {
            'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Camera'],
            'Fashion': ['Shoes', 'Shirt', 'Jeans', 'Dress', 'Jacket'],
            'Home': ['Sofa', 'Table', 'Lamp', 'Mirror', 'Cushion'],
            'Books': ['Fiction', 'Non-Fiction', 'Comics', 'Textbook', 'Biography']
        }
        
        self.brands = ['Apple', 'Samsung', 'Nike', 'Adidas', 'IKEA', 'Sony', 'Dell', 'HP']
        self.adjectives = ['Premium', 'Professional', 'Comfortable', 'Durable', 'Lightweight']
    
    def generate_product(self, product_id):
        category = random.choice(list(self.categories.keys()))
        subcategory = random.choice(self.categories[category])
        brand = random.choice(self.brands)
        adjective = random.choice(self.adjectives)
        
        name = f"{brand} {adjective} {subcategory}"
        description = f"High-quality {subcategory.lower()} from {brand}. Features {adjective.lower()} design with excellent performance and reliability."
        
        return {
            'id': product_id,
            'name': name,
            'description': description,
            'category': category,
            'brand': brand,
            'price': round(random.uniform(20, 500), 2),
            'rating': round(random.uniform(3.0, 5.0), 1)
        }
    
    def generate_dataset(self, size=1000):
        products = []
        for i in range(size):
            products.append(self.generate_product(i + 1))
        return pd.DataFrame(products)

# Simple Search System
class SimpleSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.vectors = None
        self.df = None
    
    def index_data(self, df):
        """Index the dataset for search"""
        self.df = df.copy()
        
        # Combine text fields for search
        search_texts = []
        for _, row in df.iterrows():
            text = f"{row['name']} {row['description']} {row['category']} {row['brand']}"
            search_texts.append(text)
        
        # Create TF-IDF vectors
        self.vectors = self.vectorizer.fit_transform(search_texts)
        return True
    
    def search(self, query, top_k=10):
        """Search for products"""
        if self.vectors is None or self.df is None:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                result = self.df.iloc[idx].to_dict()
                result['similarity'] = similarities[idx]
                results.append(result)
        
        return results

# Main App
def main():
    st.title("🔍 Simple Product Search")
    st.markdown("Generate products and search through them with text similarity")
    
    # Initialize components
    generator = SimpleProductGenerator()
    search_system = SimpleSearch()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        dataset_size = st.selectbox("Dataset Size", [100, 500, 1000, 2000], index=2)
        max_results = st.slider("Max Results", 5, 20, 10)
    
    # Tabs
    tab1, tab2 = st.tabs(["Generate Data", "Search Products"])
    
    with tab1:
        st.header("Generate Product Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate Dataset", type="primary"):
                with st.spinner(f"Generating {dataset_size} products..."):
                    df = generator.generate_dataset(dataset_size)
                    st.session_state.dataset = df
                    
                    # Index for search
                    search_system.index_data(df)
                    st.session_state.search_system = search_system
                    
                    st.success(f"✅ Generated {len(df)} products!")
                    
                    # Show summary
                    st.subheader("Dataset Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Products", len(df))
                    with col2:
                        st.metric("Categories", df['category'].nunique())
                    with col3:
                        st.metric("Avg Price", f"${df['price'].mean():.2f}")
                    
                    # Category distribution
                    st.bar_chart(df['category'].value_counts())
        
        with col2:
            if 'dataset' in st.session_state:
                df = st.session_state.dataset
                st.metric("Current Dataset", f"{len(df)} products")
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"products_{len(df)}.csv",
                    "text/csv"
                )
        
        # Show sample data
        if 'dataset' in st.session_state:
            with st.expander("Sample Data"):
                st.dataframe(st.session_state.dataset.head(10))
    
    with tab2:
        st.header("Search Products")
        
        if 'search_system' in st.session_state:
            # Search interface
            query = st.text_input(
                "Search products:",
                placeholder="e.g., 'premium smartphone' or 'comfortable shoes'"
            )
            
            # Example searches
            st.write("**Try these examples:**")
            examples = ["premium smartphone", "comfortable shoes", "professional laptop", "modern furniture"]
            cols = st.columns(len(examples))
            
            for i, example in enumerate(examples):
                with cols[i]:
                    if st.button(example, key=f"ex_{i}"):
                        query = example
            
            # Perform search
            if query:
                search_system = st.session_state.search_system
                results = search_system.search(query, max_results)
                
                if results:
                    st.subheader(f"Found {len(results)} results for: '{query}'")
                    
                    for i, result in enumerate(results, 1):
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{i}. {result['name']}**")
                                st.write(result['description'])
                                st.caption(f"Category: {result['category']} | Brand: {result['brand']}")
                            
                            with col2:
                                st.metric("Price", f"${result['price']}")
                                st.metric("Rating", f"{result['rating']}⭐")
                            
                            with col3:
                                similarity_pct = result['similarity'] * 100
                                st.metric("Match", f"{similarity_pct:.1f}%")
                            
                            st.divider()
                else:
                    st.warning("No results found. Try different search terms.")
        else:
            st.info("👆 Please generate a dataset first in the 'Generate Data' tab")

# File upload option
def upload_dataset():
    """Allow users to upload their own dataset"""
    st.subheader("Upload Your Own Dataset")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['name', 'description']
            if all(col in df.columns for col in required_cols):
                st.success(f"✅ Loaded {len(df)} products!")
                st.session_state.dataset = df
                
                # Index for search
                search_system = SimpleSearch()
                search_system.index_data(df)
                st.session_state.search_system = search_system
                
                st.dataframe(df.head())
            else:
                st.error(f"CSV must contain columns: {required_cols}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

if __name__ == "__main__":
    # Add upload option in sidebar
    with st.sidebar:
        st.markdown("---")
        upload_dataset()
    
    main()