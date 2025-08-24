import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Configure page
st.set_page_config(page_title="🔍 Universal Search", layout="wide")

# Custom CSS for clean design (✅ adjusted font colors)
st.markdown("""
<style>
.search-box {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 20px 0;
    color: black;  /* text color inside search box */
}
.result-item {
    background: #f8f9fa;
    padding: 15px;
    margin: 10px 0;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    transition: all 0.3s ease;
    color: black;  /* result text color */
}
.result-item:hover {
    background: #e9ecef;
    transform: translateX(5px);
}
.score-badge {
    background: linear-gradient(45deg, #007bff, #0056b3);
    color: white !important;  /* keep score text white */
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

class UniversalSearch:
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.items = []
        
    @st.cache_resource
    def load_model(_self):
        """Load the sentence transformer model"""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    @st.cache_data
    def create_embeddings(_self, items_tuple):
        """Create embeddings for all items"""
        items_list = list(items_tuple)
        model = _self.load_model()
        return model.encode(items_list, show_progress_bar=False)
    
    def search(self, query, items, top_k=10, threshold=0.1):
        """Search for query in items"""
        if not query.strip():
            return []
        
        if self.model is None:
            self.model = self.load_model()
        
        items_tuple = tuple(items)
        if self.embeddings is None or len(self.items) != len(items):
            self.embeddings = self.create_embeddings(items_tuple)
            self.items = items
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        results = []
        for i, (item, score) in enumerate(zip(items, similarities)):
            if score >= threshold:
                results.append((item, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# Initialize search engine
@st.cache_resource
def get_search_engine():
    return UniversalSearch()

search_engine = get_search_engine()

# Main interface
st.title("🔍 Universal Semantic Search")
st.markdown("*Search anything - products, concepts, ideas, or descriptions*")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("", placeholder="Search for anything... (e.g., 'wireless headphones', 'comfortable shoes', 'gaming laptop')")
with col2:
    search_clicked = st.button("🔍 Search", type="primary")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Results to show", 1, 20, 8)
    threshold = st.slider("Similarity threshold", 0.0, 0.8, 0.1)
    st.markdown("---")
    st.subheader("🚀 Quick Searches")
    quick_searches = [
        "smartphone", "laptop computer", "running shoes",
        "wireless headphones", "coffee maker", "winter jacket"
    ]
    for quick_query in quick_searches:
        if st.button(f"🔍 {quick_query}", key=f"quick_{quick_query}"):
            st.session_state.search_query = quick_query

# Dataset
dataset = [
    "iPhone 15 Pro Max - Latest Apple smartphone with titanium design",
    "Samsung Galaxy S24 Ultra - Premium Android phone with S Pen",
    "MacBook Air M3 - Lightweight laptop for professionals",
    "Dell XPS 13 - Ultra-portable Windows laptop",
    "iPad Pro 12.9 - Professional tablet for creative work",
    "Sony WH-1000XM5 - Premium noise-canceling headphones",
    "AirPods Pro - Wireless earbuds with spatial audio",
    "Nintendo Switch OLED - Portable gaming console",
    "PlayStation 5 - Next-gen gaming console",
    "Canon EOS R5 - Professional mirrorless camera",
    "Nike Air Max 270 - Comfortable running sneakers",
    "Adidas Ultraboost 22 - High-performance athletic shoes",
    "Levi's 501 Jeans - Classic denim for everyday wear",
    "Patagonia Down Jacket - Warm winter outdoor clothing",
    "Ray-Ban Aviator Sunglasses - Classic eyewear style",
    "Rolex Submariner - Luxury diving watch",
    "North Face Backpack - Durable hiking and travel bag",
    "Allbirds Tree Runners - Eco-friendly casual shoes",
    "Dyson V15 Vacuum - Powerful cordless cleaning device",
    "Instant Pot Duo - Multi-functional electric pressure cooker",
    "Nespresso VertuoLine - Premium coffee brewing machine",
    "KitchenAid Stand Mixer - Professional baking equipment",
    "Roomba i7+ - Automatic robot vacuum cleaner",
    "Philips Hue Smart Bulbs - Color-changing LED lighting",
    "Nest Thermostat - Smart home temperature control",
    "Weber Genesis Grill - Premium outdoor cooking equipment",
    "Peloton Bike+ - Interactive home fitness equipment",
    "Fitbit Charge 6 - Advanced health tracking wearable",
    "Yoga Mat Premium - Non-slip exercise surface",
    "Dumbbells Set - Adjustable weight training equipment",
    "Tennis Racket Wilson - Professional sports equipment",
    "Swimming Goggles Speedo - Water sports eyewear",
    "Mountain Bike Trek - Outdoor cycling adventure",
    "Running Shoes Brooks - Cushioned athletic footwear",
    "The Great Gatsby - Classic American literature",
    "Python Programming Guide - Technical learning book",
    "Cookbook Italian Cuisine - Recipe collection",
    "Meditation for Beginners - Mindfulness guide",
    "History of World War II - Educational documentary",
    "Jazz Music Collection - Vintage audio recordings",
    "Science Fiction Novel - Futuristic adventure story",
    "Art History Textbook - Academic reference material",
    "Skincare Routine Set - Complete facial care products",
    "Electric Toothbrush Oral-B - Advanced dental hygiene",
    "Hair Dryer Dyson - Professional styling tool",
    "Massage Gun Theragun - Muscle recovery device",
    "Essential Oils Set - Aromatherapy wellness products",
    "Protein Powder Whey - Fitness nutrition supplement",
    "Face Mask Hydrating - Beauty treatment product",
    "Sunscreen SPF 50 - Skin protection cosmetic",
    "Cloud Computing Solutions - Digital technology services",
    "Digital Marketing Strategy - Business promotion methods",
    "Sustainable Living Tips - Eco-friendly lifestyle guide",
    "Investment Portfolio Management - Financial planning service",
    "Language Learning Course - Educational skill development",
    "Virtual Reality Experience - Immersive technology entertainment",
    "Artificial Intelligence Tools - Machine learning applications",
    "Renewable Energy Systems - Clean power solutions"
]

search_query = st.session_state.get('search_query', query)
if search_query or search_clicked:
    if search_query:
        with st.spinner("🔍 Searching..."):
            start_time = time.time()
            results = search_engine.search(search_query, dataset, top_k, threshold)
            search_time = time.time() - start_time
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Results Found", len(results))
        with col2:
            st.metric("Search Time", f"{search_time:.3f}s")
        with col3:
            avg_score = np.mean([score for _, score in results]) if results else 0
            st.metric("Avg Similarity", f"{avg_score:.3f}")
        
        if results:
            st.markdown("### 🎯 Search Results")
            for i, (item, score) in enumerate(results):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="result-item">
                        <strong style="color: black;">{i+1}.</strong> {item}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    score_percent = int(score * 100)
                    st.markdown(f"""
                    <div class="score-badge">
                        {score_percent}%
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No results found. Try:")
            st.markdown("• Using different keywords")
            st.markdown("• Lowering the similarity threshold")
            st.markdown("• Trying one of the quick search options")
    
    if 'search_query' in st.session_state:
        del st.session_state.search_query

st.markdown("---")
st.markdown("💡 **Tips:** This search understands meaning, not just keywords. Try searching for concepts like 'something to keep me warm' or 'device for entertainment'")

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info(f"**{len(dataset)}** items available for search")
    st.markdown("Categories: Electronics, Fashion, Home, Sports, Books, Health, Concepts")
    st.markdown("**Note:** Results are based on semantic similarity, not exact matches.")