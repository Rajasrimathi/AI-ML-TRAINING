import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

class ReadyDatasets:
    """Generate ready-to-use datasets for semantic search testing"""
    
    def __init__(self):
        # Comprehensive product data for realistic datasets
        self.product_data = {
            'smartphones': {
                'brands': ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi', 'Huawei', 'Sony', 'Motorola', 'Nokia', 'Oppo'],
                'models': ['Pro', 'Max', 'Ultra', 'Plus', 'Mini', 'Lite', 'Standard', 'Edge', 'Note', 'Galaxy'],
                'features': ['5G connectivity', 'wireless charging', 'multiple cameras', 'OLED display', 'fast charging', 'water resistant', 'face recognition', 'fingerprint sensor', 'high refresh rate', 'premium build'],
                'descriptions': [
                    'flagship smartphone with cutting-edge technology and premium design',
                    'advanced mobile device featuring latest processor and camera system',
                    'high-performance smartphone with exceptional battery life and display quality',
                    'premium mobile phone with professional-grade camera and fast performance'
                ]
            },
            'laptops': {
                'brands': ['Apple', 'Dell', 'HP', 'Lenovo', 'ASUS', 'Acer', 'MSI', 'Razer', 'Microsoft', 'Alienware'],
                'models': ['MacBook', 'ThinkPad', 'Pavilion', 'Inspiron', 'ZenBook', 'ROG', 'Surface', 'Blade'],
                'features': ['SSD storage', 'backlit keyboard', 'touchscreen', 'dedicated graphics', 'long battery life', 'lightweight design', 'fast processor', 'premium build quality', 'high resolution display', 'fast charging'],
                'descriptions': [
                    'powerful laptop designed for professional work and creative tasks',
                    'high-performance notebook with advanced features and sleek design',
                    'versatile laptop perfect for business, gaming, and multimedia use',
                    'premium portable computer with exceptional performance and build quality'
                ]
            },
            'headphones': {
                'brands': ['Sony', 'Bose', 'Apple', 'Sennheiser', 'Audio-Technica', 'Beats', 'JBL', 'Beyerdynamic', 'AKG', 'Shure'],
                'models': ['WH', 'QuietComfort', 'AirPods', 'HD', 'ATH', 'Studio', 'Live', 'DT', 'K', 'SRH'],
                'features': ['noise cancellation', 'wireless connectivity', 'long battery life', 'premium sound quality', 'comfortable fit', 'quick charge', 'voice assistant', 'foldable design', 'touch controls', 'multipoint connection'],
                'descriptions': [
                    'premium wireless headphones with superior sound quality and comfort',
                    'noise-cancelling headphones perfect for travel and focused listening',
                    'high-fidelity audio experience with advanced driver technology',
                    'professional-grade headphones for audiophiles and content creators'
                ]
            },
            'clothing': {
                'brands': ['Nike', 'Adidas', 'Zara', 'H&M', 'Uniqlo', 'Gap', 'Levi\'s', 'Calvin Klein', 'Tommy Hilfiger', 'Ralph Lauren'],
                'categories': ['shirts', 'jeans', 'dresses', 'jackets', 'shoes', 'accessories', 'activewear', 'formal wear', 'casual wear', 'outerwear'],
                'features': ['comfortable fit', 'breathable fabric', 'durable construction', 'stylish design', 'versatile styling', 'easy care', 'premium materials', 'classic style', 'modern design', 'all-season wear'],
                'descriptions': [
                    'stylish and comfortable clothing perfect for everyday wear',
                    'premium fashion item with attention to detail and quality craftsmanship',
                    'versatile wardrobe essential that combines style and functionality',
                    'contemporary design with classic appeal for modern lifestyle'
                ]
            },
            'home_decor': {
                'brands': ['IKEA', 'West Elm', 'Pottery Barn', 'CB2', 'Wayfair', 'Target', 'HomeGoods', 'Crate & Barrel', 'Williams Sonoma', 'Restoration Hardware'],
                'categories': ['furniture', 'lighting', 'rugs', 'wall art', 'storage', 'bedding', 'kitchen', 'bathroom', 'outdoor', 'decor accessories'],
                'features': ['modern design', 'space-saving', 'easy assembly', 'durable materials', 'stylish appearance', 'functional design', 'eco-friendly', 'versatile use', 'premium quality', 'affordable pricing'],
                'descriptions': [
                    'modern home furnishing that combines style and functionality',
                    'designer piece that adds elegance and character to any space',
                    'practical home solution with contemporary aesthetic appeal',
                    'premium home accessory crafted with attention to detail and quality'
                ]
            }
        }
    
    def generate_tech_dataset(self, size=500):
        """Generate technology-focused dataset"""
        products = []
        categories = ['smartphones', 'laptops', 'headphones']
        
        for i in range(size):
            category = random.choice(categories)
            data = self.product_data[category]
            
            brand = random.choice(data['brands'])
            if category == 'smartphones':
                model_parts = [random.choice(data['models']), str(random.randint(10, 15))]
                if random.random() > 0.5:
                    model_parts.append(random.choice(['Pro', 'Max', 'Ultra']))
            elif category == 'laptops':
                model_parts = [random.choice(data['models']), str(random.randint(13, 17))]
            else:  # headphones
                model_parts = [random.choice(data['models']), str(random.randint(100, 1000))]
            
            name = f"{brand} {' '.join(model_parts)}"
            
            base_desc = random.choice(data['descriptions'])
            features = random.sample(data['features'], random.randint(3, 6))
            feature_text = ', '.join(features)
            
            description = f"{base_desc.capitalize()}. Features {feature_text}. Perfect for professionals, students, and tech enthusiasts."
            
            if category == 'smartphones':
                storage_options = [64, 128, 256, 512, 1024]
                storage = random.choice(storage_options)
                specs = f"{storage}GB storage, {random.choice([6, 8, 12])}GB RAM"
                price_base = 299 + (storage / 64 * 200)
            elif category == 'laptops':
                ram_options = [8, 16, 32]
                ssd_options = [256, 512, 1024]
                ram = random.choice(ram_options)
                ssd = random.choice(ssd_options)
                specs = f"{ram}GB RAM, {ssd}GB SSD, {random.choice(['Intel', 'AMD', 'Apple'])} processor"
                price_base = 599 + (ram * 50) + (ssd / 256 * 200)
            else:  # headphones
                driver_size = random.choice([40, 45, 50, 53])
                battery_life = random.choice([20, 30, 40, 50])
                specs = f"{driver_size}mm drivers, {battery_life}h battery life"
                price_base = 99 + random.randint(50, 400)
            
            price = round(price_base * random.uniform(0.8, 1.3), 2)
            
            products.append({
                'id': i + 1,
                'name': name,
                'description': description,
                'category': 'Electronics',
                'subcategory': category.title(),
                'brand': brand,
                'specifications': specs,
                'features': feature_text,
                'price': price,
                'currency': 'USD',
                'rating': round(random.uniform(3.5, 5.0), 1),
                'review_count': random.randint(10, 2000),
                'availability': random.choice(['In Stock', 'Limited Stock', 'Out of Stock']),
                'color': random.choice(['Black', 'White', 'Silver', 'Blue', 'Red', 'Gray', 'Rose Gold']),
                'warranty_months': random.choice([12, 24, 36]),
                'release_date': (datetime.now() - timedelta(days=random.randint(0, 730))).strftime('%Y-%m-%d'),
                'tags': ', '.join(random.sample(features + [brand, category], min(5, len(features) + 2)))
            })
        
        return pd.DataFrame(products)
    
    def generate_ecommerce_dataset(self, size=1000):
        """Generate comprehensive e-commerce dataset"""
        products = []
        all_categories = list(self.product_data.keys())
        
        for i in range(size):
            category_key = random.choice(all_categories)
            data = self.product_data[category_key]
            
            brand = random.choice(data['brands'])
            
            if category_key in ['smartphones', 'laptops', 'headphones']:
                model = random.choice(data['models'])
                name = f"{brand} {model} {random.randint(100, 999)}"
            else:
                category_item = random.choice(data['categories'])
                adjective = random.choice(['Premium', 'Classic', 'Modern', 'Essential', 'Luxury'])
                name = f"{brand} {adjective} {category_item}"
            
            base_desc = random.choice(data['descriptions'])
            features = random.sample(data['features'], random.randint(2, 5))
            
            description = f"{base_desc.capitalize()}. Featuring {', '.join(features[:3])}. "
            
            if category_key == 'clothing':
                materials = ['cotton', 'polyester', 'wool', 'silk', 'linen', 'denim']
                sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
                description += f"Made from high-quality {random.choice(materials)}. Available in sizes {'-'.join(random.sample(sizes, 3))}."
            elif category_key == 'home_decor':
                dimensions = f"{random.randint(10, 200)}x{random.randint(10, 200)}x{random.randint(5, 100)}cm"
                materials = ['wood', 'metal', 'glass', 'ceramic', 'fabric', 'plastic']
                description += f"Dimensions: {dimensions}. Crafted from {random.choice(materials)}."
            
            price_ranges = {
                'smartphones': (199, 1299),
                'laptops': (399, 2999),
                'headphones': (29, 599),
                'clothing': (15, 299),
                'home_decor': (25, 899)
            }
            
            price_min, price_max = price_ranges[category_key]
            price = round(random.uniform(price_min, price_max), 2)
            
            category_mapping = {
                'smartphones': 'Electronics',
                'laptops': 'Electronics', 
                'headphones': 'Electronics',
                'clothing': 'Fashion',
                'home_decor': 'Home & Garden'
            }
            
            products.append({
                'id': i + 1,
                'name': name,
                'description': description,
                'category': category_mapping[category_key],
                'subcategory': category_key.replace('_', ' ').title(),
                'brand': brand,
                'price': price,
                'currency': 'USD',
                'rating': round(random.uniform(2.5, 5.0), 1),
                'review_count': random.randint(0, 5000),
                'availability': random.choices(
                    ['In Stock', 'Limited Stock', 'Out of Stock', 'Pre-order'],
                    weights=[70, 15, 10, 5]
                )[0],
                'features': ', '.join(features),
                'tags': ', '.join(random.sample(features + [brand], min(6, len(features) + 1))),
                'color': random.choice(['Black', 'White', 'Blue', 'Red', 'Green', 'Gray', 'Brown', 'Pink', 'Yellow', 'Purple']),
                'weight_kg': round(random.uniform(0.1, 25.0), 2),
                'dimensions_cm': f"{random.randint(5, 150)}x{random.randint(5, 150)}x{random.randint(2, 100)}",
                'material': random.choice(['Plastic', 'Metal', 'Wood', 'Glass', 'Fabric', 'Leather', 'Ceramic', 'Rubber']),
                'country_origin': random.choice(['USA', 'China', 'Germany', 'Japan', 'South Korea', 'Taiwan', 'India', 'Vietnam']),
                'warranty_months': random.choice([6, 12, 24, 36, 60]),
                'eco_friendly': random.choice([True, False]),
                'release_date': (datetime.now() - timedelta(days=random.randint(0, 1095))).strftime('%Y-%m-%d'),
                'sku': f"{brand[:3].upper()}-{category_key[:3].upper()}-{i+1:06d}"
            })
        
        return pd.DataFrame(products)
    
    def generate_fashion_dataset(self, size=2000):
        """Generate fashion-focused dataset"""
        products = []
        
        fashion_categories = {
            'clothing': ['shirts', 'jeans', 'dresses', 'jackets', 'sweaters', 'shorts', 'skirts', 'blazers'],
            'shoes': ['sneakers', 'boots', 'sandals', 'heels', 'flats', 'loafers', 'athletic shoes'],
            'accessories': ['bags', 'watches', 'jewelry', 'belts', 'scarves', 'hats', 'sunglasses']
        }
        
        for i in range(size):
            main_category = random.choice(list(fashion_categories.keys()))
            subcategory = random.choice(fashion_categories[main_category])
            
            brand = random.choice(self.product_data['clothing']['brands'])
            
            style_words = ['Classic', 'Modern', 'Vintage', 'Urban', 'Casual', 'Formal', 'Designer', 'Premium', 'Essential', 'Signature']
            style = random.choice(style_words)
            name = f"{brand} {style} {subcategory.title()}"
            
            fit_types = ['slim fit', 'regular fit', 'relaxed fit', 'tailored fit', 'oversized']
            occasions = ['casual wear', 'formal occasions', 'business meetings', 'weekend outings', 'special events']
            
            description = f"Stylish {subcategory} with {random.choice(fit_types)} design. "
            description += f"Perfect for {random.choice(occasions)}. "
            
            materials = ['cotton', 'polyester', 'wool', 'silk', 'linen', 'denim', 'leather', 'synthetic blend']
            care_instructions = ['machine washable', 'dry clean only', 'hand wash recommended', 'delicate cycle']
            
            description += f"Made from high-quality {random.choice(materials)}. {random.choice(care_instructions).title()}."
            
            sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL'] if main_category == 'clothing' else ['5', '6', '7', '8', '9', '10', '11', '12']
            colors = ['Black', 'White', 'Blue', 'Red', 'Green', 'Gray', 'Brown', 'Pink', 'Navy', 'Beige']
            
            price_ranges = {
                'clothing': (25, 299),
                'shoes': (35, 399),
                'accessories': (15, 599)
            }
            
            price_min, price_max = price_ranges[main_category]
            price = round(random.uniform(price_min, price_max), 2)
            
            products.append({
                'id': i + 1,
                'name': name,
                'description': description,
                'category': 'Fashion',
                'subcategory': main_category.title(),
                'item_type': subcategory,
                'brand': brand,
                'price': price,
                'currency': 'USD',
                'sizes_available': ', '.join(random.sample(sizes, random.randint(3, 6))),
                'colors_available': ', '.join(random.sample(colors, random.randint(2, 5))),
                'material': random.choice(materials),
                'care_instructions': random.choice(care_instructions),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'review_count': random.randint(5, 1500),
                'availability': random.choice(['In Stock', 'Limited Stock', 'Out of Stock']),
                'season': random.choice(['Spring/Summer', 'Fall/Winter', 'All Season']),
                'gender': random.choice(['Men', 'Women', 'Unisex']),
                'age_group': random.choice(['Adult', 'Teen', 'Kids']),
                'style': random.choice(['Casual', 'Formal', 'Business', 'Sporty', 'Trendy', 'Classic']),
                'tags': f"{subcategory}, {brand}, {random.choice(style_words)}, fashion",
                'eco_friendly': random.choice([True, False]),
                'made_in': random.choice(['USA', 'China', 'Bangladesh', 'Vietnam', 'India', 'Turkey', 'Italy']),
                'release_date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(products)

def save_datasets():
    """Generate and save multiple smaller datasets"""
    generator = ReadyDatasets()
    
    datasets = {
        'tech_products_500.csv': generator.generate_tech_dataset(500),
        'ecommerce_products_1000.csv': generator.generate_ecommerce_dataset(1000),
        'fashion_products_2000.csv': generator.generate_fashion_dataset(2000)
    }
    
    for filename, df in datasets.items():
        df.to_csv(filename, index=False)
        print(f"Saved {filename}: {len(df):,} products, {len(df.columns)} columns")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
        print(f"File size: ~{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")

if __name__ == "__main__":
    print("Generating smaller datasets for semantic search testing...\n")
    save_datasets()
    
    print("✅ All datasets generated successfully!")
    print("\nDataset descriptions:")
    print("- tech_products_500.csv: Technology-focused products (smartphones, laptops, headphones)")
    print("- ecommerce_products_1000.csv: Mixed e-commerce catalog across multiple categories")
    print("- fashion_products_2000.csv: Fashion-focused products (clothing, shoes, accessories)")