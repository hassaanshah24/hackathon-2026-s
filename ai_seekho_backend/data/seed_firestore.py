import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("seed_firestore")

# Local imports
from ai_seekho_backend.config.firebase_config import db

DATA_DIR = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)
MOCK_FILE_PATH = DATA_DIR / "providers_mock.json"

# Pakistani Provider Names pool
FIRST_NAMES = ["Muhammad", "Tariq", "Hassan", "Bilal", "Sajid", "Kamran", "Zahid", "Arsalan", "Farhan", "Noman", "Ayesha", "Yasmin", "Faisal", "Imran", "Zeeshan", "Waqas", "Usman", "Raza", "Asif", "Adnan"]
LAST_NAMES = ["Khan", "Ahmed", "Mahmood", "Bibi", "Ali", "Shah", "Begum", "Sheikh", "Qureshi", "Malik", "Butt", "Riaz", "Javed", "Siddiqui", "Abbasi", "Dar", "Lodhi", "Mughal", "Gill", "Chaudhry"]

# Islamabad Sector locations mapping (Sector -> Latitude, Longitude center)
SECTORS = {
    "G-13": (33.649, 72.973),
    "G-11": (33.666, 73.010),
    "F-11": (33.684, 72.998),
    "F-10": (33.693, 73.018),
    "E-11": (33.702, 72.986),
    "I-8": (33.655, 73.076),
    "H-13": (33.633, 72.965),
    "G-10": (33.674, 73.024),
    "G-9": (33.685, 73.042)
}

# Category specialized mappings
CATEGORIES = {
    "ac_repair": {
        "specializations": ["inverter_ac", "split_ac", "gas_refill", "compressor_fixing"],
        "base_range": (600, 1000),
        "per_km": 40
    },
    "plumbing": {
        "specializations": ["leak_detection", "pipe_fitting", "geyser_repair", "drain_cleaning"],
        "base_range": (400, 800),
        "per_km": 30
    },
    "electrical": {
        "specializations": ["wiring_checks", "ups_installation", "generator_servicing", "short_circuit_fixing"],
        "base_range": (500, 900),
        "per_km": 35
    },
    "tutoring": {
        "specializations": ["matric_maths", "fsc_physics", "olevel_english", "secondary_science"],
        "base_range": (800, 1500),
        "per_km": 20
    },
    "beauty": {
        "specializations": ["bridal_makeup", "hair_styling", "threading_facial", "henna_art"],
        "base_range": (1000, 2500),
        "per_km": 45
    },
    "driving": {
        "specializations": ["manual_car", "automatic_sedan", "intercity_trips", "chaffeur_service"],
        "base_range": (600, 1200),
        "per_km": 50
    },
    "mechanics": {
        "specializations": ["engine_tuning", "brake_servicing", "suspension_repair", "efi_scanning"],
        "base_range": (800, 1800),
        "per_km": 40
    },
    "general_home": {
        "specializations": ["sofa_cleaning", "woodworking", "pest_control", "wall_painting"],
        "base_range": (500, 1200),
        "per_km": 30
    }
}

# Review text corpus in mixed Urdu / English
REVIEW_CORPUS = [
    {"text": "Bohat acha kaam kiya. Waqt per aaye aur boht professional thay.", "rating": 5},
    {"text": "Satisfied with the service. Fair price and neat job.", "rating": 4},
    {"text": "Highly recommended! Inverter issue successfully resolved.", "rating": 5},
    {"text": "Kaam tou theek tha, lekin thora late aaye.", "rating": 3},
    {"text": "Acha behave tha, overall badhiya experience raha.", "rating": 5},
    {"text": "Cheated me on material cost. Price was too high.", "rating": 2},
    {"text": "Best technician in G-13. Extremely skilled.", "rating": 5},
    {"text": "UPS was fixed in 15 mins. Clean work.", "rating": 5},
    {"text": "Double bookings problem. Kept me waiting for 2 hours.", "rating": 2},
    {"text": "Bahut mahir karigar hain. Highly professional team.", "rating": 5}
]

def generate_availability_slots():
    slots = []
    base_date = datetime.now()
    # Generate 5 available slots over the next 3 days
    for day in range(0, 3):
        target_date = base_date + timedelta(days=day)
        for hour in [9, 11, 14, 16]:
            if random.random() > 0.4:
                slot_time = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slots.append(slot_time.isoformat())
    return sorted(slots)

def generate_mock_providers(count=55):
    providers = []
    
    for i in range(1, count + 1):
        pid = f"P{i:03d}"
        
        # In informal setups, primary provider categories
        primary_category = random.choice(list(CATEGORIES.keys()))
        cat_meta = CATEGORIES[primary_category]
        
        # Sub categories (20% chance of double category)
        service_categories = [primary_category]
        if random.random() < 0.2:
            second_cat = random.choice(list(CATEGORIES.keys()))
            if second_cat not in service_categories:
                service_categories.append(second_cat)
                
        # Specialized skills list
        specializations = random.sample(cat_meta["specializations"], k=random.randint(1, 3))
        
        # Sector and precise randomized coordinate
        sector = random.choice(list(SECTORS.keys()))
        base_lat, base_lng = SECTORS[sector]
        lat = base_lat + random.uniform(-0.008, 0.008)
        lng = base_lng + random.uniform(-0.008, 0.008)
        
        # Experience metrics
        experience = random.randint(1, 15)
        
        # Phone generation
        phone = f"+92-300-{random.randint(1000000, 9999999)}"
        
        # Rating distributions
        rating = round(random.uniform(4.0, 5.0), 2)
        if random.random() < 0.1: # occasional poor provider
            rating = round(random.uniform(2.8, 3.9), 2)
            
        rating_count = random.randint(10, 300)
        on_time = round(random.uniform(0.75, 0.99), 2)
        cancellation = round(random.uniform(0.01, 0.12), 2)
        
        # Base rates PKR
        base_rate = random.randint(cat_meta["base_range"][0], cat_meta["base_range"][1])
        base_rate = (base_rate // 50) * 50 # round to nearest 50
        
        # Generate reviews
        recent_reviews = []
        review_count = random.randint(2, 5)
        for _ in range(review_count):
            template = random.choice(REVIEW_CORPUS)
            rev_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
            recent_reviews.append({
                "text": template["text"],
                "rating": template["rating"],
                "date": rev_date
            })
            
        # Compile
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        
        provider = {
            "pid": pid,
            "name": name,
            "phone": phone,
            "service_categories": service_categories,
            "specializations": specializations,
            "experience_years": experience,
            "rating": rating,
            "rating_count": rating_count,
            "on_time_score": on_time,
            "cancellation_rate": cancellation,
            "base_rate_pkr": base_rate,
            "per_km_rate": cat_meta["per_km"],
            "location": {
                "area": sector,
                "lat": round(lat, 5),
                "lng": round(lng, 5)
            },
            "availability_slots": generate_availability_slots(),
            "verified": random.random() > 0.1,
            "risk_score": round(random.uniform(0.0, 0.2), 2),
            "recent_reviews": recent_reviews
        }
        
        providers.append(provider)
        
    return providers

def seed():
    logger.info("Generating mock provider dataset...")
    providers = generate_mock_providers(60)
    
    # Save locally to JSON file
    with open(MOCK_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(providers, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully generated local fallback database file at: {MOCK_FILE_PATH}")
    
    # Try uploading to Firestore in the cloud
    if db:
        logger.info("Seeding Firestore cloud database...")
        try:
            batch = db.batch()
            col_ref = db.collection("providers")
            
            # First, clean existing providers to avoid duplicates in demo
            existing = col_ref.stream()
            for doc in existing:
                doc.reference.delete()
                
            # Upload generated list
            for idx, p in enumerate(providers):
                doc_ref = col_ref.document(p["pid"])
                batch.set(doc_ref, p)
                
                # Firestore batch limit is 500 writes, we have 60 so we are totally safe
            
            batch.commit()
            logger.info("Firestore cloud collection 'providers' successfully seeded!")
        except Exception as e:
            logger.error(f"Error seeding Firestore cloud database: {e}")
    else:
        logger.warning("Firestore client is uninitialized. Skipping cloud database seed. Local JSON fallback is fully active.")

if __name__ == "__main__":
    seed()
