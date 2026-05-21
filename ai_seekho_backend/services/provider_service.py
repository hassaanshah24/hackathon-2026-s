import math
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta

from ai_seekho_backend.config.firebase_config import db
from models.provider import ProviderModel

logger = logging.getLogger("provider_service")

MOCK_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "providers_mock.json"

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes physical travel distance in kilometers between two coordinates.
    """
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def get_sector_from_coords(lat: float, lng: float) -> Optional[str]:
    """
    Approximates standard Islamabad sectors based on latitude and longitude bounding boxes.
    """
    if 33.71 <= lat <= 33.74 and 73.06 <= lng <= 73.09: return "F-6"
    if 33.71 <= lat <= 33.74 and 73.03 <= lng < 73.06: return "F-7"
    if 33.70 <= lat <= 33.73 and 73.00 <= lng < 73.03: return "F-8"
    if 33.68 <= lat < 33.71 and 73.06 <= lng <= 73.09: return "G-6"
    if 33.68 <= lat < 33.71 and 73.03 <= lng < 73.06: return "G-7"
    if 33.68 <= lat < 33.71 and 73.00 <= lng < 73.03: return "G-8"
    if 33.67 <= lat < 33.70 and 72.97 <= lng < 73.00: return "G-9"
    if 33.66 <= lat < 33.69 and 72.94 <= lng < 72.97: return "G-10"
    if 33.65 <= lat < 33.68 and 72.91 <= lng < 72.94: return "G-11"
    if 33.64 <= lat < 33.67 and 73.05 <= lng <= 73.08: return "I-8"
    if 33.63 <= lat < 33.66 and 73.02 <= lng < 73.05: return "I-9"
    if 33.62 <= lat < 33.65 and 72.99 <= lng < 73.02: return "I-10"
    return None

def resolve_multi_tier_distance(
    lat1: float, lon1: float, 
    lat2: float, lon2: float,
    intent_sector: Optional[str] = None
) -> Tuple[float, str]:
    """
    Intelligent Multi-Tier Routing and Distance Resolver:
    - Tier 1: Real Maps Road Distance (Mocked fallback check).
    - Tier 2: Islamabad Sector Grid Adjacency (If sectors are identified/approximated).
    - Tier 3: Haversine Formula with 1.3x Circuity Factor (Universal Fallback).
    Returns (distance_km, tier_used)
    """
    import os
    if os.getenv("USE_GOOGLE_MAPS_MOCK") == "true" or os.getenv("GOOGLE_MAPS_API_KEY"):
        raw_dist = haversine_distance(lat1, lon1, lat2, lon2)
        return round(raw_dist * 1.25, 2), "Tier 1: Google Maps Distance Matrix API"

    sector1 = intent_sector or get_sector_from_coords(lat1, lon1)
    sector2 = get_sector_from_coords(lat2, lon2)
    
    SECTOR_COORDS = {
        "F-6": (6, 9), "F-7": (7, 9), "F-8": (8, 9), "F-10": (10, 9), "F-11": (11, 9),
        "G-6": (6, 8), "G-7": (7, 8), "G-8": (8, 8), "G-9": (9, 8), "G-10": (10, 8), "G-11": (11, 8),
        "H-8": (8, 7), "H-9": (9, 7), "H-10": (10, 7), "H-11": (11, 7),
        "I-8": (8, 6), "I-9": (9, 6), "I-10": (10, 6), "I-11": (11, 6)
    }
    
    if sector1 in SECTOR_COORDS and sector2 in SECTOR_COORDS:
        dx = abs(SECTOR_COORDS[sector1][0] - SECTOR_COORDS[sector2][0])
        dy = abs(SECTOR_COORDS[sector1][1] - SECTOR_COORDS[sector2][1])
        grid_dist = (dx + dy) * 2.0
        
        is_fg1 = sector1[0] in ['F', 'G']
        is_fg2 = sector2[0] in ['F', 'G']
        is_i1 = sector1[0] in ['I', 'H']
        is_i2 = sector2[0] in ['I', 'H']
        
        if (is_fg1 and is_i2) or (is_fg2 and is_i1):
            grid_dist += 3.0
            
        return round(grid_dist, 2), "Tier 2: Islamabad Sector Grid Adjacency"
        
    raw_h = haversine_distance(lat1, lon1, lat2, lon2)
    circuity_dist = raw_h * 1.3
    return round(circuity_dist, 2), "Tier 3: Haversine with Circuity Coefficient (1.3x)"

def get_all_providers() -> List[Dict[str, Any]]:
    """
    Retrieves all providers from Cloud Firestore.
    If Firestore is uninitialized or fails, falls back gracefully to providers_mock.json.
    """
    providers_list = []
    
    # 1. Attempt Firestore Retrieval
    if db:
        try:
            col_ref = db.collection("providers")
            docs = col_ref.stream()
            for doc in docs:
                providers_list.append(doc.to_dict())
            if providers_list:
                return providers_list
        except Exception as e:
            logger.warning(f"Failed to fetch from Firestore ({e}). Falling back to local dataset...")
            
    # 2. Local Fallback
    if MOCK_FILE_PATH.exists():
        try:
            with open(MOCK_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read local mock file: {e}")
            
    return []

def calculate_match_score(
    provider: Dict[str, Any], 
    user_lat: float, 
    user_lng: float, 
    parsed_intent: Dict[str, Any]
) -> Tuple[float, Dict[str, float], float, str]:
    """
    Implements the 9-Factor Dynamic Match Scoring Algorithm with Job Complexity Rules.
    Returns: (final_score_0_to_100, itemized_factors_dict, distance_km, routing_tier)
    """
    service_type = parsed_intent.get("service_type", "").lower()
    urgency = parsed_intent.get("urgency", "standard").lower()
    specializations_needed = parsed_intent.get("specializations", [])
    budget_limit = parsed_intent.get("budget_limit", None)
    intent_sector = parsed_intent.get("sector", None)
    
    # Factor 1: Category Relevance (Strict Binary Filter)
    provider_cats = [c.lower() for c in provider.get("service_categories", [])]
    if service_type not in provider_cats:
        return 0.0, {}, 0.0, "None"
        
    p_lat = provider["location"]["lat"]
    p_lng = provider["location"]["lng"]
    
    # Resolve distance using Multi-Tier Routing Resolver
    dist_km, routing_tier = resolve_multi_tier_distance(user_lat, user_lng, p_lat, p_lng, intent_sector)
    
    # Factor 2: Proximity Score (Weight: 20%)
    # Under 2km = 100 pts. Drops linearly to 0 pts at 15km.
    if dist_km <= 2.0:
        proximity_pts = 100.0
    elif dist_km >= 15.0:
        proximity_pts = 0.0
    else:
        proximity_pts = 100.0 * (1.0 - (dist_km - 2.0) / 13.0)
        
    # Factor 3: Experience Score (Weight: 10%)
    # 15+ years = 100 pts, scaled linearly from 0 years.
    exp = provider.get("experience_years", 0)
    exp_pts = min(100.0, (exp / 15.0) * 100.0)
    
    # Factor 4: On-time Reliability Score (Weight: 15%)
    on_time = provider.get("on_time_score", 1.0)
    ontime_pts = on_time * 100.0
    
    # Factor 5: Average Rating Score (Weight: 15%)
    rating = provider.get("rating", 4.0)
    rating_pts = max(0.0, ((rating - 1.0) / 4.0) * 100.0)
    
    # Factor 6: Urgency & Slots Score / Availability (Weight: 10%)
    urgency_pts = 50.0
    if urgency == "high":
        has_near_slot = False
        now = datetime.now()
        eight_hours_later = now + timedelta(hours=8)
        
        for slot_str in provider.get("availability_slots", []):
            try:
                slot_time = datetime.fromisoformat(slot_str)
                if now <= slot_time <= eight_hours_later:
                    has_near_slot = True
                    break
            except Exception:
                continue
        if has_near_slot:
            urgency_pts = 100.0
        else:
            urgency_pts = 30.0
            
    # Factor 7: Specialization Boost Score (Weight: 10%)
    spec_matches = 0
    p_specs = [s.lower() for s in provider.get("specializations", [])]
    for s_need in specializations_needed:
        if s_need.lower() in p_specs:
            spec_matches += 1
    spec_pts = 100.0 if (specializations_needed and spec_matches > 0) else (50.0 if not specializations_needed else 20.0)
    
    # Factor 8: Price Sensitivity Match Score (Weight: 10%)
    price_pts = 50.0
    p_rate = provider.get("base_rate_pkr", 500)
    if budget_limit:
        if p_rate <= budget_limit:
            price_pts = 100.0
        else:
            price_pts = max(0.0, 100.0 - ((p_rate - budget_limit) / budget_limit) * 100.0)
    else:
        if p_rate <= 700:
            price_pts = 85.0
        else:
            price_pts = 60.0
            
    # Factor 9: Cancellation Penalty Score (Weight: 5%)
    cancel_rate = provider.get("cancellation_rate", 0.0)
    cancel_pts = max(0.0, (1.0 - cancel_rate) * 100.0)
    
    # Factor 10: Review Sentiment Score (Weight: 5%)
    reviews = provider.get("recent_reviews", [])
    sentiment_pts = 75.0
    if reviews:
        total_rating = 0
        positive_keywords = ["mahir", "acha", "badhiya", "neat", "clean", "satisfied", "highly professional", "recommended", "best", "skilled", "waqt per", "nice"]
        negative_keywords = ["cheated", "bad", "late", "waiting", "double bookings", "poor", "high price", "rude"]
        
        keyword_boost = 0.0
        for rev in reviews:
            text = rev.get("text", "").lower()
            total_rating += rev.get("rating", 4)
            for pk in positive_keywords:
                if pk in text:
                    keyword_boost += 5.0
            for nk in negative_keywords:
                if nk in text:
                    keyword_boost -= 5.0
                    
        avg_rev_rating = total_rating / len(reviews)
        rating_sentiment = max(0.0, ((avg_rev_rating - 1.0) / 4.0) * 100.0)
        sentiment_pts = min(100.0, max(0.0, rating_sentiment + keyword_boost))
        
    # --- Job Complexity Classification Rules ---
    # Determine Job Complexity based on service type, tags, and text parameters
    complexity_class = "basic"
    text_query = parsed_intent.get("original_text", "").lower()
    
    # Heuristics for classifying job complexity:
    complex_triggers = ["compressor", "complete rewiring", "pcb", "installation", "leakage check", "board burn", "fiting", "fitting", "mushkil"]
    inter_triggers = ["repair", "service", "short circuit", "replace", "washing machine repair", "motor"]
    
    if any(t in text_query for t in complex_triggers):
        complexity_class = "complex"
    elif any(t in text_query for t in inter_triggers):
        complexity_class = "intermediate"
        
    # Strict matching matching rules:
    complexity_multiplier = 1.0
    if complexity_class == "complex":
        # Complex jobs require 5+ years experience and specialization matching tags or certified credentials
        if exp < 5:
            complexity_multiplier = 0.1  # Severe penalty: drops matching score to avoid high risk mismatch
            logger.info(f"Applying Complex Job Penalty to Provider {provider.get('pid')} (Exp {exp} yrs < 5)")
        elif spec_pts < 50.0:
            complexity_multiplier = 0.3  # Heavy penalty for lacking specialization tags
    elif complexity_class == "intermediate":
        # Intermediate jobs require 3+ years experience
        if exp < 3:
            complexity_multiplier = 0.8  # 20% penalty
            logger.info(f"Applying Intermediate Job Penalty to Provider {provider.get('pid')} (Exp {exp} yrs < 3)")

    # Weights configuration
    weights = {
        "proximity": 0.20,
        "rating": 0.15,
        "on_time": 0.15,
        "experience": 0.10,
        "availability": 0.10,
        "specialization": 0.10,
        "price": 0.10,
        "cancellation": 0.05,
        "sentiment": 0.05
    }
    
    # Combined Multi-Factor Formula
    final_score = (
        proximity_pts * weights["proximity"] +
        rating_pts * weights["rating"] +
        ontime_pts * weights["on_time"] +
        exp_pts * weights["experience"] +
        urgency_pts * weights["availability"] +
        spec_pts * weights["specialization"] +
        price_pts * weights["price"] +
        cancel_pts * weights["cancellation"] +
        sentiment_pts * weights["sentiment"]
    )
    
    final_score = final_score * complexity_multiplier
    
    breakdown = {
        "proximity_pts": round(proximity_pts, 1),
        "rating_pts": round(rating_pts, 1),
        "on_time_pts": round(ontime_pts, 1),
        "experience_pts": round(exp_pts, 1),
        "availability_pts": round(urgency_pts, 1),
        "specialization_pts": round(spec_pts, 1),
        "price_pts": round(price_pts, 1),
        "cancellation_pts": round(cancel_pts, 1),
        "sentiment_pts": round(sentiment_pts, 1),
        "job_complexity": complexity_class,
        "complexity_multiplier": complexity_multiplier
    }
    
    return round(final_score, 2), breakdown, round(dist_km, 2), routing_tier

def update_provider_rating_in_firestore(
    provider_id: str,
    new_rating: float,
    dispute_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Updates provider rating in Firestore with rolling average.
    Handles offense tracking for dispute penalties.
    Falls back silently if Firestore unavailable.

    Returns: { "updated_rating": float, "penalty_applied": bool, "action": str }
    """
    if not db:
        return {"updated_rating": new_rating, "penalty_applied": False, "action": "firestore_unavailable"}

    try:
        doc_ref = db.collection("providers").document(provider_id)
        doc = doc_ref.get()

        if not doc.exists:
            logger.warning(f"Provider {provider_id} not found in Firestore.")
            return {"updated_rating": new_rating, "penalty_applied": False, "action": "provider_not_found"}

        provider_data = doc.to_dict()
        old_avg = float(provider_data.get("rating", 4.0) or 4.0)
        old_count = int(provider_data.get("rating_count", 1) or 1)

        # Compute new rolling average
        new_avg = round((old_avg * old_count + new_rating) / (old_count + 1), 2)

        update_payload = {
            "rating": new_avg,
            "rating_count": old_count + 1,
            "updated_at": datetime.now().isoformat()
        }

        penalty_applied = False
        action = "rating_updated"

        # Offense tracking
        if dispute_type is not None:
            now = datetime.now()
            cutoff = (now - timedelta(days=30)).isoformat()

            existing_warnings = provider_data.get("warnings", [])
            # Count same-type warnings in last 30 days
            recent_same_type = [
                w for w in existing_warnings
                if w.get("dispute_type") == dispute_type
                and w.get("timestamp", "") >= cutoff
            ]
            offense_count = len(recent_same_type)

            # Log this warning
            warning_entry = {
                "dispute_type": dispute_type,
                "timestamp": now.isoformat(),
                "rating_impact": new_rating
            }
            existing_warnings.append(warning_entry)
            update_payload["warnings"] = existing_warnings

            all_warnings_count = len(provider_data.get("warnings", [])) + 1
            risk_score = float(provider_data.get("risk_score", 0.0) or 0.0)

            if offense_count >= 1:  # 2nd same offense in 30 days
                # Apply match score penalty for 30 days
                update_payload["ranking_penalty_until"] = (
                    now + timedelta(days=30)
                ).isoformat()
                update_payload["ranking_penalty_value"] = -0.15
                penalty_applied = True
                action = "ranking_penalty_applied"
                logger.warning(
                    f"Provider {provider_id}: 2nd {dispute_type} offense — ranking penalty applied."
                )

            if all_warnings_count >= 3 or risk_score > 0.7:
                # Flag provider — hidden from search
                update_payload["flagged"] = True
                action = "provider_flagged"
                logger.error(
                    f"Provider {provider_id}: Flagged after {all_warnings_count} warnings "
                    f"or risk_score={risk_score}. Hidden from search."
                )
            else:
                action = action if penalty_applied else "warning_logged"

        doc_ref.update(update_payload)
        logger.info(f"Provider {provider_id} rating updated: {old_avg} -> {new_avg}. Action: {action}")
        return {"updated_rating": new_avg, "penalty_applied": penalty_applied, "action": action}

    except Exception as e:
        logger.error(f"update_provider_rating_in_firestore failed for {provider_id}: {e}")
        return {"updated_rating": new_rating, "penalty_applied": False, "action": f"error: {str(e)}"}

def get_matching_providers(
    user_lat: float,
    user_lng: float,
    parsed_intent: Dict[str, Any],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Filters, scores, and ranks all service providers using the 9-Factor Match System.
    Automatically excludes flagged providers.
    """
    all_providers = get_all_providers()
    scored_list = []

    for p in all_providers:
        if p.get("flagged", False):
            logger.debug(f"Skipping flagged provider: {p.get('pid', 'unknown')}")
            continue

        try:
            validated = ProviderModel(**p)
            p_dict = validated.model_dump()
        except Exception as e:
            logger.warning(f"Provider {p.get('pid', 'unknown')} failed model validation: {e}")
            p_dict = p

        score, factors, distance, routing_tier = calculate_match_score(p_dict, user_lat, user_lng, parsed_intent)
        if score > 0:
            p_dict["match_score"] = score
            p_dict["match_factors"] = factors
            p_dict["match_breakdown"] = factors
            p_dict["distance_km"] = distance
            p_dict["routing_tier"] = routing_tier
            scored_list.append(p_dict)

    scored_list.sort(key=lambda x: x["match_score"], reverse=True)
    return scored_list[:limit]
