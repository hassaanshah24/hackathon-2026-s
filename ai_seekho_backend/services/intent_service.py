import os
import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from ai_seekho_backend.config.settings import settings
from models.intent import IntentParseRequest

logger = logging.getLogger("intent_service")

# Configure Gemini if API Key is available
gemini_available = False
if settings.GEMINI_API_KEY:
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        gemini_available = True
        logger.info("Gemini API successfully configured for Intent Parsing.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
else:
    logger.warning("GEMINI_API_KEY not found in settings. Running in heuristic-fallback mode.")

# Category keywords mapping for heuristic fallback parser
HEURISTICS = {
    "ac_repair": ["ac", "air condition", "inverter", "gas refill", "cooling", "ac kharab", "thanda"],
    "plumbing": ["leak", "pipe", "tap", "flush", "plumber", "geyser", "drain", "pani", "nalka"],
    "electrical": ["wiring", "ups", "generator", "short circuit", "fan", "board", "bijli", "light", "switch"],
    "tutoring": ["maths", "physics", "olevel", "matric", "teacher", "academy", "parhana", "study", "tutor"],
    "beauty": ["makeup", "bridal", "threading", "facial", "henna", "parlor", "shadi", "hair style"],
    "driving": ["driver", "car", "manual", "automatic", "sedan", "intercity", "gari", "chalana"],
    "mechanics": ["engine", "brake", "suspension", "efi", "tuning", "mobile oil", "gari repair"],
    "general_home": ["sofa", "carpenter", "pest", "paint", "safai", "wood", "furniture", "diwar"]
}

def _compute_confidence_and_followup(
    service_type: str,
    urgency: str,
    location_mention,
    budget_limit,
    specializations: list,
    used_heuristic: bool = False
) -> Dict[str, Any]:
    """
    Computes real confidence score based on field completeness.
    Returns dict with 'confidence' and 'follow_up_question'.
    """
    if used_heuristic:
        return {
            "confidence": 0.70,
            "follow_up_question": None
        }

    # Count how many of the 5 key fields are present
    fields = [
        service_type and service_type != "general_home",  # service_type meaningful
        urgency in ["high", "standard"],                   # urgency detected
        location_mention is not None,                       # location present
        budget_limit is not None,                           # budget present
        len(specializations) > 0                            # specializations present
    ]
    present_count = sum(1 for f in fields if f)

    if present_count == 5:
        confidence = 0.95
    elif present_count == 4:
        confidence = 0.80
    elif present_count == 3:
        confidence = 0.65
    else:
        confidence = 0.55

    follow_up_question = None
    if confidence < 0.70:
        # Ask for the single most important missing field
        if not location_mention:
            follow_up_question = (
                "Aap ka ghar kaunse sector mein hai? (maslan G-13, F-10, E-11)"
            )
        elif not budget_limit and urgency == "standard":
            follow_up_question = (
                "Aap ka approximate budget kya hai? (maslan PKR 1000-2000)"
            )
        elif service_type == "general_home" and not specializations:
            follow_up_question = (
                "Aap ko exactly kya kaam karwana hai? Thodi aur tafseelaat batayein."
            )
        else:
            follow_up_question = (
                "Kya aap apni zaroorat ke baare mein thodi aur tafseelaat de sakte hain?"
            )

    return {
        "confidence": confidence,
        "follow_up_question": follow_up_question
    }


def parse_intent_heuristically(query: str) -> Dict[str, Any]:
    """
    Highly resilient fallback heuristic parser when Gemini API is unavailable.
    Includes real confidence scoring and follow_up_question.
    """
    query_lower = query.lower()

    # 1. Determine service type by keyword matching
    service_type = "general_home"  # default baseline
    max_matches = 0
    for cat, keywords in HEURISTICS.items():
        matches = sum(1 for kw in keywords if kw in query_lower)
        if matches > max_matches:
            max_matches = matches
            service_type = cat

    # 2. Determine urgency
    urgency = "standard"
    urgency_keywords = ["jaldi", "foran", "urgent", "immediately", "emergency", "abbi", "broken", "leakage", "blasts"]
    if any(ukw in query_lower for ukw in urgency_keywords):
        urgency = "high"

    # 3. Detect specialized items
    specializations = []
    if "inverter" in query_lower:
        specializations.append("inverter_ac")
    if "geyser" in query_lower:
        specializations.append("geyser_repair")
    if "leak" in query_lower:
        specializations.append("leak_detection")
    if "ups" in query_lower:
        specializations.append("ups_installation")

    # 4. Extract budget limits (e.g. "budget 1000", "under 1500")
    budget_limit = None
    words = query_lower.split()
    for idx, w in enumerate(words):
        if w in ["budget", "under", "rs", "pkr", "rate"]:
            if idx + 1 < len(words):
                try:
                    num = int(''.join(filter(str.isdigit, words[idx + 1])))
                    if num > 100:
                        budget_limit = num
                        break
                except ValueError:
                    pass

    # 5. Location mentions
    location_mention = None
    for sec in ["g-13", "g-11", "f-11", "f-10", "e-11", "i-8", "h-13", "g-10", "g-9"]:
        if sec in query_lower:
            location_mention = sec.upper()
            break

    confidence_data = _compute_confidence_and_followup(
        service_type=service_type,
        urgency=urgency,
        location_mention=location_mention,
        budget_limit=budget_limit,
        specializations=specializations,
        used_heuristic=True  # heuristic always gives 0.70
    )

    return {
        "service_type": service_type,
        "urgency": urgency,
        "specializations": specializations,
        "budget_limit": budget_limit,
        "location_mention": location_mention,
        "urdu_reasoning": "Heuristic fallback se text analyze kiya gaya hai kyunki Gemini API connected nahi hai.",
        "english_reasoning": "Parsed using keyword fallback matching due to Gemini API offline status.",
        "confidence": confidence_data["confidence"],
        "follow_up_question": confidence_data["follow_up_question"]
    }

def parse_user_intent(query: str) -> Dict[str, Any]:
    """
    Parses natural language queries by delegating to the Gemini 1.5 model.
    Falls back to heuristic rules if offline or rate-limited.
    """
    if not gemini_available:
        return parse_intent_heuristically(query)
        
    prompt = f"""
    You are the Intent Parsing Agent for 'AI Seekho', an on-demand services matching application in Islamabad, Pakistan.
    Your task is to analyze the following user query. The query can be written in English, Urdu, Roman Urdu (Urdu written in Latin script), or a code-switched mix.

    User Query: "{query}"

    Extract the following details as a strict JSON object:
    1. "service_type": Must be exactly one of: ["ac_repair", "plumbing", "electrical", "tutoring", "beauty", "driving", "mechanics", "general_home"]. Map AC-related problems to "ac_repair", water leakage/pipe issues to "plumbing", generator/UPS/wiring to "electrical", school/college tutoring to "tutoring", salon/makeup to "beauty", drivers to "driving", car repair/tuning to "mechanics", and woodwork/painting/sofa cleaning to "general_home".
    2. "urgency": Either "high" (if the user shows panic, uses words like 'jaldi', 'foran', 'immediately', 'urgent', 'abbi', 'emergency', 'pani beh raha hai') or "standard".
    3. "specializations": An array of specific specialization tags requested, e.g. ["inverter_ac", "leak_detection", "ups_installation", "matric_maths"].
    4. "budget_limit": An integer representing PKR budget if mentioned (e.g. "1500 tak", "under 1000", "budget 2000"). Otherwise null.
    5. "location_mention": Extracted sector label in Islamabad (e.g. "G-11", "G-13", "F-11") if mentioned. Otherwise null.
    6. "urdu_reasoning": A 1-2 sentence explanation in Roman Urdu (Urdu in English alphabet) explaining your match decision. Make it sound empathetic and helpful.
    7. "english_reasoning": A 1-2 sentence explanation in standard English detailing the logical extraction details.

    Return ONLY the raw JSON string with no markdown blocks or backticks.
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        parsed = json.loads(response.text.strip())
        
        # Schema validation safeguards
        valid_cats = ["ac_repair", "plumbing", "electrical", "tutoring", "beauty", "driving", "mechanics", "general_home"]
        if parsed.get("service_type") not in valid_cats:
            # Re-verify heuristics
            fallback = parse_intent_heuristically(query)
            parsed["service_type"] = fallback["service_type"]
            
        parsed["urgency"] = parsed.get("urgency", "standard")
        if parsed["urgency"] not in ["high", "standard"]:
            parsed["urgency"] = "standard"
            
        parsed["specializations"] = parsed.get("specializations", [])
        parsed["budget_limit"] = parsed.get("budget_limit", None)
        parsed["location_mention"] = parsed.get("location_mention", None)

        # Compute real confidence based on field completeness
        confidence_data = _compute_confidence_and_followup(
            service_type=parsed.get("service_type", "general_home"),
            urgency=parsed.get("urgency", "standard"),
            location_mention=parsed.get("location_mention"),
            budget_limit=parsed.get("budget_limit"),
            specializations=parsed.get("specializations", []),
            used_heuristic=False
        )
        parsed["confidence"] = confidence_data["confidence"]
        parsed["follow_up_question"] = confidence_data["follow_up_question"]

        return parsed

    except Exception as e:
        logger.warning(f"Gemini intent parsing failed ({e}). Reverting to fallback heuristics...")
        return parse_intent_heuristically(query)
