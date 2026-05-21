"""
agents/shared/tools.py
======================
All callable tools for the AI Seekho multi-agent system.

Each tool is defined as:
  1. A Python function with type annotations and docstring
  2. A genai.protos.FunctionDeclaration for Gemini function calling
  3. All declarations aggregated in ALL_TOOL_DECLARATIONS

The central execute_tool() dispatcher maps Gemini function call names
to actual Python function calls and wraps all calls in try/except.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from ai_seekho_backend.config.firebase_config import db
from services.intent_service import parse_user_intent
from services.provider_service import get_matching_providers
from services.pricing_service import generate_price_quote
from services.scheduling_service import validate_provider_schedule

logger = logging.getLogger("tools")

# ---------------------------------------------------------------------------
# Tool 1: understand_request_tool
# ---------------------------------------------------------------------------

def understand_request(query: str) -> Dict[str, Any]:
    """
    Parses a natural-language service request in Urdu/Roman Urdu/English.
    Returns structured intent with confidence and optional follow-up question.

    Args:
        query: Raw user query string.

    Returns:
        dict with service_type, urgency, specializations, budget_limit,
        location_mention, confidence, urdu_reasoning, english_reasoning,
        and follow_up_question if confidence < 0.70.
    """
    result = parse_user_intent(query)

    # Ensure all expected keys exist
    result.setdefault("confidence", 0.70)
    result.setdefault("follow_up_question", None)
    result.setdefault("service_type", "general_home")
    result.setdefault("urgency", "standard")
    result.setdefault("specializations", [])
    result.setdefault("budget_limit", None)
    result.setdefault("location_mention", None)
    result.setdefault("urdu_reasoning", "")
    result.setdefault("english_reasoning", "")

    confidence = result.get("confidence", 0.70)
    if confidence < 0.70 and not result.get("follow_up_question"):
        # Generate appropriate follow-up question for the most important missing field
        if not result.get("location_mention"):
            result["follow_up_question"] = (
                "Aap ka ghar kaunse sector mein hai? (maslan G-13, F-10)"
            )
        elif not result.get("budget_limit") and result.get("urgency") == "standard":
            result["follow_up_question"] = (
                "Aap ka approximate budget kya hai? (maslan PKR 1000-2000)"
            )
        elif result.get("service_type") == "general_home" and not result.get("specializations"):
            result["follow_up_question"] = (
                "Aap ko exactly kya kaam karwana hai? Thodi aur tafseelaat batayein."
            )
        else:
            result["follow_up_question"] = (
                "Kya aap apni zaroorat ke baare mein thodi aur tafseelaat de sakte hain?"
            )

    return result


understand_request_declaration = genai.protos.FunctionDeclaration(
    name="understand_request_tool",
    description=(
        "Parses a user's service request in Urdu, Roman Urdu, English, or mixed language. "
        "Returns structured intent including service type, urgency, specializations, "
        "budget limit, location, confidence score, and a Roman Urdu follow-up question "
        "if confidence is below 0.70."
    ),
    parameters=genai.protos.Schema(
        type_=genai.protos.Type.OBJECT,
        properties={
            "query": genai.protos.Schema(
                type_=genai.protos.Type.STRING,
                description="The raw user query string in any supported language."
            )
        },
        required=["query"]
    )
)

# ---------------------------------------------------------------------------
# Tool 2: search_providers_tool
# ---------------------------------------------------------------------------

def search_providers(
    user_lat: float,
    user_lng: float,
    parsed_intent: Dict[str, Any],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Finds and ranks service providers using the 9-factor matching algorithm.

    Args:
        user_lat: User's latitude.
        user_lng: User's longitude.
        parsed_intent: Structured intent dict from understand_request_tool.
        limit: Maximum providers to return (default 5).

    Returns:
        List of provider dicts with match_score, match_factors, distance_km appended.
    """
    providers = get_matching_providers(
        user_lat=user_lat,
        user_lng=user_lng,
        parsed_intent=parsed_intent,
        limit=limit
    )
    # Filter out flagged providers
    providers = [p for p in providers if not p.get("flagged", False)]
    return providers


search_providers_declaration = genai.protos.FunctionDeclaration(
    name="search_providers_tool",
    description=(
        "Finds and ranks service providers using the 9-factor matching algorithm. "
        "Filters out flagged providers. Returns list with match scores and distance."
    ),
    parameters=genai.protos.Schema(
        type_=genai.protos.Type.OBJECT,
        properties={
            "user_lat": genai.protos.Schema(
                type_=genai.protos.Type.NUMBER,
                description="User's latitude coordinate."
            ),
            "user_lng": genai.protos.Schema(
                type_=genai.protos.Type.NUMBER,
                description="User's longitude coordinate."
            ),
            "parsed_intent": genai.protos.Schema(
                type_=genai.protos.Type.OBJECT,
                description="Structured intent dict from understand_request_tool."
            ),
            "limit": genai.protos.Schema(
                type_=genai.protos.Type.INTEGER,
                description="Maximum number of providers to return. Default is 5."
            )
        },
        required=["user_lat", "user_lng", "parsed_intent"]
    )
)

# ---------------------------------------------------------------------------
# Tool 3: generate_price_quote_tool
# ---------------------------------------------------------------------------

def generate_price_quote_tool(
    provider: Dict[str, Any],
    distance_km: float,
    parsed_intent: Dict[str, Any],
    user_lat: float,
    user_lng: float
) -> Dict[str, Any]:
    """
    Generates a dynamic price quote for a provider using the pricing formula.

    Args:
        provider: Provider dict (must include pid, base_rate_pkr, location).
        distance_km: Distance between user and provider in km.
        parsed_intent: Structured intent dict.
        user_lat: User's latitude (for budget alternative calculation).
        user_lng: User's longitude (for budget alternative calculation).

    Returns:
        PriceQuoteResponse serialized as dict.
    """
    quote_response = generate_price_quote(
        provider=provider,
        distance_km=distance_km,
        parsed_intent=parsed_intent,
        user_lat=user_lat,
        user_lng=user_lng
    )
    return quote_response.model_dump()


generate_price_quote_declaration = genai.protos.FunctionDeclaration(
    name="generate_price_quote_tool",
    description=(
        "Generates a dynamic price quote using the formula: "
        "Total = ((Base + Urgency + Complexity) × Surge) + VisitFee + DistanceFee − Loyalty. "
        "Also finds a budget alternative if one exists."
    ),
    parameters=genai.protos.Schema(
        type_=genai.protos.Type.OBJECT,
        properties={
            "provider": genai.protos.Schema(
                type_=genai.protos.Type.OBJECT,
                description="Provider dict with pid, base_rate_pkr, location."
            ),
            "distance_km": genai.protos.Schema(
                type_=genai.protos.Type.NUMBER,
                description="Distance between user and provider in kilometers."
            ),
            "parsed_intent": genai.protos.Schema(
                type_=genai.protos.Type.OBJECT,
                description="Structured intent dict with service_type, urgency, specializations."
            ),
            "user_lat": genai.protos.Schema(
                type_=genai.protos.Type.NUMBER,
                description="User's latitude for budget alternative search."
            ),
            "user_lng": genai.protos.Schema(
                type_=genai.protos.Type.NUMBER,
                description="User's longitude for budget alternative search."
            )
        },
        required=["provider", "distance_km", "parsed_intent", "user_lat", "user_lng"]
    )
)

# ---------------------------------------------------------------------------
# Tool 4: validate_slot_tool
# ---------------------------------------------------------------------------

def validate_slot(
    provider_id: str,
    requested_time: str,
    provider_slots: List[str]
) -> Dict[str, Any]:
    """
    Validates provider availability for a requested time slot.

    Args:
        provider_id: Firestore provider document ID.
        requested_time: ISO 8601 datetime string for the requested slot.
        provider_slots: List of provider's general availability slot ISO strings.

    Returns:
        dict with available (bool), message (str), next_available_slot (str|None).
    """
    available, message, next_slot = validate_provider_schedule(
        provider_id=provider_id,
        requested_time_str=requested_time,
        provider_slots=provider_slots
    )
    return {
        "available": available,
        "message": message,
        "next_available_slot": next_slot
    }


validate_slot_declaration = genai.protos.FunctionDeclaration(
    name="validate_slot_tool",
    description=(
        "Validates if a provider is available for the requested time slot. "
        "Checks general availability and Firestore for double-booking conflicts. "
        "Returns next available slot if the requested time is taken."
    ),
    parameters=genai.protos.Schema(
        type_=genai.protos.Type.OBJECT,
        properties={
            "provider_id": genai.protos.Schema(
                type_=genai.protos.Type.STRING,
                description="Provider's unique ID."
            ),
            "requested_time": genai.protos.Schema(
                type_=genai.protos.Type.STRING,
                description="Requested appointment time in ISO 8601 format."
            ),
            "provider_slots": genai.protos.Schema(
                type_=genai.protos.Type.ARRAY,
                items=genai.protos.Schema(type_=genai.protos.Type.STRING),
                description="List of provider's general availability slots as ISO strings."
            )
        },
        required=["provider_id", "requested_time", "provider_slots"]
    )
)

# ---------------------------------------------------------------------------
# Tool 5: create_booking_tool
# ---------------------------------------------------------------------------

def create_booking(
    user_id: str,
    provider_id: str,
    service_type: str,
    scheduled_time: str,
    location_address: str,
    lat: float,
    lng: float,
    price_quote: Dict[str, Any],
    intent_raw: str,
    intent_parsed: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Creates a confirmed booking record in Firestore.
    Validates slot availability first — returns conflict info if unavailable.

    Args:
        user_id: User's ID string.
        provider_id: Provider's pid string.
        service_type: Service category string.
        scheduled_time: ISO 8601 appointment time.
        location_address: Human-readable address string.
        lat: Location latitude.
        lng: Location longitude.
        price_quote: Price quote dict from generate_price_quote_tool.
        intent_raw: Original user query string.
        intent_parsed: Structured intent dict.

    Returns:
        dict with bid, status, booking dict — or conflict info if unavailable.
    """
    from services.provider_service import get_all_providers

    # 1. Find provider's availability slots
    providers = get_all_providers()
    provider_slots = []
    for p in providers:
        if p.get("pid") == provider_id:
            provider_slots = p.get("availability_slots", [])
            break

    # 2. Validate slot first
    available, msg, next_slot = validate_provider_schedule(
        provider_id=provider_id,
        requested_time_str=scheduled_time,
        provider_slots=provider_slots
    )

    if not available:
        return {
            "available": False,
            "message": msg,
            "suggested_slot": next_slot,
            "bid": None,
            "status": "conflict"
        }

    # 3. Resolve provider display name for booking history UI
    provider_name = "Unknown Provider"
    for p in providers:
        if p.get("pid") == provider_id:
            provider_name = p.get("name", provider_name)
            break

    # 4. Create booking
    bid = f"BK-{uuid.uuid4().hex[:6].upper()}"
    now_iso = datetime.now().isoformat()

    from services.notification_service import simulate_booking_notifications

    total_pkr = 0
    if isinstance(price_quote, dict):
        q = price_quote.get("quote") or price_quote
        if isinstance(q, dict):
            inner = q.get("quote") if isinstance(q.get("quote"), dict) else q
            total_pkr = inner.get("total_pkr") or inner.get("total") or 0

    notifications = simulate_booking_notifications(
        bid, provider_name, scheduled_time, total_pkr
    )

    booking_payload = {
        "bid": bid,
        "user_id": user_id,
        "provider_id": provider_id,
        "provider_name": provider_name,
        "service_type": service_type,
        "status": "confirmed",
        "scheduled_time": scheduled_time,
        "location": {
            "address": location_address,
            "lat": lat,
            "lng": lng
        },
        "price_quote": price_quote,
        "intent_raw": intent_raw,
        "intent_parsed": intent_parsed,
        "notifications": notifications,
        "created_at": now_iso,
        "updated_at": now_iso
    }

    if db:
        try:
            db.collection("bookings").document(bid).set(booking_payload)
            logger.info(f"Booking '{bid}' created in Firestore.")
        except Exception as e:
            logger.error(f"Firestore booking write failed: {e}")

    return {
        "available": True,
        "bid": bid,
        "status": "pending",
        "booking": booking_payload
    }


create_booking_declaration = genai.protos.FunctionDeclaration(
    name="create_booking_tool",
    description=(
        "Creates a confirmed booking record in Firestore after validating slot availability. "
        "If a conflict exists, returns conflict info with suggested alternative slot instead of creating a booking."
    ),
    parameters=genai.protos.Schema(
        type_=genai.protos.Type.OBJECT,
        properties={
            "user_id": genai.protos.Schema(type_=genai.protos.Type.STRING, description="User's ID."),
            "provider_id": genai.protos.Schema(type_=genai.protos.Type.STRING, description="Provider's pid."),
            "service_type": genai.protos.Schema(type_=genai.protos.Type.STRING, description="Service category."),
            "scheduled_time": genai.protos.Schema(type_=genai.protos.Type.STRING, description="ISO 8601 appointment time."),
            "location_address": genai.protos.Schema(type_=genai.protos.Type.STRING, description="Human-readable address."),
            "lat": genai.protos.Schema(type_=genai.protos.Type.NUMBER, description="Location latitude."),
            "lng": genai.protos.Schema(type_=genai.protos.Type.NUMBER, description="Location longitude."),
            "price_quote": genai.protos.Schema(type_=genai.protos.Type.OBJECT, description="Price quote dict."),
            "intent_raw": genai.protos.Schema(type_=genai.protos.Type.STRING, description="Original user query."),
            "intent_parsed": genai.protos.Schema(type_=genai.protos.Type.OBJECT, description="Structured intent dict.")
        },
        required=["user_id", "provider_id", "service_type", "scheduled_time",
                  "location_address", "lat", "lng", "price_quote", "intent_raw", "intent_parsed"]
    )
)

# ---------------------------------------------------------------------------
# Tool 6: compute_refund_amount_tool
# ---------------------------------------------------------------------------

def compute_refund_amount(
    dispute_type: str,
    total_paid: int,
    base_fee: int,
    description: str = ""
) -> Dict[str, Any]:
    """
    Computes the refund amount for a dispute using the deterministic refund table.

    Args:
        dispute_type: One of: no_show, overrun, quality, price, cancellation.
        total_paid: Total amount the user paid in PKR.
        base_fee: Base service fee from the original quote in PKR.
        description: User's dispute description (used for quality severity analysis).

    Returns:
        dict with resolution_type, amount_pkr, and reasoning.
    """
    desc_lower = description.lower()

    if dispute_type == "no_show":
        amount = total_paid + 100  # 100% refund + PKR 100 inconvenience credit
        return {
            "resolution_type": "full_refund_plus_credit",
            "amount_pkr": amount,
            "reasoning": (
                f"Provider did not show up. Full refund of PKR {total_paid} + "
                f"PKR 100 inconvenience credit = PKR {amount}."
            )
        }

    elif dispute_type == "overrun":
        amount = int(total_paid * 0.20)
        return {
            "resolution_type": "partial_refund",
            "amount_pkr": amount,
            "reasoning": (
                f"Service ran significantly over the agreed time. "
                f"20% partial refund = PKR {amount}."
            )
        }

    elif dispute_type == "quality":
        # Severity keywords affect refund percentage (30-50%)
        high_severity_keywords = [
            "dangerous", "fire", "flood", "broke", "damaged", "ruined",
            "kharab", "toota", "jalsa", "aag", "pani", "nuksan"
        ]
        medium_severity_keywords = [
            "incomplete", "poor", "bad", "wrong", "adhoora", "galat",
            "kharaab", "bura", "problem"
        ]
        severity_boost = 0
        for kw in high_severity_keywords:
            if kw in desc_lower:
                severity_boost = 20
                break
        if severity_boost == 0:
            for kw in medium_severity_keywords:
                if kw in desc_lower:
                    severity_boost = 10
                    break
        refund_pct = 0.30 + (severity_boost / 100)
        amount = int(total_paid * refund_pct)
        return {
            "resolution_type": "partial_refund",
            "amount_pkr": amount,
            "reasoning": (
                f"Quality complaint assessed at {int(refund_pct*100)}% severity. "
                f"Partial refund = PKR {amount}."
            )
        }

    elif dispute_type == "price":
        # Try to compute exact overcharge; fall back to 30% if unknown
        overcharge = total_paid - base_fee
        if overcharge > 0 and overcharge < total_paid:
            amount = overcharge
            return {
                "resolution_type": "overcharge_refund",
                "amount_pkr": amount,
                "reasoning": (
                    f"Exact overcharge detected: PKR {total_paid} charged vs "
                    f"PKR {base_fee} quoted. Refund = PKR {amount}."
                )
            }
        else:
            amount = int(total_paid * 0.30)
            return {
                "resolution_type": "flat_refund",
                "amount_pkr": amount,
                "reasoning": (
                    f"Price dispute with insufficient quote data. "
                    f"30% flat refund = PKR {amount}."
                )
            }

    elif dispute_type == "cancellation":
        amount = 150  # Travel allowance to provider
        return {
            "resolution_type": "travel_allowance",
            "amount_pkr": amount,
            "reasoning": (
                "Provider cancelled after travel. PKR 150 travel allowance awarded to provider. "
                "User receives full booking refund."
            )
        }

    else:
        return {
            "resolution_type": "no_action",
            "amount_pkr": 0,
            "reasoning": f"Unknown dispute type: {dispute_type}. No automated resolution applied."
        }


compute_refund_declaration = genai.protos.FunctionDeclaration(
    name="compute_refund_amount_tool",
    description=(
        "Computes the deterministic refund amount for a dispute using the refund table: "
        "no_show=100%+PKR100, overrun=20%, quality=30-50%, price=exact overcharge or 30%, "
        "cancellation=PKR150 travel allowance."
    ),
    parameters=genai.protos.Schema(
        type_=genai.protos.Type.OBJECT,
        properties={
            "dispute_type": genai.protos.Schema(
                type_=genai.protos.Type.STRING,
                description="Dispute category: no_show, overrun, quality, price, or cancellation."
            ),
            "total_paid": genai.protos.Schema(
                type_=genai.protos.Type.INTEGER,
                description="Total amount paid by user in PKR."
            ),
            "base_fee": genai.protos.Schema(
                type_=genai.protos.Type.INTEGER,
                description="Original quoted base service fee in PKR."
            ),
            "description": genai.protos.Schema(
                type_=genai.protos.Type.STRING,
                description="User's dispute description for severity analysis."
            )
        },
        required=["dispute_type", "total_paid", "base_fee"]
    )
)

# ---------------------------------------------------------------------------
# Tool 7: update_provider_reputation_tool
# ---------------------------------------------------------------------------

def update_provider_reputation(
    provider_id: str,
    new_rating: float,
    dispute_type: Optional[str] = None,
    offense_count: int = 0
) -> Dict[str, Any]:
    """
    Updates provider's rolling average rating and applies offense penalties.

    Args:
        provider_id: Provider's pid string.
        new_rating: New rating value between 1.0 and 5.0.
        dispute_type: Optional dispute type for offense tracking.
        offense_count: Number of previous offenses of the same type.

    Returns:
        dict with updated_rating, penalty_applied, action.
    """
    from services.provider_service import update_provider_rating_in_firestore
    return update_provider_rating_in_firestore(
        provider_id=provider_id,
        new_rating=new_rating,
        dispute_type=dispute_type
    )


update_reputation_declaration = genai.protos.FunctionDeclaration(
    name="update_provider_reputation_tool",
    description=(
        "Updates a provider's rolling average rating in Firestore. "
        "Tracks offense history and applies penalties: "
        "1st offense=warning, 2nd same offense in 30 days=match score penalty, "
        "3rd offense or fraud=flagged and hidden from search."
    ),
    parameters=genai.protos.Schema(
        type_=genai.protos.Type.OBJECT,
        properties={
            "provider_id": genai.protos.Schema(
                type_=genai.protos.Type.STRING,
                description="Provider's unique ID."
            ),
            "new_rating": genai.protos.Schema(
                type_=genai.protos.Type.NUMBER,
                description="New rating value between 1.0 and 5.0."
            ),
            "dispute_type": genai.protos.Schema(
                type_=genai.protos.Type.STRING,
                description="Optional dispute category that triggered this update."
            ),
            "offense_count": genai.protos.Schema(
                type_=genai.protos.Type.INTEGER,
                description="Number of previous offenses of this type."
            )
        },
        required=["provider_id", "new_rating"]
    )
)

# ---------------------------------------------------------------------------
# ALL_TOOL_DECLARATIONS — passed to GenerativeModel(tools=...)
# ---------------------------------------------------------------------------

ALL_TOOL_DECLARATIONS = [
    understand_request_declaration,
    search_providers_declaration,
    generate_price_quote_declaration,
    validate_slot_declaration,
    create_booking_declaration,
    compute_refund_declaration,
    update_reputation_declaration,
]

# ---------------------------------------------------------------------------
# Central Dispatcher
# ---------------------------------------------------------------------------

_TOOL_REGISTRY = {
    "understand_request_tool": understand_request,
    "search_providers_tool": search_providers,
    "generate_price_quote_tool": generate_price_quote_tool,
    "validate_slot_tool": validate_slot,
    "create_booking_tool": create_booking,
    "compute_refund_amount_tool": compute_refund_amount,
    "update_provider_reputation_tool": update_provider_reputation,
}


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """
    Central dispatcher. Called by agent run loops to execute tool calls
    returned by the Gemini API function calling response.
    Maps tool_name string → actual function call → returns result dict.
    Wraps all calls in try/except and returns {"error": str} on failure.

    Args:
        tool_name: String name of the tool to call (matches declaration name).
        tool_args: Dict of arguments to pass to the tool function.

    Returns:
        Result dict from the tool, or {"error": str} on any failure.
    """
    if tool_name not in _TOOL_REGISTRY:
        logger.error(f"Unknown tool requested: '{tool_name}'")
        return {"error": f"Unknown tool: '{tool_name}'. Valid tools: {list(_TOOL_REGISTRY.keys())}"}

    try:
        logger.info(f"Executing tool: {tool_name} with args: {list(tool_args.keys())}")
        result = _TOOL_REGISTRY[tool_name](**tool_args)
        logger.info(f"Tool '{tool_name}' completed successfully.")
        return result if isinstance(result, dict) else {"result": result}
    except TypeError as e:
        logger.error(f"Tool '{tool_name}' called with wrong arguments: {e}")
        return {"error": f"Argument error in {tool_name}: {str(e)}"}
    except Exception as e:
        logger.error(f"Tool '{tool_name}' execution failed: {e}", exc_info=True)
        return {"error": f"Tool execution failed: {str(e)}"}
