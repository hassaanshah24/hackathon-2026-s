"""Booking queries, provider ops, chat messages, auto-reschedule."""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ai_seekho_backend.config.firebase_config import db
from services.provider_service import get_all_providers, get_matching_providers
from services.scheduling_service import validate_provider_schedule

logger = logging.getLogger("booking_service")

DEFAULT_PROVIDER_ID = "P001"


def _bookings_collection():
    if not db:
        return None
    return db.collection("bookings")


def list_user_bookings(user_id: str) -> List[Dict[str, Any]]:
    if not db:
        return []
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = _bookings_collection().where(filter=FieldFilter("user_id", "==", user_id)).stream()
        rows = [doc.to_dict() for doc in q]
        rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return rows
    except Exception as e:
        logger.error(f"list_user_bookings: {e}")
        return []


def list_provider_bookings(provider_id: str) -> List[Dict[str, Any]]:
    if not db:
        return []
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = _bookings_collection().where(
            filter=FieldFilter("provider_id", "==", provider_id)
        ).stream()
        rows = [doc.to_dict() for doc in q]
        rows.sort(key=lambda x: x.get("scheduled_time", ""))
        return rows
    except Exception as e:
        logger.error(f"list_provider_bookings: {e}")
        return []


def get_booking(bid: str) -> Optional[Dict[str, Any]]:
    if not db:
        return None
    try:
        doc = _bookings_collection().document(bid).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        logger.error(f"get_booking {bid}: {e}")
        return None


def provider_dashboard_stats(provider_id: str) -> Dict[str, Any]:
    bookings = list_provider_bookings(provider_id)
    today = datetime.now().date().isoformat()
    completed = [b for b in bookings if b.get("status") == "completed"]
    upcoming = [
        b for b in bookings
        if b.get("status") in ("pending", "confirmed", "en_route", "in_progress")
    ]
    earnings_today = sum(
        _extract_total_pkr(b) for b in completed if (b.get("updated_at") or "")[:10] == today
    )
    providers = get_all_providers()
    rating = 4.8
    name = "Provider"
    for p in providers:
        if p.get("pid") == provider_id:
            rating = float(p.get("rating", 4.8))
            name = p.get("name", name)
            break
    return {
        "provider_id": provider_id,
        "provider_name": name,
        "earnings_today_pkr": earnings_today,
        "jobs_completed_today": len([b for b in completed if (b.get("updated_at") or "")[:10] == today]),
        "upcoming_count": len(upcoming),
        "avg_rating": rating,
        "upcoming": upcoming[:10],
        "active": [b for b in bookings if b.get("status") in ("en_route", "in_progress")][:10],
    }


def _extract_total_pkr(booking: Dict[str, Any]) -> int:
    pq = booking.get("price_quote") or {}
    quote = pq.get("quote") or pq
    if isinstance(quote, dict):
        nested = quote.get("quote") if isinstance(quote.get("quote"), dict) else quote
        return int(nested.get("total_pkr") or nested.get("total") or 0)
    return 0


def patch_booking_fields(bid: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    fields["updated_at"] = datetime.now().isoformat()
    if not db:
        return {"bid": bid, "warning": "firestore_unavailable", **fields}
    _bookings_collection().document(bid).update(fields)
    return {"bid": bid, **fields}


def list_chat_messages(bid: str) -> List[Dict[str, Any]]:
    if not db:
        return []
    try:
        docs = _bookings_collection().document(bid).collection("messages").order_by("created_at").stream()
        return [d.to_dict() for d in docs]
    except Exception as e:
        logger.error(f"list_chat_messages: {e}")
        return []


def post_chat_message(
    bid: str,
    sender_id: str,
    sender_role: str,
    text: str,
) -> Dict[str, Any]:
    msg = {
        "id": f"MSG-{uuid.uuid4().hex[:6].upper()}",
        "bid": bid,
        "sender_id": sender_id,
        "sender_role": sender_role,
        "text": text,
        "created_at": datetime.now().isoformat(),
    }
    if db:
        try:
            _bookings_collection().document(bid).collection("messages").document(msg["id"]).set(msg)
        except Exception as e:
            logger.error(f"post_chat_message: {e}")
    return msg


def auto_reschedule_after_provider_cancel(
    bid: str,
    user_lat: float = 33.649,
    user_lng: float = 72.973,
) -> Dict[str, Any]:
    """Find alternate provider + slot when provider cancels."""
    booking = get_booking(bid)
    if not booking:
        return {"status": "error", "message": "Booking not found"}

    old_pid = booking.get("provider_id")
    parsed = booking.get("intent_parsed") or {"service_type": booking.get("service_type", "general_home")}
    matches = get_matching_providers(user_lat, user_lng, parsed, limit=5)
    alternate = None
    for m in matches:
        if m.get("pid") != old_pid:
            alternate = m
            break

    if not alternate:
        patch_booking_fields(bid, {"status": "cancelled", "cancel_reason": "provider_cancelled_no_alternate"})
        return {
            "status": "no_alternate",
            "message": "Koi alternate provider available nahi — search again.",
            "booking": get_booking(bid),
        }

    slots = alternate.get("availability_slots") or []
    new_time = slots[0] if slots else (datetime.now() + timedelta(days=1)).replace(
        hour=10, minute=0, second=0, microsecond=0
    ).isoformat()

    ok, msg, suggested = validate_provider_schedule(
        alternate.get("pid"), new_time, slots
    )
    if not ok and suggested:
        new_time = suggested

    update = {
        "status": "confirmed",
        "provider_id": alternate.get("pid"),
        "provider_name": alternate.get("name", "Alternate Provider"),
        "scheduled_time": new_time,
        "rescheduled_from_provider_cancel": True,
        "previous_provider_id": old_pid,
        "distance_km": alternate.get("distance_km", booking.get("distance_km")),
    }
    patch_booking_fields(bid, update)
    return {
        "status": "rescheduled",
        "message": f"Naya provider: {alternate.get('name')} — {new_time}",
        "alternate_provider": alternate,
        "booking": get_booking(bid),
    }


def estimate_eta_minutes(distance_km: float, status: str) -> Optional[int]:
    """Rough ETA from distance when en_route (no live GPS)."""
    if status != "en_route":
        return None
    if distance_km <= 0:
        return 15
    # ~25 km/h average urban speed + 5 min buffer
    return max(8, int((distance_km / 25.0) * 60) + 5)
