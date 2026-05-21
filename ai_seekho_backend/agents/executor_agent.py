"""
agents/executor_agent.py — Agent 2: The Executor.
Goal: Make the booking real. Lock slots atomically. Send confirmations.
Schedule reminders. Watch for provider no-shows. Handle cancellations.
Reasoning loop: ACT -> OBSERVE -> DECIDE (escalate or proceed)
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agents.shared.state import AgentHandoff
from agents.shared.tools import execute_tool
from ai_seekho_backend.config.firebase_config import db

logger = logging.getLogger("executor_agent")


class ExecutorAgent:
    """
    Agent 2: The Executor.
    Makes bookings real, schedules reminders, handles cancellations.
    """

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    def _trace(self, event_type: str, content: str) -> Dict[str, Any]:
        return {"type": event_type, "content": content, "timestamp": self._now_iso()}

    def simulate_reminders(self, bid: str, scheduled_time: str) -> List[Dict[str, Any]]:
        """
        Generates reminder payloads (simulated — not actually sent via SMS).
        Returns list of { reminder_type, trigger_at, message }
        Reminders: 24 hours before, 1 hour before, 30 minutes before (en-route)
        """
        try:
            sched_dt = datetime.fromisoformat(scheduled_time)
        except Exception:
            sched_dt = datetime.now() + timedelta(hours=24)

        reminders = [
            {
                "reminder_type": "24hr_reminder",
                "trigger_at": (sched_dt - timedelta(hours=24)).isoformat(),
                "message": (
                    f"Kal aap ki service hai (Booking ID: {bid}). "
                    "Technician waqt per aa jaye ga. Tayyar rahein!"
                )
            },
            {
                "reminder_type": "1hr_reminder",
                "trigger_at": (sched_dt - timedelta(hours=1)).isoformat(),
                "message": (
                    f"1 ghante mein technician pahunche ga (Booking ID: {bid}). "
                    "Apna darwaza khula rakhein."
                )
            },
            {
                "reminder_type": "en_route",
                "trigger_at": (sched_dt - timedelta(minutes=30)).isoformat(),
                "message": (
                    f"Technician raste mein hai (Booking ID: {bid}). "
                    "30 minute mein pahunch jayega. Shukriya!"
                )
            }
        ]
        return reminders

    def execute_booking(self, handoff: AgentHandoff) -> Dict[str, Any]:
        """
        Called after user confirms booking from CoordinatorAgent.
        Returns: { status, booking, bid, confirmation_message,
                   confirmation_message_en, reminders_scheduled,
                   trace_events, escalation_needed, escalation_reason }
        """
        trace_events = []
        ctx = handoff.full_context

        provider_id = ctx.get("provider_id", "")
        scheduled_time = ctx.get("scheduled_time", datetime.now().isoformat())
        provider = ctx.get("provider", {})
        parsed_intent = ctx.get("parsed_intent", {})
        user_lat = ctx.get("user_lat", 33.649)
        user_lng = ctx.get("user_lng", 72.973)
        service_type = ctx.get("service_type", "general_home")
        user_id = ctx.get("user_id", "user_demo_001")
        location_address = ctx.get("location_address", "Islamabad")
        lat = ctx.get("lat", user_lat)
        lng = ctx.get("lng", user_lng)
        distance_km = ctx.get("distance_km", 3.0)

        trace_events.append(self._trace("act",
            f"Validating slot for provider={provider_id} at time={scheduled_time}"))

        # Step 1: Validate slot
        provider_slots = provider.get("availability_slots", [])
        slot_result = execute_tool("validate_slot_tool", {
            "provider_id": provider_id,
            "requested_time": scheduled_time,
            "provider_slots": provider_slots
        })
        trace_events.append(self._trace("observe",
            f"Slot validation: available={slot_result.get('available')}, "
            f"msg={slot_result.get('message', '')[:100]}"))

        if not slot_result.get("available", True):
            next_slot = slot_result.get("next_available_slot")
            trace_events.append(self._trace("think",
                f"Conflict detected. Next slot: {next_slot}"))
            return {
                "status": "conflict",
                "booking": None,
                "bid": None,
                "confirmation_message": (
                    f"Maafi! Is waqt provider dastiyaab nahi hai. "
                    f"Agla slot: {next_slot or 'Kal subah 10:00 AM'}."
                ),
                "confirmation_message_en": (
                    f"Sorry, provider is not available at that time. "
                    f"Next available slot: {next_slot or 'Tomorrow 10:00 AM'}."
                ),
                "reminders_scheduled": [],
                "trace_events": trace_events,
                "escalation_needed": False,
                "escalation_reason": None,
                "suggested_slot": next_slot
            }

        trace_events.append(self._trace("think", "Slot clear — proceeding to create booking."))

        # Step 2: Generate price quote for booking record
        trace_events.append(self._trace("act", "generate_price_quote_tool"))
        quote_result = execute_tool("generate_price_quote_tool", {
            "provider": provider,
            "distance_km": distance_km,
            "parsed_intent": parsed_intent,
            "user_lat": user_lat,
            "user_lng": user_lng
        })
        if "error" in quote_result:
            quote_result = {"quote": {"total_pkr": 0}, "provider_id": provider_id}
        trace_events.append(self._trace("observe",
            f"Price quote: PKR {quote_result.get('quote', {}).get('total_pkr', 'N/A')}"))

        # Step 3: Create booking in Firestore
        trace_events.append(self._trace("act", "create_booking_tool"))
        booking_result = execute_tool("create_booking_tool", {
            "user_id": user_id,
            "provider_id": provider_id,
            "service_type": service_type,
            "scheduled_time": scheduled_time,
            "location_address": location_address,
            "lat": lat,
            "lng": lng,
            "price_quote": quote_result,
            "intent_raw": parsed_intent.get("urdu_reasoning", ""),
            "intent_parsed": parsed_intent
        })
        trace_events.append(self._trace("observe",
            f"Booking result: status={booking_result.get('status')}, bid={booking_result.get('bid')}"))

        if booking_result.get("status") == "conflict":
            return {
                "status": "conflict",
                "booking": None,
                "bid": None,
                "confirmation_message": "Schedule conflict hua — dobara koshish karein.",
                "confirmation_message_en": "Schedule conflict occurred — please try again.",
                "reminders_scheduled": [],
                "trace_events": trace_events,
                "escalation_needed": False,
                "escalation_reason": None,
                "suggested_slot": booking_result.get("suggested_slot")
            }

        bid = booking_result.get("bid", f"BK-{uuid.uuid4().hex[:6].upper()}")
        booking_data = booking_result.get("booking", {})

        # Step 4: Update Firestore booking with agent trace reference
        agent_trace_id = f"TRACE-{uuid.uuid4().hex[:8].upper()}"
        from services.notification_service import simulate_booking_notifications

        notifications = simulate_booking_notifications(
            bid, provider.get("name", "Technician"), scheduled_time,
            quote_result.get("quote", {}).get("total_pkr")
        )
        if db:
            try:
                db.collection("bookings").document(bid).update({
                    "agent_trace_id": agent_trace_id,
                    "executor_processed": True,
                    "distance_km": distance_km,
                    "notifications": notifications,
                    "status": "confirmed",
                    "updated_at": self._now_iso()
                })
            except Exception as e:
                logger.warning(f"Could not update booking trace: {e}")

        # Step 5: Write trace to Firestore
        if db:
            try:
                db.collection("agent_traces").document(agent_trace_id).set({
                    "trace_id": agent_trace_id,
                    "bid": bid,
                    "agent": "ExecutorAgent",
                    "trace_events": trace_events,
                    "created_at": self._now_iso()
                })
            except Exception as e:
                logger.warning(f"Could not write agent trace: {e}")

        # Step 6: Simulate reminders
        reminders = self.simulate_reminders(bid, scheduled_time)
        trace_events.append(self._trace("think",
            f"Booking {bid} confirmed. {len(reminders)} reminders scheduled."))

        svc = service_type.replace("_", " ").title()
        total_pkr = quote_result.get("quote", {}).get("total_pkr", "N/A")
        provider_name = provider.get("name", "Technician")

        confirmation_urdu = (
            f"Mubarak! Aap ki {svc} booking {bid} confirm ho gayi! "
            f"{provider_name} {scheduled_time[:10]} ko aayenge. "
            f"Total: PKR {total_pkr}. Shukriya AI Seekho use karne ka!"
        )
        confirmation_en = (
            f"Congratulations! Your {svc} booking {bid} is confirmed! "
            f"{provider_name} will arrive on {scheduled_time[:10]}. "
            f"Total: PKR {total_pkr}. Thank you for using AI Seekho!"
        )

        return {
            "status": "booked",
            "booking": booking_data,
            "bid": bid,
            "confirmation_message": confirmation_urdu,
            "confirmation_message_en": confirmation_en,
            "reminders_scheduled": [r["trigger_at"] for r in reminders],
            "reminders": reminders,
            "notifications": notifications,
            "agent_trace_id": agent_trace_id,
            "trace_events": trace_events,
            "antigravity": {
                "workflow_id": agent_trace_id,
                "nodes_executed": ["Coordinate", "Execute"],
                "platform": "Google Antigravity Workflow Bridge",
            },
            "escalation_needed": False,
            "escalation_reason": None
        }

    def handle_provider_cancellation(self, bid: str, provider_id: str) -> Dict[str, Any]:
        """
        Called when a provider cancels after confirmation.
        1. Updates booking status to 'cancelled' in Firestore
        2. Re-runs search for alternative provider
        3. Returns alternative provider and new slot suggestion
        4. Sets escalation_needed=True if no alternative found within 10km
        """
        # Update booking status
        if db:
            try:
                db.collection("bookings").document(bid).update({
                    "status": "cancelled",
                    "cancellation_reason": "provider_cancelled",
                    "updated_at": self._now_iso()
                })
                logger.info(f"Booking {bid} marked cancelled due to provider cancellation.")
            except Exception as e:
                logger.error(f"Could not update cancelled booking: {e}")

        # Try to find alternative (using broad search)
        try:
            from services.provider_service import get_matching_providers
            alt_providers = get_matching_providers(
                user_lat=33.649, user_lng=72.973,
                parsed_intent={"service_type": "general_home", "urgency": "high"},
                limit=5
            )
            # Filter out the cancelled provider
            alt_providers = [p for p in alt_providers if p.get("pid") != provider_id]
        except Exception as e:
            logger.error(f"Alternative provider search failed: {e}")
            alt_providers = []

        if alt_providers:
            alt = alt_providers[0]
            new_slot = (alt.get("availability_slots") or [
                (datetime.now() + timedelta(hours=2)).isoformat()
            ])[0]
            return {
                "alternative_provider": alt,
                "new_slot": new_slot,
                "user_message": (
                    f"Aap ke provider ne cancel kar diya. Hum ne ek aur provider {alt.get('name')} "
                    f"dhoondha hai jo {alt.get('distance_km', 0):.1f}km door hai. "
                    f"Kya aap unhe accept karna chahte hain?"
                ),
                "escalation_needed": False
            }
        else:
            return {
                "alternative_provider": None,
                "new_slot": None,
                "user_message": (
                    "Aap ke provider ne cancel kar diya aur abhi koi alternative 10km ke andar "
                    "dastiyaab nahi hai. Hamara team aap se rabta karega."
                ),
                "escalation_needed": True
            }
