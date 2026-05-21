"""
agents/guardian_agent.py — Agent 3: The Guardian.
Goal: Ensure the user got what they paid for.
Collects feedback, resolves disputes fairly, updates provider reputation.
Fairness principle: "Would a reasonable senior operations manager make this call?"
Escalate when: refund > PKR 2000, contradictory evidence,
or user says 'manager se baat karni hai'.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import google.generativeai as genai

from ai_seekho_backend.config.settings import settings
from ai_seekho_backend.config.firebase_config import db
from agents.shared.prompts import GUARDIAN_SYSTEM_PROMPT
from agents.shared.tools import execute_tool, ALL_TOOL_DECLARATIONS

logger = logging.getLogger("guardian_agent")

ESCALATION_PHRASES = [
    "manager se baat", "manager chahiye", "manager ko bulao",
    "upar wale se baat", "complaint", "police", "court",
    "yeh nahi chalta", "nahi maanta", "fraud"
]


class GuardianAgent:
    """
    Agent 3: The Guardian.
    Handles feedback collection and dispute resolution with Gemini reasoning.
    """

    def __init__(self):
        self._model = None
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=GUARDIAN_SYSTEM_PROMPT,
                    tools=ALL_TOOL_DECLARATIONS
                )
                logger.info("GuardianAgent: Gemini model initialized.")
            except Exception as e:
                logger.error(f"GuardianAgent init error: {e}")

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    def _trace(self, event_type: str, content: str) -> Dict[str, Any]:
        return {"type": event_type, "content": content, "timestamp": self._now_iso()}

    def _needs_escalation(self, description: str, amount_pkr: int) -> tuple:
        """Returns (needs_escalation: bool, reason: str|None)."""
        desc_lower = description.lower()
        if amount_pkr > 2000:
            return True, f"Refund amount PKR {amount_pkr} exceeds PKR 2000 — requires human review."
        for phrase in ESCALATION_PHRASES:
            if phrase in desc_lower:
                return True, f"User requested human manager: '{phrase}' detected."
        return False, None

    def collect_feedback(
        self, bid: str, rating: float, comment: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Saves feedback to Firestore and updates provider reputation.
        Returns: { saved, new_provider_rating, message }
        """
        saved = False
        new_provider_rating = rating
        provider_id = None

        # 1. Fetch booking to get provider_id
        if db:
            try:
                doc = db.collection("bookings").document(bid).get()
                if doc.exists:
                    booking_data = doc.to_dict()
                    provider_id = booking_data.get("provider_id")
            except Exception as e:
                logger.warning(f"Could not fetch booking for feedback: {e}")

        # 2. Save feedback document
        feedback_id = f"FB-{uuid.uuid4().hex[:6].upper()}"
        feedback_payload = {
            "feedback_id": feedback_id,
            "booking_id": bid,
            "user_id": user_id,
            "rating": rating,
            "comment": comment,
            "provider_id": provider_id,
            "created_at": self._now_iso()
        }
        if db:
            try:
                db.collection("feedback").document(feedback_id).set(feedback_payload)
                saved = True
                logger.info(f"Feedback {feedback_id} saved for booking {bid}.")
            except Exception as e:
                logger.error(f"Feedback save failed: {e}")

        # 3. Update provider reputation
        if provider_id:
            rep_result = execute_tool("update_provider_reputation_tool", {
                "provider_id": provider_id,
                "new_rating": rating
            })
            new_provider_rating = rep_result.get("updated_rating", rating)

        if rating >= 4.0:
            message = "Aap ka feedback mila! Shukriya. Provider ko mila raha hai."
        elif rating >= 3.0:
            message = "Feedback mila. Hum provider ki improvement ensure karenge."
        else:
            message = "Aap ki shikayat note ho gayi. Provider ko warning di ja rahi hai."

        return {
            "saved": saved,
            "feedback_id": feedback_id,
            "new_provider_rating": new_provider_rating,
            "message": message
        }

    def resolve_dispute(
        self,
        dispute_id: str,
        booking_id: str,
        dispute_type: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Uses Gemini + GUARDIAN_SYSTEM_PROMPT to reason through the dispute.
        Returns resolution with bilingual explanation.
        """
        trace_events = []
        trace_events.append(self._trace("think",
            f"Resolving dispute {dispute_id}: type={dispute_type}, booking={booking_id}"))

        # 1. Fetch booking data
        booking_data = {}
        provider_id = None
        total_paid = 0
        base_fee = 500

        if db:
            try:
                doc = db.collection("bookings").document(booking_id).get()
                if doc.exists:
                    booking_data = doc.to_dict()
                    provider_id = booking_data.get("provider_id")
                    quote = booking_data.get("price_quote", {})
                    if isinstance(quote, dict):
                        q_inner = quote.get("quote", quote)
                        total_paid = q_inner.get("total_pkr", 0)
                        base_fee = q_inner.get("base_service_fee", 500)
                trace_events.append(self._trace("observe",
                    f"Booking fetched: provider={provider_id}, total_paid=PKR {total_paid}"))
            except Exception as e:
                logger.warning(f"Booking fetch failed: {e}")
                trace_events.append(self._trace("observe", f"Booking fetch failed: {e}"))

        # 2. Fetch provider history
        provider_history = {}
        offense_count = 0
        if provider_id and db:
            try:
                pdoc = db.collection("providers").document(provider_id).get()
                if pdoc.exists:
                    provider_history = pdoc.to_dict()
                    warnings = provider_history.get("warnings", [])
                    offense_count = sum(
                        1 for w in warnings if w.get("dispute_type") == dispute_type
                    )
                trace_events.append(self._trace("observe",
                    f"Provider history: risk_score={provider_history.get('risk_score', 0)}, "
                    f"offense_count={offense_count}"))
            except Exception as e:
                logger.warning(f"Provider history fetch failed: {e}")

        # 3. Compute refund using tool (deterministic, auditable)
        trace_events.append(self._trace("act", "compute_refund_amount_tool"))
        refund_result = execute_tool("compute_refund_amount_tool", {
            "dispute_type": dispute_type,
            "total_paid": int(total_paid),
            "base_fee": int(base_fee),
            "description": description
        })
        amount_pkr = refund_result.get("amount_pkr", 0)
        resolution_type = refund_result.get("resolution_type", "no_action")
        trace_events.append(self._trace("observe",
            f"Refund computed: {resolution_type} = PKR {amount_pkr}"))

        # 4. Check escalation
        needs_escalation, escalation_reason = self._needs_escalation(description, amount_pkr)
        if needs_escalation:
            trace_events.append(self._trace("think",
                f"ESCALATION required: {escalation_reason}"))

        # 5. Update provider reputation (if not escalated)
        provider_action = "none"
        if provider_id and not needs_escalation:
            trace_events.append(self._trace("act", "update_provider_reputation_tool"))
            # Apply a penalty rating for dispute
            penalty_rating = max(1.0, (provider_history.get("rating", 4.0) or 4.0) - 0.5)
            rep_result = execute_tool("update_provider_reputation_tool", {
                "provider_id": provider_id,
                "new_rating": penalty_rating,
                "dispute_type": dispute_type,
                "offense_count": offense_count
            })
            provider_action = rep_result.get("action", "warning_logged")
            trace_events.append(self._trace("observe",
                f"Provider reputation updated: {provider_action}"))

        # 6. Use Gemini to generate empathetic bilingual explanation (if available)
        user_message_urdu = ""
        user_message_en = ""

        if self._model and not needs_escalation:
            try:
                gemini_prompt = (
                    f"A customer dispute has been resolved. Generate a warm, clear explanation.\n"
                    f"Dispute type: {dispute_type}\n"
                    f"Resolution: {resolution_type}\n"
                    f"Amount: PKR {amount_pkr}\n"
                    f"User description: {description[:200]}\n"
                    f"Reasoning: {refund_result.get('reasoning', '')}\n\n"
                    f"Write TWO messages:\n"
                    f"1. URDU_MSG: A 2-3 sentence message in Roman Urdu (empathetic, clear)\n"
                    f"2. ENGLISH_MSG: The same message in English\n"
                    f"Format: URDU_MSG: <text>\nENGLISH_MSG: <text>"
                )
                resp = self._model.generate_content(gemini_prompt)
                text = resp.text if resp.text else ""
                for line in text.split("\n"):
                    if line.startswith("URDU_MSG:"):
                        user_message_urdu = line.replace("URDU_MSG:", "").strip()
                    elif line.startswith("ENGLISH_MSG:"):
                        user_message_en = line.replace("ENGLISH_MSG:", "").strip()
                trace_events.append(self._trace("think", "Gemini generated bilingual messages."))
            except Exception as e:
                logger.warning(f"Gemini message generation failed: {e}")

        # Fallback messages
        if not user_message_urdu:
            if needs_escalation:
                user_message_urdu = (
                    f"Aap ka case ({dispute_id}) hamare team ko bheja ja raha hai. "
                    f"24 ghante mein aap se rabta kiya jayega. Maafi chahte hain."
                )
                user_message_en = (
                    f"Your case ({dispute_id}) has been escalated to our team. "
                    f"We will contact you within 24 hours. We apologize for the inconvenience."
                )
            else:
                user_message_urdu = (
                    f"Aap ki {dispute_type} shikayat review ho gayi. "
                    f"{refund_result.get('reasoning', '')} "
                    f"PKR {amount_pkr} wapas kiye ja rahe hain. Shukriya!"
                )
                user_message_en = (
                    f"Your {dispute_type} dispute has been reviewed. "
                    f"{refund_result.get('reasoning', '')} "
                    f"PKR {amount_pkr} will be refunded. Thank you!"
                )

        # 7. Write resolution to Firestore
        resolution_payload = {
            "dispute_id": dispute_id,
            "booking_id": booking_id,
            "dispute_type": dispute_type,
            "description": description,
            "resolution_type": resolution_type,
            "amount_pkr": amount_pkr,
            "reasoning": refund_result.get("reasoning", ""),
            "provider_id": provider_id,
            "provider_action": provider_action,
            "escalation_needed": needs_escalation,
            "escalation_reason": escalation_reason,
            "user_message_urdu": user_message_urdu,
            "user_message_en": user_message_en,
            "trace_events": trace_events,
            "resolved_at": self._now_iso()
        }
        if db:
            try:
                db.collection("disputes").document(dispute_id).set(resolution_payload)
                # Update booking status to disputed or resolved
                db.collection("bookings").document(booking_id).update({
                    "status": "escalated" if needs_escalation else "disputed",
                    "dispute_id": dispute_id,
                    "updated_at": self._now_iso()
                })
                logger.info(f"Dispute {dispute_id} resolved and saved to Firestore.")
            except Exception as e:
                logger.error(f"Dispute Firestore write failed: {e}")

        return {
            "resolved": not needs_escalation,
            "escalation_needed": needs_escalation,
            "escalation_reason": escalation_reason,
            "resolution": {
                "type": resolution_type,
                "amount_pkr": amount_pkr,
                "reasoning": refund_result.get("reasoning", "")
            },
            "provider_action": provider_action,
            "user_message_urdu": user_message_urdu,
            "user_message_en": user_message_en,
            "trace_events": trace_events
        }
