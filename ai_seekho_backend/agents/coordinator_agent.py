"""
agents/coordinator_agent.py — Agent 1: The Coordinator.
Goal: Understand user needs, find best provider, get confirmation.
Reasoning loop: THINK -> ACT (tool call) -> OBSERVE -> THINK -> ACT -> PAUSE (human)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from ai_seekho_backend.config.settings import settings
from agents.shared.state import CoordinatorState, AgentHandoff
from agents.shared.prompts import MULTILINGUAL_PARSER_PROMPT
from agents.shared.tools import ALL_TOOL_DECLARATIONS, execute_tool

logger = logging.getLogger("coordinator_agent")

COORDINATOR_SYSTEM_PROMPT = (
    MULTILINGUAL_PARSER_PROMPT
    + "\n\nYou are the Coordinator Agent for AI Seekho. "
    "You reason step-by-step. You call tools to gather information. "
    "You never hallucinate provider data. "
    "When confidence is below 0.70, you ask ONE clarifying question before proceeding. "
    "You always present tradeoffs honestly to the user in Roman Urdu and English. "
    "Tool call sequence: understand_request_tool -> (if confident) search_providers_tool -> generate_price_quote_tool."
)

CONFIRMATION_PHRASES = [
    "confirm", "book", "yes", "ok", "okay", "theek hai", "theek",
    "haan", "ha", "book karo", "karwa do", "approved", "proceed",
    "qabool", "manzoor", "done", "go ahead", "bilkul", "zaroor",
]


class CoordinatorAgent:
    """
    Agent 1: The Coordinator.
    Reasoning loop: THINK -> ACT (tool call) -> OBSERVE -> THINK -> ACT -> PAUSE (human)
    """

    def __init__(self):
        self._model = None
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=COORDINATOR_SYSTEM_PROMPT,
                    tools=ALL_TOOL_DECLARATIONS
                )
                logger.info("CoordinatorAgent: Gemini model initialized.")
            except Exception as e:
                logger.error(f"CoordinatorAgent init error: {e}")

    def _now_iso(self) -> str:
        return datetime.now().isoformat()

    def _build_execute_handoff(
        self,
        state: "CoordinatorState",
        top_provider: Dict[str, Any],
        parsed_intent: Dict[str, Any],
        quote: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build AgentHandoff for ExecutorAgent when providers are shown.

        Always produces a valid handoff so ExecutorAgent can create a real booking
        as soon as the user confirms, without a second coordinate round-trip.

        scheduled_time: first availability_slot of top_provider, or tomorrow at 10:00 local.
        """
        from datetime import timedelta

        tomorrow_10 = (
            datetime.now() + timedelta(days=1)
        ).replace(hour=10, minute=0, second=0, microsecond=0).isoformat()

        slots = top_provider.get("availability_slots") or []
        scheduled_time = slots[0] if slots else tomorrow_10

        handoff = AgentHandoff(
            from_agent="CoordinatorAgent",
            to_agent="ExecutorAgent",
            reason="providers_ready_for_user_confirmation",
            full_context={
                "provider_id": top_provider.get("pid"),
                "provider": top_provider,
                "parsed_intent": parsed_intent,
                "user_lat": state.user_lat,
                "user_lng": state.user_lng,
                "lat": state.user_lat,
                "lng": state.user_lng,
                "session_id": state.session_id,
                "scheduled_time": scheduled_time,
                "service_type": parsed_intent.get("service_type", "general_home"),
                "user_id": parsed_intent.get("user_id", "user_demo_001"),
                "location_address": parsed_intent.get("location_mention", ""),
                "distance_km": top_provider.get("distance_km", 0.0),
                "price_quote": quote,
            },
            urgency=parsed_intent.get("urgency", "standard"),
        )
        return handoff.model_dump()

    def _trace(self, event_type: str, content: str) -> Dict[str, Any]:
        return {"type": event_type, "content": content, "timestamp": self._now_iso()}

    def _is_confirmation(self, message: str) -> bool:
        msg_lower = message.lower().strip()
        return any(phrase in msg_lower for phrase in CONFIRMATION_PHRASES)

    def _fallback_run(self, state: CoordinatorState, trace_events: List) -> Dict[str, Any]:
        """Direct tool execution when Gemini is unavailable."""
        last_user_msg = next(
            (m.get("content", "") for m in reversed(state.messages) if m.get("role") == "user"),
            ""
        )

        trace_events.append(self._trace("act", "understand_request_tool"))
        intent = execute_tool("understand_request_tool", {"query": last_user_msg})
        confidence = intent.get("confidence", 0.70)
        trace_events.append(self._trace("observe",
            f"service={intent.get('service_type')} confidence={confidence:.2f}"))

        if confidence < 0.70:
            fq = intent.get("follow_up_question",
                "Kya aap apni zaroorat ke baare mein thodi aur tafseelaat de sakte hain?")
            return {
                "action": "ask_clarification",
                "message": fq,
                "message_en": "Could you provide more details about your service request?",
                "providers": None, "quote": None,
                "trace_events": trace_events, "confidence": confidence,
                "updated_state": state.model_copy(update={"confidence": confidence, "extracted_fields": intent}),
                "handoff": None
            }

        trace_events.append(self._trace("act", "search_providers_tool"))
        providers_raw = execute_tool("search_providers_tool", {
            "user_lat": state.user_lat, "user_lng": state.user_lng,
            "parsed_intent": intent, "limit": 5
        })
        # execute_tool wraps non-dict returns as {"result": value}
        if isinstance(providers_raw, list):
            providers = providers_raw
        elif isinstance(providers_raw, dict) and "result" in providers_raw:
            providers = providers_raw["result"] if isinstance(providers_raw["result"], list) else []
        else:
            providers = []
        trace_events.append(self._trace("observe", f"Found {len(providers)} providers."))

        if not providers:
            return {
                "action": "ask_clarification",
                "message": "Aap ke qareeb koi provider nahi mila. Kya aap apna area confirm kar sakte hain?",
                "message_en": "No providers found near you. Please verify your location.",
                "providers": [], "quote": None,
                "trace_events": trace_events, "confidence": confidence,
                "updated_state": state.model_copy(update={"confidence": confidence, "extracted_fields": intent}),
                "handoff": None
            }

        top = providers[0]
        trace_events.append(self._trace("act", f"generate_price_quote_tool(pid={top.get('pid')})"))
        quote = execute_tool("generate_price_quote_tool", {
            "provider": top, "distance_km": top.get("distance_km", 3.0),
            "parsed_intent": intent, "user_lat": state.user_lat, "user_lng": state.user_lng
        })
        if "error" in quote:
            quote = None
        trace_events.append(self._trace("observe",
            f"Quote: PKR {quote.get('quote', {}).get('total_pkr', 'N/A') if quote else 'N/A'}"))

        svc = intent.get("service_type", "service").replace("_", " ").title()
        message = (
            f"Humne {len(providers)} {svc} providers dhoondhe! "
            f"Sab se behtar {top.get('name', 'Provider')} hain "
            f"({top.get('distance_km', 0):.1f}km, rating {top.get('rating', 4.0):.1f}). "
            f"Kya aap booking confirm karna chahte hain?"
        )
        message_en = (
            f"Found {len(providers)} {svc} providers! "
            f"Best match: {top.get('name', 'Provider')} "
            f"({top.get('distance_km', 0):.1f}km, {top.get('rating', 4.0):.1f}★). Confirm booking?"
        )
        return {
            "action": "show_providers",
            "message": message, "message_en": message_en,
            "providers": providers, "quote": quote,
            "trace_events": trace_events, "confidence": confidence,
            "updated_state": state.model_copy(update={
                "confidence": confidence, "extracted_fields": intent,
                "shortlisted_providers": providers, "current_step": "show_providers"
            }),
            "handoff": self._build_execute_handoff(state, top, intent, quote)
        }

    def run(self, state: CoordinatorState) -> Dict[str, Any]:
        """
        Main entry point. Returns action dict with trace_events.
        Returns:
            { "action": str, "message": str, "message_en": str,
              "providers": list|None, "quote": dict|None,
              "trace_events": list, "confidence": float,
              "updated_state": CoordinatorState, "handoff": dict|None }
        """
        trace_events = []
        last_user_msg = next(
            (m.get("content", "") for m in reversed(state.messages) if m.get("role") == "user"),
            ""
        )
        trace_events.append(self._trace("think", f"Processing: '{last_user_msg[:80]}'"))

        # Check for booking confirmation
        if self._is_confirmation(last_user_msg) and state.shortlisted_providers:
            trace_events.append(self._trace("think", "Confirmation detected — creating AgentHandoff."))
            top = state.shortlisted_providers[0]
            default_slot = datetime.now().replace(
                hour=10, minute=0, second=0, microsecond=0).isoformat()
            scheduled = (
                top.get("availability_slots", [default_slot])[0]
                if top.get("availability_slots") else default_slot
            )
            handoff = AgentHandoff(
                from_agent="CoordinatorAgent",
                to_agent="ExecutorAgent",
                reason="User confirmed booking",
                full_context={
                    "provider_id": top.get("pid"),
                    "provider": top,
                    "parsed_intent": state.extracted_fields,
                    "user_lat": state.user_lat,
                    "user_lng": state.user_lng,
                    "session_id": state.session_id,
                    "scheduled_time": scheduled,
                    "service_type": state.extracted_fields.get("service_type", "general_home"),
                    "user_id": state.extracted_fields.get("user_id", "user_demo_001"),
                    "location_address": state.extracted_fields.get("location_mention", "Islamabad"),
                    "lat": state.user_lat, "lng": state.user_lng,
                    "distance_km": top.get("distance_km", 3.0),
                },
                urgency=state.extracted_fields.get("urgency", "standard")
            )
            return {
                "action": "confirm_booking",
                "message": "Bohat acha! Aap ki booking confirm ho rahi hai. Shukriya!",
                "message_en": "Excellent! Your booking is being confirmed. Thank you!",
                "providers": state.shortlisted_providers, "quote": None,
                "trace_events": trace_events, "confidence": state.confidence,
                "updated_state": state.model_copy(update={"current_step": "confirmed"}),
                "handoff": handoff.model_dump()
            }

        if not self._model:
            return self._fallback_run(state, trace_events)

        # --- Gemini agentic loop ---
        try:
            history = []
            for msg in state.messages[:-1]:
                role = msg.get("role", "user")
                if role in ("user", "model"):
                    history.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

            chat = self._model.start_chat(history=history)
            response = chat.send_message(last_user_msg)

            intent_result = None
            providers_result = None
            quote_result = None
            confidence = state.confidence
            max_iterations = 8
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                # Check for function call
                fn_call = None
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "function_call") and part.function_call.name:
                            fn_call = part.function_call
                            break

                if fn_call:
                    tool_name = fn_call.name
                    tool_args = dict(fn_call.args)
                    trace_events.append(self._trace("act", f"{tool_name}({list(tool_args.keys())})"))
                    tool_result = execute_tool(tool_name, tool_args)
                    trace_events.append(self._trace("observe", str(tool_result)[:300]))

                    if tool_name == "understand_request_tool":
                        intent_result = tool_result
                        confidence = tool_result.get("confidence", 0.70)
                        trace_events.append(self._trace("think",
                            f"Confidence={confidence:.2f}. "
                            f"{'Ask clarification.' if confidence < 0.70 else 'Proceeding to search.'}"))
                    elif tool_name == "search_providers_tool":
                        # execute_tool wraps non-dict returns as {"result": value}
                        if isinstance(tool_result, list):
                            providers_result = tool_result
                        elif isinstance(tool_result, dict) and "result" in tool_result:
                            providers_result = tool_result["result"] if isinstance(tool_result["result"], list) else []
                        else:
                            providers_result = []
                    elif tool_name == "generate_price_quote_tool":
                        quote_result = tool_result

                    response = chat.send_message(
                        genai.protos.Content(parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"result": tool_result}
                            )
                        )])
                    )
                else:
                    # Gemini returned text — done
                    text = ""
                    if response.candidates:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, "text"):
                                text += part.text
                    trace_events.append(self._trace("think", f"Agent decision: {text[:200]}"))
                    break

            # Determine action from collected tool results
            if intent_result and confidence < 0.70:
                fq = intent_result.get("follow_up_question",
                    "Kya aap apni zaroorat ke baare mein thodi aur tafseelaat de sakte hain?")
                return {
                    "action": "ask_clarification",
                    "message": fq,
                    "message_en": "Could you provide more details?",
                    "providers": None, "quote": None,
                    "trace_events": trace_events, "confidence": confidence,
                    "updated_state": state.model_copy(update={
                        "confidence": confidence,
                        "extracted_fields": intent_result,
                        "current_step": "clarifying"
                    }),
                    "handoff": None
                }

            if providers_result is not None and providers_result and providers_result[0].get("pid"):
                svc = (intent_result or {}).get("service_type", "service").replace("_", " ").title()
                top = providers_result[0]
                message = (
                    f"Humne {len(providers_result)} {svc} providers dhoondhe! "
                    f"Sab se behtar {top.get('name', 'Provider')} "
                    f"({top.get('distance_km', 0):.1f}km, {top.get('rating', 4.0):.1f}). "
                    f"Kya confirm karna chahte hain?"
                )
                message_en = (
                    f"Found {len(providers_result)} {svc} providers! "
                    f"Best: {top.get('name', 'Provider')} "
                    f"({top.get('distance_km', 0):.1f}km, {top.get('rating', 4.0):.1f}★). Confirm?"
                )
                effective_intent = intent_result or state.extracted_fields
                return {
                    "action": "show_providers",
                    "message": message, "message_en": message_en,
                    "providers": providers_result, "quote": quote_result,
                    "trace_events": trace_events, "confidence": confidence,
                    "updated_state": state.model_copy(update={
                        "confidence": confidence,
                        "extracted_fields": effective_intent,
                        "shortlisted_providers": providers_result,
                        "current_step": "show_providers"
                    }),
                    "handoff": self._build_execute_handoff(
                        state, top, effective_intent, quote_result
                    )
                }

            # No clear result — fallback
            trace_events.append(self._trace("think", "No definitive result — running fallback tools."))
            return self._fallback_run(state, trace_events)

        except Exception as e:
            logger.error(f"CoordinatorAgent.run() Gemini error: {e}", exc_info=True)
            trace_events.append(self._trace("think", f"Gemini error — fallback: {str(e)[:100]}"))
            return self._fallback_run(state, trace_events)
