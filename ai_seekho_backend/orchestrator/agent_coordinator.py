import time
import logging
from typing import Dict, Any, List
from datetime import datetime
import uuid

from ai_seekho_backend.config.firebase_config import db
from models.agent_trace import AgentTraceModel, TraceStep
from services.intent_service import parse_user_intent
from services.provider_service import get_matching_providers
from services.pricing_service import generate_price_quote

logger = logging.getLogger("agent_coordinator")

def run_orchestrated_matching(
    query: str, 
    user_lat: float, 
    user_lng: float, 
    session_id: str = "session-default"
) -> Dict[str, Any]:
    """
    Executes the entire agentic matching pipeline and generates a rich trace timeline.
    """
    trace_id = f"TR-{uuid.uuid4().hex[:8].upper()}"
    start_total = time.perf_counter()
    steps: List[TraceStep] = []
    
    # ----------------------------------------------------
    # STEP 1: Intent Parsing Agent (Gemini/Fallback)
    # ----------------------------------------------------
    step_1_start = time.perf_counter()
    intent_parsed = parse_user_intent(query)
    step_1_lat = int((time.perf_counter() - step_1_start) * 1000)
    
    step_1_trace = TraceStep(
        step=1,
        agent="IntentParsingAgent",
        action="parse_natural_query",
        reasoning=intent_parsed.get("urdu_reasoning", ""),
        confidence=0.95 if "Heuristic" not in intent_parsed.get("urdu_reasoning", "") else 0.80,
        latency_ms=step_1_lat,
        tools_used=["Gemini_1.5_Flash", "HeuristicRegexFallback"],
        timestamp=datetime.now().isoformat()
    )
    steps.append(step_1_trace)
    
    # ----------------------------------------------------
    # STEP 2: Multi-Factor Matching Agent
    # ----------------------------------------------------
    step_2_start = time.perf_counter()
    matching_providers = get_matching_providers(user_lat, user_lng, intent_parsed, limit=5)
    step_2_lat = int((time.perf_counter() - step_2_start) * 1000)
    
    p_names = ", ".join([p["name"] for p in matching_providers]) if matching_providers else "Koi nahi"
    urdu_match_reason = (
        f"Aap ke sector ke qareeb aur is kaam ke mahir technicians dhoond liye hain. "
        f"Sab se behtareen match: {matching_providers[0]['name']} hai jis ka score {matching_providers[0]['match_score']}% hai."
        if matching_providers else "Hume koi relevant provider aap ke area me nahi mila."
    )
    
    step_2_trace = TraceStep(
        step=2,
        agent="MatchingAgent",
        action="evaluate_8_factor_score",
        reasoning=urdu_match_reason,
        confidence=0.98 if matching_providers else 0.50,
        latency_ms=step_2_lat,
        tools_used=["HaversineDistanceMath", "EightFactorRanker"],
        timestamp=datetime.now().isoformat()
    )
    steps.append(step_2_trace)
    
    # ----------------------------------------------------
    # STEP 3: Pricing Engine & Budget Options
    # ----------------------------------------------------
    step_3_start = time.perf_counter()
    
    primary_quote = None
    if matching_providers:
        best_p = matching_providers[0]
        # Calculate dynamic quote for primary match
        primary_quote = generate_price_quote(
            provider=best_p,
            distance_km=best_p["distance_km"],
            parsed_intent=intent_parsed,
            user_lat=user_lat,
            user_lng=user_lng
        )
        
    step_3_lat = int((time.perf_counter() - step_3_start) * 1000)
    
    pricing_reason = ""
    if primary_quote:
        pricing_reason = (
            f"Primary technician {best_p['name']} ka dynamic quote PKR {primary_quote.quote.total_pkr} calculate hua. "
        )
        if primary_quote.budget_alternative:
            pricing_reason += f"Aap ke liye sasta alternative P003 bhi mojood hai jo PKR {primary_quote.budget_alternative.total_pkr} me kaam kar sakta hai."
    else:
        pricing_reason = "Koi matches nahi hain tou pricing compute nahi ho saki."
        
    step_3_trace = TraceStep(
        step=3,
        agent="PricingAgent",
        action="calculate_surge_and_discount",
        reasoning=pricing_reason,
        confidence=1.0 if primary_quote else 0.0,
        latency_ms=step_3_lat,
        tools_used=["DynamicSurgeMultiplier", "BudgetAlternativeFinder"],
        timestamp=datetime.now().isoformat()
    )
    steps.append(step_3_trace)
    
    # ----------------------------------------------------
    # Finalize Orchestration Payload
    # ----------------------------------------------------
    total_lat = int((time.perf_counter() - start_total) * 1000)
    
    trace_model = AgentTraceModel(
        trace_id=trace_id,
        booking_id=None,
        session_id=session_id,
        steps=steps,
        total_latency_ms=total_lat,
        status="completed" if matching_providers else "failed",
        created_at=datetime.now().isoformat()
    )
    
    trace_payload = trace_model.model_dump()
    
    # Write trace timeline to Firestore in real-time
    if db:
        try:
            db.collection("agent_traces").document(trace_id).set(trace_payload)
            logger.info(f"Successfully posted real-time reasoning trace '{trace_id}' to Firestore.")
        except Exception as e:
            logger.error(f"Failed to upload reasoning trace to Firestore: {e}")
            
    return {
        "trace_id": trace_id,
        "parsed_intent": intent_parsed,
        "matching_providers": matching_providers,
        "primary_quote": primary_quote.model_dump() if primary_quote else None,
        "steps": [s.model_dump() for s in steps],
        "total_latency_ms": total_lat
    }


def yield_orchestrated_matching(
    query: str, 
    user_lat: float, 
    user_lng: float, 
    session_id: str = "session-default"
):
    """
    Generator that executes the agentic matching pipeline and yields each TraceStep 
    progressively to enable the frontend 'Reasoning Glass' loading visual.
    """
    trace_id = f"TR-{uuid.uuid4().hex[:8].upper()}"
    start_total = time.perf_counter()
    steps: List[TraceStep] = []
    
    # ----------------------------------------------------
    # STEP 1: Intent Parsing Agent (Gemini/Fallback)
    # ----------------------------------------------------
    step_1_start = time.perf_counter()
    intent_parsed = parse_user_intent(query)
    step_1_lat = int((time.perf_counter() - step_1_start) * 1000)
    
    step_1_trace = TraceStep(
        step=1,
        agent="IntentParsingAgent",
        action="parse_natural_query",
        reasoning=intent_parsed.get("urdu_reasoning", ""),
        confidence=0.95 if "Heuristic" not in intent_parsed.get("urdu_reasoning", "") else 0.80,
        latency_ms=step_1_lat,
        tools_used=["Gemini_1.5_Flash", "HeuristicRegexFallback"],
        timestamp=datetime.now().isoformat()
    )
    steps.append(step_1_trace)
    yield {"event": "step_completed", "step": step_1_trace.model_dump()}
    
    # ----------------------------------------------------
    # STEP 2: Multi-Factor Matching Agent
    # ----------------------------------------------------
    step_2_start = time.perf_counter()
    matching_providers = get_matching_providers(user_lat, user_lng, intent_parsed, limit=5)
    step_2_lat = int((time.perf_counter() - step_2_start) * 1000)
    
    p_names = ", ".join([p["name"] for p in matching_providers]) if matching_providers else "Koi nahi"
    urdu_match_reason = (
        f"Aap ke sector ke qareeb aur is kaam ke mahir technicians dhoond liye hain. "
        f"Sab se behtareen match: {matching_providers[0]['name']} hai jis ka score {matching_providers[0]['match_score']}% hai."
        if matching_providers else "Hume koi relevant provider aap ke area me nahi mila."
    )
    
    step_2_trace = TraceStep(
        step=2,
        agent="MatchingAgent",
        action="evaluate_8_factor_score",
        reasoning=urdu_match_reason,
        confidence=0.98 if matching_providers else 0.50,
        latency_ms=step_2_lat,
        tools_used=["HaversineDistanceMath", "EightFactorRanker"],
        timestamp=datetime.now().isoformat()
    )
    steps.append(step_2_trace)
    yield {"event": "step_completed", "step": step_2_trace.model_dump()}
    
    # ----------------------------------------------------
    # STEP 3: Pricing Engine & Budget Options
    # ----------------------------------------------------
    step_3_start = time.perf_counter()
    
    primary_quote = None
    if matching_providers:
        best_p = matching_providers[0]
        # Calculate dynamic quote for primary match
        primary_quote = generate_price_quote(
            provider=best_p,
            distance_km=best_p["distance_km"],
            parsed_intent=intent_parsed,
            user_lat=user_lat,
            user_lng=user_lng
        )
        
    step_3_lat = int((time.perf_counter() - step_3_start) * 1000)
    
    pricing_reason = ""
    if primary_quote:
        pricing_reason = (
            f"Primary technician {best_p['name']} ka dynamic quote PKR {primary_quote.quote.total_pkr} calculate hua. "
        )
        if primary_quote.budget_alternative:
            pricing_reason += f"Aap ke liye sasta alternative P003 bhi mojood hai jo PKR {primary_quote.budget_alternative.total_pkr} me kaam kar sakta hai."
    else:
        pricing_reason = "Koi matches nahi hain tou pricing compute nahi ho saki."
        
    step_3_trace = TraceStep(
        step=3,
        agent="PricingAgent",
        action="calculate_surge_and_discount",
        reasoning=pricing_reason,
        confidence=1.0 if primary_quote else 0.0,
        latency_ms=step_3_lat,
        tools_used=["DynamicSurgeMultiplier", "BudgetAlternativeFinder"],
        timestamp=datetime.now().isoformat()
    )
    steps.append(step_3_trace)
    yield {"event": "step_completed", "step": step_3_trace.model_dump()}
    
    # ----------------------------------------------------
    # Finalize Orchestration Payload
    # ----------------------------------------------------
    total_lat = int((time.perf_counter() - start_total) * 1000)
    
    trace_model = AgentTraceModel(
        trace_id=trace_id,
        booking_id=None,
        session_id=session_id,
        steps=steps,
        total_latency_ms=total_lat,
        status="completed" if matching_providers else "failed",
        created_at=datetime.now().isoformat()
    )
    
    trace_payload = trace_model.model_dump()
    
    # Write trace timeline to Firestore in real-time
    if db:
        try:
            db.collection("agent_traces").document(trace_id).set(trace_payload)
            logger.info(f"Successfully posted real-time reasoning trace '{trace_id}' to Firestore.")
        except Exception as e:
            logger.error(f"Failed to upload reasoning trace to Firestore: {e}")
            
    yield {
        "event": "orchestration_completed",
        "result": {
            "trace_id": trace_id,
            "parsed_intent": intent_parsed,
            "matching_providers": matching_providers,
            "primary_quote": primary_quote.model_dump() if primary_quote else None,
            "steps": [s.model_dump() for s in steps],
            "total_latency_ms": total_lat
        }
    }
