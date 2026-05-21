import os
import json
import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports
from ai_seekho_backend.config.settings import settings
from ai_seekho_backend.config.firebase_config import db
from ai_seekho_backend.orchestrator.agent_coordinator import run_orchestrated_matching, yield_orchestrated_matching
from ai_seekho_backend.services.provider_service import get_all_providers
from ai_seekho_backend.services.dispute_service import mediate_dispute
from ai_seekho_backend.services.scheduling_service import validate_provider_schedule
from ai_seekho_backend.models.booking import BookingModel

# New agent imports
from ai_seekho_backend.agents.coordinator_agent import CoordinatorAgent
from ai_seekho_backend.agents.executor_agent import ExecutorAgent
from ai_seekho_backend.agents.guardian_agent import GuardianAgent
from ai_seekho_backend.agents.shared.state import CoordinatorState, AgentHandoff

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai_seekho_main")

app = FastAPI(
    title="AI Seekho Backend Engine",
    description="Multi-Agent Local Services Marketplace Orchestrated with Google ADK Logic",
    version="1.0.0"
)

# Enable CORS for frontend integrations (Flutter, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections registry
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"WebSocket client registered for session: {session_id}")

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket client disconnected from session: {session_id}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send broadcast websocket packet: {e}")

manager = ConnectionManager()

# ----------------------------------------------------
# Request Schemas
# ----------------------------------------------------
class MatchRequest(BaseModel):
    query: str
    lat: float
    lng: float
    session_id: Optional[str] = "session-default"

class BookingCreateRequest(BaseModel):
    user_id: str
    provider_id: str
    service_type: str
    scheduled_time: str
    location_address: str
    lat: float
    lng: float
    price_quote: Dict[str, Any]
    intent_raw: str
    intent_parsed: Dict[str, Any]

class DisputeCreateRequest(BaseModel):
    booking_id: str
    type: str
    description: str

# ----------------------------------------------------
# HTTP API Endpoints
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": "AI Seekho Engine",
        "firebase_active": db is not None,
        "gemini_api_active": bool(settings.GEMINI_API_KEY)
    }

@app.post("/api/match")
async def match_technicians(req: MatchRequest):
    """
    HTTP endpoint to trigger the multi-factor agentic matchmaking process.
    Also broadcasts real-time updates over active WebSockets matching the session_id.
    """
    logger.info(f"Received matching request: Query='{req.query}' @ ({req.lat}, {req.lng})")
    
    # Run orchestrator
    try:
        # Broadcast initiation
        await manager.broadcast_to_session(req.session_id, {
            "event": "orchestration_started",
            "message": "Initializing AI Seekho Agent Orchestrator..."
        })
        
        result = run_orchestrated_matching(
            query=req.query,
            user_lat=req.lat,
            user_lng=req.lng,
            session_id=req.session_id
        )
        
        # Broadcast final result
        await manager.broadcast_to_session(req.session_id, {
            "event": "orchestration_completed",
            "trace_id": result["trace_id"],
            "steps": result["steps"]
        })
        
        return result
    except Exception as e:
        logger.error(f"Error in matchmaking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/providers")
def get_providers():
    """
    Fetches the loaded service provider records.
    """
    providers = get_all_providers()
    return {"count": len(providers), "providers": providers}

@app.post("/api/booking/create")
def create_booking(req: BookingCreateRequest):
    """
    Creates a physical booking record inside Firestore.
    """
    # 1. Fetch provider details to validate availability slots
    providers = get_all_providers()
    provider_slots = []
    provider_found = False
    for p in providers:
        if p.get("pid") == req.provider_id:
            provider_slots = p.get("availability_slots", [])
            provider_found = True
            break
            
    if not provider_found:
        raise HTTPException(status_code=404, detail=f"Provider with ID '{req.provider_id}' not found.")
        
    # 2. Run schedule & double-booking validation
    is_available, msg, _next_slot = validate_provider_schedule(req.provider_id, req.scheduled_time, provider_slots)
    if not is_available:
        raise HTTPException(status_code=400, detail=msg)
        
    bid = f"BK-{uuid.uuid4().hex[:6].upper()}"
    
    booking_payload = {
        "bid": bid,
        "user_id": req.user_id,
        "provider_id": req.provider_id,
        "service_type": req.service_type,
        "status": "pending",
        "scheduled_time": req.scheduled_time,
        "location": {
            "address": req.location_address,
            "lat": req.lat,
            "lng": req.lng
        },
        "price_quote": req.price_quote,
        "intent_raw": req.intent_raw,
        "intent_parsed": req.intent_parsed,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    if db:
        try:
            db.collection("bookings").document(bid).set(booking_payload)
            logger.info(f"Successfully posted Booking '{bid}' to Firestore.")
        except Exception as e:
            logger.error(f"Failed to post Booking to Firestore: {e}")
            raise HTTPException(status_code=500, detail="Firestore database failure")
            
    return {"status": "success", "booking": booking_payload}

@app.post("/api/dispute/create")
def create_dispute(req: DisputeCreateRequest):
    """
    Submits and mediations a customer service dispute.
    """
    dispute_id = f"DS-{uuid.uuid4().hex[:6].upper()}"
    try:
        payload = mediate_dispute(
            dispute_id=dispute_id,
            booking_id=req.booking_id,
            dispute_type=req.type,
            description=req.description
        )
        return {"status": "success", "dispute": payload}
    except Exception as e:
        logger.error(f"Error mediating dispute: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trace/{trace_id}")
def get_trace(trace_id: str):
    """
    Retrieves the logical agent tracing timeline for a given ID.
    """
    if db:
        try:
            ref = db.collection("agent_traces").document(trace_id).get()
            if ref.exists:
                return ref.to_dict()
        except Exception as e:
            logger.error(f"Failed to read trace from Firestore: {e}")
            
    raise HTTPException(status_code=404, detail="Trace record not found")

# ----------------------------------------------------
# WebSocket Reasoning Stream Endpoint
# ----------------------------------------------------
@app.websocket("/ws/trace/{session_id}")
async def websocket_trace_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time WebSocket endpoint. Clients connect to listen for orchestrator timelines.
    """
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                query = payload.get("query", "")
                lat = payload.get("lat", 33.649)
                lng = payload.get("lng", 72.973)
                
                await websocket.send_json({
                    "event": "orchestration_started",
                    "message": "Initializing AI Seekho Agent Orchestrator..."
                })
                
                # Execute pipeline progressively and yield traces in real-time
                for step_update in yield_orchestrated_matching(
                    query=query,
                    user_lat=lat,
                    user_lng=lng,
                    session_id=session_id
                ):
                    event_type = step_update.get("event")
                    if event_type == "step_completed":
                        await websocket.send_json({
                            "event": "step_completed",
                            "step": step_update["step"]
                        })
                    elif event_type == "orchestration_completed":
                        result = step_update["result"]
                        await websocket.send_json({
                            "event": "orchestration_completed",
                            "trace_id": result["trace_id"],
                            "matching_providers": result["matching_providers"][:3],
                            "primary_quote": result["primary_quote"],
                            "steps": result["steps"]
                        })
                
            except Exception as e:
                await websocket.send_json({
                    "event": "error",
                    "message": f"Pipeline failure: {str(e)}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)


# ====================================================================
# NEW v1 Agent Endpoints — Lazy-initialized singletons
# ====================================================================

_coordinator: CoordinatorAgent = None
_executor: ExecutorAgent = None
_guardian: GuardianAgent = None


def _get_coordinator() -> CoordinatorAgent:
    global _coordinator
    if _coordinator is None:
        _coordinator = CoordinatorAgent()
    return _coordinator


def _get_executor() -> ExecutorAgent:
    global _executor
    if _executor is None:
        _executor = ExecutorAgent()
    return _executor


def _get_guardian() -> GuardianAgent:
    global _guardian
    if _guardian is None:
        _guardian = GuardianAgent()
    return _guardian


# ── Request Schemas ───────────────────────────────────────────────

class CoordinateRequest(BaseModel):
    query: str
    lat: float
    lng: float
    session_id: str = "session-default"
    conversation_history: Optional[List[Dict[str, Any]]] = None


class ExecuteRequest(BaseModel):
    handoff: Dict[str, Any]


class ResolveDisputeRequest(BaseModel):
    booking_id: str
    dispute_type: str
    description: str
    user_id: str


class FeedbackRequest(BaseModel):
    booking_id: str
    rating: float
    comment: str
    user_id: str


class BookingStatusRequest(BaseModel):
    status: str


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/api/v1/agent/coordinate")
async def agent_coordinate(req: CoordinateRequest):
    """
    Runs the CoordinatorAgent: understands the request, finds providers, generates quotes.
    Supports multi-turn via conversation_history.
    Writes agent trace to Firestore agent_traces collection.
    """
    try:
        # Build messages list from conversation history + current query
        messages = []
        if req.conversation_history:
            messages.extend(req.conversation_history)
        messages.append({"role": "user", "content": req.query})

        state = CoordinatorState(
            messages=messages,
            session_id=req.session_id,
            user_lat=req.lat,
            user_lng=req.lng
        )

        # Broadcast initiation
        await manager.broadcast_to_session(req.session_id, {
            "event": "orchestration_started",
            "message": "Initializing AI Seekho Agent Coordinator..."
        })

        coordinator = _get_coordinator()
        result = coordinator.run(state)

        # Broadcast progressive trace events to simulate real-time thought logs
        for evt in result.get("trace_events", []):
            evt_type = evt.get("type", "think").upper()
            evt_content = evt.get("content", "")
            reasoning_msg = f"[{evt_type}] {evt_content}"
            
            await manager.broadcast_to_session(req.session_id, {
                "event": "step_completed",
                "step": {
                    "reasoning": reasoning_msg,
                    "timestamp": evt.get("timestamp", datetime.now().isoformat())
                }
            })
            await asyncio.sleep(0.4)

        # Broadcast completion
        await manager.broadcast_to_session(req.session_id, {
            "event": "orchestration_completed"
        })

        # Write trace to Firestore
        trace_id = f"COORD-{uuid.uuid4().hex[:8].upper()}"
        if db:
            try:
                db.collection("agent_traces").document(trace_id).set({
                    "trace_id": trace_id,
                    "session_id": req.session_id,
                    "agent": "CoordinatorAgent",
                    "query": req.query,
                    "action": result.get("action"),
                    "confidence": result.get("confidence"),
                    "trace_events": result.get("trace_events", []),
                    "created_at": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Coordinate trace write failed: {e}")

        # Serialize updated_state (CoordinatorState → dict)
        updated = result.get("updated_state")
        result_out = {
            "action": result.get("action"),
            "message": result.get("message"),
            "message_en": result.get("message_en"),
            "providers": result.get("providers"),
            "quote": result.get("quote"),
            "trace_events": result.get("trace_events", []),
            "confidence": result.get("confidence", 0.0),
            "trace_id": trace_id,
            "handoff": result.get("handoff"),
            "updated_state": updated.model_dump() if updated else None
        }
        return result_out

    except Exception as e:
        logger.error(f"agent_coordinate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CoordinatorAgent error: {str(e)}")


@app.post("/api/v1/agent/execute")
async def agent_execute(req: ExecuteRequest):
    """
    Runs the ExecutorAgent: locks the booking slot, creates Firestore record,
    schedules simulated reminders. Called after user confirms from CoordinatorAgent.
    """
    try:
        handoff = AgentHandoff(**req.handoff)
        executor = _get_executor()
        result = executor.execute_booking(handoff)
        return result
    except Exception as e:
        logger.error(f"agent_execute error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ExecutorAgent error: {str(e)}")


@app.post("/api/v1/agent/resolve")
async def agent_resolve(req: ResolveDisputeRequest):
    """
    Runs the GuardianAgent: resolves the dispute using Gemini reasoning + deterministic
    refund table. Escalates to human if refund > PKR 2000 or manager requested.
    """
    try:
        dispute_id = f"DS-{uuid.uuid4().hex[:6].upper()}"
        guardian = _get_guardian()
        result = guardian.resolve_dispute(
            dispute_id=dispute_id,
            booking_id=req.booking_id,
            dispute_type=req.dispute_type,
            description=req.description
        )
        return result
    except Exception as e:
        logger.error(f"agent_resolve error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GuardianAgent error: {str(e)}")


@app.post("/api/v1/feedback/submit")
async def submit_feedback(req: FeedbackRequest):
    """
    Submits booking feedback and updates provider reputation via GuardianAgent.
    Returns saved status and new provider rating.
    """
    try:
        guardian = _get_guardian()
        result = guardian.collect_feedback(
            bid=req.booking_id,
            rating=req.rating,
            comment=req.comment,
            user_id=req.user_id
        )
        return result
    except Exception as e:
        logger.error(f"submit_feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@app.get("/api/v1/bookings")
async def get_user_bookings(user_id: str):
    """
    Returns all bookings for a user_id from Firestore.
    Falls back to empty list if Firestore is unavailable.
    """
    if not db:
        return {"bookings": [], "count": 0, "source": "firestore_unavailable"}

    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
        query = db.collection("bookings").where(
            filter=FieldFilter("user_id", "==", user_id)
        ).stream()
        bookings = []
        for doc in query:
            b = doc.to_dict()
            bookings.append(b)

        # Sort by created_at descending
        bookings.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return {"bookings": bookings, "count": len(bookings)}

    except Exception as e:
        logger.error(f"get_user_bookings error: {e}")
        return {"bookings": [], "count": 0, "error": str(e)}


@app.patch("/api/v1/booking/{bid}/status")
async def update_booking_status(bid: str, req: BookingStatusRequest):
    """
    Updates booking status. Valid statuses: confirmed, en_route, in_progress, completed, cancelled.
    """
    valid_statuses = ["confirmed", "en_route", "in_progress", "completed", "cancelled", "pending", "disputed"]
    if req.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{req.status}'. Must be one of: {valid_statuses}"
        )

    if not db:
        return {"bid": bid, "new_status": req.status, "updated_at": datetime.now().isoformat(),
                "warning": "Firestore unavailable — status not persisted"}

    try:
        updated_at = datetime.now().isoformat()
        db.collection("bookings").document(bid).update({
            "status": req.status,
            "updated_at": updated_at
        })
        logger.info(f"Booking {bid} status updated to '{req.status}'.")
        return {"bid": bid, "new_status": req.status, "updated_at": updated_at}
    except Exception as e:
        logger.error(f"update_booking_status error: {e}")
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")


# ── New Agent-Stream WebSocket ────────────────────────────────────

@app.websocket("/ws/agent-stream")
async def websocket_agent_stream(websocket: WebSocket):
    """
    New real-time WebSocket endpoint that streams CoordinatorAgent trace events.
    Accepts: { query, lat, lng, session_id, conversation_history? }
    Yields: { event, content, timestamp } for each THINK/ACT/OBSERVE step.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                query = payload.get("query", "")
                lat = float(payload.get("lat", 33.649))
                lng = float(payload.get("lng", 72.973))
                session_id = payload.get("session_id", f"ws-{uuid.uuid4().hex[:8]}")
                history = payload.get("conversation_history", [])

                await websocket.send_json({
                    "event": "thinking",
                    "content": "AI Seekho Agent activated. Analyzing your request...",
                    "timestamp": datetime.now().isoformat()
                })

                messages = list(history) + [{"role": "user", "content": query}]
                state = CoordinatorState(
                    messages=messages,
                    session_id=session_id,
                    user_lat=lat,
                    user_lng=lng
                )

                coordinator = _get_coordinator()

                # Stream trace events from coordinator
                # Since run() is synchronous, we emit events after the call
                await websocket.send_json({
                    "event": "tool_call",
                    "content": "Calling understand_request_tool...",
                    "timestamp": datetime.now().isoformat()
                })

                result = coordinator.run(state)

                # Replay all trace events
                for evt in result.get("trace_events", []):
                    event_type = evt.get("type", "think")
                    ws_event = {
                        "thinking": "thinking",
                        "think": "thinking",
                        "act": "tool_call",
                        "observe": "tool_result",
                        "decision": "decision"
                    }.get(event_type, "thinking")

                    await websocket.send_json({
                        "event": ws_event,
                        "content": evt.get("content", ""),
                        "timestamp": evt.get("timestamp", datetime.now().isoformat())
                    })

                # Final completed event with full result
                await websocket.send_json({
                    "event": "completed",
                    "content": result.get("message", ""),
                    "action": result.get("action"),
                    "confidence": result.get("confidence", 0.0),
                    "providers": result.get("providers"),
                    "quote": result.get("quote"),
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"agent-stream error: {e}", exc_info=True)
                await websocket.send_json({
                    "event": "error",
                    "content": f"Agent pipeline error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        logger.info("Agent-stream WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Agent-stream fatal error: {e}")
