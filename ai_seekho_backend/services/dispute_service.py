import logging
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai

from ai_seekho_backend.config.settings import settings
from models.dispute import DisputeModel, DisputeResolution
from ai_seekho_backend.config.firebase_config import db

logger = logging.getLogger("dispute_service")

# Configure Gemini if API Key is available
gemini_available = False
if settings.GEMINI_API_KEY:
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        gemini_available = True
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")

def run_mediation_rules(dispute_type: str, booking_quote: Dict[str, Any], description: str) -> DisputeResolution:
    """
    Executes core deterministic business logic for dispute payouts.
    """
    total_paid = booking_quote.get("total_pkr", 1000)
    base_fee = booking_quote.get("base_service_fee", 500)
    visit_fee = booking_quote.get("visit_fee", 200)
    
    res_type = "none"
    amount = 0
    reason = ""
    
    if dispute_type == "no_show":
        # 100% full refund + warn provider
        res_type = "refund"
        amount = total_paid
        reason = "Provider did not show up. 100% refund applied to customer wallet, provider issued a critical warning warning."
        
    elif dispute_type == "overrun":
        # Late/Overrun: 20% discount on total price
        res_type = "compensation"
        amount = int(total_paid * 0.20)
        reason = "Technician arrived significantly late or overrun slot. 20% delay compensation refund credited to customer."
        
    elif dispute_type == "quality":
        # Quality Complaint: 50% partial refund for materials or quality deviation
        res_type = "refund"
        amount = int(total_paid * 0.50)
        reason = "Quality complaint validated. 50% partial restitution refund applied to user balance."
        
    elif dispute_type == "price":
        # Price Deviation: provider charged extra. Refund the deviation difference or 30% flat refund.
        res_type = "refund"
        amount = int(base_fee * 0.40) # refund base deviation factor
        reason = "Provider overcharged. The difference of rate deviation is compensated back to customer."
        
    elif dispute_type == "cancellation":
        # Late customer cancellation: Pay PKR 150 travel allowance to provider
        res_type = "warning"
        amount = 150
        reason = "Customer cancelled at short notice. PKR 150 mobilization charge credited to technician's wallet to cover travel costs."
        
    return DisputeResolution(
        type=res_type,
        amount_pkr=amount,
        reasoning=reason
    )

def mediate_dispute(dispute_id: str, booking_id: str, dispute_type: str, description: str) -> Dict[str, Any]:
    """
    Combines rule-based settlement math with a Gemini-generated empathetic resolution explanation.
    """
    # 1. Fetch booking to understand financials
    booking_quote = {"total_pkr": 1200, "base_service_fee": 600, "visit_fee": 200}
    if db:
        try:
            b_ref = db.collection("bookings").document(booking_id).get()
            if b_ref.exists:
                b_data = b_ref.to_dict()
                booking_quote = b_data.get("price_quote", booking_quote)
        except Exception as e:
            logger.warning(f"Could not load booking financials from Firestore: {e}")
            
    # 2. Run rule engine
    resolution = run_mediation_rules(dispute_type, booking_quote, description)
    
    # 3. Draft letter using Gemini (with heuristic fallback)
    urdu_explanation = f"Piyare customer, hume dukh hai k aap ko pareshani hui. Humne action le kar {resolution.type.upper()} settlement of PKR {resolution.amount_pkr} jari kar di hai."
    english_explanation = f"Dear valued client, we sincerely apologize for the inconvenience. We have mediated the dispute and issued a {resolution.type.upper()} of PKR {resolution.amount_pkr}."
    
    if gemini_available:
        prompt = f"""
        You are the Dispute Resolution Agent for 'AI Seekho', an empathetic service platform in Pakistan.
        A customer has submitted a dispute. 
        Dispute details:
        - Incident Type: {dispute_type}
        - Customer Description: "{description}"
        - Automated Decision outcome: {resolution.type} of PKR {resolution.amount_pkr}

        Formulate two resolution responses to comfort the customer:
        1. "urdu_explanation": A response written in Roman Urdu (Urdu written in the Latin alphabet) that sounds extremely polite, reassuring, apologetic, and friendly (e.g. 'Hum aapse maafi chahte hain...'). Explain the settlement of PKR {resolution.amount_pkr} and what actions we took.
        2. "english_explanation": A highly professional, warm customer relations response in standard English detailing the refund/compensation actions taken.

        Return ONLY a raw JSON object containing "urdu_explanation" and "english_explanation". No markdown tags.
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            parsed = json.loads(response.text.strip())
            urdu_explanation = parsed.get("urdu_explanation", urdu_explanation)
            english_explanation = parsed.get("english_explanation", english_explanation)
        except Exception as e:
            logger.warning(f"Failed to generate empathetic dispute letter via Gemini: {e}")
            
    resolution.reasoning = f"{english_explanation} | Roman Urdu: {urdu_explanation}"
    
    dispute_payload = {
        "dispute_id": dispute_id,
        "booking_id": booking_id,
        "type": dispute_type,
        "description": description,
        "evidence": [],
        "status": "resolved",
        "resolution": resolution.model_dump(),
        "created_at": "2026-05-18T12:00:00Z"
    }
    
    # 4. Save dispute and update booking status in Firestore
    if db:
        try:
            # Save dispute
            db.collection("disputes").document(dispute_id).set(dispute_payload)
            # Update booking status to 'disputed'
            db.collection("bookings").document(booking_id).update({"status": "disputed"})
        except Exception as e:
            logger.error(f"Failed to commit dispute records to Firestore: {e}")
            
    return dispute_payload
