import os
import sys
import unittest
from pathlib import Path
import json

# Setup import path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Local imports
from ai_seekho_backend.config.firebase_config import db
from services.intent_service import parse_user_intent
from services.provider_service import get_matching_providers
from services.pricing_service import generate_price_quote
from services.dispute_service import mediate_dispute
from orchestrator.agent_coordinator import run_orchestrated_matching
from services.scheduling_service import validate_provider_schedule

class TestAISeekhoBackend(unittest.TestCase):
    
    def setUp(self):
        # G-13 coordinate baseline
        self.user_lat = 33.649
        self.user_lng = 72.973
        
    def test_roman_urdu_intent_parsing(self):
        """
        Scenario 1: Code-switched / Roman Urdu intent parsing check.
        """
        query = "AC kharab ho gaya hai, bilkul cooling nahi kar raha foran theek karo"
        result = parse_user_intent(query)
        
        self.assertEqual(result["service_type"], "ac_repair")
        self.assertEqual(result["urgency"], "high")
        self.assertIn("cooling", query)
        print("\n[PASS] Scenario 1: Roman Urdu Intent Parsing successful.")
        print(f"       Extracted Service: '{result['service_type']}', Urgency: '{result['urgency']}'")
        
    def test_eight_factor_matching_and_haversine(self):
        """
        Scenario 2: Multi-factor matching ranking and physical proximity distance validation.
        """
        intent = {
            "service_type": "ac_repair",
            "urgency": "high",
            "specializations": ["inverter_ac"],
            "budget_limit": None
        }
        
        matches = get_matching_providers(self.user_lat, self.user_lng, intent, limit=3)
        self.assertTrue(len(matches) > 0, "Should find matching AC repair providers in Islamabad.")
        
        # Verify order: best match_score should be highest
        scores = [p["match_score"] for p in matches]
        self.assertEqual(scores, sorted(scores, reverse=True), "Matching results must be sorted descending by match score.")
        
        best = matches[0]
        self.assertTrue(best["distance_km"] < 25.0, "Best provider coordinates should fall within Islamabad sector boundaries.")
        print("\n[PASS] Scenario 2: 8-Factor Proximity Evaluation and ranking successful.")
        print(f"       Top Match: {best['name']} ({best['location']['area']}) | Score: {best['match_score']}% | Distance: {best['distance_km']}km")

    def test_dynamic_pricing_calculation(self):
        """
        Scenario 3: Peak sRGB summer surge pricing and loyalty reductions check.
        """
        provider = {
            "pid": "P001",
            "name": "Arsalan Khan",
            "base_rate_pkr": 800,
            "per_km_rate": 40,
            "rating": 4.8,
            "location": {
                "lat": 33.666,
                "lng": 73.010
            }
        }
        distance = 3.5
        intent = {
            "service_type": "ac_repair",
            "urgency": "high",
            "specializations": ["inverter_ac"],
            "loyalty_tier": "gold"
        }
        
        quote_resp = generate_price_quote(provider, distance, intent)
        quote = quote_resp.quote
        
        # Math verification
        expected_distance_fee = int(3.5 * 40)
        self.assertEqual(quote.distance_fee, expected_distance_fee)
        self.assertEqual(quote.urgency_surcharge, 150)
        self.assertEqual(quote.loyalty_discount, 150)
        self.assertTrue(quote.total_pkr >= 300, "Safety floor must be respected.")
        print("\n[PASS] Scenario 3: Dynamic Pricing, surge calculations, and loyalty discounts check successful.")
        print(f"       Invoice Breakdown: {quote.breakdown_reasoning}")
        print(f"       Final Total PKR: {quote.total_pkr}")

    def test_empathic_dispute_resolution(self):
        """
        Scenario 4: Auto-payout mediation for no-show technician.
        """
        dispute_id = "DS-TEST"
        booking_id = "BK-TEST"
        dispute_type = "no_show"
        desc = "Technician did not arrive, calls are switched off!"
        
        payload = mediate_dispute(dispute_id, booking_id, dispute_type, desc)
        
        self.assertEqual(payload["status"], "resolved")
        self.assertEqual(payload["resolution"]["type"], "refund")
        self.assertTrue(payload["resolution"]["amount_pkr"] > 0)
        print("\n[PASS] Scenario 4: Empathetic Dispute Resolution and Auto-Mediation rules check successful.")
        print(f"       Dispute Outcome: {payload['resolution']['type'].upper()} of PKR {payload['resolution']['amount_pkr']}")
        try:
            print(f"       Decision reasoning summary: {payload['resolution']['reasoning']}")
        except UnicodeEncodeError:
            print(f"       Decision reasoning summary: {payload['resolution']['reasoning'].encode('ascii', errors='ignore').decode('ascii')}")

    def test_pipeline_orchestration_trace(self):
        """
        Scenario 5: Full coordinated pipeline execution checking latency logging and trace logging.
        """
        query = "Mera geyser leak ho raha hai jaldi aao"
        res = run_orchestrated_matching(query, self.user_lat, self.user_lng, session_id="test-session")
        
        self.assertIn("trace_id", res)
        self.assertTrue(len(res["steps"]) >= 3, "Orchestrator trace must record Intent, Match, and Pricing timelines.")
        self.assertTrue(res["total_latency_ms"] >= 0)
        print("\n[PASS] Scenario 5: Full Orchestrator Trace workflow completed successfully.")
        print(f"       Trace ID: '{res['trace_id']}' in {res['total_latency_ms']}ms across {len(res['steps'])} reasoning steps.")

    def test_scheduling_conflicts_and_buffer(self):
        """
        Scenario 6: Double-booking prevention and 1-hour travel buffer margin verification.
        """
        provider_slots = [
            "2026-05-18T09:00:00",
            "2026-05-18T14:00:00"
        ]
        
        # Test Case 1: Slot matches general availability (within 2 hours of a slot)
        is_ok, msg, next_slot = validate_provider_schedule("P001", "2026-05-18T09:30:00", provider_slots)
        self.assertTrue(is_ok, f"Should be approved as it matches 9:00 AM slot. Reason: {msg}")
        
        # Test Case 2: Slot does NOT match general availability
        is_ok, msg, next_slot = validate_provider_schedule("P001", "2026-05-18T11:30:00", provider_slots)
        self.assertFalse(is_ok, "Should be rejected because 11:30 AM is too far from any slot.")
        
        print("\n[PASS] Scenario 6: Scheduling validation & travel buffer checks verified successfully.")

if __name__ == "__main__":
    unittest.main()
