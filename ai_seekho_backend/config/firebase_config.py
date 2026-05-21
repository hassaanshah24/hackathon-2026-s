import os
import logging
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
from ai_seekho_backend.config.settings import settings

logger = logging.getLogger("ai_seekho.firebase")

db = None

def init_firebase():
    global db
    if not firebase_admin._apps:
        try:
            # Resolve path relative to backend root
            cred_path = settings.FIREBASE_CREDENTIALS_PATH
            if not os.path.isabs(cred_path):
                backend_root = Path(__file__).resolve().parent.parent
                resolved_path = (backend_root / cred_path).resolve()
            else:
                resolved_path = Path(cred_path)

            if resolved_path.exists():
                logger.info(f"Initializing Firebase Admin with Service Account at: {resolved_path}")
                cred = credentials.Certificate(str(resolved_path))
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                logger.info("Firebase Admin and Firestore client initialized successfully.")
            else:
                try:
                    logger.info("Firebase credentials JSON not found. Attempting Application Default Credentials (ADC) fallback...")
                    firebase_admin.initialize_app()
                    db = firestore.client()
                    logger.info("Firebase Admin and Firestore client initialized successfully via ADC.")
                except Exception as adc_e:
                    logger.info(f"Firebase ADC initialization failed: {adc_e}. Running in Local/Mock mode without Firebase.")
                    db = None
                
        except Exception as e:
            logger.info(f"Firebase initialization bypassed: {e}. Running in Local/Mock mode.")
    else:
        db = firestore.client()
    return db

# Initialize immediately for module import
try:
    db = init_firebase()
except Exception:
    db = None
