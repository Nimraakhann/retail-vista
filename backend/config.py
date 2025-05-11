import os

# API URLs
API_BASE_URL = os.getenv('API_BASE_URL', 'https://retailvista.duckdns.org')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://retailvista.netlify.app')

# API Endpoints
API_ENDPOINTS = {
    'check_camera_status': f"{API_BASE_URL}/api/check-camera-status/",
    'create_shoplifting_alert': f"{API_BASE_URL}/api/create-shoplifting-alert/",
    'update_detection_data': f"{API_BASE_URL}/api/update-detection-data/",
    'update_alert_evidence': f"{API_BASE_URL}/api/update-alert-evidence/",
    'update_age_gender_data': f"{API_BASE_URL}/api/update-age-gender-data/",
    'shoplifting_in_progress': f"{API_BASE_URL}/api/shoplifting-in-progress/",
}

# Frontend URLs
FRONTEND_ENDPOINTS = {
    'reset_password': f"{FRONTEND_URL}/reset-password",
    'verify_email': f"{FRONTEND_URL}/verify-email",
} 