"""
Backend service for ECG LLM Demo
Receives webhook requests from Streamlit, processes LLM calls via n8n, and stores results for polling.
"""

import os
import json
import uuid
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit

# In-memory storage for job results
# In production, use Redis or a database
jobs: Dict[str, Dict[str, Any]] = {}

# Configuration
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/336107ed-9ef6-48f3-afdc-92c57edf0ca9")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://n8n:5678")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://backend:5000")  # URL that n8n can reach
N8N_API_KEY = os.getenv("N8N_API_KEY", "")
N8N_USER = os.getenv("N8N_USER", "admin")
N8N_PASSWORD = os.getenv("N8N_PASSWORD", "admin")
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))  # 5 minutes default


def get_n8n_auth_config():
    """Get n8n authentication configuration."""
    if N8N_API_KEY:
        return None, {"X-N8N-API-KEY": N8N_API_KEY}
    else:
        return (N8N_USER, N8N_PASSWORD), {}


# Note: Streamlit calls n8n webhook directly, n8n sends result back to backend


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "ecg-llm-backend"})


@app.route("/webhook/result", methods=["POST"])
def result_webhook():
    """
    Webhook endpoint that n8n calls to send the LLM result back.
    
    Expected payload from n8n:
    {
        "job_id": "...",  # Job ID that was passed from Streamlit
        "success": true/false,
        "data": {...},  # LLM response data
        "error": "...",  # Error message if success is false
        "error_type": "...",  # Optional error type
    }
    
    Returns:
    {
        "status": "received"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        job_id = data.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id is required"}), 400
        
        # Check if job exists
        if job_id not in jobs:
            logger.warning(f"Received result for unknown job_id: {job_id}")
            # Create job if it doesn't exist (in case n8n sends result before Streamlit creates job)
            jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
        
        # Update job with result
        success = data.get("success", True)
        jobs[job_id]["status"] = "completed" if success else "error"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Handle data field - can be string or object
        data_field = data.get("data")
        logger.info(f"Received data field type: {type(data_field)}, value preview: {str(data_field)[:100] if data_field else 'None'}")
        
        if isinstance(data_field, str):
            # If data is a string, wrap it in an object for consistency
            data_field = {"text": data_field}
            logger.info(f"Converted string data to object with 'text' key")
        elif data_field is None:
            data_field = {}
            logger.warning(f"Data field is None, using empty dict")
        
        jobs[job_id]["result"] = {
            "success": success,
            "data": data_field,
            "error": data.get("error"),
            "error_type": data.get("error_type"),
            "raw_response": data  # Store full response for debugging
        }
        
        logger.info(f"Received result for job {job_id}: success={success}, data keys: {list(data_field.keys()) if isinstance(data_field, dict) else 'not a dict'}")
        
        return jsonify({
            "status": "received",
            "job_id": job_id
        }), 200
        
    except Exception as e:
        logger.exception("Error in result_webhook")
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """
    Get job status and result. Called by Streamlit to poll for results.
    
    Returns:
    {
        "job_id": "...",
        "status": "pending|completed|error",
        "result": {...}  # Only present when status is completed or error
    }
    """
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job.get("created_at", datetime.now().isoformat())
    }
    
    if "completed_at" in job:
        response["completed_at"] = job["completed_at"]
    
    # Include result if job is completed or errored
    if job["status"] in ["completed", "error"]:
        response["result"] = job.get("result", {})
    
    return jsonify(response)


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all jobs (for debugging)."""
    return jsonify({
        "jobs": list(jobs.keys()),
        "count": len(jobs)
    })


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    """Delete a job (cleanup)."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    del jobs[job_id]
    return jsonify({"message": "Job deleted"}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"Starting backend server on port {port}")
    logger.info(f"Backend will receive results from n8n at: {BACKEND_BASE_URL}/webhook/result")
    app.run(host="0.0.0.0", port=port, debug=False)

