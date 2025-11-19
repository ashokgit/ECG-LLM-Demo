"""
Streamlit App for ECG Data Visualization and LLM Clinical Notes Generation
Displays ECG data and integrates with n8n agent nodes for clinical report generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import requests
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import neurokit2 as nk
import sys
import time
import base64
import logging
import uuid
sys.path.append(os.path.dirname(__file__))
from generate_ecg_data import generate_all_samples

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ECG LLM Clinical Notes Demo",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ecg_data' not in st.session_state:
    st.session_state.ecg_data = None
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None
if 'llm_results' not in st.session_state:
    st.session_state.llm_results = {}


def generate_ecg_data(selected_conditions: List[Dict[str, Any]], sampling_rate: int = 1000, duration: int = 10) -> Dict[str, Any]:
    """Generate ECG data on the fly."""
    samples = generate_all_samples(selected_conditions, sampling_rate, duration)
    
    return {
        "samples": samples,
        "metadata": {
            "generated_by": "neurokit2",
            "purpose": "LLM testing for clinical notes and ECG reports",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sampling_rate": sampling_rate,
            "duration": duration
        }
    }


def load_ecg_data(file_path: str = "ecg_samples.json") -> Dict[str, Any]:
    """Load ECG data from JSON file."""
    try:
        # Check if path exists and is a file (not a directory)
        if not os.path.isfile(file_path):
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, IsADirectoryError):
        return None


def plot_ecg_signal(signal: List[float], sampling_rate: int, title: str = "ECG Signal", duration: int = 10):
    """Plot ECG signal using Plotly."""
    time_axis = np.linspace(0, duration, len(signal))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal,
        mode='lines',
        name='ECG',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig


def get_n8n_auth() -> Optional[tuple]:
    """
    Get n8n basic auth credentials from environment variables.
    
    Returns:
        Tuple of (username, password) or None if not configured
    """
    username = os.getenv("N8N_USER", "admin")
    password = os.getenv("N8N_PASSWORD", "admin")
    return (username, password)


def get_n8n_headers() -> Dict[str, str]:
    """
    Get n8n API authentication headers.
    Prioritizes API key if available, otherwise returns empty dict for basic auth.
    
    Returns:
        Dict with headers (X-N8N-API-KEY if available, otherwise empty)
    """
    api_key = os.getenv("N8N_API_KEY")
    if api_key:
        return {"X-N8N-API-KEY": api_key}
    return {}


def get_n8n_auth_config():
    """
    Get n8n authentication configuration.
    Returns tuple of (auth, headers) where:
    - auth: tuple for basic auth or None if using API key
    - headers: dict with API key header or empty dict
    """
    api_key = os.getenv("N8N_API_KEY", "").strip()
    if api_key:
        # Use API key authentication
        return None, {"X-N8N-API-KEY": api_key}
    else:
        # Use basic auth (fallback)
        auth = get_n8n_auth()
        return auth, {}


def trigger_n8n_workflow(webhook_url: str, sample_data: Dict[str, Any], prompt_type: str = "clinical_note", timeout: int = 300) -> Dict[str, Any]:
    """
    Trigger n8n workflow via webhook and return execution ID if available.
    
    Args:
        webhook_url: n8n webhook URL
        sample_data: ECG sample data
        prompt_type: Type of prompt
        timeout: Request timeout in seconds (default: 300 for LLM processing)
    
    Returns:
        Dict with execution_id, response data, and success status
    """
    payload = {
        "ecg_data": sample_data,
        "prompt_type": prompt_type,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        # Try to parse JSON response
        response_data = {}
        if response.content:
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                # If not JSON, treat as text
                response_data = {"message": response.text}
        
        # Try to extract execution ID from response
        execution_id = None
        if isinstance(response_data, dict):
            execution_id = (response_data.get('executionId') or 
                          response_data.get('execution_id') or 
                          response_data.get('id') or
                          response_data.get('executionId'))
        
        # Also check response headers for execution ID (some n8n versions include it)
        if not execution_id:
            execution_id = response.headers.get('X-Execution-Id') or response.headers.get('x-execution-id')
        
        return {
            "success": True,
            "execution_id": execution_id,
            "data": response_data,
            "status_code": response.status_code,
            "response_headers": dict(response.headers)
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "execution_id": None
        }


def get_n8n_execution_status(execution_id: str, n8n_base_url: str) -> Dict[str, Any]:
    """
    Get execution status from n8n API.
    
    Args:
        execution_id: n8n execution ID
        n8n_base_url: Base URL for n8n (e.g., http://n8n:5678)
    
    Returns:
        Execution status and data
    """
    auth, headers = get_n8n_auth_config()
    # Include includeData=true to get the full execution data with resultData
    api_url = f"{n8n_base_url}/api/v1/executions/{execution_id}?includeData=true"
    
    try:
        response = requests.get(api_url, auth=auth, headers=headers, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        
        # Log the full response structure for debugging
        logging.info(f"[n8n API] Execution {execution_id} response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
        logging.info(f"[n8n API] Full response structure: {json.dumps(response_data, indent=2, default=str)[:2000]}")
        
        return {
            "success": True,
            "data": response_data
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"[n8n API] Error getting execution status: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def find_latest_execution(n8n_base_url: str, workflow_id: Optional[str] = None, limit: int = 5, 
                          retries: int = 3, retry_delay: int = 1) -> Dict[str, Any]:
    """
    Find the latest execution(s) from n8n API.
    
    Args:
        n8n_base_url: Base URL for n8n
        workflow_id: Optional workflow ID to filter by
        limit: Number of executions to retrieve
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        Latest execution data with detailed error info
    """
    auth, headers = get_n8n_auth_config()
    api_url = f"{n8n_base_url}/api/v1/executions"
    params = {"limit": limit}
    if workflow_id:
        params["workflowId"] = workflow_id
    
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(api_url, auth=auth, headers=headers, params=params, timeout=30)
            
            # Check for authentication errors
            if response.status_code == 401:
                auth_method = "N8N_API_KEY" if os.getenv("N8N_API_KEY") else "N8N_USER and N8N_PASSWORD"
                return {
                    "success": False,
                    "error": f"Authentication failed (401). Check {auth_method}. Response: {response.text[:200]}",
                    "status_code": 401
                }
            elif response.status_code == 403:
                return {
                    "success": False,
                    "error": f"Access forbidden (403). Check API permissions. Response: {response.text[:200]}",
                    "status_code": 403
                }
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("data") and len(data["data"]) > 0:
                # Find the most recent execution that's not finished yet, or the most recent one
                executions = data["data"]
                # Try to find a running execution first
                for exec in executions:
                    if not exec.get("finished", True):
                        return {
                            "success": True,
                            "execution": exec
                        }
                # Otherwise return the most recent
                return {
                    "success": True,
                    "execution": executions[0]
                }
            
            # If no executions found and this is not the last attempt, wait and retry
            if attempt < retries - 1:
                time.sleep(retry_delay)
                continue
            
            return {
                "success": False,
                "error": f"No executions found in n8n. This might mean: 1) The workflow hasn't started yet, 2) API access is restricted, or 3) No executions exist.",
                "status_code": response.status_code,
                "response_data": data
            }
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < retries - 1:
                time.sleep(retry_delay)
                continue
    
    return {
        "success": False,
        "error": f"Failed to fetch executions after {retries} attempts: {last_error}",
        "status_code": None
    }


def poll_n8n_execution(execution_id: Optional[str], n8n_base_url: str, max_wait: int = 300, 
                       poll_interval: int = 2, progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """
    Poll n8n execution until it completes or times out.
    
    Args:
        execution_id: Execution ID to poll (if None, will try to find latest)
        n8n_base_url: Base URL for n8n
        max_wait: Maximum time to wait in seconds
        poll_interval: Time between polls in seconds
        progress_callback: Optional callback function(status_text, progress) for progress updates
    
    Returns:
        Final execution result
    """
    start_time = time.time()
    
    # If no execution ID provided, try to find the latest execution
    if not execution_id:
        if progress_callback:
            progress_callback("Finding latest execution... (waiting for workflow to start)", 0.1)
        
        # Wait a moment for n8n to create the execution
        time.sleep(1)
        
        latest = find_latest_execution(n8n_base_url, retries=5, retry_delay=1)
        if latest.get("success"):
            execution_id = latest["execution"].get("id")
            if progress_callback:
                progress_callback(f"Found execution: {execution_id[:8] if execution_id else 'N/A'}...", 0.2)
        else:
            error_detail = latest.get("error", "Unknown error")
            status_code = latest.get("status_code")
            response_data = latest.get("response_data")
            
            detailed_error = f"Could not find execution ID. {error_detail}"
            if status_code:
                detailed_error += f" (HTTP {status_code})"
            
            return {
                "success": False,
                "error": detailed_error,
                "error_details": {
                    "status_code": status_code,
                    "response_data": response_data,
                    "suggestion": "Make sure: 1) N8N_API_KEY (or N8N_USER and N8N_PASSWORD) are correct, 2) The workflow is active, 3) The workflow has executed at least once"
                }
            }
    
    poll_count = 0
    while time.time() - start_time < max_wait:
        elapsed = time.time() - start_time
        progress = min(0.2 + (elapsed / max_wait) * 0.7, 0.9)  # Progress from 20% to 90%
        
        if progress_callback:
            status_msg = f"Step 2/3: Polling execution status... ({int(elapsed)}s elapsed)"
            progress_callback(status_msg, progress)
        
        status = get_n8n_execution_status(execution_id, n8n_base_url)
        
        if not status.get("success"):
            logging.error(f"[Polling] Failed to get execution status: {status.get('error')}")
            return status
        
        exec_data = status.get("data", {})
        
        # Log the execution data structure
        logging.info(f"[Polling] Execution data type: {type(exec_data)}")
        logging.info(f"[Polling] Execution data keys: {list(exec_data.keys()) if isinstance(exec_data, dict) else 'Not a dict'}")
        
        # Check multiple possible completion indicators
        finished = exec_data.get("finished", False)
        stopped_at = exec_data.get("stoppedAt")
        status_field = exec_data.get("status")
        mode = exec_data.get("mode")
        
        logging.info(f"[Polling] finished={finished}, stoppedAt={stopped_at}, status={status_field}, mode={mode}")
        
        # Check if execution is complete - multiple ways to detect this
        is_complete = False
        if finished:
            is_complete = True
            logging.info(f"[Polling] Execution marked as finished=True")
        elif stopped_at is not None:
            is_complete = True
            logging.info(f"[Polling] Execution has stoppedAt timestamp: {stopped_at}")
        elif status_field in ["success", "error", "crashed"]:
            is_complete = True
            logging.info(f"[Polling] Execution status indicates completion: {status_field}")
        elif mode == "manual" and stopped_at:
            is_complete = True
            logging.info(f"[Polling] Manual execution completed")
        
        if is_complete:
            # Execution completed
            logging.info(f"[Polling] Execution {execution_id} is complete!")
            if progress_callback:
                progress_callback("Step 3/3: Extracting result...", 0.95)
            return {
                "success": True,
                "execution_id": execution_id,
                "data": exec_data,
                "finished": True
            }
        
        logging.info(f"[Polling] Execution {execution_id} still running (poll #{poll_count + 1})...")
        
        poll_count += 1
        # Wait before next poll
        time.sleep(poll_interval)
    
    # Timeout
    return {
        "success": False,
        "error": f"Execution did not complete within {max_wait} seconds",
        "execution_id": execution_id,
        "timeout": True
    }


def extract_result_from_execution(execution_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract the LLM result from n8n execution data.
    
    Args:
        execution_data: Execution data from n8n API
    
    Returns:
        Extracted result data or None
    """
    # Log the full execution data structure for debugging
    logging.info(f"[Extract] Execution data type: {type(execution_data)}")
    logging.info(f"[Extract] Execution data keys: {list(execution_data.keys()) if isinstance(execution_data, dict) else 'Not a dict'}")
    logging.info(f"[Extract] Full execution data structure: {json.dumps(execution_data, indent=2, default=str)[:2000]}")
    
    # n8n execution data structure can be:
    # Option 1: execution_data.data.resultData.runData[NodeName][0].data.main[0][0].json
    # Option 2: execution_data.resultData.runData[NodeName][0].data.main[0][0].json
    # Option 3: The execution_data itself might be the metadata, and we need to fetch full data
    
    try:
        # Try Option 1: nested in "data" key
        result_data = execution_data.get("data", {}).get("resultData", {})
        if not result_data:
            # Try Option 2: direct "resultData" key
            result_data = execution_data.get("resultData", {})
        
        logging.info(f"[Extract] resultData keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'No resultData'}")
        
        if not result_data:
            logging.warning(f"[Extract] No resultData found in execution. Execution might need to be fetched with includeData parameter.")
            return None
        
        run_data = result_data.get("runData", {})
        logging.info(f"[Extract] runData keys: {list(run_data.keys()) if isinstance(run_data, dict) else 'No runData'}")
        
        if not run_data:
            return None
        
        # Find the last node (usually the LLM node or response node)
        # Try common node names
        node_names = list(run_data.keys())
        logging.info(f"[Extract] Found nodes: {node_names}")
        
        if not node_names:
            return None
        
        # Get the last node's output (usually "Message a model" or similar)
        last_node = node_names[-1]
        logging.info(f"[Extract] Using last node: {last_node}")
        node_output = run_data.get(last_node, [])
        
        if node_output and len(node_output) > 0:
            node_data = node_output[0].get("data", {})
            main_data = node_data.get("main", [])
            
            logging.info(f"[Extract] Node output structure: {json.dumps(node_output[0] if node_output else {}, indent=2, default=str)[:1000]}")
            
            if main_data and len(main_data) > 0 and len(main_data[0]) > 0:
                extracted = main_data[0][0].get("json", {})
                logging.info(f"[Extract] Extracted result keys: {list(extracted.keys()) if isinstance(extracted, dict) else 'Not a dict'}")
                return extracted
        
        return None
    except Exception as e:
        logging.error(f"[Extract] Error extracting result: {str(e)}", exc_info=True)
        return None


def call_n8n_with_backend_polling(n8n_webhook_url: str, backend_url: str, sample_data: Dict[str, Any], 
                                   prompt_type: str = "clinical_note", max_wait: int = 300,
                                   progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """
    Call n8n webhook and poll backend for results.
    
    Flow:
    1. Generate job_id
    2. Call n8n webhook with job_id in payload
    3. Poll backend API for result
    4. Return result when available
    
    Args:
        n8n_webhook_url: n8n webhook URL to trigger workflow
        backend_url: Backend API base URL (e.g., http://backend:5000)
        sample_data: ECG sample data
        prompt_type: Type of prompt
        max_wait: Maximum time to wait for completion in seconds
        progress_callback: Optional callback function(status_text, progress) for progress updates
    
    Returns:
        Result with LLM response data
    """
    # Generate job_id
    job_id = str(uuid.uuid4())
    
    # Step 1: Trigger n8n workflow with job_id
    if progress_callback:
        progress_callback("Step 1/2: Triggering n8n workflow...", 0.1)
    
    payload = {
        "job_id": job_id,
        "ecg_data": sample_data,
        "prompt_type": prompt_type,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(n8n_webhook_url, json=payload, timeout=30)
        response.raise_for_status()
        logger.info(f"Triggered n8n workflow with job_id: {job_id}")
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to trigger n8n workflow: {str(e)}",
            "job_id": job_id
        }
    
    # Step 2: Poll backend for result
    if progress_callback:
        progress_callback("Step 2/2: Waiting for LLM response...", 0.3)
    
    start_time = time.time()
    poll_interval = 2  # Poll every 2 seconds
    poll_count = 0
    
    while time.time() - start_time < max_wait:
        elapsed = time.time() - start_time
        progress = min(0.3 + (elapsed / max_wait) * 0.65, 0.95)
        
        if progress_callback:
            status_msg = f"Polling backend for result... ({int(elapsed)}s elapsed)"
            progress_callback(status_msg, progress)
        
        # Poll backend
        try:
            poll_url = f"{backend_url}/api/jobs/{job_id}"
            poll_response = requests.get(poll_url, timeout=10)
            poll_response.raise_for_status()
            job_status = poll_response.json()
            
            status = job_status.get("status")
            
            if status == "completed":
                if progress_callback:
                    progress_callback("Complete!", 1.0)
                result = job_status.get("result", {})
                return {
                    "success": True,
                    "data": result.get("data", {}),
                    "job_id": job_id,
                    "raw_result": result
                }
            elif status == "error":
                if progress_callback:
                    progress_callback("Error occurred", 1.0)
                result = job_status.get("result", {})
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "error_type": result.get("error_type"),
                    "job_id": job_id,
                    "raw_result": result
                }
            # If status is "pending", continue polling
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error polling backend (attempt {poll_count + 1}): {str(e)}")
            # Continue polling even if one request fails
        
        poll_count += 1
        time.sleep(poll_interval)
    
    # Timeout
    return {
        "success": False,
        "error": f"Job did not complete within {max_wait} seconds",
        "job_id": job_id,
        "timeout": True
    }


def call_n8n_agent_async(webhook_url: str, sample_data: Dict[str, Any], prompt_type: str = "clinical_note", 
                          n8n_base_url: Optional[str] = None, max_wait: int = 300,
                          progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """
    Call n8n agent asynchronously using polling pattern.
    
    This function:
    1. Triggers the workflow via webhook
    2. Gets execution ID from response or finds latest execution
    3. Polls execution status until complete
    4. Extracts and returns the result
    
    Args:
        webhook_url: n8n webhook URL
        sample_data: ECG sample data
        prompt_type: Type of prompt
        n8n_base_url: Base URL for n8n API (defaults to webhook URL base)
        max_wait: Maximum time to wait for completion in seconds
        progress_callback: Optional callback function(status_text, progress) for progress updates
    
    Returns:
        Result with LLM response data
    """
    # Extract base URL from webhook URL if not provided
    if not n8n_base_url:
        from urllib.parse import urlparse
        parsed = urlparse(webhook_url)
        n8n_base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    # Step 1: Trigger workflow
    if progress_callback:
        progress_callback("Step 1/3: Triggering workflow...", 0.1)
    trigger_result = trigger_n8n_workflow(webhook_url, sample_data, prompt_type)
    
    if not trigger_result.get("success"):
        return trigger_result
    
    execution_id = trigger_result.get("execution_id")
    
    # Step 2: Poll for completion
    poll_result = poll_n8n_execution(execution_id, n8n_base_url, max_wait=max_wait, 
                                     progress_callback=progress_callback)
    
    if not poll_result.get("success"):
        return poll_result
    
    # Step 3: Extract result
    if progress_callback:
        progress_callback("Step 3/3: Extracting LLM response...", 0.95)
    execution_data = poll_result.get("data", {})
    result_data = extract_result_from_execution(execution_data)
    
    if result_data:
        if progress_callback:
            progress_callback("Complete!", 1.0)
        return {
            "success": True,
            "data": result_data,
            "execution_id": poll_result.get("execution_id")
        }
    else:
        # Fallback: return the raw execution data
        if progress_callback:
            progress_callback("Complete! (using raw data)", 1.0)
        return {
            "success": True,
            "data": execution_data,
            "execution_id": poll_result.get("execution_id"),
            "note": "Could not extract result, returning raw execution data"
        }


def call_n8n_agent(webhook_url: str, sample_data: Dict[str, Any], prompt_type: str = "clinical_note", timeout: int = 300) -> Dict[str, Any]:
    """
    Call n8n agent node to generate clinical notes or reports.
    
    Note: This function waits for the LLM to complete. Make sure your n8n webhook
    is configured to "Respond When Last Node Finishes" (not "Respond Immediately").
    The timeout is set to 300 seconds (5 minutes) to allow for longer LLM responses.
    
    Args:
        webhook_url: n8n webhook URL for agent node
        sample_data: ECG sample data
        prompt_type: Type of prompt (clinical_note, report, analysis)
        timeout: Request timeout in seconds (default: 300 for LLM responses)
    
    Returns:
        Response from n8n agent
    """
    payload = {
        "ecg_data": sample_data,
        "prompt_type": prompt_type,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Try POST first (most common for webhooks)
        # Increased timeout to 300 seconds (5 minutes) for LLM processing
        response = requests.post(webhook_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json() if response.content else {"message": "Success"},
            "status_code": response.status_code
        }
    except requests.exceptions.HTTPError as e:
        # If POST fails with 404 (not registered for POST), try GET
        should_try_get = False
        error_msg = ""
        
        if e.response:
            status_code = e.response.status_code
            # Try to extract error message from response
            try:
                error_data = e.response.json()
                error_msg = error_data.get('message', '')
            except:
                error_msg = e.response.text or ''
            
            # Check if it's a method not allowed error
            if status_code == 404 and ('not registered for POST' in error_msg or 'Did you mean to make a GET' in error_msg):
                should_try_get = True
            elif status_code == 405:  # Method Not Allowed
                should_try_get = True
        
        if should_try_get:
            try:
                # For GET, we need to flatten the payload or send as query params
                # n8n webhooks can receive data via query params
                # Send key fields as query params
                get_params = {
                    'condition': sample_data.get('condition', ''),
                    'heart_rate': str(sample_data.get('indicators', {}).get('heart_rate_mean', '')),
                    'prompt_type': prompt_type,
                    'timestamp': datetime.now().isoformat()
                }
                response = requests.get(webhook_url, params=get_params, timeout=timeout)
                response.raise_for_status()
                return {
                    "success": True,
                    "data": response.json() if response.content else {"message": "Success"},
                    "status_code": response.status_code,
                    "note": "Used GET method (webhook configured for GET, not POST)"
                }
            except requests.exceptions.RequestException as e2:
                return {
                    "success": False,
                    "error": f"POST failed (404), GET also failed: {str(e2)}",
                    "status_code": getattr(e2.response, 'status_code', None) if hasattr(e2, 'response') else None
                }
        
        # Return the original error if we shouldn't try GET or GET also failed
        return {
            "success": False,
            "error": str(e),
            "status_code": e.response.status_code if e.response else None
        }
    except requests.exceptions.Timeout as e:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds. The LLM may need more time to generate the response. Please try again or check your n8n workflow configuration.",
            "status_code": None,
            "timeout": True
        }
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        # Check if it's a timeout error
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return {
                "success": False,
                "error": f"Request timed out. The LLM may need more time. Please ensure your n8n webhook is configured to 'Respond When Last Node Finishes'.",
                "status_code": None,
                "timeout": True
            }
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }


def extract_llm_text(response: Any) -> str:
    """
    Extract text content from various LLM response formats.
    
    Handles:
    - Gemini API format: [{"content": {"parts": [{"text": "..."}]}}]
    - Direct text strings
    - Dict with keys like 'clinical_note', 'report', 'text', 'message'
    - Nested structures
    
    Args:
        response: The response from the LLM/n8n workflow
        
    Returns:
        Extracted text content, or empty string if not found
    """
    if isinstance(response, str):
        return response
    
    if isinstance(response, list) and len(response) > 0:
        # Handle Gemini API format: [{"content": {"parts": [{"text": "..."}]}}]
        first_item = response[0]
        if isinstance(first_item, dict):
            # Try Gemini format: content.parts[0].text
            if 'content' in first_item:
                content = first_item['content']
                if isinstance(content, dict) and 'parts' in content:
                    parts = content['parts']
                    if isinstance(parts, list) and len(parts) > 0:
                        first_part = parts[0]
                        if isinstance(first_part, dict) and 'text' in first_part:
                            return first_part['text']
            
            # Try direct 'text' key
            if 'text' in first_item:
                return first_item['text']
            
            # Try 'message' key
            if 'message' in first_item:
                return first_item['message']
    
    if isinstance(response, dict):
        # Try common keys
        for key in ['clinical_note', 'report', 'text', 'message', 'content']:
            if key in response:
                value = response[key]
                if isinstance(value, str):
                    return value
                # Recursively try to extract from nested structure
                if isinstance(value, (dict, list)):
                    nested_text = extract_llm_text(value)
                    if nested_text:
                        return nested_text
        
        # Try to find any string value in the dict
        for value in response.values():
            if isinstance(value, str) and len(value) > 10:  # Likely the text content
                return value
            elif isinstance(value, (dict, list)):
                nested_text = extract_llm_text(value)
                if nested_text:
                    return nested_text
    
    return ""


def clean_markdown_text(text: str) -> str:
    """
    Clean and format markdown text for better rendering.
    
    Args:
        text: Raw markdown text
        
    Returns:
        Cleaned markdown text
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    cleaned = text.strip()
    
    # Fix common markdown issues:
    # - Remove triple backticks if they wrap the entire content (Streamlit handles code blocks differently)
    # - Ensure proper line breaks
    # - Fix any escaped characters
    
    # Remove markdown code fence if it wraps the entire content
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.split("\n")
        if len(lines) > 2 and lines[0].startswith("```"):
            # Remove first and last line (code fence markers)
            cleaned = "\n".join(lines[1:-1])
    
    # Ensure proper spacing around headers
    # Add blank line before headers if missing
    cleaned = re.sub(r'\n(#{1,6}\s)', r'\n\n\1', cleaned)
    
    # Fix multiple consecutive blank lines (max 2)
    cleaned = re.sub(r'\n{3,}', r'\n\n', cleaned)
    
    return cleaned.strip()


def get_prompt_type_display_name(prompt_type: str) -> str:
    """Get a human-readable display name for a prompt type."""
    display_names = {
        "clinical_note": "Clinical Note",
        "detailed_report": "Detailed Report",
        "analysis": "Analysis",
        "summary": "Summary"
    }
    return display_names.get(prompt_type, prompt_type.replace("_", " ").title())


def display_sample_info(sample: Dict[str, Any]):
    """Display detailed information about an ECG sample."""
    st.subheader(f"Condition: {sample['condition'].replace('_', ' ').title()}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Heart Rate (mean)", f"{sample['indicators']['heart_rate_mean']:.1f} bpm")
    with col2:
        st.metric("HRV RMSSD", f"{sample['indicators']['hrv_rmssd']:.2f} ms")
    with col3:
        st.metric("R Peaks Count", sample['indicators']['r_peaks_count'])
    with col4:
        st.metric("Signal Quality", f"{sample['indicators']['quality_mean']:.2f}")
    
    # Additional metrics
    with st.expander("Detailed Indicators"):
        indicators_df = pd.DataFrame([sample['indicators']])
        st.dataframe(indicators_df, use_container_width=True)


def main():
    """Main Streamlit app."""
    st.title("❤️ ECG LLM Clinical Notes Demo")
    st.markdown("**Interactive ECG Data Visualization and Clinical Report Generation**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # ECG Condition Selection
        st.subheader("📋 ECG Conditions")
        
        # Define available conditions
        available_conditions = {
            "Normal": {"name": "normal", "heart_rate": 70, "noise": 0.0, "description": "Normal sinus rhythm"},
            "Tachycardia": {"name": "tachycardia", "heart_rate": 120, "noise": 0.0, "description": "Fast heart rate (>100 bpm)"},
            "Bradycardia": {"name": "bradycardia", "heart_rate": 50, "noise": 0.0, "description": "Slow heart rate (<60 bpm)"},
            "Noisy Normal": {"name": "noisy_normal", "heart_rate": 70, "noise": 0.5, "description": "Normal rhythm with noise"},
            "Atrial Fibrillation": {"name": "atrial_fibrillation", "heart_rate": 80, "noise": 0.1, "description": "Irregular heart rhythm"}
        }
        
        # Initialize selected conditions in session state
        if 'selected_condition_keys' not in st.session_state:
            st.session_state.selected_condition_keys = list(available_conditions.keys())
        
        # Simple checkbox-based selection
        st.markdown("**Select Conditions to Generate**")
        
        # Use checkboxes in a clean layout
        selected_condition_keys = []
        for condition_key in available_conditions.keys():
            checkbox_key = f"condition_{condition_key}"
            # Initialize default value from session state if first time
            default_value = condition_key in st.session_state.selected_condition_keys
            
            # Display checkbox - Streamlit automatically manages state via key
            if st.checkbox(
                condition_key,
                value=default_value,
                key=checkbox_key,
                help=available_conditions[condition_key]["description"]
            ):
                selected_condition_keys.append(condition_key)
        
        # Update session state
        st.session_state.selected_condition_keys = selected_condition_keys
        
        # Show selection summary
        if selected_condition_keys:
            st.info(f"✓ {len(selected_condition_keys)} condition(s) selected")
        else:
            st.warning("⚠️ Please select at least one condition to generate.")
        
        # Generate or Load data
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Generate ECG Data", use_container_width=True, type="primary", disabled=len(selected_condition_keys) == 0):
                with st.spinner("Generating ECG samples... This may take a moment."):
                    try:
                        # Build conditions list from selected keys
                        selected_conditions = [available_conditions[key] for key in selected_condition_keys]
                        data = generate_ecg_data(selected_conditions)
                        st.session_state.ecg_data = data
                        st.success(f"✓ Generated {len(data['samples'])} samples")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating data: {str(e)}")
        
        with col2:
            if st.button("📂 Load from File", use_container_width=True):
                data = load_ecg_data()
                if data:
                    st.session_state.ecg_data = data
                    st.success(f"✓ Loaded {len(data['samples'])} samples")
                    st.rerun()
                else:
                    st.warning("No data file found. Use Generate instead.")
        
        # Webhook configuration
        st.subheader("🔗 n8n Integration")
        # Get default webhook URL from environment variable or use default
        default_webhook_url = os.getenv("N8N_AGENT_WEBHOOK_URL", "")
        # Check if we're in Docker (service name available)
        in_docker = os.path.exists("/.dockerenv") or os.getenv("N8N_WEBHOOK_URL", "").startswith("http://n8n")
        
        if default_webhook_url:
            # Replace localhost with n8n if in Docker
            if in_docker and "localhost" in default_webhook_url:
                default_webhook_url = default_webhook_url.replace("localhost", "n8n")
            help_text = f"Current: {default_webhook_url}\n\nEnter n8n webhook URL (use 'n8n' instead of 'localhost' when in Docker)"
        else:
            help_text = "Enter your n8n webhook URL for agent nodes (or set N8N_AGENT_WEBHOOK_URL env var)"
            default_webhook_url = "http://n8n:5678/webhook/your-path" if in_docker else "http://localhost:5678/webhook/your-path"
        
        n8n_webhook_url = st.text_input(
            "n8n Agent Webhook URL",
            value=default_webhook_url,
            help=help_text
        )
        
        # Important note about webhook configuration
        with st.expander("⚠️ Important: Architecture & n8n Configuration"):
            st.markdown("""
            **The app uses a backend polling pattern:**
            
            **Flow:**
            1. Streamlit sends request to n8n webhook (with `job_id`)
            2. n8n processes LLM request
            3. n8n sends result to backend webhook at `http://backend:5000/webhook/result`
            4. Streamlit polls backend API for result
            
            **n8n Workflow Configuration:**
            - **Webhook Node**: Receives request from Streamlit (includes `job_id`, `ecg_data`, `prompt_type`)
            - **LLM Node**: Processes the request
            - **HTTP Request Node**: Sends result back to backend
              - **IMPORTANT**: Use `http://backend:5000/webhook/result` (port 5000, NOT 5001)
              - Port 5001 is only for host machine access, not for Docker-to-Docker communication
              - Method: POST
              - Enable "Send Body" toggle
              - Body should include:
                ```json
                {
                  "job_id": "{{ $json.job_id }}",
                  "success": true,
                  "data": { ... LLM response ... }
                }
                ```
            
            **Environment Variables:**
            - `BACKEND_URL`: Backend API URL (default: http://backend:5000 inside Docker, or http://localhost:5001 on host)
            - `N8N_AGENT_WEBHOOK_URL`: n8n webhook URL to trigger workflow
            
            **Benefits:**
            - Predictable payload format from backend
            - Exact LLM errors can be shown in UI
            - Easy to parse markdown responses
            - No need to poll n8n API directly
            """)
            
            # Test backend connection button
            if st.button("🔍 Test Backend Connection", use_container_width=True):
                in_docker = os.path.exists("/.dockerenv") or os.getenv("N8N_WEBHOOK_URL", "").startswith("http://n8n")
                
                # Use appropriate URL based on environment
                if in_docker:
                    backend_url = os.getenv("BACKEND_URL", "http://backend:5000")
                else:
                    # On host, backend is mapped to port 5001
                    backend_url = os.getenv("BACKEND_URL", "http://localhost:5001")
                
                with st.spinner("Testing backend connection..."):
                    try:
                        health_url = f"{backend_url}/health"
                        response = requests.get(health_url, timeout=5)
                        response.raise_for_status()
                        health_data = response.json()
                        st.success(f"✅ Backend connection successful! Status: {health_data.get('status', 'unknown')}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Backend connection failed: {str(e)}")
                        st.info(f"💡 Check that backend service is running at {backend_url}")
                        st.info("💡 In Docker, use 'backend' instead of 'localhost' for BACKEND_URL")
        
        prompt_type = st.selectbox(
            "Prompt Type",
            ["clinical_note", "detailed_report", "analysis", "summary"],
            help="Select the type of clinical note/report to generate"
        )
    
    # Main content
    if st.session_state.ecg_data is None:
        st.info("👈 Click 'Generate ECG Data' in the sidebar to get started.")
        st.markdown("""
        ### Getting Started:
        1. Click **"Generate ECG Data"** in the sidebar to create ECG samples
        2. Select a sample from the dropdown to visualize
        3. Configure n8n webhook URL and generate clinical notes
        4. View LLM-generated clinical reports
        
        **Note:** Data is generated on-the-fly using neurokit2. No pre-generated files needed!
        """)
    else:
        data = st.session_state.ecg_data
        samples = data['samples']
        
        # Sample selection
        st.header("📊 ECG Samples Overview")
        sample_names = [f"{s['condition'].replace('_', ' ').title()} (HR: {s['indicators']['heart_rate_mean']:.1f} bpm)" 
                       for s in samples]
        
        selected_idx = st.selectbox(
            "Select ECG Sample",
            range(len(samples)),
            format_func=lambda x: sample_names[x]
        )
        
        selected_sample = samples[selected_idx]
        st.session_state.selected_sample = selected_sample
        
        # Display sample information
        display_sample_info(selected_sample)
        
        # ECG Signal Visualization
        st.header("📈 ECG Signal Visualization")
        ecg_fig = plot_ecg_signal(
            selected_sample['ecg_signal'],
            selected_sample['sampling_rate'],
            title=f"ECG Signal - {selected_sample['condition'].replace('_', ' ').title()}",
            duration=selected_sample['duration']
        )
        st.plotly_chart(ecg_fig, use_container_width=True)
        
        # R Peaks visualization
        if len(selected_sample['indicators']['r_peaks']) > 0:
            st.subheader("📍 R Peaks Detection")
            r_peaks = selected_sample['indicators']['r_peaks']
            time_axis = np.linspace(0, selected_sample['duration'], len(selected_sample['ecg_signal']))
            
            r_peaks_fig = go.Figure()
            r_peaks_fig.add_trace(go.Scatter(
                x=time_axis,
                y=selected_sample['ecg_signal'],
                mode='lines',
                name='ECG',
                line=dict(color='#1f77b4', width=1)
            ))
            r_peaks_fig.add_trace(go.Scatter(
                x=time_axis[r_peaks],
                y=np.array(selected_sample['ecg_signal'])[r_peaks],
                mode='markers',
                name='R Peaks',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
            r_peaks_fig.update_layout(
                title="ECG Signal with R Peaks",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude (mV)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(r_peaks_fig, use_container_width=True)
        
        # RR Intervals visualization
        if len(selected_sample['indicators']['rr_intervals']) > 0:
            st.subheader("⏱️ RR Intervals")
            rr_intervals = selected_sample['indicators']['rr_intervals']
            rr_fig = go.Figure()
            rr_fig.add_trace(go.Scatter(
                x=list(range(len(rr_intervals))),
                y=rr_intervals,
                mode='lines+markers',
                name='RR Intervals',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ))
            rr_fig.update_layout(
                title="RR Intervals Over Time",
                xaxis_title="Interval Number",
                yaxis_title="RR Interval (ms)",
                height=300,
                template="plotly_white"
            )
            st.plotly_chart(rr_fig, use_container_width=True)
        
        # LLM Clinical Notes Section
        st.header("🤖 LLM Clinical Notes Generation")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Generate clinical notes and reports using n8n agent integration")
        with col2:
            if st.button("🚀 Generate Clinical Note", use_container_width=True, type="primary"):
                # Determine which webhook URL to use
                env_webhook = os.getenv("N8N_AGENT_WEBHOOK_URL", "")
                in_docker = os.path.exists("/.dockerenv") or os.getenv("N8N_WEBHOOK_URL", "").startswith("http://n8n")
                
                # Prefer environment variable, but allow manual override
                if env_webhook and (not n8n_webhook_url or n8n_webhook_url == default_webhook_url or "your-path" in n8n_webhook_url):
                    webhook_to_use = env_webhook
                elif n8n_webhook_url and n8n_webhook_url != default_webhook_url and "your-path" not in n8n_webhook_url:
                    webhook_to_use = n8n_webhook_url
                    # Convert localhost to n8n if in Docker
                    if in_docker and "localhost" in webhook_to_use:
                        webhook_to_use = webhook_to_use.replace("localhost", "n8n")
                        st.info(f"🔧 Converted localhost to n8n: {webhook_to_use}")
                else:
                    webhook_to_use = env_webhook if env_webhook else None
                
                if webhook_to_use:
                    # Get backend URL from environment or use default
                    # Note: Inside Docker, use backend:5000. On host, use localhost:5001
                    backend_url = os.getenv("BACKEND_URL", "http://backend:5000")
                    in_docker = os.path.exists("/.dockerenv") or os.getenv("N8N_WEBHOOK_URL", "").startswith("http://n8n")
                    
                    # Convert localhost to backend if in Docker
                    if in_docker and "localhost" in backend_url:
                        backend_url = backend_url.replace("localhost", "backend")
                    
                    # Create progress containers
                    status_container = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Progress callback function
                    def update_progress(text: str, progress: float):
                        status_text.text(text)
                        progress_bar.progress(progress)
                    
                    status_container.info("🔄 Starting workflow execution...")
                    
                    try:
                        # Use new backend polling pattern
                        result = call_n8n_with_backend_polling(
                            webhook_to_use, 
                            backend_url,
                            selected_sample, 
                            prompt_type,
                            max_wait=300,
                            progress_callback=update_progress
                        )
                        
                        status_container.empty()
                        progress_bar.empty()
                        status_text.empty()
                        
                        if result['success']:
                            # Store result with prompt_type for dynamic header display
                            st.session_state.llm_results[selected_idx] = {
                                'data': result['data'],
                                'prompt_type': prompt_type
                            }
                            job_id = result.get('job_id', 'N/A')
                            display_name = get_prompt_type_display_name(prompt_type)
                            st.success(f"✓ {display_name} generated successfully! (Job ID: {job_id[:8]}...)")
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            is_timeout = result.get('timeout', False)
                            
                            if is_timeout:
                                st.error(f"⏱️ Timeout Error: {error_msg}")
                                st.warning("""
                                **Possible Solutions:**
                                1. **Check n8n Workflow**: Verify the workflow completes and sends result to backend
                                2. **Check Backend Connection**: Ensure BACKEND_URL is set correctly (http://backend:5000 in Docker, http://localhost:5001 on host)
                                3. **Check n8n Configuration**: Make sure n8n workflow sends result to backend webhook at `/webhook/result`
                                4. **Increase Timeout**: The current timeout is 5 minutes
                                5. **Try Again**: Sometimes LLM APIs can be slow - try the request again
                                """)
                            else:
                                st.error(f"✗ Error: {error_msg}")
                                
                                # Show detailed error information if available
                                raw_result = result.get('raw_result', {})
                                if raw_result:
                                    with st.expander("🔍 Error Details"):
                                        st.json(raw_result)
                                
                                # Provide helpful hints
                                if "Failed to trigger n8n workflow" in error_msg:
                                    st.warning("""
                                    **Troubleshooting Steps:**
                                    
                                    1. **Check n8n Webhook URL**: Verify the webhook URL is correct and the workflow is active
                                    2. **Check n8n Workflow**: Make sure the workflow is **Active** in n8n
                                    3. **Check Network**: Ensure Streamlit can reach n8n (use 'n8n' instead of 'localhost' in Docker)
                                    """)
                                elif "Job did not complete" in error_msg:
                                    st.warning("""
                                    **Troubleshooting Steps:**
                                    
                                    1. **Check Backend Connection**: Verify BACKEND_URL is set correctly
                                    2. **Check n8n Workflow**: Ensure n8n sends result to backend at `http://backend:5000/webhook/result`
                                    3. **Check n8n Workflow Configuration**: The workflow should include a webhook node that POSTs to the backend
                                    4. **Check Backend Logs**: Look for errors in backend service logs
                                    """)
                                elif "404" in error_msg:
                                    st.info("💡 Tip: Make sure the backend service is running and BACKEND_URL is correct")
                                elif "Connection refused" in error_msg or "localhost" in error_msg:
                                    st.info("💡 Tip: When running in Docker, use 'backend:5000'. On host, use 'localhost:5001' (port 5000 is used by macOS AirPlay)")
                    except Exception as e:
                        status_container.empty()
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"✗ Unexpected error: {str(e)}")
                        st.exception(e)
                else:
                    st.warning("⚠ Please configure n8n webhook URL in the sidebar or set N8N_AGENT_WEBHOOK_URL environment variable")
        
        # Display LLM results
        if selected_idx in st.session_state.llm_results:
            # Get stored result (may be old format or new format with prompt_type)
            stored_result = st.session_state.llm_results[selected_idx]
            
            # Handle both old format (just data) and new format (dict with data and prompt_type)
            if isinstance(stored_result, dict) and 'data' in stored_result:
                result = stored_result['data']
                stored_prompt_type = stored_result.get('prompt_type', prompt_type)
            else:
                # Old format - just the data
                result = stored_result
                stored_prompt_type = prompt_type
            
            # Get display name for the prompt type
            display_name = get_prompt_type_display_name(stored_prompt_type)
            st.subheader(f"📝 Generated {display_name}")
            
            # The result from backend should be the LLM response data
            # Extract text from various response formats (including Gemini API format)
            extracted_text = extract_llm_text(result)
            
            if extracted_text:
                # Clean and format the markdown text
                cleaned_text = clean_markdown_text(extracted_text)
                
                # Display options
                col1, col2 = st.columns([1, 4])
                with col1:
                    view_mode = st.radio(
                        "View Mode",
                        ["Formatted", "Raw Markdown"],
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                
                # Display the markdown in a styled container
                if view_mode == "Formatted":
                    # Use a container with custom styling for better presentation
                    # Add custom CSS for better markdown rendering
                    st.markdown("""
                    <style>
                    .clinical-note-container {
                        background-color: #f8f9fa;
                        padding: 1.5rem;
                        border-radius: 0.5rem;
                        border-left: 4px solid #1f77b4;
                        margin: 1rem 0;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Render markdown with proper formatting
                    # Wrap in a container div for styling
                    st.markdown('<div class="clinical-note-container">', unsafe_allow_html=True)
                    st.markdown(cleaned_text, unsafe_allow_html=False)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Show raw markdown in a code block
                    st.code(cleaned_text, language="markdown")
                
                # Additional options
                with st.expander("🔍 View Raw Response (JSON)"):
                    st.json(result)
                
                # Download button for the clinical note
                st.download_button(
                    label=f"📥 Download {display_name}",
                    data=cleaned_text,
                    file_name=f"{stored_prompt_type}_{selected_sample['condition']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            else:
                # Fallback: display as JSON if we can't extract text
                st.warning("⚠️ Could not extract text from response. Showing raw format:")
                st.json(result)
        
        # Comparison Table
        st.header("📋 All Samples Comparison")
        comparison_data = []
        for sample in samples:
            comparison_data.append({
                "Condition": sample['condition'].replace('_', ' ').title(),
                "Heart Rate (bpm)": f"{sample['indicators']['heart_rate_mean']:.1f}",
                "HRV RMSSD (ms)": f"{sample['indicators']['hrv_rmssd']:.2f}",
                "R Peaks": sample['indicators']['r_peaks_count'],
                "Quality": f"{sample['indicators']['quality_mean']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

