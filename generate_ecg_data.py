"""
ECG Data Generator for LLM Clinical Notes Testing
Generates synthetic ECG data using neurokit2 and sends it to n8n webhook
"""

import neurokit2 as nk
import numpy as np
import requests
import json
import random
from datetime import datetime
from typing import List, Dict, Any
import sys


def generate_ecg_sample(condition: Dict[str, Any], sampling_rate: int = 1000, duration: int = 10) -> Dict[str, Any]:
    """
    Generate a single ECG sample for a given condition.
    
    Args:
        condition: Dictionary with condition parameters (name, heart_rate, noise)
        sampling_rate: Sampling rate in Hz
        duration: Duration in seconds
    
    Returns:
        Dictionary containing ECG signal and processed indicators
    """
    try:
        # Generate random seed for each generation to get different data
        random_seed = random.randint(0, 2**31 - 1)
        
        # Simulate ECG signal
        if condition["name"] == "atrial_fibrillation":
            # For AFib, use ecgsyn method with added irregularity
            ecg_signal = nk.ecg_simulate(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=condition["heart_rate"],
                noise=condition["noise"],
                method="ecgsyn",
                random_state=random_seed
            )
        else:
            # Use ecgsyn method for single lead, or extract first lead from multilead
            ecg_signal = nk.ecg_simulate(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=condition["heart_rate"],
                noise=condition["noise"],
                method="ecgsyn",
                random_state=random_seed
            )
            # If multilead is needed, uncomment below and use first lead
            # ecg_multilead = nk.ecg_simulate(
            #     duration=duration,
            #     sampling_rate=sampling_rate,
            #     heart_rate=condition["heart_rate"],
            #     noise=condition["noise"],
            #     method="multilead",
            #     random_state=42
            # )
            # ecg_signal = ecg_multilead[:, 0]  # Extract first lead (Lead I)
        
        # Process the ECG to extract indicators
        processed_ecg, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        
        # Calculate RR intervals
        r_peaks = info["ECG_R_Peaks"]
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000 if len(r_peaks) > 1 else np.array([0])
        
        # Extract key indicators
        indicators = {
            "heart_rate_mean": float(processed_ecg["ECG_Rate"].mean()) if "ECG_Rate" in processed_ecg.columns else 0.0,
            "heart_rate_std": float(processed_ecg["ECG_Rate"].std()) if "ECG_Rate" in processed_ecg.columns else 0.0,
            "hrv_rmssd": float(np.sqrt(np.mean(np.diff(rr_intervals)**2))) if len(rr_intervals) > 1 else 0.0,
            "hrv_mean_nni": float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0.0,
            "r_peaks": [int(peak) for peak in r_peaks],
            "r_peaks_count": len(r_peaks),
            "quality_mean": float(processed_ecg["ECG_Quality"].mean()) if "ECG_Quality" in processed_ecg.columns else 0.0,
            "rr_intervals": [float(interval) for interval in rr_intervals],
        }
        
        return {
            "condition": condition["name"],
            "ecg_signal": ecg_signal.tolist(),
            "sampling_rate": sampling_rate,
            "duration": duration,
            "indicators": indicators
        }
    
    except Exception as e:
        print(f"Error generating sample for {condition['name']}: {str(e)}")
        raise


def generate_all_samples(conditions: List[Dict[str, Any]], sampling_rate: int = 1000, duration: int = 10) -> List[Dict[str, Any]]:
    """
    Generate ECG samples for all conditions.
    
    Args:
        conditions: List of condition dictionaries
        sampling_rate: Sampling rate in Hz
        duration: Duration in seconds
    
    Returns:
        List of sample dictionaries
    """
    samples = []
    for cond in conditions:
        print(f"Generating sample for condition: {cond['name']}")
        sample = generate_ecg_sample(cond, sampling_rate, duration)
        samples.append(sample)
    return samples


def send_to_webhook(webhook_url: str, samples: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> bool:
    """
    Send ECG samples to n8n webhook.
    
    Args:
        webhook_url: n8n webhook URL
        samples: List of ECG samples
        metadata: Optional metadata dictionary
    
    Returns:
        True if successful, False otherwise
    """
    if metadata is None:
        metadata = {
            "generated_by": "neurokit2",
            "purpose": "LLM testing for clinical notes and ECG reports",
            "date": datetime.now().strftime("%Y-%m-%d")
        }
    
    payload = {
        "samples": samples,
        "metadata": metadata
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=30)
        response.raise_for_status()
        print("✓ Payload sent successfully!")
        print(f"Response status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Error sending payload: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def main():
    """Main function to generate and send ECG data."""
    # Define parameters
    sampling_rate = 1000  # Hz
    duration = 10  # seconds
    
    # List of conditions to simulate
    conditions = [
        {"name": "normal", "heart_rate": 70, "noise": 0.0},
        {"name": "tachycardia", "heart_rate": 120, "noise": 0.0},
        {"name": "bradycardia", "heart_rate": 50, "noise": 0.0},
        {"name": "noisy_normal", "heart_rate": 70, "noise": 0.5},
        {"name": "atrial_fibrillation", "heart_rate": 80, "noise": 0.1}
    ]
    
    # Webhook URL - replace with your actual n8n webhook URL
    webhook_url = "https://your-n8n-webhook-url-here"
    
    # Check if webhook URL is provided
    if webhook_url == "https://your-n8n-webhook-url-here":
        print("⚠ Warning: Please update the webhook_url variable with your actual n8n webhook URL")
        print("You can also pass it as a command line argument: python generate_ecg_data.py <webhook_url>")
        if len(sys.argv) > 1:
            webhook_url = sys.argv[1]
        else:
            print("Skipping webhook send. Data generation completed.")
            webhook_url = None
    
    # Generate samples
    print("Generating ECG samples...")
    samples = generate_all_samples(conditions, sampling_rate, duration)
    print(f"✓ Generated {len(samples)} samples")
    
    # Save to local JSON file as backup
    output_file = "ecg_samples.json"
    with open(output_file, 'w') as f:
        json.dump({
            "samples": samples,
            "metadata": {
                "generated_by": "neurokit2",
                "purpose": "LLM testing for clinical notes and ECG reports",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "sampling_rate": sampling_rate,
                "duration": duration
            }
        }, f, indent=2)
    print(f"✓ Saved samples to {output_file}")
    
    # Send to webhook if URL is provided
    if webhook_url:
        print(f"\nSending data to webhook: {webhook_url}")
        send_to_webhook(webhook_url, samples)
    else:
        print("\nTo send data to webhook, update webhook_url or pass it as argument")


if __name__ == "__main__":
    main()

