# ECG LLM Clinical Notes Demo

A comprehensive demo application for teaching LLM applications in clinical settings, specifically for generating clinical notes and reports from ECG data. This project uses neurokit2 to generate synthetic ECG data and integrates with n8n for LLM-powered clinical note generation.

## Features

- **ECG Data Generation**: Generate synthetic ECG signals with various conditions (normal, tachycardia, bradycardia, atrial fibrillation, etc.)
- **Interactive Visualization**: Streamlit-based UI for visualizing ECG signals, R peaks, and RR intervals
- **n8n Integration**: Send ECG data to n8n webhooks and receive LLM-generated clinical notes
- **Clinical Indicators**: Extract and display key ECG indicators (heart rate, HRV, signal quality, etc.)

## Prerequisites

### Option 1: Docker (Recommended)
- Docker and Docker Compose installed
- No need to install Python or n8n separately

### Option 2: Local Installation
- Python 3.8 or higher
- n8n instance with webhook nodes configured (optional, for LLM integration)

## Installation

### Docker Installation (Recommended)

1. Clone or navigate to this repository:
```bash
cd ECG-LLM-Demo
```

2. (Optional) Create `.env` file from example:
```bash
cp env.example .env
# Edit .env to set your n8n credentials
```

3. Generate ECG data (before starting containers):
```bash
# Option A: Run locally if you have Python installed
python generate_ecg_data.py

# Option B: Run in a temporary container
docker run --rm -v $(pwd):/app -w /app python:3.11-slim sh -c "pip install -r requirements.txt && python generate_ecg_data.py"
```

4. Start all services with Docker Compose:
```bash
docker-compose up -d
```

This will start:
- **n8n** at `http://localhost:5678` (default credentials: admin/admin)
- **Streamlit App** at `http://localhost:8501`

5. Access the applications:
   - Streamlit: http://localhost:8501
   - n8n: http://localhost:5678

6. Stop services:
```bash
docker-compose down
```

7. View logs:
```bash
docker-compose logs -f
```

### Local Installation

1. Clone or navigate to this repository:
```bash
cd ECG-LLM-Demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Docker Usage

1. **Access n8n**: Open http://localhost:5678 and log in
2. **Create Workflows**: Set up webhook nodes as described in [N8N_SETUP.md](N8N_SETUP.md)
3. **Access Streamlit**: Open http://localhost:8501
4. **Load Data**: Click "Load ECG Data" in the sidebar
5. **Generate Notes**: Use the webhook URL from n8n (format: `http://n8n:5678/webhook/your-path`)

**Note**: When using Docker, the Streamlit app can access n8n using the service name `n8n` instead of `localhost`. The webhook URL should be: `http://n8n:5678/webhook/your-path`

### Local Usage

### Step 1: Generate ECG Data

Generate ECG samples with various conditions:

```bash
python generate_ecg_data.py
```

Or specify a webhook URL to send data directly:

```bash
python generate_ecg_data.py https://your-n8n-webhook-url-here
```

This will:
- Generate ECG samples for 5 different conditions
- Save data to `ecg_samples.json`
- Optionally send data to n8n webhook

### Step 2: Configure n8n Webhooks

1. **Data Collection Webhook**: Create a webhook node in n8n to receive ECG data
   - Update `webhook_url` in `generate_ecg_data.py` with your webhook URL
   - This webhook receives the generated ECG samples

2. **Agent Node Webhook**: Create an agent node in n8n for LLM processing
   - The agent should accept ECG data and generate clinical notes
   - Update the webhook URL in the Streamlit sidebar

### Step 3: Run Streamlit App

Launch the interactive Streamlit application:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 4: Use the Application

1. **Load Data**: Click "Load ECG Data" in the sidebar
2. **Select Sample**: Choose an ECG sample from the dropdown
3. **Visualize**: View ECG signals, R peaks, and RR intervals
4. **Generate Notes**: Configure n8n webhook URL and click "Generate Clinical Note"
5. **View Results**: See LLM-generated clinical notes in the app

## Project Structure

```
ECG-LLM-Demo/
├── generate_ecg_data.py    # ECG data generation script
├── streamlit_app.py         # Streamlit web application
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image for Streamlit app
├── docker-compose.yml       # Docker Compose configuration
├── .dockerignore           # Docker ignore file
├── env.example             # Environment variables example
├── N8N_SETUP.md           # n8n workflow setup guide
├── README.md               # This file
└── ecg_samples.json        # Generated ECG data (created after running generator)
```

## ECG Conditions

The generator creates samples for the following conditions:

- **Normal**: Regular heart rhythm at 70 bpm
- **Tachycardia**: Elevated heart rate at 120 bpm
- **Bradycardia**: Low heart rate at 50 bpm
- **Noisy Normal**: Normal rhythm with added noise (0.5)
- **Atrial Fibrillation**: Irregular rhythm at 80 bpm with variability

## Data Format

Each ECG sample includes:

```json
{
  "condition": "normal",
  "ecg_signal": [array of signal values],
  "sampling_rate": 1000,
  "duration": 10,
  "indicators": {
    "heart_rate_mean": 70.0,
    "heart_rate_std": 2.5,
    "hrv_rmssd": 45.2,
    "hrv_mean_nni": 857.1,
    "r_peaks": [100, 1100, 2100, ...],
    "r_peaks_count": 10,
    "quality_mean": 0.95,
    "rr_intervals": [1000.0, 1000.0, ...]
  }
}
```

## n8n Integration

### Webhook Payload Format

**For Data Collection:**
```json
{
  "samples": [array of ECG samples],
  "metadata": {
    "generated_by": "neurokit2",
    "purpose": "LLM testing for clinical notes and ECG reports",
    "date": "2025-11-18"
  }
}
```

**For Agent Node (Clinical Note Generation):**
```json
{
  "ecg_data": {single ECG sample},
  "prompt_type": "clinical_note",
  "timestamp": "2025-11-18T10:30:00"
}
```

### Expected Agent Response

The agent node should return a JSON response with clinical notes:

```json
{
  "clinical_note": "Patient presents with regular sinus rhythm...",
  "findings": ["Normal heart rate", "Regular rhythm"],
  "recommendations": ["Continue monitoring"]
}
```

Or a simple string response that will be displayed directly.

## Customization

### Adding New Conditions

Edit `generate_ecg_data.py` to add new conditions:

```python
conditions = [
    # ... existing conditions ...
    {"name": "ventricular_tachycardia", "heart_rate": 150, "noise": 0.2}
]
```

### Modifying Sampling Parameters

Adjust in `generate_ecg_data.py`:

```python
sampling_rate = 1000  # Hz
duration = 10  # seconds
```

### Customizing Streamlit UI

Edit `streamlit_app.py` to modify the interface, add new visualizations, or change the layout.

## Troubleshooting

### Docker Issues

**Containers won't start:**
- Check if ports 5678 and 8501 are already in use
- Verify Docker and Docker Compose are installed and running
- Check logs: `docker-compose logs`

**n8n not accessible:**
- Wait for n8n to fully start (check health status)
- Verify credentials in `.env` file
- Check n8n logs: `docker-compose logs n8n`

**Streamlit can't connect to n8n:**
- Use service name `n8n` instead of `localhost` in webhook URLs
- Verify both containers are on the same network: `docker network ls`
- Check network connectivity: `docker-compose exec streamlit-app ping n8n`

**ECG data not loading:**
- Ensure `ecg_samples.json` exists in the project root
- Check file permissions: `ls -la ecg_samples.json`
- Regenerate data if needed

### General Issues

**Data Not Loading:**
- Ensure `ecg_samples.json` exists (run `generate_ecg_data.py` first)
- Check file permissions
- In Docker: Verify volume mount is correct

**Webhook Errors:**
- Verify webhook URLs are correct
- Check n8n workflow is active
- Ensure network connectivity
- In Docker: Use `http://n8n:5678` instead of `http://localhost:5678`

**Missing Dependencies:**
- Run `pip install -r requirements.txt`
- Check Python version (3.8+)
- In Docker: Rebuild image: `docker-compose build --no-cache`

## Educational Use

This project is designed for teaching:
- LLM applications in healthcare
- ECG signal processing
- Clinical note generation
- API/webhook integrations
- Data visualization in medical contexts

## License

This project is for educational purposes.

## Support

For issues or questions, please check:
- neurokit2 documentation: https://neurokit2.readthedocs.io/
- Streamlit documentation: https://docs.streamlit.io/
- n8n documentation: https://docs.n8n.io/

