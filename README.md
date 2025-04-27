# Drone Anomaly Detection System

A comprehensive platform for analyzing drone telemetry data and identifying potential security threats using multi-dimensional anomaly detection and AI-powered analysis.

## Overview

This system helps security professionals identify suspicious drone activities by analyzing telemetry data through multiple detection algorithms and providing detailed threat assessments. It combines statistical detection methods with AI interpretation to provide actionable intelligence.

## Features

- **Multi-dimensional Anomaly Detection**: Employs six specialized tests:
  - Distance-to-pilot analysis (beyond visual line of sight detection)
  - Speed anomaly detection
  - Turn rate analysis
  - Burst detection (sudden changes in flight patterns)
  - Cluster-based anomaly detection
  - Machine learning with Isolation Forest

- **AI-Powered Analysis**: Utilizes OpenAI's o3 model to provide:
  - Detailed threat assessments
  - Threat prioritization (1-5 scale)
  - Contextual interpretation of anomalies
  - Recommended actions and contingency plans

- **Interactive Web Interface**:
  - Upload and analyze drone telemetry data
  - Visual map of drone flight paths with anomaly indicators
  - AI-powered chat for follow-up questions about the analysis
  - Downloadable annotated results

- **Comprehensive Reporting**:
  - Threat profiles for anomalous drones
  - Priority-based sorting of potential threats
  - Detailed rationale for threat assessments
  - Action recommendations for security response

## Installation

### Prerequisites

- Python 3.8+
- FastAPI
- Required Python packages (listed in requirements.txt)
- OpenAI API key

### Setup

1. Clone this repository:
2. Install dependencies: pip install -r requirements.txt
3. Create a `keys.env` file with your OpenAI API key: OPENAI_API_KEY=your_api_key_here

## Usage

1. Start the server: uvicorn main:app --reload
2. Open your browser and navigate to `http://localhost:8000`

3. Upload a drone telemetry CSV file with the required columns:
   - lat, lng, ts (timestamp)
   - pilot_lat, pilot_lon
   - speed_mps, turn_rate_deg_s, pilot_dist_m
   - drone_id

4. View the analysis results, including:
   - Map visualization of drone paths
   - Anomaly detection results
   - AI-generated threat assessments

5. Ask follow-up questions using the chat interface

6. Download the annotated CSV for further analysis

## Data Format

The system works with CSV files containing drone telemetry data. The required columns are:

- `lat`, `lng`: Drone coordinates (latitude/longitude)
- `ts`: Timestamp of detection
- `pilot_lat`, `pilot_lon`: Pilot/controller coordinates
- `speed_mps`: Speed in meters per second
- `turn_rate_deg_s`: Turn rate in degrees per second
- `pilot_dist_m`: Distance to pilot in meters
- `drone_id`: Unique identifier for each drone

Additional columns will be preserved in the output.

## Technical Details

### Anomaly Detection Methods

1. **Distance Test**: Flags drones operating beyond 2000m (likely beyond visual line of sight)
2. **Speed Test**: Flags drones exceeding 50 m/s (~180 km/h)
3. **Turn Rate Test**: Flags extreme maneuvers exceeding 90 degrees/second
4. **Burst Detection**: Uses change point detection to identify sudden changes in distance patterns
5. **Clustering**: Identifies multivariate outliers using HDBSCAN
6. **Isolation Forest**: Machine learning approach for identifying anomalies

### AI Analysis

The system uses OpenAI's o3 model to provide detailed threat assessments, including:

- Summary of the threat
- Detailed rationale for the assessment
- Priority level (1-5)
- Threat profile categorization
- Recommended actions and contingency plans

## Acknowledgments

- This project utilizes the following open-source libraries: pandas, numpy, scipy, pyproj, ruptures, hdbscan, scikit-learn, plotly, and FastAPI.
- AI capabilities powered by OpenAI's API.
- This project was created for the 2025 National Security Hackathon in SF: https://cerebralvalley.ai/e/national-security-hackathon-5a6fa1dc

