# ============================  main.py  ============================
"""
Drone Anomaly Detection System

A comprehensive platform for analyzing drone telemetry data and identifying potential security threats.

Key components:
• Data processing: Works with dedrone_kepler_clean.csv format
• Required columns: lat, lng, ts, PilotLatitude, PilotLongitude, speed_mps, turn_rate_deg_s, pilot_dist_m, drone_id
• Multi-dimensional anomaly detection: Six specialized tests (distance, speed, turn rate, burst detection,
  cluster anomaly detection, and isolation forest machine learning)
• AI-powered analysis: Utilizes OpenAI's o3 model (with temperature=1 for creative responses)
• Interactive visualization: Web interface with mapping and follow-up question capability
• Threat prioritization: 1-5 priority scale with detailed threat assessment and recommended actions
"""

# ─── Standard libs ───────────────────────────────────────────────
import os, io, time, json, tempfile, traceback
from io import BytesIO
# ─── Third-party ─────────────────────────────────────────────────
import pandas as pd
import numpy as np
from scipy.stats import median_abs_deviation
import pyproj, ruptures, hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
# ─── FastAPI ────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
# ─── OpenAI o3 ───────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv("keys.env", override=True)
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ────────────────────────────────────────────────────────────────

app = FastAPI(title="Drone-Anomaly Demo + o3")

# ================================================================
# Helpers
# ================================================================
def great_circle(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on Earth's surface.
    
    Args:
        lon1 (float): Longitude of first point in degrees
        lat1 (float): Latitude of first point in degrees
        lon2 (float): Longitude of second point in degrees
        lat2 (float): Latitude of second point in degrees
        
    Returns:
        float: Distance between the points in meters using WGS84 ellipsoid model
    """
    geod = pyproj.Geod(ellps="WGS84")  # Use WGS84 ellipsoid for Earth's shape
    _, _, d = geod.inv(lon1, lat1, lon2, lat2)  # Calculate inverse geodesic
    return d  # Returns distance in meters


def brief_llm(drone_id: str, row: pd.Series) -> dict:
    """
    Ask OpenAI o3 (T=1) for a deeper assessment.
    Returns a dict: {summary, rationale[], priority, action}
    """
    metrics = (
        f"* Distance-to-pilot …… {row.pilot_dist_m:.0f} m\n"
        f"* Speed …………………… {row.speed_mps:.1f} m/s\n"
        f"* Turn rate ………………… {row.turn_rate_deg_s:.1f} °/s\n"
        f"* Approach rate ………… {row.dist_rate:.1f} m/s\n"
        f"* Heading-Δ rate ………… {row.bearing_rate:.1f} °/s\n"
        f"* Robust-Z(distance) … {row.z_dist:.2f}\n"
        f"* Composite votes ……… {row.votes}\n"
    )

    prompt = (
        "ROLE: Senior counter-UAS analyst with expertise in drone threat assessment and airspace security\n\n"
        "TASK: Given one drone telemetry snapshot with anomaly indicators, produce a comprehensive threat assessment in **JSON**.\n\n"
        "Input metrics:\n"
        f"{metrics}\n"
        "Context for interpretation:\n"
        "• Distance-to-pilot > 2000m often indicates beyond visual line of sight operation (illegal in many jurisdictions)\n"
        "• Speed > 50 m/s (~180 km/h) exceeds typical consumer drone capabilities\n"
        "• High turn rates or sudden bearing changes suggest unusual flight patterns or evasive maneuvers\n"
        "• Negative approach rate means the drone is moving toward the pilot/target\n"
        "• High z-scores indicate statistical outliers in the dataset\n\n"
        "Judging rubric:\n"
        "• Consider multiple possible intentions (recreational hobbyist, professional photographer, careless operator, deliberate interference, malicious reconnaissance, data collection, potential weaponization, GPS spoofing)\n"
        "• Evaluate operator skill level and likely equipment sophistication\n"
        "• Analyze flight pattern in context of the surroundings and access restrictions\n"
        "• Assign a priority level from 1-5 (5 = immediate threat requiring intervention, 4 = high concern, 3 = moderate anomaly, 2 = minor concern, 1 = likely benign)\n"
        "• Provide 3-5 detailed rationale factors with specific metric interpretations\n"
        "• Recommend concrete and specific actions with clear escalation paths if needed\n\n"
        "Return ONLY a JSON object with these keys:\n"
        "  summary   – concise but informative assessment (≤ 50 words)\n"
        "  rationale – list of strings explaining your reasoning (3-5 items, each ≤ 30 words)\n"
        "  priority  – integer 1-5 with justification\n"
        "  threat_profile – most likely type of threat (e.g., 'Unauthorized surveillance', 'Airspace violation', 'Potential smuggling')\n" 
        "  action    – detailed recommended response with specific steps (≤ 150 chars)\n"
        "  contingency – secondary action if primary recommendation fails or situation escalates"
    )

    resp = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=1
    )
    return json.loads(resp.choices[0].message.content)


def qa_llm(context_json: str, question: str) -> str:
    """
    Generate detailed answers to follow-up questions about the drone analysis.
    """
    prompt = (
        "You are an expert counter-UAS (Unmanned Aircraft System) analyst specializing in drone threat assessment. "
        "You provide detailed, precise, and actionable insights based on the drone anomaly data. "
        "When answering questions:\n"
        "1. Reference specific metrics and values from the context when relevant\n"
        "2. Explain your reasoning and interpretation clearly\n"
        "3. Offer detailed perspectives on potential risks, intentions, and appropriate responses\n"
        "4. When discussing actions or interventions, provide specific steps and consider the urgency level\n"
        "5. If asked to compare multiple drones, highlight key differences in their anomaly profiles\n"
        "6. If the question is outside the scope of the data provided, clearly state the limitations"
    )

    resp = client.chat.completions.create(
        model="o3",
        messages=[
            {"role":"system", "content": prompt},
            {"role":"user", 
             "content":f"Context (drone anomaly data):\n{context_json}\n\nQuestion:\n{question}"}
        ],
        temperature=1  # Slightly lower temperature for more focused answers
    )
    return resp.choices[0].message.content.strip()


# ================================================================
# Core analysis
# ================================================================
def analyse(df_raw: pd.DataFrame):
    """Perform comprehensive anomaly detection analysis on drone telemetry data.
    
    This function processes raw telemetry data through multiple detection algorithms to identify
    suspicious drone behavior, calculates anomaly scores, and generates AI-powered threat assessments.
    
    Args:
        df_raw (pd.DataFrame): Raw drone telemetry data from CSV upload
        
    Returns:
        tuple: (processed_dataframe, threat_briefs, context_json, map_html)
            - processed_dataframe: DataFrame with all calculations and anomaly flags
            - threat_briefs: List of AI-assessed threat summaries for highest-priority drones
            - context_json: JSON string of threat summaries for follow-up questions
            - map_html: HTML for interactive map visualization of drone paths
    """
    # ── Normalize column names for consistent processing ─────────────────────
    df_raw.columns = df_raw.columns.str.lower()  # Convert all column names to lowercase
    df = df_raw.rename(columns={  # Standardize column names across different data sources
        "dronelatitude":"lat",
        "dronelongitude":"lng",
        "detectiontime":"ts",
        "pilotlatitude":"pilot_lat",
        "pilotlongitude":"pilot_lon",
        "serialnumber":"drone_id"
    })
    # ── Validate required columns and clean data ─────────────────────
    # Check for required columns - will raise an error if any are missing
    for need in ["lat","lng","ts","pilot_lat","pilot_lon",
                 "speed_mps","turn_rate_deg_s","pilot_dist_m","drone_id"]:
        if need not in df.columns:
            raise ValueError(f"Missing column: {need}")
            
    # Convert timestamp strings to datetime objects with UTC timezone
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    
    # Remove rows with missing values in essential columns
    df = df.dropna(subset=["lat","lng","ts","pilot_lat","pilot_lon",
                           "speed_mps","turn_rate_deg_s","pilot_dist_m"])
    
    # Sort data by drone ID and timestamp for proper sequential analysis
    df = df.sort_values(["drone_id","ts"])

    # ── Calculate derived features for each drone separately ─────────────
    minis=[]  # List to store processed dataframes for each drone
    for did, g in df.groupby("drone_id"):  # Process each drone separately
        g=g.copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Calculate time difference between consecutive points in seconds
        g["dt"] = g["ts"].diff().dt.total_seconds().fillna(0)
        
        # Calculate change in distance to pilot between consecutive points
        g["dist_change"] = g["pilot_dist_m"].diff().fillna(0)
        
        # Calculate rate of distance change (negative means approaching pilot)
        g["dist_rate"] = g["dist_change"]/g["dt"].replace(0,np.nan)
        
        # Calculate bearing (angle) from pilot to drone
        bearing = np.degrees(np.arctan2(g["lng"]-g["pilot_lon"],
                                         g["lat"]-g["pilot_lat"]))
        
        # Calculate change in bearing (normalized to -180 to 180 degrees)
        bearing_diff = (bearing.diff().fillna(0)+180)%360 - 180
        
        # Calculate rate of bearing change (indicates turning behavior)
        g["bearing_rate"] = bearing_diff/g["dt"].replace(0,np.nan)
        
        # Replace NaN values with 0 for rate calculations
        g.fillna({"dist_rate":0,"bearing_rate":0}, inplace=True)
        
        minis.append(g)  # Add processed drone data to list
    
    # Combine all drone dataframes back into one
    df = pd.concat(minis)

    # ── Calculate robust z-score for distance (outlier detection) ────────────────
    # Use median and median absolute deviation for robustness against extreme values
    med = df.pilot_dist_m.median()  # Median is less affected by outliers than mean
    mad = median_abs_deviation(df.pilot_dist_m, nan_policy="omit")  # Robust measure of dispersion
    
    # Calculate robust z-score: values > 3 or < -3 are typically considered outliers
    # 1.4826 factor makes MAD consistent with standard deviation for normal distributions
    df["z_dist"] = (df.pilot_dist_m-med)/(1.4826*mad)

    # ── Apply primary anomaly detection tests ────────────────────────
    # Test 1: Distance - Flag drones operating beyond 2000m (likely beyond visual line of sight)
    df["dist_fail"] = df.pilot_dist_m > 2000
    
    # Test 2: Speed - Flag drones exceeding 50 m/s (~180 km/h, beyond typical consumer capabilities)
    df["speed_fail"] = df.speed_mps > 50
    
    # Test 3: Turn rate - Flag extreme maneuvers exceeding 90 degrees/second
    df["turn_fail"] = df.turn_rate_deg_s.abs() > 90
    # ── Test 4: Burst detection (sudden changes in distance pattern) ────────────
    # Initialize burst flag column
    df["burst"] = False
    
    # Process each drone separately for change point detection
    for did,g in df.groupby("drone_id"):
        # Skip drones with too few data points
        if len(g)<6: continue
        
        # Use kernel change point detection to find sudden changes in distance pattern
        # This identifies abrupt shifts that could indicate evasive maneuvers or automated waypoints
        cps = ruptures.KernelCPD(kernel="linear").fit(g.pilot_dist_m.values).predict(2)[:-1]
        
        # Mark detected change points as True in the burst column
        df.loc[g.iloc[cps].index,"burst"]=True
    # ── Test 5: Clustering-based anomaly detection ────────────
    # Select features for multivariate anomaly detection
    feats = df[["speed_mps","turn_rate_deg_s","pilot_dist_m"]]
    
    # Standardize features to give equal weight (mean=0, std=1)
    scaled = StandardScaler().fit_transform(feats)
    
    # Apply HDBSCAN clustering algorithm to identify outliers
    # Points labeled as -1 are considered noise/outliers not belonging to any cluster
    # min_cluster_size=10: minimum points to form a cluster
    # min_samples=5: minimum points in neighborhood to form core points
    df["clu_noise"] = hdbscan.HDBSCAN(min_cluster_size=10,min_samples=5)\
                         .fit_predict(scaled) == -1
    # ── Test 6: Machine learning-based anomaly detection with Isolation Forest ────────────
    # Isolation Forest works by isolating observations through recursive partitioning
    # Anomalies require fewer partitions to isolate, resulting in shorter path lengths
    
    # Set contamination=0.05 expects approximately 5% of the data to be anomalous
    # random_state=42 ensures reproducible results
    iso = IsolationForest(contamination=0.05, random_state=42).fit(feats)
    
    # Flag points with decision function < -0.15 as anomalies
    # The decision function returns the negative average path length
    # More negative values indicate stronger anomalies
    df["iso_flag"] = iso.decision_function(feats) < -0.15

    # ── Calculate composite anomaly score by summing test results ────────────
    # Combine results from all detection methods into a single voting score
    test_cols = ["dist_fail","speed_fail","turn_fail","burst","clu_noise"]
    
    # Sum all boolean flags (True=1, False=0) plus the isolation forest flag
    # Higher vote counts indicate more anomalous behavior detected by multiple methods
    df["votes"] = df[test_cols].sum(axis=1) + df.iso_flag.astype(int)

    # ── Identify most anomalous point for each drone ───────────────────
    # For each drone, find the row with the highest anomaly score (votes)
    # Then select the top 5 drones with the highest maximum anomaly scores
    worst = df.loc[df.groupby("drone_id").votes.idxmax()].nlargest(5,"votes")

    # ── Generate AI-powered threat assessments for top anomalous drones ────────────
    # For each of the top anomalous drones, generate a detailed threat assessment using LLM
    briefs=[]
    for _,r in worst.iterrows():
        # Call OpenAI API to get detailed analysis of the drone's behavior
        b = brief_llm(r.drone_id,r)
        briefs.append({
            "id": r.drone_id,
            "votes": int(r.votes),
            "ts": r.ts,
            "summary": b["summary"],
            "action": b["action"],
            "priority": b.get("priority", 0),  # Extract priority (use 0 as default for sorting)
            "threat_profile": b.get("threat_profile", "Unknown threat type"),  # Extract threat profile
            "rationale": b.get("rationale", []),  # Extract rationale list
            "contingency": b.get("contingency", ""),  # Extract contingency plan
            "pilot_dist": f"{r.pilot_dist_m:.0f} m",
            "speed": f"{r.speed_mps:.1f} m/s"
        })
    
    # Sort briefs by priority (highest first)
    briefs.sort(key=lambda x: (0 if isinstance(x["priority"], str) else int(x["priority"])), reverse=True)

    ctx_json = json.dumps(briefs, default=str, indent=2)

    # ── Generate interactive map visualization of anomalous drone paths ───────────
    map_html=""
    if briefs:
        # Extract drone IDs from the briefs list
        ids=[b["id"] for b in briefs]
        
        # Filter data to only include the identified anomalous drones
        dplot=df[df.drone_id.isin(ids)]
        
        # Create interactive map using Plotly Express
        # - Points colored by anomaly score (votes)
        # - Hover shows drone ID
        # - Uses OpenStreetMap as base layer
        fig=px.scatter_mapbox(dplot, lat="lat", lon="lng",
                              color=dplot.votes.astype(str),
                              hover_name="drone_id",
                              mapbox_style="open-street-map",zoom=10)
        
        # Convert plot to HTML for embedding in web response
        # Use CDN for plotly.js to reduce response size
        map_html=fig.to_html(full_html=False, include_plotlyjs='cdn')

    return df, briefs, ctx_json, map_html

# ================================================================
# FastAPI routes
# ================================================================
cache = {}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Process uploaded drone telemetry CSV file and return analysis results.
    
    This endpoint handles the file upload, processes the data through the analysis pipeline,
    and returns a formatted HTML response with visualization and interactive elements.
    
    Args:
        file (UploadFile): The uploaded CSV file containing drone telemetry data
        
    Returns:
        HTMLResponse: Rendered HTML page with analysis results, visualization, and chat interface
        
    Raises:
        HTTPException: If file processing or analysis fails
    """
    try:
        df_raw = pd.read_csv(BytesIO(await file.read()))
        df, briefs, ctx, map_html = analyse(df_raw)
    except Exception:
        return HTMLResponse(f"<pre>{traceback.format_exc()}</pre>",500)

    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".csv",mode="w")
    df.to_csv(tmp.name,index=False)
    cache["csv"]=tmp.name
    cache["ctx"]=ctx

    # Create styled cards for each drone brief
    drone_cards = "".join(
        f'''
        <div class="drone-card priority-{b.get('priority', 'unknown')}">
            <div class="card-header">
                <h3>Drone {b['id']}</h3>
                <span class="priority-badge">Priority: {b.get('priority', 'N/A')}</span>
                <span class="votes-badge">Votes: {b['votes']}</span>
            </div>
            <div class="card-body">
                <p class="threat-profile">{b.get('threat_profile', 'Unknown threat type')}</p>
                <p class="summary"><strong>Summary:</strong> {b['summary']}</p>
                <div class="metrics">
                    <span><i class="fas fa-map-marker-alt"></i> {b['pilot_dist']}</span>
                    <span><i class="fas fa-tachometer-alt"></i> {b['speed']}</span>
                    <span><i class="far fa-clock"></i> {b['ts']}</span>
                </div>
                <div class="action-box">
                    <p><strong>Recommended Action:</strong> {b['action']}</p>
                    {f'<p><strong>Contingency:</strong> {b.get("contingency", "No contingency provided")}</p>' if b.get('contingency') else ''}
                </div>
                {f'<div class="rationale"><strong>Rationale:</strong><ul>{"<li>".join([""] + b.get("rationale", ["No rationale provided"]))}</ul></div>' if b.get('rationale') else ''}
            </div>
        </div>''' for b in briefs
    )
    
    # Create the HTML with styling - separating the script part to avoid f-string issues
    html_content = f'''
    <html>
    <head>
        <title>Drone Anomaly Analysis</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css" rel="stylesheet">
        <style>
            body {{font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; color: #333;}}
            h1, h2 {{color: #2c3e50; margin-bottom: 20px;}}
            h1 {{border-bottom: 2px solid #3498db; padding-bottom: 10px;}}
            .container {{max-width: 1200px; margin: 0 auto;}}
            .drone-card {{background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; overflow: hidden;}}
            .card-header {{background: #f8f9fa; padding: 15px; border-bottom: 1px solid #eee; display: flex; align-items: center; justify-content: space-between;}}
            .card-header h3 {{margin: 0; color: #2c3e50;}}
            .card-body {{padding: 20px;}}
            .priority-badge, .votes-badge {{padding: 5px 10px; border-radius: 4px; font-size: 14px; font-weight: bold;}}
            .priority-badge {{background-color: #e74c3c; color: white;}}
            .votes-badge {{background-color: #3498db; color: white;}}
            .priority-5 .card-header {{background-color: #ff5252; color: white;}}
            .priority-5 .card-header h3 {{color: white;}}
            .priority-4 .card-header {{background-color: #ff9800;}}
            .priority-3 .card-header {{background-color: #ffc107;}}
            .priority-2 .card-header {{background-color: #8bc34a;}}
            .priority-1 .card-header {{background-color: #4caf50;}}
            .priority-unknown .card-header {{background-color: #607d8b;}}
            .summary {{font-size: 16px; line-height: 1.5; margin-bottom: 15px;}}
            .metrics {{display: flex; gap: 15px; margin-bottom: 15px; color: #7f8c8d;}}
            .metrics span {{display: flex; align-items: center; gap: 5px;}}
            .action-box {{background-color: #f1f8e9; border-left: 4px solid #8bc34a; padding: 10px 15px; margin-bottom: 15px;}}
            .threat-profile {{background-color: #e3f2fd; border-radius: 4px; padding: 8px 12px; display: inline-block; margin-bottom: 10px; font-weight: bold; color: #0d47a1;}}
            .rationale {{background-color: #fafafa; padding: 10px 15px; border-radius: 4px;}}
            .rationale ul {{margin-top: 5px; padding-left: 20px;}}
            .rationale li {{margin-bottom: 5px;}}
            .action-buttons {{margin-top: 20px; display: flex; gap: 10px;}}
            .action-buttons a {{text-decoration: none; padding: 10px 15px; border-radius: 4px; display: inline-block; font-weight: bold; transition: all 0.3s;}}
            .download-btn {{background-color: #2ecc71; color: white;}}
            .download-btn:hover {{background-color: #27ae60;}}
            .back-btn {{background-color: #95a5a6; color: white;}}
            .back-btn:hover {{background-color: #7f8c8d;}}
            .map-container {{margin: 30px 0; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}}
            
            /* Chat section styling */
            .chat-section {{background-color: white; border-radius: 8px; padding: 25px; margin: 30px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}}
            #q {{width: 100%; padding: 12px; border: 1px solid #dce4ec; border-radius: 4px; font-size: 16px; margin-bottom: 10px;}}
            #log {{background-color: #f8f9fa; border: 1px solid #dce4ec; border-radius: 4px; padding: 15px; height: 300px; overflow-y: auto; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; margin-top: 15px;}}
            .user-message {{color: #2c3e50; margin-bottom: 10px;}}
            .ai-message {{color: #27ae60; margin-bottom: 15px; border-left: 3px solid #27ae60; padding-left: 10px;}}
            button {{background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background-color 0.3s;}}
            button:hover {{background-color: #2980b9;}}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Drone Anomaly Analysis</h1>
            <h2>Top Anomalous Drones</h2>
            <div class="drone-list">
                {drone_cards}
            </div>
            
            <div class="map-container">
                {map_html}
            </div>
            
            <div class="chat-section">
                <h2><i class="fas fa-comments"></i> Ask Follow-up Questions</h2>
                <p>Ask questions about the analysis results to get AI-powered insights.</p>
                <form id="chat" onsubmit="send(); return false;">
                    <input id="q" placeholder="e.g., Which drone poses the highest threat and why?" autocomplete="off">
                    <button type="submit"><i class="fas fa-paper-plane"></i> Ask</button>
                </form>
                <div id="log"></div>
                <p><a href="/docs" style="color: #3498db;"><i class="fas fa-book"></i> Read detailed technical documentation</a> about our anomaly detection algorithms.</p>
            </div>
            
            <div class="action-buttons">
                <a href="/download" class="download-btn"><i class="fas fa-download"></i> Download Annotated CSV</a>
                <a href="/" class="back-btn"><i class="fas fa-arrow-left"></i> Back to Home</a>
            </div>
        </div>
    '''
    
    # Add JavaScript separately (not in f-string)
    script_part = '''
        
    <script>
        async function send(){
            const txt=document.getElementById('q').value;
            document.getElementById('q').value='';
            if(!txt) return;
            const log=document.getElementById('log');
            
            // Add user message with styling
            const userDiv = document.createElement('div');
            userDiv.className = 'user-message';
            userDiv.innerHTML = '<strong>You:</strong> ' + txt;
            log.appendChild(userDiv);
            
            // Show loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'ai-message';
            loadingDiv.innerHTML = '<strong>AI:</strong> <em>Thinking...</em>';
            log.appendChild(loadingDiv);
            
            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({question: txt})
                });
                
                const data = await res.json();
                
                // Replace loading indicator with actual response
                loadingDiv.innerHTML = '<strong>AI:</strong> ' + data.answer;
            } catch (error) {
                loadingDiv.innerHTML = '<strong>AI:</strong> <em>Sorry, an error occurred while processing your question.</em>';
            }
            
            log.scrollTop = log.scrollHeight;
        }
    </script>
    </body>
    </html>
    '''
    
    # Combine the HTML content and script
    html = html_content + script_part
    return HTMLResponse(html)

@app.post("/ask")
async def ask(req: Request):
    """Handle follow-up questions about the drone anomaly analysis.
    
    This endpoint receives natural language questions from the user interface,
    passes them to the LLM along with the analysis context, and returns the AI response.
    
    Args:
        req (Request): FastAPI request object containing the question JSON
        
    Returns:
        JSONResponse: Contains the answer to the user's question
        
    Raises:
        HTTPException: If question processing fails or no analysis data is available
    """
    q=(await req.json()).get("question","")
    if not q: return JSONResponse({"error":"no question"},400)
    if "ctx" not in cache: return JSONResponse({"error":"run analysis first"},400)
    ans = qa_llm(cache["ctx"], q)
    return JSONResponse({"answer":ans})

@app.get("/download")
async def download():
    """Provide the annotated CSV file with analysis results for download.
    
    This endpoint returns the processed CSV file that includes all anomaly detection flags,
    calculated features, and anomaly scores for further analysis in external tools.
    
    Returns:
        FileResponse: The annotated CSV file with analysis results
        
    Raises:
        HTTPException: If no analysis has been performed yet
    """
    path = cache.get("csv")
    if not path: return HTMLResponse("No CSV",404)
    return FileResponse(path, filename="annotated.csv")

@app.get("/docs")
def docs():
    """Provide detailed documentation about the anomaly detection tests."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drone Anomaly Detection: Technical Documentation</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f7fa;
                color: #2c3e50;
                line-height: 1.6;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                padding: 30px 20px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }
            .test-card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .test-card h3 {
                color: #2980b9;
                margin-top: 0;
            }
            .back-btn {
                display: inline-block;
                background-color: #95a5a6;
                color: white;
                text-decoration: none;
                padding: 10px 15px;
                border-radius: 4px;
                font-weight: bold;
                margin-top: 20px;
            }
            .back-btn:hover {
                background-color: #7f8c8d;
            }
            code {
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
                color: #e74c3c;
            }
            .test-group {
                border-left: 3px solid #2ecc71;
                padding-left: 15px;
                margin: 30px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1><i class="fas fa-shield-alt"></i> Drone Anomaly Detection System</h1>
            <p>This documentation explains the technical details of the six anomaly detection tests used in our system to identify suspicious drone behavior.</p>
            
            <div class="test-group">
                <h2>Core Anomaly Detection Tests</h2>
                
                <div class="test-card">
                    <h3><i class="fas fa-ruler-horizontal"></i> Distance Test</h3>
                    <p><strong>Description:</strong> Flags drones operating beyond visual line of sight (BVLOS) limits.</p>
                    <p><strong>Implementation:</strong> Calculates geodesic distance between drone and pilot positions using pyproj.</p>
                    <p><strong>Threshold:</strong> <code>pilot_dist_m > 2000</code> - Flags drones more than 2km from their pilot, which often exceeds legal VLOS requirements.</p>
                    <p><strong>Significance:</strong> BVLOS operations typically require special authorization and may indicate unauthorized usage.</p>
                </div>
                
                <div class="test-card">
                    <h3><i class="fas fa-tachometer-alt"></i> Speed Test</h3>
                    <p><strong>Description:</strong> Identifies drones operating at abnormally high speeds.</p>
                    <p><strong>Implementation:</strong> Monitors the <code>speed_mps</code> value and compares against threshold.</p>
                    <p><strong>Threshold:</strong> <code>speed_mps > 50</code> - Flags drones exceeding 50 m/s (~180 km/h), which is beyond typical consumer drone capabilities.</p>
                    <p><strong>Significance:</strong> Unusually high speeds may indicate military-grade equipment, modified consumer drones, or sensor errors.</p>
                </div>
                
                <div class="test-card">
                    <h3><i class="fas fa-redo"></i> Turn Rate Test</h3>
                    <p><strong>Description:</strong> Detects erratic flight patterns with abnormally rapid turns.</p>
                    <p><strong>Implementation:</strong> Calculates rate of bearing change in degrees per second.</p>
                    <p><strong>Threshold:</strong> <code>turn_rate_deg_s.abs() > 90</code> - Flags drones making turns faster than 90 degrees per second.</p>
                    <p><strong>Significance:</strong> Rapid directional changes may indicate evasive maneuvers, aggressive flying, or loss of control.</p>
                </div>
            </div>
            
            <div class="test-group">
                <h2>Advanced Statistical Methods</h2>
                
                <div class="test-card">
                    <h3><i class="fas fa-chart-line"></i> Burst Detection</h3>
                    <p><strong>Description:</strong> Identifies sudden changes in flight behavior patterns.</p>
                    <p><strong>Implementation:</strong> Uses the ruptures library with KernelCPD (change point detection) algorithm on distance data.</p>
                    <p><strong>Algorithm:</strong> <code>ruptures.KernelCPD(kernel="linear")</code> to detect significant shifts in time series data.</p>
                    <p><strong>Significance:</strong> Sudden changes in distance patterns may indicate a change in mission, evasive behavior, or hand-off between operators.</p>
                </div>
                
                <div class="test-card">
                    <h3><i class="fas fa-braille"></i> Cluster Analysis</h3>
                    <p><strong>Description:</strong> Uses density-based clustering to identify spatial outliers.</p>
                    <p><strong>Implementation:</strong> HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) on standardized features.</p>
                    <p><strong>Features used:</strong> <code>["speed_mps", "turn_rate_deg_s", "pilot_dist_m"]</code> with StandardScaler normalization.</p>
                    <p><strong>Parameters:</strong> <code>min_cluster_size=10, min_samples=5</code></p>
                    <p><strong>Significance:</strong> Points labeled as noise (-1) represent multivariate statistical outliers across multiple dimensions.</p>
                </div>
                
                <div class="test-card">
                    <h3><i class="fas fa-tree"></i> Isolation Forest</h3>
                    <p><strong>Description:</strong> Machine learning algorithm specialized for anomaly detection.</p>
                    <p><strong>Implementation:</strong> sklearn's IsolationForest on the same feature set as cluster analysis.</p>
                    <p><strong>Parameters:</strong> <code>contamination=0.05, random_state=42</code></p>
                    <p><strong>Threshold:</strong> <code>decision_function(feats) < -0.15</code></p>
                    <p><strong>Significance:</strong> Adds weight to the anomaly vote by detecting points that can be easily isolated through recursive feature partitioning.</p>
                </div>
            </div>
            
            <div class="test-card">
                <h3><i class="fas fa-vote-yea"></i> Composite Scoring System</h3>
                <p><strong>Description:</strong> Multi-factor voting system that combines results from all tests.</p>
                <p><strong>Implementation:</strong> <code>df["votes"] = df[test_cols].sum(axis=1) + df.iso_flag.astype(int)</code></p>
                <p><strong>Scoring:</strong> Each failed test adds one vote to the anomaly score. Higher scores indicate more suspicious behavior.</p>
                <p><strong>Analysis:</strong> The LLM assesses the full context including vote count to assign priority levels (1-5) and recommend appropriate actions.</p>
            </div>
            
            <a href="/" class="back-btn"><i class="fas fa-arrow-left"></i> Back to Home</a>
        </div>
    </body>
    </html>
    """)

@app.get("/")
def home():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drone Anomaly Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f7fa;
                color: #2c3e50;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                padding: 30px 20px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            h2:before {
                content: '';
                width: 20px;
                height: 20px;
                background-color: #3498db;
                display: inline-block;
                border-radius: 50%;
                text-align: center;
                color: white;
                line-height: 20px;
                font-size: 14px;
            }
            .upload-section, .chat-section {
                background-color: white;
                border-radius: 8px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .form-group {
                margin-bottom: 15px;
            }
            input[type="file"] {
                border: 1px solid #dce4ec;
                padding: 10px;
                border-radius: 4px;
                width: 100%;
                max-width: 400px;
                background-color: #fafafa;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            #q {
                width: 100%;
                padding: 12px;
                border: 1px solid #dce4ec;
                border-radius: 4px;
                font-size: 16px;
                margin-bottom: 10px;
            }
            #log {
                background-color: #f8f9fa;
                border: 1px solid #dce4ec;
                border-radius: 4px;
                padding: 15px;
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin-top: 15px;
            }
            .user-message {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .ai-message {
                color: #27ae60;
                margin-bottom: 15px;
                border-left: 3px solid #27ae60;
                padding-left: 10px;
            }
            .divider {
                height: 1px;
                background-color: #ecf0f1;
                margin: 30px 0;
            }
            .feature-list {
                margin-top: 20px;
                padding-left: 20px;
            }
            .feature-list li {
                margin-bottom: 8px;
                list-style-type: none;
                position: relative;
                padding-left: 25px;
            }
            .feature-list li:before {
                content: '✓';
                position: absolute;
                left: 0;
                color: #27ae60;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1><i class="fas fa-drone"></i> Drone Anomaly Detection System</h1>
            
            <div class="upload-section">
                <h2><i class="fas fa-upload"></i> Upload Drone Telemetry Data</h2>
                <p>Upload a CSV file containing drone telemetry data for analysis.</p>
                <form action="/analyze" method="post" enctype="multipart/form-data" class="form-group">
                    <input type="file" name="file" accept=".csv">
                    <button type="submit"><i class="fas fa-chart-line"></i> Analyze Data</button>
                </form>
            </div>
            
            <div class="divider"></div>
            
            <div class="features-section">
                <h2><i class="fas fa-list"></i> System Features</h2>
                <ul class="feature-list">
                    <li>Advanced anomaly detection using 6 specialized tests (distance, speed, turn rate, burst detection, clustering, and machine learning)</li>
                    <li>AI-powered analysis of drone behavior patterns</li>
                    <li>Detailed threat assessment with priority levels</li>
                    <li>Interactive visualization of drone flight paths</li>
                    <li>Downloadable annotated results for further analysis</li>
                </ul>
            </div>
        </div>
        
        <script>
            async function send(){
                const txt=document.getElementById('q').value;
                document.getElementById('q').value='';
                if(!txt) return;
                const log=document.getElementById('log');
                
                // Add user message with styling
                const userDiv = document.createElement('div');
                userDiv.className = 'user-message';
                userDiv.innerHTML = '<strong>You:</strong> ' + txt;
                log.appendChild(userDiv);
                
                // Show loading indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'ai-message';
                loadingDiv.innerHTML = '<strong>AI:</strong> <em>Thinking...</em>';
                log.appendChild(loadingDiv);
                
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({question: txt})
                    });
                    
                    const data = await res.json();
                    
                    // Replace loading indicator with actual response
                    loadingDiv.innerHTML = '<strong>AI:</strong> ' + data.answer;
                } catch (error) {
                    loadingDiv.innerHTML = '<strong>AI:</strong> <em>Sorry, an error occurred while processing your question.</em>';
                }
                
                log.scrollTop = log.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

@app.get("/analyze")
def analyze():
    """
    Renders the main application interface for the Drone Anomaly Detection System.
    
    This endpoint serves the static HTML landing page that includes:
    - File upload functionality for drone telemetry data
    - Chat interface for AI-powered follow-up questions
    - System features overview and documentation links
    - Clear UI elements for user interaction
    
    Returns:
        HTMLResponse: Fully-rendered HTML page with embedded CSS and JavaScript
    """
    
    This endpoint serves the static HTML page that includes:
    - File upload functionality for drone telemetry data
    - Chat interface for AI-powered follow-up questions
    - System features overview
    - Interactive UI elements for user interaction
    
    Returns:
        HTMLResponse: Fully-rendered HTML page with embedded CSS and JavaScript
    """
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drone Anomaly Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f7fa;
                color: #2c3e50;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                padding: 30px 20px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            h2:before {
                content: '';
                width: 20px;
                height: 20px;
                background-color: #3498db;
                display: inline-block;
                border-radius: 50%;
                text-align: center;
                color: white;
                line-height: 20px;
                font-size: 14px;
            }
            .upload-section, .chat-section {
                background-color: white;
                border-radius: 8px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            .form-group {
                margin-bottom: 15px;
            }
            input[type="file"] {
                border: 1px solid #dce4ec;
                padding: 10px;
                border-radius: 4px;
                width: 100%;
                max-width: 400px;
                background-color: #fafafa;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            #q {
                width: 100%;
                padding: 12px;
                border: 1px solid #dce4ec;
                border-radius: 4px;
                font-size: 16px;
                margin-bottom: 10px;
            }
            #log {
                background-color: #f8f9fa;
                border: 1px solid #dce4ec;
                border-radius: 4px;
                padding: 15px;
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin-top: 15px;
            }
            .user-message {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .ai-message {
                color: #27ae60;
                margin-bottom: 15px;
                border-left: 3px solid #27ae60;
                padding-left: 10px;
            }
            .divider {
                height: 1px;
                background-color: #ecf0f1;
                margin: 30px 0;
            }
            .feature-list {
                margin-top: 20px;
                padding-left: 20px;
            }
            .feature-list li {
                margin-bottom: 8px;
                list-style-type: none;
                position: relative;
                padding-left: 25px;
            }
            .feature-list li:before {
                content: '✓';
                position: absolute;
                left: 0;
                color: #27ae60;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1><i class="fas fa-drone"></i> Drone Anomaly Detection System</h1>
            
            <div class="upload-section">
                <h2><i class="fas fa-upload"></i> Upload Drone Telemetry Data</h2>
                <p>Upload a CSV file containing drone telemetry data for analysis.</p>
                <form action="/analyze" method="post" enctype="multipart/form-data" class="form-group">
                    <input type="file" name="file" accept=".csv">
                    <button type="submit"><i class="fas fa-chart-line"></i> Analyze Data</button>
                </form>
            </div>
            
            <div class="divider"></div>
            
            <div class="chat-section">
                <h2><i class="fas fa-comments"></i> Ask Follow-up Questions</h2>
                <p>Ask questions about the analysis results and get AI-powered insights.</p>
                <h2>2  Ask follow-up questions</h2>
                <p><em>Note: This feature will be available after analyzing data.</em></p>
            </div>
            
            <div class="divider"></div>
            
            <div class="features-section">
                <h2><i class="fas fa-list"></i> System Features</h2>
                <ul class="feature-list">
                    <li>Advanced anomaly detection using 6 specialized tests (distance, speed, turn rate, burst detection, clustering, and machine learning)</li>
                    <li>AI-powered analysis of drone behavior patterns</li>
                    <li>Detailed threat assessment with priority levels</li>
                    <li>Interactive visualization of drone flight paths</li>
                    <li>Downloadable annotated results for further analysis</li>
                </ul>
            </div>
        </div>
        
        <script>
            async function send(){
                const txt=document.getElementById('q').value;
                document.getElementById('q').value='';
                if(!txt) return;
                const log=document.getElementById('log');
                
                // Add user message with styling
                const userDiv = document.createElement('div');
                userDiv.className = 'user-message';
                userDiv.innerHTML = '<strong>You:</strong> ' + txt;
                log.appendChild(userDiv);
                
                // Show loading indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'ai-message';
                loadingDiv.innerHTML = '<strong>AI:</strong> <em>Thinking...</em>';
                log.appendChild(loadingDiv);
                
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({question: txt})
                    });
                    
                    const data = await res.json();
                    
                    // Replace loading indicator with actual response
                    loadingDiv.innerHTML = '<strong>AI:</strong> ' + data.answer;
                } catch (error) {
                    loadingDiv.innerHTML = '<strong>AI:</strong> <em>Sorry, an error occurred while processing your question.</em>';
                }
                
                log.scrollTop = log.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

# Run the FastAPI application with: uvicorn main:app --reload
# The server will listen on port 8000 by default, which can be changed via command line parameters
# ================================================================= End of analyze function