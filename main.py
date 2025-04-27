# ============================  main.py  ============================
"""
Drone-Anomaly Demo • tuned for dedrone_kepler_clean.csv
• Uses columns: lat, lng, ts, PilotLatitude, PilotLongitude,
                speed_mps, turn_rate_deg_s, pilot_dist_m, drone_id
• Five-test vote (distance, speed, heading-rate, burst, cluster outlier)
• LLM summaries via OpenAI o3  (temperature = 1)
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
    geod = pyproj.Geod(ellps="WGS84")
    _, _, d = geod.inv(lon1, lat1, lon2, lat2)
    return d                   # metres


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
        temperature=0.7  # Slightly lower temperature for more focused answers
    )
    return resp.choices[0].message.content.strip()


# ================================================================
# Core analysis
# ================================================================
def analyse(df_raw: pd.DataFrame):
    # ── column normalise ─────────────────────
    df_raw.columns = df_raw.columns.str.lower()
    df = df_raw.rename(columns={
        "dronelatitude":"lat",
        "dronelongitude":"lng",
        "detectiontime":"ts",
        "pilotlatitude":"pilot_lat",
        "pilotlongitude":"pilot_lon",
        "serialnumber":"drone_id"
    })
    for need in ["lat","lng","ts","pilot_lat","pilot_lon",
                 "speed_mps","turn_rate_deg_s","pilot_dist_m","drone_id"]:
        if need not in df.columns:
            raise ValueError(f"Missing column: {need}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["lat","lng","ts","pilot_lat","pilot_lon",
                           "speed_mps","turn_rate_deg_s","pilot_dist_m"])
    df = df.sort_values(["drone_id","ts"])

    # ── extra features per drone ─────────────
    minis=[]
    for did, g in df.groupby("drone_id"):
        g=g.copy()
        g["dt"] = g["ts"].diff().dt.total_seconds().fillna(0)
        g["dist_change"] = g["pilot_dist_m"].diff().fillna(0)
        g["dist_rate"] = g["dist_change"]/g["dt"].replace(0,np.nan)
        # heading rate
        bearing = np.degrees(np.arctan2(g["lng"]-g["pilot_lon"],
                                        g["lat"]-g["pilot_lat"]))
        bearing_diff = (bearing.diff().fillna(0)+180)%360 - 180
        g["bearing_rate"] = bearing_diff/g["dt"].replace(0,np.nan)
        g.fillna({"dist_rate":0,"bearing_rate":0}, inplace=True)
        minis.append(g)
    df = pd.concat(minis)

    # ── robust z for distance ────────────────
    med = df.pilot_dist_m.median()
    mad = median_abs_deviation(df.pilot_dist_m, nan_policy="omit")
    df["z_dist"] = (df.pilot_dist_m-med)/(1.4826*mad)

    # ── anomaly tests ────────────────────────
    df["dist_fail"]   = df.pilot_dist_m > 2000
    df["speed_fail"]  = df.speed_mps > 50
    df["turn_fail"]   = df.turn_rate_deg_s.abs() > 90
    # burst flag
    df["burst"] = False
    for did,g in df.groupby("drone_id"):
        if len(g)<6: continue
        cps = ruptures.KernelCPD(kernel="linear").fit(g.pilot_dist_m.values).predict(2)[:-1]
        df.loc[g.iloc[cps].index,"burst"]=True
    # cluster outlier
    feats = df[["speed_mps","turn_rate_deg_s","pilot_dist_m"]]
    scaled = StandardScaler().fit_transform(feats)
    df["clu_noise"] = hdbscan.HDBSCAN(min_cluster_size=10,min_samples=5)\
                         .fit_predict(scaled) == -1
    # isolate forest (adds weight)
    iso = IsolationForest(contamination=0.05, random_state=42).fit(feats)
    df["iso_flag"] = iso.decision_function(feats) < -0.15

    test_cols = ["dist_fail","speed_fail","turn_fail","burst","clu_noise"]
    df["votes"] = df[test_cols].sum(axis=1) + df.iso_flag.astype(int)

    # ── worst row per drone ───────────────────
    worst = df.loc[df.groupby("drone_id").votes.idxmax()].nlargest(5,"votes")

    # ── LLM briefs ────────────────────────────
    briefs=[]
    for _,r in worst.iterrows():
        b = brief_llm(r.drone_id,r)
        briefs.append({"id":r.drone_id,"votes":int(r.votes),"ts":r.ts,
                       "summary":b["summary"],"action":b["action"],
                       "pilot_dist":f"{r.pilot_dist_m:.0f} m",
                       "speed":f"{r.speed_mps:.1f} m/s"})

    ctx_json = json.dumps(briefs, default=str, indent=2)

    # ── minimal map ───────────────────────────
    map_html=""
    if briefs:
        ids=[b["id"] for b in briefs]
        dplot=df[df.drone_id.isin(ids)]
        fig=px.scatter_mapbox(dplot, lat="lat", lon="lng",
                              color=dplot.votes.astype(str),
                              hover_name="drone_id",
                              mapbox_style="open-street-map",zoom=10)
        map_html=fig.to_html(full_html=False, include_plotlyjs='cdn')

    return df, briefs, ctx_json, map_html

# ================================================================
# FastAPI routes
# ================================================================
cache = {}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
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
    
    # Create the complete HTML with styling
    html = f'''
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
            
            <div class="action-buttons">
                <a href="/download" class="download-btn"><i class="fas fa-download"></i> Download Annotated CSV</a>
                <a href="/" class="back-btn"><i class="fas fa-arrow-left"></i> Back to Home</a>
            </div>
        </div>
    </body>
    </html>
    '''
    return HTMLResponse(html)

@app.post("/ask")
async def ask(req: Request):
    q=(await req.json()).get("question","")
    if not q: return JSONResponse({"error":"no question"},400)
    if "ctx" not in cache: return JSONResponse({"error":"run analysis first"},400)
    ans = qa_llm(cache["ctx"], q)
    return JSONResponse({"answer":ans})

@app.get("/download")
def download():
    path = cache.get("csv")
    if not path: return HTMLResponse("No CSV",404)
    return FileResponse(path, filename="annotated.csv")

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
            
            <div class="chat-section">
                <h2><i class="fas fa-comments"></i> Ask Follow-up Questions</h2>
                <p>Ask questions about the analysis results and get AI-powered insights.</p>
                <form id="chat" onsubmit="send(); return false;">
                    <input id="q" placeholder="e.g., Which drone is the highest priority?" autocomplete="off">
                    <button type="submit"><i class="fas fa-paper-plane"></i> Ask</button>
                </form>
                <div id="log"></div>
            </div>
            
            <div class="divider"></div>
            
            <div class="features-section">
                <h2><i class="fas fa-list"></i> System Features</h2>
                <ul class="feature-list">
                    <li>Advanced anomaly detection using multiple tests</li>
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
                userDiv.innerHTML = `<strong>You:</strong> ${txt}`;
                log.appendChild(userDiv);
                
                // Show loading indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'ai-message';
                loadingDiv.innerHTML = `<strong>AI:</strong> <em>Thinking...</em>`;
                log.appendChild(loadingDiv);
                
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: txt})
                    });
                    
                    const data = await res.json();
                    
                    // Replace loading indicator with actual response
                    loadingDiv.innerHTML = `<strong>AI:</strong> ${data.answer}`;
                } catch (error) {
                    loadingDiv.innerHTML = `<strong>AI:</strong> <em>Sorry, an error occurred while processing your question.</em>`;
                }
                
                log.scrollTop = log.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

# uvicorn main:app --reload  (port defaults to 8000; change if needed)
# =================================================================