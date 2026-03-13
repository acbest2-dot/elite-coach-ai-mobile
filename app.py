import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import json
import polyline
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import calendar as cal_module

# ============================================================
# CONFIGURAZIONE
# ============================================================
REDIRECT_URI = "https://elite-ai-coach-mobile.streamlit.app/"  # ← aggiorna con il tuo URL mobile

def get_secret(key):
    return st.secrets.get(key) or os.getenv(key)

CLIENT_ID     = get_secret("STRAVA_CLIENT_ID")
CLIENT_SECRET = get_secret("STRAVA_CLIENT_SECRET")
GEMINI_KEY    = get_secret("GOOGLE_API_KEY")
GROK_KEY      = get_secret("GROK_API_KEY") or get_secret("XAI_API_KEY") or ""
MAPBOX_TOKEN  = get_secret("MAPBOX_TOKEN") or ""
GSHEET_ID     = get_secret("GSHEET_ID") or ""
GSHEET_CREDS  = get_secret("GSHEET_CREDENTIALS") or ""

# ── AI Provider ──
_ai_client     = None
_ai_client_v1a = None
_ai_sdk_mode   = None

if GEMINI_KEY:
    try:
        import google.genai as genai_new
        from google.genai import types as genai_types
        _ai_client    = genai_new.Client(api_key=GEMINI_KEY)
        _ai_client_v1a = genai_new.Client(
            api_key=GEMINI_KEY,
            http_options=genai_types.HttpOptions(api_version="v1alpha")
        )
        _ai_sdk_mode = "new"
    except Exception:
        pass
    if _ai_sdk_mode is None:
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=GEMINI_KEY)
            _ai_client   = genai_old
            _ai_sdk_mode = "old"
        except ImportError:
            pass

if _ai_sdk_mode is None and GROK_KEY:
    try:
        from openai import OpenAI as _OpenAI
        _ai_client   = _OpenAI(api_key=GROK_KEY, base_url="https://api.x.ai/v1")
        _ai_sdk_mode = "grok"
    except ImportError:
        pass

st.set_page_config(
    page_title="Elite Coach",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS MOBILE
# ============================================================
st.markdown("""
<style>
  /* Reset e base mobile */
  * { box-sizing: border-box; }
  html, body, [data-testid="stAppViewContainer"] {
      background: #f0f2f6;
  }
  /* Nascondi sidebar toggle su mobile */
  [data-testid="collapsedControl"] { display: none !important; }
  section[data-testid="stSidebar"] { display: none !important; }

  /* Padding ridotto per mobile */
  .block-container {
      padding: 0 0 100px 0 !important;
      max-width: 100% !important;
  }

  /* Header app */
  .mob-header {
      background: linear-gradient(135deg, #1565C0, #0D47A1);
      color: white;
      padding: 16px 20px 12px;
      margin-bottom: 0;
      position: sticky;
      top: 0;
      z-index: 999;
  }
  .mob-header h1 {
      font-size: 20px;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.3px;
  }
  .mob-header p {
      font-size: 12px;
      opacity: 0.8;
      margin: 2px 0 0;
  }

  /* Bottom Navigation Bar */
  .bottom-nav {
      position: fixed;
      bottom: 0; left: 0; right: 0;
      background: #ffffff;
      border-top: 1px solid #e0e0e0;
      display: flex;
      justify-content: space-around;
      align-items: center;
      padding: 6px 0 10px;
      z-index: 1000;
      box-shadow: 0 -2px 12px rgba(0,0,0,0.08);
  }
  .nav-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      cursor: pointer;
      padding: 4px 12px;
      border-radius: 12px;
      transition: background 0.15s;
      text-decoration: none;
  }
  .nav-item.active { background: #E3F2FD; }
  .nav-icon { font-size: 22px; line-height: 1; }
  .nav-label { font-size: 10px; font-weight: 600; color: #555; }
  .nav-item.active .nav-label { color: #1565C0; }

  /* Card generica */
  .mob-card {
      background: #ffffff;
      border-radius: 16px;
      padding: 16px;
      margin: 12px 12px 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .mob-card-title {
      font-size: 12px;
      font-weight: 700;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
  }

  /* Metriche grandi */
  .big-metric {
      text-align: center;
      padding: 8px;
  }
  .big-metric .val {
      font-size: 36px;
      font-weight: 900;
      line-height: 1;
  }
  .big-metric .lbl {
      font-size: 11px;
      color: #888;
      font-weight: 600;
      margin-top: 2px;
  }

  /* Attività card mobile */
  .act-card {
      background: #fff;
      border-radius: 14px;
      padding: 14px 16px;
      margin: 8px 12px 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      border-left: 4px solid #ccc;
  }
  .act-title {
      font-size: 15px;
      font-weight: 700;
      color: #1a1a1a;
      margin-bottom: 4px;
  }
  .act-meta {
      font-size: 12px;
      color: #777;
      margin-bottom: 8px;
  }
  .act-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
  }
  .act-pill {
      background: #f5f5f5;
      border-radius: 20px;
      padding: 4px 10px;
      font-size: 12px;
      color: #333;
      font-weight: 500;
  }
  .act-pill b { color: #e94560; }

  /* Zone badge */
  .zone-chip {
      display: inline-block;
      border-radius: 20px;
      padding: 2px 10px;
      font-size: 11px;
      font-weight: 700;
  }

  /* AI box */
  .ai-box {
      background: #f8f9fa;
      border-left: 4px solid #2196F3;
      border-radius: 0 12px 12px 0;
      padding: 14px 16px;
      margin: 8px 12px;
      color: #212529;
      font-size: 15px;
      line-height: 1.75;
  }

  /* Chat bubbles */
  .chat-user {
      background: #1565C0;
      color: white;
      border-radius: 18px 18px 4px 18px;
      padding: 10px 14px;
      margin: 4px 12px 4px 48px;
      font-size: 14px;
      line-height: 1.5;
  }
  .chat-ai {
      background: #ffffff;
      color: #1a1a1a;
      border-radius: 18px 18px 18px 4px;
      padding: 10px 14px;
      margin: 4px 48px 4px 12px;
      font-size: 14px;
      line-height: 1.5;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .chat-label {
      font-size: 10px;
      color: #999;
      margin: 0 12px 2px;
      font-weight: 600;
  }

  /* Buttons grandi touch-friendly */
  div[data-testid="stButton"] > button {
      min-height: 48px !important;
      font-size: 15px !important;
      border-radius: 12px !important;
      font-weight: 600 !important;
  }

  /* Input touch-friendly */
  div[data-testid="stTextInput"] input,
  div[data-testid="stChatInput"] textarea {
      font-size: 16px !important;
      min-height: 48px !important;
  }

  /* Slider touch */
  div[data-testid="stSlider"] { padding: 0 12px; }

  /* Section padding */
  .sec-pad { padding: 0 12px; }

  /* Divider sottile */
  .mob-divider {
      height: 1px;
      background: #eeeeee;
      margin: 12px;
  }

  /* Status badge */
  .status-badge {
      display: inline-block;
      border-radius: 20px;
      padding: 4px 14px;
      font-size: 13px;
      font-weight: 700;
  }

  /* Calendario mobile */
  .cal-day-act {
      background: #fff;
      border-radius: 10px;
      padding: 4px 6px;
      margin: 2px 0;
      font-size: 11px;
      display: flex;
      align-items: center;
      gap: 4px;
  }
  .cal-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
  }

  /* Hide streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SPORT INFO
# ============================================================
SPORT_INFO = {
    "Run":              {"icon": "🏃", "label": "Corsa",         "color": "#FF4B4B"},
    "TrailRun":         {"icon": "🏔️", "label": "Trail Run",     "color": "#FF7043"},
    "Ride":             {"icon": "🚴", "label": "Ciclismo",       "color": "#2196F3"},
    "VirtualRide":      {"icon": "🖥️", "label": "Ciclismo V.",   "color": "#42A5F5"},
    "MountainBikeRide": {"icon": "🚵", "label": "MTB",           "color": "#1565C0"},
    "BackcountrySki":   {"icon": "🎿", "label": "Sci Alpinismo", "color": "#4FC3F7"},
    "AlpineSki":        {"icon": "⛷️", "label": "Sci Alpino",    "color": "#81D4FA"},
    "Hike":             {"icon": "🥾", "label": "Escursionismo", "color": "#4CAF50"},
    "Walk":             {"icon": "🚶", "label": "Camminata",     "color": "#8BC34A"},
    "Workout":          {"icon": "💪", "label": "Allenamento",   "color": "#FF9800"},
    "Swim":             {"icon": "🏊", "label": "Nuoto",         "color": "#00BCD4"},
}

def get_sport_info(a_type, name=""):
    if a_type == "Ride" and name:
        n = name.lower()
        if any(k in n for k in ["mtb","mountain","gravel","sterrato","trail","enduro"]):
            a_type = "MountainBikeRide"
    return SPORT_INFO.get(a_type, {"icon": "🏅", "label": a_type, "color": "#9E9E9E"})

# ============================================================
# METRICHE
# ============================================================
def format_metrics(row):
    a_type = row["type"]
    dist   = row["distance"] / 1000
    time   = row["moving_time"]
    elev   = row.get("total_elevation_gain", 0) or 0
    hr_avg = row.get("average_heartrate")
    hr_max = row.get("max_heartrate")
    watts  = row.get("average_watts")
    hrs    = int(time // 3600)
    mins   = int((time % 3600) // 60)
    dur_str = f"{hrs}h {mins:02d}m" if hrs > 0 else f"{mins}m"
    if a_type in ("Ride", "VirtualRide", "MountainBikeRide"):
        speed   = dist / (time / 3600) if time > 0 else 0
        pace_str = f"{speed:.1f} km/h"
    else:
        pace_raw = time / dist if dist > 0 else 0
        pace_str = f"{int(pace_raw // 60)}:{int(pace_raw % 60):02d} /km"
    return {
        "dist_str":  f"{dist:.1f} km",
        "pace_str":  pace_str,
        "dur_str":   dur_str,
        "elev":      f"{elev:.0f} m",
        "hr_avg":    f"{hr_avg:.0f}" if pd.notna(hr_avg) else "—",
        "hr_max":    f"{hr_max:.0f}" if pd.notna(hr_max) else "—",
        "watts":     f"{watts:.0f} W" if pd.notna(watts) else "—",
        "dist_km":   dist,
        "time_sec":  time,
    }

def get_hr_zone(hr_pct):
    if hr_pct < 0.60: return 1, "#4CAF50", "Z1"
    if hr_pct < 0.70: return 2, "#8BC34A", "Z2"
    if hr_pct < 0.80: return 3, "#FFC107", "Z3"
    if hr_pct < 0.90: return 4, "#FF9800", "Z4"
    return 5, "#F44336", "Z5"

def get_zone_for_activity(row, fc_max):
    hr = row.get("average_heartrate")
    if pd.notna(hr) and fc_max > 0:
        return get_hr_zone(hr / fc_max)
    return 0, "#9E9E9E", "N/A"

# ============================================================
# TSS / FITNESS
# ============================================================
def calc_tss(row, u):
    dur   = row["moving_time"] / 60
    hr    = row["average_heartrate"] if pd.notna(row.get("average_heartrate")) else 0
    watts = row["average_watts"]     if pd.notna(row.get("average_watts"))     else 0
    ftp   = u.get("ftp", 200)
    if hr > 0 and u["fc_max"] > u["fc_min"]:
        intensity = (hr - u["fc_min"]) / (u["fc_max"] - u["fc_min"])
        intensity = max(0.0, min(intensity, 1.0))
        return (dur * hr * intensity) / (u["fc_max"] * 60) * 100
    if watts > 0 and ftp > 0:
        IF = watts / ftp
        return (row["moving_time"] * watts * IF) / (ftp * 3600) * 100
    return dur * 0.4

def compute_fitness(df):
    daily = df.groupby(df["start_date"].dt.date)["tss"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    ctl   = daily.ewm(span=42, adjust=False).mean()
    atl   = daily.ewm(span=7,  adjust=False).mean()
    tsb   = ctl - atl
    df_dates   = df["start_date"].dt.date.map(lambda d: pd.Timestamp(d))
    ctl_mapped = df_dates.map(ctl)
    atl_mapped = df_dates.map(atl)
    tsb_mapped = df_dates.map(tsb)
    return ctl_mapped, atl_mapped, tsb_mapped, ctl, atl, tsb, daily

def calc_vo2max_estimate(df_sorted):
    runs = df_sorted[
        (df_sorted["type"].isin(["Run","TrailRun"])) &
        (df_sorted["distance"] >= 5000)
    ].copy()
    if runs.empty:
        return None
    best_vo2 = 0
    for _, row in runs.iterrows():
        dist_m   = row["distance"]
        time_min = row["moving_time"] / 60
        if time_min <= 0: continue
        vel = dist_m / time_min
        pct = 0.8 + 0.1894393 * np.exp(-0.012778 * time_min) + \
              0.2989558 * np.exp(-0.1932605 * time_min)
        vo2 = (-4.60 + 0.182258 * vel + 0.000104 * vel**2)
        vo2max = vo2 / pct if pct > 0 else 0
        if vo2max > best_vo2:
            best_vo2 = vo2max
    return round(best_vo2, 1) if best_vo2 > 0 else None

# ============================================================
# GOOGLE SHEETS — Cache persistente
# ============================================================
def _get_gsheet_client():
    """Restituisce il client gspread se configurato, altrimenti None."""
    if not GSHEET_ID or not GSHEET_CREDS:
        return None, None
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = json.loads(GSHEET_CREDS)
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet  = client.open_by_key(GSHEET_ID)
        return client, sheet
    except Exception as e:
        return None, None

def gsheet_load_activities() -> list:
    """Carica attività dal Google Sheet. Ritorna lista di dict o []."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return []
    try:
        ws = sheet.worksheet("activities")
        records = ws.get_all_records()
        return records
    except Exception:
        return []

def gsheet_save_activities(activities: list):
    """Salva/aggiorna attività nel Google Sheet (sovrascrittura completa)."""
    _, sheet = _get_gsheet_client()
    if sheet is None or not activities:
        return False
    try:
        try:
            ws = sheet.worksheet("activities")
            ws.clear()
        except Exception:
            ws = sheet.add_worksheet(title="activities", rows=10000, cols=60)

        if not activities:
            return True

        # Prendi tutte le chiavi come header
        all_keys = set()
        for a in activities:
            all_keys.update(a.keys())
        # Escludi campi complessi non serializzabili in sheet
        exclude = {"map", "segment_efforts", "best_efforts", "splits_metric",
                   "splits_standard", "laps", "photos", "gear", "stats_visibility",
                   "hide_from_home", "similar_activities"}
        headers = sorted([k for k in all_keys if k not in exclude])

        rows = [headers]
        for a in activities:
            row = []
            for h in headers:
                val = a.get(h, "")
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                row.append(str(val) if val is not None else "")
            rows.append(row)

        ws.update(rows, "A1")

        # Salva metadata (data ultimo sync)
        try:
            meta_ws = sheet.worksheet("meta")
        except Exception:
            meta_ws = sheet.add_worksheet(title="meta", rows=10, cols=5)
        meta_ws.update([["last_sync", datetime.now().isoformat()]], "A1")
        return True
    except Exception as e:
        return False

def gsheet_get_last_sync() -> datetime | None:
    """Ritorna datetime dell'ultimo sync, o None."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return None
    try:
        meta_ws = sheet.worksheet("meta")
        val = meta_ws.cell(1, 2).value
        return datetime.fromisoformat(val) if val else None
    except Exception:
        return None

def gsheet_needs_sync() -> bool:
    """True se non abbiamo mai sincronizzato o l'ultimo sync è > 24h fa."""
    last = gsheet_get_last_sync()
    if last is None:
        return True
    return (datetime.now() - last) > timedelta(hours=24)

# ============================================================
# STRAVA AUTH + FETCH
# ============================================================
def refresh_token_if_needed():
    token_info = st.session_state.get("strava_token_info", {})
    if not token_info:
        return False
    if datetime.now(timezone.utc).timestamp() < token_info.get("expires_at", 0):
        return True
    refresh_tok = token_info.get("refresh_token")
    if not refresh_tok:
        return False
    res = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token", "refresh_token": refresh_tok,
    }).json()
    if "access_token" in res:
        st.session_state.strava_token_info = res
        return True
    return False

def _fetch_page(access_token, page, after_ts=0):
    params = f"per_page=200&page={page}"
    if after_ts:
        params += f"&after={after_ts}"
    r = requests.get(
        f"https://www.strava.com/api/v3/athlete/activities?{params}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return r.json() if r.status_code == 200 else []

def load_all_from_strava(access_token):
    """Scarica tutto lo storico da Strava con paginazione."""
    all_acts = []
    page = 1
    while True:
        batch = _fetch_page(access_token, page)
        if not batch:
            break
        all_acts.extend(batch)
        if len(batch) < 200:
            break
        page += 1
    return all_acts

def load_new_from_strava(access_token, after_ts):
    """Scarica solo le attività successive a after_ts."""
    new_acts = []
    page = 1
    while True:
        batch = _fetch_page(access_token, page, after_ts=after_ts)
        if not batch:
            break
        new_acts.extend(batch)
        if len(batch) < 200:
            break
        page += 1
    return new_acts

@st.cache_data(ttl=300)
def fetch_athlete(access_token):
    r = requests.get(
        "https://www.strava.com/api/v3/athlete",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return r.json() if r.status_code == 200 else {}

# ============================================================
# MAPPA 2D
# ============================================================
def draw_map(encoded_polyline, height=220):
    if not encoded_polyline:
        return None
    try:
        pts = polyline.decode(encoded_polyline)
        if not pts:
            return None
        clat = sum(p[0] for p in pts) / len(pts)
        clon = sum(p[1] for p in pts) / len(pts)
        m = folium.Map(location=[clat, clon], zoom_start=12,
                       tiles="CartoDB positron")
        folium.PolyLine(pts, color="#2196F3", weight=3, opacity=0.9).add_to(m)
        folium.CircleMarker(pts[0],  radius=5, color="#4CAF50", fill=True).add_to(m)
        folium.CircleMarker(pts[-1], radius=5, color="#F44336", fill=True).add_to(m)
        return m
    except Exception:
        return None

# ============================================================
# AI
# ============================================================
def ai_generate(prompt: str, max_tokens: int = 1500) -> str:
    if _ai_sdk_mode is None:
        return "⚠️ Nessun provider AI configurato. Aggiungi GOOGLE_API_KEY nei Secrets."
    try:
        if _ai_sdk_mode == "new":
            resp = _ai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return resp.text
        elif _ai_sdk_mode == "old":
            model = _ai_client.GenerativeModel("gemini-1.5-flash")
            return model.generate_content(prompt).text
        elif _ai_sdk_mode == "grok":
            resp = _ai_client.chat.completions.create(
                model="grok-3-fast",
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Errore AI: {e}"

def ai_deep(prompt: str) -> str:
    """Analisi approfondita — Gemini 2.5 Pro con fallback."""
    if _ai_sdk_mode == "new":
        for model in ["gemini-2.5-pro-preview-05-06", "gemini-2.5-pro-preview-03-25",
                      "gemini-2.0-flash"]:
            try:
                resp = _ai_client.models.generate_content(
                    model=model, contents=prompt,
                )
                return resp.text
            except Exception:
                continue
    return ai_generate(prompt, max_tokens=2000)

# ============================================================
# SESSION STATE
# ============================================================
for key, val in {
    "strava_token_info":  {},
    "messages":           [],
    "user_data":          {"peso": 75.0, "fc_min": 50, "fc_max": 190, "ftp": 200},
    "activities_cache":   [],
    "activities_last_ts": 0,
    "activities_token":   "",
    "mob_menu":           "dashboard",
    "selected_act_id":    None,
    "gsheet_loaded":      False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# OAUTH
# ============================================================
if "code" in st.query_params and not st.session_state.strava_token_info.get("access_token"):
    res = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "code": st.query_params["code"], "grant_type": "authorization_code",
    }).json()
    if "access_token" in res:
        st.session_state.strava_token_info = res
        st.rerun()

# ============================================================
# BOTTOM NAV BAR
# ============================================================
NAV_ITEMS = [
    ("dashboard", "📊", "Home"),
    ("fitness",   "💪", "Fitness"),
    ("storico",   "📅", "Storico"),
    ("chat",      "💬", "Coach"),
    ("profilo",   "👤", "Profilo"),
]

def render_bottom_nav():
    nav_html = '<div class="bottom-nav">'
    for key, icon, label in NAV_ITEMS:
        active = "active" if st.session_state.mob_menu == key else ""
        nav_html += (
            f'<div class="nav-item {active}" onclick="">'
            f'<span class="nav-icon">{icon}</span>'
            f'<span class="nav-label">{label}</span>'
            f'</div>'
        )
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)

    # Pulsanti reali (invisibili sotto la nav bar per catturare i click)
    cols = st.columns(5)
    for i, (key, icon, label) in enumerate(NAV_ITEMS):
        with cols[i]:
            if st.button(f"{icon}", key=f"nav_{key}", use_container_width=True,
                         help=label, type="secondary"):
                st.session_state.mob_menu = key
                st.session_state.selected_act_id = None
                st.rerun()

# ============================================================
# LOGIN PAGE
# ============================================================
token_ok = refresh_token_if_needed()

if not token_ok:
    st.markdown("""
    <div style="text-align:center;padding:60px 24px 40px">
        <div style="font-size:64px;margin-bottom:16px">🏆</div>
        <h1 style="font-size:26px;font-weight:900;color:#1565C0;margin-bottom:8px">Elite AI Coach</h1>
        <p style="color:#666;font-size:15px;margin-bottom:32px">Il tuo coach AI personale.<br>Connetti Strava per iniziare.</p>
    </div>
    """, unsafe_allow_html=True)

    if CLIENT_ID:
        auth_url = (
            f"https://www.strava.com/oauth/authorize"
            f"?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
            f"&response_type=code&scope=read,activity:read_all"
        )
        st.markdown(f"""
        <div style="text-align:center;padding:0 24px">
            <a href="{auth_url}" style="
                display:block;background:#FC4C02;color:white;
                border-radius:14px;padding:16px;font-size:17px;
                font-weight:700;text-decoration:none;
                box-shadow:0 4px 16px rgba(252,76,2,0.35)">
                🔗 Connetti Strava
            </a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Configura STRAVA_CLIENT_ID e STRAVA_CLIENT_SECRET nei Secrets.")
    st.stop()

# ============================================================
# DATI STRAVA + GSHEET CACHE
# ============================================================
access_token = st.session_state.strava_token_info["access_token"]
athlete      = fetch_athlete(access_token)

# ── Logica caricamento con Google Sheets ──
_gsheet_ok = bool(GSHEET_ID and GSHEET_CREDS)

if not st.session_state.activities_cache:
    # Prima prova dal Google Sheet
    if _gsheet_ok and not st.session_state.gsheet_loaded:
        with st.spinner("📊 Carico storico dalla cache..."):
            sheet_data = gsheet_load_activities()
        if sheet_data:
            st.session_state.activities_cache = sheet_data
            st.session_state.gsheet_loaded    = True
            # Controlla se serve aggiornamento da Strava
            if gsheet_needs_sync():
                st.toast("🔄 Aggiorno con le ultime attività da Strava...", icon="⏳")
                # Trova timestamp più recente nella cache
                _dates = []
                for a in sheet_data:
                    try:
                        d = datetime.fromisoformat(str(a.get("start_date","")).replace("Z",""))
                        _dates.append(int(d.timestamp()))
                    except Exception:
                        pass
                _last_ts = max(_dates) if _dates else 0
                new_acts = load_new_from_strava(access_token, after_ts=_last_ts)
                if new_acts:
                    existing_ids = {a["id"] for a in sheet_data}
                    added = [a for a in new_acts if a["id"] not in existing_ids]
                    if added:
                        merged = sheet_data + added
                        st.session_state.activities_cache = merged
                        gsheet_save_activities(merged)
                        st.toast(f"✅ {len(added)} nuove attività sincronizzate", icon="🏃")
                else:
                    # Solo aggiorna timestamp sync
                    gsheet_save_activities(sheet_data)
        else:
            # Nessun dato nel sheet → carica tutto da Strava
            with st.spinner("⏳ Primo caricamento storico da Strava (30-60 sec)..."):
                raw = load_all_from_strava(access_token)
            st.session_state.activities_cache = raw
            st.session_state.gsheet_loaded    = True
            if _gsheet_ok and raw:
                with st.spinner("💾 Salvo in cache persistente..."):
                    gsheet_save_activities(raw)
                st.toast(f"✅ {len(raw)} attività caricate e salvate", icon="🏃")
    else:
        # Nessun GSheet configurato → carica da Strava direttamente
        with st.spinner("⏳ Carico storico da Strava..."):
            raw = load_all_from_strava(access_token)
        st.session_state.activities_cache = raw
        if raw:
            st.toast(f"✅ {len(raw)} attività caricate", icon="🏃")

raw = st.session_state.activities_cache
if not raw:
    st.error("Impossibile recuperare le attività da Strava.")
    st.stop()

# ── Build DataFrame ──
df = pd.DataFrame(raw)
df["start_date"] = pd.to_datetime(df.get("start_date_local", df.get("start_date",""))).dt.tz_localize(None)
df = df.sort_values("start_date").reset_index(drop=True)

for col in ["average_heartrate","max_heartrate","average_watts",
            "total_elevation_gain","average_cadence","kilojoules",
            "calories","suffer_score","distance","moving_time"]:
    if col not in df.columns:
        df[col] = np.nan
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")

u = st.session_state.user_data
df["tss"] = df.apply(lambda row: calc_tss(row, u), axis=1)
ctl_s, atl_s, tsb_s, ctl_daily, atl_daily, tsb_daily, tss_daily = compute_fitness(df)
df["ctl"]        = ctl_s.values
df["atl"]        = atl_s.values
df["tsb"]        = tsb_s.values
df["zone_num"]   = df.apply(lambda r: get_zone_for_activity(r, u["fc_max"])[0], axis=1)
df["zone_color"] = df.apply(lambda r: get_zone_for_activity(r, u["fc_max"])[1], axis=1)
df["zone_label"] = df.apply(lambda r: get_zone_for_activity(r, u["fc_max"])[2], axis=1)

current_ctl = float(df["ctl"].iloc[-1])
current_atl = float(df["atl"].iloc[-1])
current_tsb = float(df["tsb"].iloc[-1])
vo2max_val  = calc_vo2max_estimate(df)

if current_tsb > 10:   status_color, status_label = "#4CAF50", "🟢 In Forma"
elif current_tsb > -5: status_color, status_label = "#FF9800", "🟡 Stabile"
elif current_tsb > -20:status_color, status_label = "#FF5722", "🟠 Affaticato"
else:                   status_color, status_label = "#F44336", "🔴 Sovraccarico"

last_act = df.iloc[-1]

# ============================================================
# HEADER
# ============================================================
athlete_name = athlete.get("firstname", "Atleta")
st.markdown(f"""
<div class="mob-header">
    <h1>🏆 Elite Coach</h1>
    <p>Ciao {athlete_name} · {status_label}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DETTAGLIO ATTIVITÀ (intercetta tutto)
# ============================================================
if st.session_state.selected_act_id is not None:
    _sid = st.session_state.selected_act_id
    _srow_df = df[df["id"] == _sid] if "id" in df.columns else pd.DataFrame()
    if _srow_df.empty:
        try:
            _srow_df = df.iloc[[int(_sid)]]
        except Exception:
            pass

    if not _srow_df.empty:
        row  = _srow_df.iloc[0]
        s    = get_sport_info(row["type"], row.get("name",""))
        m    = format_metrics(row)
        z_n, z_c, z_l = get_zone_for_activity(row, u["fc_max"])

        # Back button
        st.markdown('<div class="sec-pad">', unsafe_allow_html=True)
        if st.button("← Indietro", use_container_width=True):
            st.session_state.selected_act_id = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Header attività
        st.markdown(f"""
        <div class="mob-card" style="border-left:4px solid {s['color']}">
            <div style="font-size:22px;font-weight:900;color:{s['color']}">{s['icon']} {row['name']}</div>
            <div style="font-size:13px;color:#777;margin-top:4px">
                {row['start_date'].strftime('%A %d %B %Y · %H:%M')} · {s['label']}
            </div>
            <div style="margin-top:8px">
                <span class="zone-chip" style="background:{z_c}22;color:{z_c};border:1px solid {z_c}44">{z_l}</span>
                <span style="font-size:13px;color:#555;margin-left:8px">TSS: <b>{row['tss']:.0f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metriche principali — 2x3 grid
        st.markdown("""<div class="mob-card">
        <div class="mob-card-title">📊 Statistiche</div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("📏 Distanza", m["dist_str"])
        c2.metric("⏱️ Durata",   m["dur_str"])
        c3.metric("⚡ Ritmo",    m["pace_str"])
        c4, c5, c6 = st.columns(3)
        c4.metric("⛰️ D+",       m["elev"])
        c5.metric("❤️ FC avg",   m["hr_avg"])
        c6.metric("⚡ Watt",     m["watts"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Mappa
        poly = row.get("map", {})
        poly = poly.get("summary_polyline") if isinstance(poly, dict) else None
        if poly:
            mobj = draw_map(poly, height=220)
            if mobj:
                st.markdown('<div style="margin:8px 12px 0">', unsafe_allow_html=True)
                st_folium(mobj, width=None, height=220, key="det_map")
                st.markdown('</div>', unsafe_allow_html=True)

        # Zone FC
        st.markdown("""<div class="mob-card">
        <div class="mob-card-title">❤️ Zone FC</div>""", unsafe_allow_html=True)
        hr_avg = row.get("average_heartrate")
        fc_max = u["fc_max"]
        _hr_zones = [(1,"#4CAF50","Z1",0.00,0.60),(2,"#8BC34A","Z2",0.60,0.70),
                     (3,"#FFC107","Z3",0.70,0.80),(4,"#FF9800","Z4",0.80,0.90),
                     (5,"#F44336","Z5",0.90,1.00)]
        _hz = st.columns(5)
        for _zi, (_zn,_zc,_zl,_zlo,_zhi) in enumerate(_hr_zones):
            _blo, _bhi = int(fc_max*_zlo), int(fc_max*_zhi)
            _cur = pd.notna(hr_avg) and fc_max>0 and _zlo <= hr_avg/fc_max < _zhi
            _bg  = "20" if _cur else "0a"
            _brd = "3" if _cur else "1"
            _active = f"<div style='font-size:9px;font-weight:900;color:{_zc}'>← qui</div>" if _cur else ""
            _hz[_zi].markdown(
                f"<div style='background:{_zc}{_bg};border:{_brd}px solid {_zc}cc;"
                f"border-radius:8px;padding:6px 4px;text-align:center'>"
                f"<div style='font-size:10px;font-weight:700;color:{_zc}'>{_zl}</div>"
                f"<div style='font-size:9px;color:#444'>{_blo}–{_bhi}</div>"
                f"{_active}</div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Analisi AI
        st.markdown("""<div class="mob-card">
        <div class="mob-card-title">🤖 Analisi Coach</div>""", unsafe_allow_html=True)
        _aid = str(row.get("id", str(row["start_date"])))
        _ck  = f"mob_ai_{_aid}"
        if _ck in st.session_state:
            st.markdown(f'<div class="ai-box">{st.session_state[_ck]}</div>',
                        unsafe_allow_html=True)
        else:
            if st.button("🤖 Chiedi al Coach", use_container_width=True, type="primary"):
                with st.spinner("Il coach sta analizzando..."):
                    _ctx = (f"Sport: {s['label']}. Distanza: {m['dist_str']}. "
                            f"Durata: {m['dur_str']}. Passo: {m['pace_str']}. "
                            f"Dislivello: {m['elev']}. FC media: {m['hr_avg']}. "
                            f"Watt: {m['watts']}. TSS: {row['tss']:.0f}. "
                            f"CTL: {current_ctl:.0f}, TSB: {current_tsb:.0f}. "
                            f"Forma attuale: {status_label}.")
                    _res = ai_deep(_ctx + "\n\nSei un coach d'élite. "
                        "Commenta questa sessione in 3 paragrafi brevi: "
                        "qualità dell'allenamento, zone di lavoro, "
                        "suggerimento concreto per la prossima sessione.")
                    st.session_state[_ck] = _res
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        render_bottom_nav()
        st.stop()

# ============================================================
# ── MENU: DASHBOARD ──────────────────────────────────────────
# ============================================================
if st.session_state.mob_menu == "dashboard":

    # ── Metriche CTL / ATL / TSB ──
    st.markdown("""<div class="mob-card">
    <div class="mob-card-title">📈 Stato Forma</div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ctl_color = "#4CAF50" if current_ctl > 60 else "#FF9800" if current_ctl > 40 else "#F44336"
        st.markdown(f"""<div class="big-metric">
            <div class="val" style="color:{ctl_color}">{current_ctl:.0f}</div>
            <div class="lbl">CTL Fitness</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        tsb_color = "#4CAF50" if current_tsb > 5 else "#FF9800" if current_tsb > -15 else "#F44336"
        st.markdown(f"""<div class="big-metric">
            <div class="val" style="color:{tsb_color}">{current_tsb:+.0f}</div>
            <div class="lbl">TSB Forma</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        atl_color = "#FF9800" if current_atl > current_ctl else "#4CAF50"
        st.markdown(f"""<div class="big-metric">
            <div class="val" style="color:{atl_color}">{current_atl:.0f}</div>
            <div class="lbl">ATL Fatica</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="text-align:center;margin-top:8px">
        <span class="status-badge" style="background:{status_color}22;color:{status_color};
              border:1px solid {status_color}44">{status_label}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafico fitness 30gg ──
    st.markdown('<div class="mob-card"><div class="mob-card-title">📊 Fitness ultimi 30 giorni</div>',
                unsafe_allow_html=True)
    chart_df = pd.DataFrame({
        "Fitness (CTL)": ctl_daily,
        "Fatica (ATL)":  atl_daily,
        "Forma (TSB)":   tsb_daily,
    }).dropna().tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Fitness (CTL)"],
        name="CTL", line=dict(color="#4CAF50", width=2)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Fatica (ATL)"],
        name="ATL", line=dict(color="#F44336", width=2)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Forma (TSB)"],
        name="TSB", line=dict(color="#2196F3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)"))
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig.update_layout(
        height=200, margin=dict(l=0,r=0,t=8,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2, font_size=11),
        xaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickfont_size=10),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickfont_size=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Ultima attività ──
    s   = get_sport_info(last_act["type"], last_act.get("name",""))
    m   = format_metrics(last_act)
    z_n, z_c, z_l = get_zone_for_activity(last_act, u["fc_max"])

    st.markdown(f"""
    <div class="act-card" style="border-left-color:{s['color']}">
        <div class="mob-card-title">🕐 Ultima Attività</div>
        <div class="act-title">{s['icon']} {last_act['name']}</div>
        <div class="act-meta">{last_act['start_date'].strftime('%d %b %Y · %H:%M')} ·
            <span class="zone-chip" style="background:{z_c}22;color:{z_c}">{z_l}</span>
        </div>
        <div class="act-pills">
            <span class="act-pill">📏 <b>{m['dist_str']}</b></span>
            <span class="act-pill">⏱️ <b>{m['dur_str']}</b></span>
            <span class="act-pill">⚡ <b>{m['pace_str']}</b></span>
            <span class="act-pill">⛰️ <b>{m['elev']}</b></span>
            <span class="act-pill">❤️ <b>{m['hr_avg']} bpm</b></span>
            <span class="act-pill">📊 TSS <b>{last_act['tss']:.0f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mappa ultima attività
    poly = last_act.get("map", {})
    poly = poly.get("summary_polyline") if isinstance(poly, dict) else None
    if poly:
        mobj = draw_map(poly)
        if mobj:
            st.markdown('<div style="margin:0 12px">', unsafe_allow_html=True)
            st_folium(mobj, width=None, height=200, key="dash_map")
            st.markdown('</div>', unsafe_allow_html=True)

    # Pulsante dettaglio
    st.markdown('<div class="sec-pad" style="margin-top:8px">', unsafe_allow_html=True)
    _act_id = last_act.get("id", last_act.name)
    if st.button("🔍 Dettaglio completo", use_container_width=True, type="secondary"):
        st.session_state.selected_act_id = _act_id
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ── MENU: FITNESS ────────────────────────────────────────────
# ============================================================
elif st.session_state.mob_menu == "fitness":
    st.markdown('<div class="sec-pad"><h3 style="margin:12px 0 4px">💪 Stato Fisico</h3></div>',
                unsafe_allow_html=True)

    # Stato forma
    st.markdown(f"""<div class="mob-card">
    <div class="mob-card-title">📈 Performance Management</div>
    <div style="display:flex;justify-content:space-around;margin:8px 0">
        <div class="big-metric">
            <div class="val" style="color:#4CAF50">{current_ctl:.0f}</div>
            <div class="lbl">CTL Fitness</div>
        </div>
        <div class="big-metric">
            <div class="val" style="color:#F44336">{current_atl:.0f}</div>
            <div class="lbl">ATL Fatica</div>
        </div>
        <div class="big-metric">
            <div class="val" style="color:#2196F3">{current_tsb:+.0f}</div>
            <div class="lbl">TSB Forma</div>
        </div>
    </div>
    <div style="text-align:center">
        <span class="status-badge" style="background:{status_color}22;color:{status_color}">
            {status_label}
        </span>
    </div>
    </div>""", unsafe_allow_html=True)

    # PMC grafico 60gg
    st.markdown('<div class="mob-card"><div class="mob-card-title">📊 PMC — 60 giorni</div>',
                unsafe_allow_html=True)
    pmc_df = pd.DataFrame({
        "CTL": ctl_daily, "ATL": atl_daily, "TSB": tsb_daily,
    }).dropna().tail(60)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df["CTL"],
        name="CTL", line=dict(color="#4CAF50", width=2.5)))
    fig2.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df["ATL"],
        name="ATL", line=dict(color="#F44336", width=2.5)))
    fig2.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df["TSB"],
        name="TSB", line=dict(color="#2196F3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.07)"))
    fig2.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig2.update_layout(
        height=220, margin=dict(l=0,r=0,t=8,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25, font_size=11),
        xaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickfont_size=10),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickfont_size=10),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # VO2max
    if vo2max_val:
        if vo2max_val >= 65:   vc, vl = "#9C27B0", "🏆 Élite"
        elif vo2max_val >= 55: vc, vl = "#4CAF50", "🥇 Molto Buono"
        elif vo2max_val >= 45: vc, vl = "#2196F3", "🥈 Buono"
        elif vo2max_val >= 35: vc, vl = "#FF9800", "🥉 Media"
        else:                   vc, vl = "#F44336", "📈 Da Migliorare"
        st.markdown(f"""<div class="mob-card">
        <div class="mob-card-title">🔬 VO2max Stimato</div>
        <div style="display:flex;align-items:center;gap:16px;padding:4px 0">
            <div style="font-size:48px;font-weight:900;color:{vc}">{vo2max_val:.0f}</div>
            <div>
                <div style="font-size:12px;color:#888">ml/kg/min · Formula Daniels</div>
                <div style="font-size:16px;font-weight:700;color:{vc};margin-top:4px">{vl}</div>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

    # Volume settimanale ultimi 8 settimane
    st.markdown('<div class="mob-card"><div class="mob-card-title">📅 Volume settimanale (TSS)</div>',
                unsafe_allow_html=True)
    weekly = tss_daily.resample("W").sum().tail(8)
    fig3   = go.Figure(go.Bar(
        x=[d.strftime("S%V") for d in weekly.index],
        y=weekly.values,
        marker_color="#2196F3",
        marker_opacity=0.8,
    ))
    fig3.update_layout(
        height=160, margin=dict(l=0,r=0,t=4,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_size=10),
        yaxis=dict(gridcolor="rgba(0,0,0,0.05)", tickfont_size=10),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Analisi fisiologica AI
    st.markdown('<div class="mob-card"><div class="mob-card-title">🤖 Analisi Fisiologica</div>',
                unsafe_allow_html=True)
    if "mob_analisi_fisica" in st.session_state:
        st.markdown(f'<div class="ai-box">{st.session_state["mob_analisi_fisica"]}</div>',
                    unsafe_allow_html=True)
    else:
        if st.button("🤖 Genera Analisi Completa", use_container_width=True, type="primary"):
            with st.spinner("Analisi fisiologica in corso..."):
                _last7 = df[df["start_date"] >= df["start_date"].max() - timedelta(days=7)]
                _ctx = (
                    f"Atleta con CTL={current_ctl:.0f}, ATL={current_atl:.0f}, "
                    f"TSB={current_tsb:.0f}, stato={status_label}. "
                    f"Ultimi 7 giorni: {len(_last7)} sessioni, "
                    f"TSS totale={_last7['tss'].sum():.0f}. "
                    f"VO2max stimato: {vo2max_val or 'N/D'} ml/kg/min. "
                    f"FTP: {u.get('ftp',200)} W, FC max: {u['fc_max']} bpm."
                )
                _res = ai_deep(_ctx + "\n\nSei un coach sportivo d'élite. "
                    "Fai un'analisi fisiologica completa in 4 paragrafi: "
                    "1) stato forma attuale, 2) tendenza allenamento, "
                    "3) punti di forza e debolezze, 4) raccomandazioni per i prossimi 7 giorni. "
                    "Sii specifico con i numeri.")
                st.session_state["mob_analisi_fisica"] = _res
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ── MENU: STORICO ────────────────────────────────────────────
# ============================================================
elif st.session_state.mob_menu == "storico":
    st.markdown('<div class="sec-pad"><h3 style="margin:12px 0 4px">📅 Storico Attività</h3></div>',
                unsafe_allow_html=True)

    # Barra di ricerca
    search_query = st.text_input("🔍 Cerca attività", placeholder="Nome, sport, data...",
                                  label_visibility="collapsed")

    # Vista toggle: calendario o lista
    view_toggle = st.radio("Vista", ["📅 Calendario", "📋 Lista"], horizontal=True,
                            label_visibility="collapsed")

    if view_toggle == "📋 Lista":
        # ── Lista attività ──
        df_filtered = df.copy()
        if search_query:
            q = search_query.lower()
            df_filtered = df_filtered[
                df_filtered["name"].str.lower().str.contains(q, na=False) |
                df_filtered["type"].str.lower().str.contains(q, na=False) |
                df_filtered["start_date"].dt.strftime("%d/%m/%Y %b").str.lower().str.contains(q, na=False)
            ]

        df_show = df_filtered.iloc[::-1].head(50)
        st.markdown(f'<div class="sec-pad" style="color:#888;font-size:12px;margin:4px 0">'
                    f'{len(df_filtered)} attività trovate</div>', unsafe_allow_html=True)

        for _, row in df_show.iterrows():
            s   = get_sport_info(row["type"], row.get("name",""))
            m   = format_metrics(row)
            z_n, z_c, z_l = get_zone_for_activity(row, u["fc_max"])
            _act_id = row.get("id", row.name)

            st.markdown(f"""
            <div class="act-card" style="border-left-color:{s['color']}">
                <div class="act-title">{s['icon']} {row['name']}</div>
                <div class="act-meta">{row['start_date'].strftime('%d %b %Y · %H:%M')} ·
                    <span class="zone-chip" style="background:{z_c}22;color:{z_c}">{z_l}</span>
                </div>
                <div class="act-pills">
                    <span class="act-pill">📏 <b>{m['dist_str']}</b></span>
                    <span class="act-pill">⏱️ <b>{m['dur_str']}</b></span>
                    <span class="act-pill">⚡ <b>{m['pace_str']}</b></span>
                    <span class="act-pill">⛰️ <b>{m['elev']}</b></span>
                    <span class="act-pill">📊 TSS <b>{row['tss']:.0f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔍 Dettaglio", key=f"list_{_act_id}", use_container_width=True):
                st.session_state.selected_act_id = _act_id
                st.rerun()

    else:
        # ── Calendario mensile semplificato ──
        now = datetime.now()
        if "cal_year"  not in st.session_state: st.session_state.cal_year  = now.year
        if "cal_month" not in st.session_state: st.session_state.cal_month = now.month

        cy, cm = st.session_state.cal_year, st.session_state.cal_month

        # Navigazione mese
        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("◀", use_container_width=True):
                if cm == 1: cm, cy = 12, cy-1
                else:       cm -= 1
                st.session_state.cal_month, st.session_state.cal_year = cm, cy
                st.rerun()
        with nav2:
            st.markdown(f"<div style='text-align:center;font-weight:700;font-size:16px;padding:8px'>"
                        f"{['','Gennaio','Febbraio','Marzo','Aprile','Maggio','Giugno','Luglio','Agosto','Settembre','Ottobre','Novembre','Dicembre'][cm]} {cy}"
                        f"</div>", unsafe_allow_html=True)
        with nav3:
            if st.button("▶", use_container_width=True):
                if cm == 12: cm, cy = 1, cy+1
                else:        cm += 1
                st.session_state.cal_month, st.session_state.cal_year = cm, cy
                st.rerun()

        # Filtro ricerca nel calendario
        df_cal = df.copy()
        if search_query:
            q = search_query.lower()
            df_cal = df_cal[
                df_cal["name"].str.lower().str.contains(q, na=False) |
                df_cal["type"].str.lower().str.contains(q, na=False)
            ]

        # Build calendario
        month_acts = df_cal[
            (df_cal["start_date"].dt.year == cy) &
            (df_cal["start_date"].dt.month == cm)
        ]
        acts_by_day = {}
        for _, row in month_acts.iterrows():
            day = row["start_date"].day
            acts_by_day.setdefault(day, []).append(row)

        weeks = cal_module.monthcalendar(cy, cm)
        day_names = ["L","M","M","G","V","S","D"]

        # Header giorni
        header_html = '<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin:8px 12px 4px">'
        for d in day_names:
            color = "#e94560" if d == "D" else "#888"
            header_html += f'<div style="text-align:center;font-size:11px;font-weight:700;color:{color}">{d}</div>'
        header_html += '</div>'
        st.markdown(header_html, unsafe_allow_html=True)

        # Celle calendario
        for week in weeks:
            row_html = '<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin:0 12px 2px">'
            for day in week:
                if day == 0:
                    row_html += '<div style="min-height:56px"></div>'
                else:
                    acts = acts_by_day.get(day, [])
                    is_today = (day == now.day and cm == now.month and cy == now.year)
                    bg = "#E3F2FD" if is_today else "#fff"
                    brd = "2px solid #1565C0" if is_today else "1px solid #eee"
                    cell_content = f'<div style="font-size:12px;font-weight:700;color:#333;line-height:1;margin-bottom:3px">{day}</div>'
                    for a in acts[:2]:
                        si = get_sport_info(a["type"], a.get("name",""))
                        km = a["distance"] / 1000
                        tss = a.get("tss", 0)
                        cell_content += (
                            f'<div class="cal-day-act">'
                            f'<div class="cal-dot" style="background:{si["color"]}"></div>'
                            f'<span style="font-size:9px;color:#333;line-height:1">'
                            f'{si["icon"]} {km:.0f}k<br>'
                            f'<span style="color:#999">{tss:.0f}tss</span></span>'
                            f'</div>'
                        )
                    if len(acts) > 2:
                        cell_content += f'<div style="font-size:9px;color:#999">+{len(acts)-2}</div>'
                    row_html += (f'<div style="background:{bg};border:{brd};border-radius:8px;'
                                 f'padding:5px 4px;min-height:56px;overflow:hidden">'
                                 f'{cell_content}</div>')
            row_html += '</div>'
            st.markdown(row_html, unsafe_allow_html=True)

        # Lista attività del mese selezionato
        if not month_acts.empty:
            st.markdown(f'<div class="mob-card" style="margin-top:8px">'
                        f'<div class="mob-card-title">{len(month_acts)} attività questo mese · '
                        f'TSS totale: {month_acts["tss"].sum():.0f}</div>',
                        unsafe_allow_html=True)
            for _, row in month_acts.iloc[::-1].iterrows():
                s_  = get_sport_info(row["type"], row.get("name",""))
                m_  = format_metrics(row)
                _id = row.get("id", row.name)
                st.markdown(f"""
                <div style="padding:6px 0;border-bottom:1px solid #f0f0f0;display:flex;
                            align-items:center;gap:8px">
                    <div style="font-size:20px">{s_['icon']}</div>
                    <div style="flex:1">
                        <div style="font-size:13px;font-weight:700;color:#1a1a1a">{row['name']}</div>
                        <div style="font-size:11px;color:#888">{row['start_date'].strftime('%d %b · %H:%M')}
                            · {m_['dist_str']} · {m_['dur_str']}</div>
                    </div>
                    <div style="font-size:12px;color:#888;font-weight:600">{row['tss']:.0f} TSS</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("→", key=f"cal_det_{_id}"):
                    st.session_state.selected_act_id = _id
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ── MENU: COACH CHAT ─────────────────────────────────────────
# ============================================================
elif st.session_state.mob_menu == "chat":
    st.markdown('<div class="sec-pad"><h3 style="margin:12px 0 4px">💬 Coach Chat</h3></div>',
                unsafe_allow_html=True)

    if _ai_sdk_mode is None:
        st.warning("⚠️ Aggiungi GOOGLE_API_KEY nei Secrets per abilitare il Coach AI.")
    else:
        # Contesto coach
        _last7 = df[df["start_date"] >= df["start_date"].max() - timedelta(days=7)]
        _ctx_sys = (
            f"Sei un coach sportivo d'élite specializzato in ciclismo, trail running e sci alpinismo. "
            f"Rispondi sempre in italiano, in modo conciso e pratico. "
            f"Dati atleta: CTL={current_ctl:.0f}, ATL={current_atl:.0f}, TSB={current_tsb:.0f}, "
            f"Forma={status_label}. FTP={u.get('ftp',200)}W, FC_max={u['fc_max']}bpm. "
            f"Ultimi 7gg: {len(_last7)} sessioni, TSS={_last7['tss'].sum():.0f}. "
            f"VO2max stimato: {vo2max_val or 'N/D'} ml/kg/min. "
            f"Ultima attività: {last_act['name']} ({last_act['start_date'].strftime('%d/%m')})."
        )

        # Mostra messaggi
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-label">Tu</div>'
                            f'<div class="chat-user">{msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label">🤖 Coach</div>'
                            f'<div class="chat-ai">{msg["content"]}</div>',
                            unsafe_allow_html=True)

        # Pulsanti suggerimento veloci
        st.markdown('<div class="sec-pad" style="margin-top:8px">', unsafe_allow_html=True)
        quick_prompts = [
            "Come sto fisicamente?",
            "Cosa fare oggi?",
            "Piano per questa settimana",
            "Analizza il mio allenamento",
        ]
        qc = st.columns(2)
        for i, qp in enumerate(quick_prompts):
            with qc[i % 2]:
                if st.button(qp, use_container_width=True, key=f"qp_{i}"):
                    st.session_state.messages.append({"role":"user","content":qp})
                    with st.spinner("Il coach risponde..."):
                        history = "\n".join([
                            f"{'Atleta' if m['role']=='user' else 'Coach'}: {m['content']}"
                            for m in st.session_state.messages[-6:]
                        ])
                        res = ai_generate(f"{_ctx_sys}\n\nConversazione:\n{history}")
                        st.session_state.messages.append({"role":"assistant","content":res})
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Input chat
        if prompt := st.chat_input("Scrivi al tuo coach..."):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.spinner("Il coach risponde..."):
                history = "\n".join([
                    f"{'Atleta' if m['role']=='user' else 'Coach'}: {m['content']}"
                    for m in st.session_state.messages[-8:]
                ])
                res = ai_generate(f"{_ctx_sys}\n\nConversazione:\n{history}")
                st.session_state.messages.append({"role":"assistant","content":res})
            st.rerun()

        if st.session_state.messages:
            st.markdown('<div class="sec-pad">', unsafe_allow_html=True)
            if st.button("🗑️ Nuova conversazione", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ── MENU: PROFILO ────────────────────────────────────────────
# ============================================================
elif st.session_state.mob_menu == "profilo":
    st.markdown('<div class="sec-pad"><h3 style="margin:12px 0 4px">👤 Profilo</h3></div>',
                unsafe_allow_html=True)

    # Info atleta Strava
    if athlete:
        st.markdown(f"""<div class="mob-card">
        <div class="mob-card-title">👤 Strava</div>
        <div style="font-size:17px;font-weight:700">{athlete.get('firstname','')} {athlete.get('lastname','')}</div>
        <div style="font-size:13px;color:#777;margin-top:4px">
            📍 {athlete.get('city','N/D')}, {athlete.get('country','N/D')}
        </div>
        <div style="font-size:13px;color:#777">
            🏃 {df[df['type'].isin(['Run','TrailRun'])].shape[0]} corse ·
            🚴 {df[df['type'].isin(['Ride','VirtualRide','MountainBikeRide'])].shape[0]} pedalate ·
            🎿 {df[df['type'].isin(['BackcountrySki','AlpineSki'])].shape[0]} uscite sci
        </div>
        </div>""", unsafe_allow_html=True)

    # Parametri fisiologici
    st.markdown('<div class="mob-card"><div class="mob-card-title">⚙️ Parametri Fisiologici</div>',
                unsafe_allow_html=True)

    with st.form("profilo_form"):
        peso  = st.number_input("Peso (kg)",    value=float(u["peso"]),   min_value=40.0, max_value=150.0, step=0.5)
        fc_min = st.number_input("FC riposo",   value=int(u["fc_min"]),   min_value=30,   max_value=80)
        fc_max = st.number_input("FC massima",  value=int(u["fc_max"]),   min_value=150,  max_value=230)
        ftp    = st.number_input("FTP (Watt)",  value=int(u.get("ftp",200)), min_value=50, max_value=500)
        if st.form_submit_button("💾 Salva", use_container_width=True, type="primary"):
            st.session_state.user_data = {
                "peso": peso, "fc_min": fc_min, "fc_max": fc_max, "ftp": ftp
            }
            st.success("✅ Profilo aggiornato!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Stats generali
    total_km  = df["distance"].sum() / 1000
    total_hrs = df["moving_time"].sum() / 3600
    total_elev = df["total_elevation_gain"].sum()
    st.markdown(f"""<div class="mob-card">
    <div class="mob-card-title">📊 Statistiche Totali</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="big-metric">
            <div class="val" style="color:#2196F3;font-size:28px">{total_km:.0f}</div>
            <div class="lbl">km totali</div>
        </div>
        <div class="big-metric">
            <div class="val" style="color:#4CAF50;font-size:28px">{total_hrs:.0f}</div>
            <div class="lbl">ore totali</div>
        </div>
        <div class="big-metric">
            <div class="val" style="color:#FF9800;font-size:28px">{total_elev/1000:.0f}k</div>
            <div class="lbl">m dislivello</div>
        </div>
        <div class="big-metric">
            <div class="val" style="color:#9C27B0;font-size:28px">{len(df)}</div>
            <div class="lbl">attività</div>
        </div>
    </div>
    </div>""", unsafe_allow_html=True)

    # Cache GSheet status
    if _gsheet_ok:
        last_sync = gsheet_get_last_sync()
        sync_str  = last_sync.strftime("%d/%m/%Y %H:%M") if last_sync else "Mai"
        st.markdown(f"""<div class="mob-card">
        <div class="mob-card-title">💾 Cache Google Sheets</div>
        <div style="font-size:13px;color:#555">Ultimo sync: <b>{sync_str}</b></div>
        <div style="font-size:12px;color:#888;margin-top:4px">
            Aggiornamento automatico ogni 24h. Dati salvati in modo persistente.
        </div>
        </div>""", unsafe_allow_html=True)
        if st.button("🔄 Forza sync da Strava", use_container_width=True):
            with st.spinner("Sincronizzazione in corso..."):
                raw_new = load_all_from_strava(access_token)
                st.session_state.activities_cache = raw_new
                gsheet_save_activities(raw_new)
            st.success(f"✅ {len(raw_new)} attività sincronizzate e salvate!")
            st.rerun()

    # Logout
    st.markdown('<div class="sec-pad" style="margin-top:12px">', unsafe_allow_html=True)
    if st.button("🚪 Disconnetti da Strava", use_container_width=True):
        for k in ["strava_token_info","activities_cache","activities_last_ts",
                  "activities_token","gsheet_loaded"]:
            st.session_state[k] = {} if k == "strava_token_info" else ([] if "cache" in k else False if "loaded" in k else 0 if "ts" in k else "")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# BOTTOM NAV (sempre visibile)
# ============================================================
render_bottom_nav()
