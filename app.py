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
REDIRECT_URI = "https://elite-ai-coach-mobile.streamlit.app"  # ← aggiorna con il tuo URL mobile

# ============================================================
# FETCH GPX ON-DEMAND DA STRAVA
# ============================================================
@st.cache_data(ttl=3600)
def fetch_activity_details_from_strava(activity_id: int, access_token: str):
    """
    Fetcha i dettagli completi di un'attività da Strava.
    Include: polyline completo, splits, dati della traccia.
    Cache: 1 ora (evita chiamate ripetute per la stessa attività).
    """
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"https://www.strava.com/api/v3/activities/{activity_id}"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data
    except Exception as e:
        st.error(f"❌ Errore nel fetch da Strava: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def fetch_activity_streams_from_strava(activity_id: int, access_token: str, 
                                       stream_types: str = "latlng,altitude,cadence,heartrate,velocity_smooth,power"):
    """
    Fetcha gli stream GPS ad alta precisione da Strava.
    Include: coordinate lat/lng, altitudine, cadenza, FC, velocità, potenza.
    Cache: 1 ora
    """
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
        
        params = {
            "keys": stream_types,
            "key_by_type": True
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        st.warning(f"⚠️ Stream GPS non disponibile: {str(e)}")
        return None

def get_secret(key):
    return st.secrets.get(key) or os.getenv(key)

CLIENT_ID     = get_secret("STRAVA_CLIENT_ID")
CLIENT_SECRET = get_secret("STRAVA_CLIENT_SECRET")
GEMINI_KEY    = get_secret("GOOGLE_API_KEY")
GROK_KEY      = get_secret("GROK_API_KEY") or get_secret("XAI_API_KEY") or ""
MAPBOX_TOKEN  = get_secret("MAPBOX_TOKEN") or ""
GSHEET_ID     = get_secret("GSHEET_ID") or ""
GSHEET_CREDS     = get_secret("GSHEET_CREDENTIALS") or ""
INTERVALS_API_KEY    = get_secret("INTERVALS_API_KEY") or ""
INTERVALS_ATHLETE_ID = get_secret("INTERVALS_ATHLETE_ID") or "0"

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
  /* ── PALETTE PREMIUM ── */
  :root {
    --navy:   #0F2744;
    --blue:   #1A56DB;
    --blue2:  #1E40AF;
    --accent: #F97316;
    --green:  #16A34A;
    --red:    #DC2626;
    --amber:  #D97706;
    --gray50: #F8FAFC;
    --gray100:#F1F5F9;
    --gray200:#E2E8F0;
    --gray400:#94A3B8;
    --gray700:#334155;
    --gray900:#0F172A;
  }

  /* ── RESET ── */
  * { box-sizing: border-box; }
  html, body, [data-testid="stAppViewContainer"] {
      background: var(--gray100) !important;
  }
  [data-testid="collapsedControl"] { display: none !important; }
  section[data-testid="stSidebar"]  { display: none !important; }
  .block-container { padding: 0 0 220px 0 !important; max-width: 100% !important; }

  /* ── FADE-IN — NON su stVerticalBlock (rompe position:fixed della nav) ── */
  /* Applicato solo al mob-header che è il primo elemento visibile */
  .mob-header {
      animation: fadeInPage 0.2s ease-out both;
  }
  @keyframes fadeInPage {
      from { opacity: 0; transform: translateY(4px); }
      to   { opacity: 1; transform: translateY(0); }
  }

  /* ── TOOLBAR HIDE ── */
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  .stDeployButton,
  #MainMenu, footer, header { display: none !important; visibility: hidden !important; }

  /* ── HEADER PREMIUM ── */
  .mob-header {
      background: linear-gradient(135deg, var(--navy) 0%, var(--blue2) 100%);
      color: white;
      padding: 14px 16px 12px;
      margin-bottom: 0;
      position: sticky;
      top: 0;
      z-index: 998;
      box-shadow: 0 2px 12px rgba(15,39,68,0.25);
  }

  /* ── HERO CARD ── */
  .hero-card {
      background: linear-gradient(145deg, var(--navy) 0%, #1e3a5f 100%);
      border-radius: 20px;
      padding: 20px 16px 16px;
      margin: 12px 12px 0;
      box-shadow: 0 8px 32px rgba(15,39,68,0.22);
      position: relative;
      overflow: hidden;
  }
  .hero-card::before {
      content: "";
      position: absolute;
      top: -40px; right: -40px;
      width: 140px; height: 140px;
      border-radius: 50%;
      background: rgba(255,255,255,0.04);
  }
  .hero-card::after {
      content: "";
      position: absolute;
      bottom: -20px; left: 40px;
      width: 80px; height: 80px;
      border-radius: 50%;
      background: rgba(255,255,255,0.03);
  }

  /* ── CARD GENERICA ── */
  .mob-card {
      background: #ffffff;
      border-radius: 18px;
      padding: 16px;
      margin: 10px 12px 0;
      box-shadow: 0 1px 6px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
  }
  .mob-card-title {
      font-size: 11px; font-weight: 700;
      color: var(--gray400);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 10px;
  }

  /* ── METRICHE ── */
  .big-metric { text-align: center; padding: 8px; }
  .big-metric .val { font-size: 36px; font-weight: 900; line-height: 1; }
  .big-metric .lbl { font-size: 11px; color: var(--gray400); font-weight: 600; margin-top: 2px; }

  /* ── ATTIVITÀ CARD ── */
  .act-card {
      border-radius: 16px;
      padding: 13px 14px 12px;
      margin: 8px 12px 0;
      border-left: 4px solid #ccc;
      background: #fff;
      box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  }
  .act-title { font-size: 15px; font-weight: 700; color: var(--gray900); margin-bottom: 3px; }
  .act-meta  { font-size: 11px; color: var(--gray400); margin-bottom: 7px; }
  .act-pills { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
  .act-pill {
      background: var(--gray100); border-radius: 20px;
      padding: 3px 9px; font-size: 11px; color: var(--gray700); font-weight: 500;
  }
  .act-pill b { color: var(--gray900); }
  .zone-chip {
      display: inline-block; border-radius: 20px;
      padding: 2px 9px; font-size: 10px; font-weight: 700;
  }

  /* ── BRIEFING AI — carta warm ── */
  .briefing-card {
      background: #FAFAF7;
      border-radius: 16px;
      padding: 16px;
      margin: 8px 0 0;
      border: 1px solid #EEF0E8;
      font-size: 14px; line-height: 1.75; color: var(--gray900);
  }
  .briefing-signature {
      margin-top: 12px; padding-top: 10px;
      border-top: 1px solid #E8EAE0;
      font-size: 11px; color: var(--gray400);
      display: flex; align-items: center; gap: 6px;
  }

  /* ── AI BOX ── */
  .ai-box {
      background: var(--gray50);
      border-left: 3px solid var(--blue);
      border-radius: 0 14px 14px 0;
      padding: 14px 16px; margin: 8px 12px;
      color: var(--gray900); font-size: 14px; line-height: 1.75;
  }

  /* ── CHAT BUBBLES con avatar ── */
  .chat-row {
      display: flex; align-items: flex-end; gap: 8px;
      margin: 6px 12px;
  }
  .chat-row.user { flex-direction: row-reverse; }
  .chat-avatar {
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px; font-weight: 700; flex-shrink: 0;
  }
  .chat-avatar.coach {
      background: linear-gradient(135deg, var(--navy), var(--blue2));
      color: white; font-size: 16px;
  }
  .chat-avatar.user-av {
      background: var(--gray200); color: var(--gray700); font-size: 12px;
  }
  .chat-bubble-wrap { display: flex; flex-direction: column; max-width: calc(100% - 48px); }
  .chat-bubble-wrap.user { align-items: flex-end; }
  .chat-user {
      background: linear-gradient(135deg, var(--blue) 0%, var(--blue2) 100%);
      color: white; border-radius: 18px 18px 4px 18px;
      padding: 10px 14px; font-size: 14px; line-height: 1.5;
      box-shadow: 0 2px 8px rgba(26,86,219,0.25);
  }
  .chat-ai {
      background: #ffffff; color: var(--gray900);
      border-radius: 18px 18px 18px 4px;
      padding: 10px 14px; font-size: 14px; line-height: 1.5;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
  }
  .chat-ts {
      font-size: 10px; color: var(--gray400);
      margin-top: 3px; padding: 0 2px;
  }

  /* ── QUICK PROMPTS card-style ── */
  .qp-card {
      background: #fff; border: 1px solid var(--gray200);
      border-radius: 14px; padding: 10px 12px;
      cursor: pointer; transition: all 0.15s;
  }
  .qp-card:hover { border-color: var(--blue); box-shadow: 0 2px 8px rgba(26,86,219,0.12); }
  .qp-card-icon { font-size: 20px; margin-bottom: 4px; }
  .qp-card-title { font-size: 13px; font-weight: 700; color: var(--gray900); }
  .qp-card-sub   { font-size: 11px; color: var(--gray400); margin-top: 2px; }

  /* ── BRIEFING sezioni card ── */
  .brief-section {
      border-radius: 14px; padding: 12px 14px; margin-bottom: 8px;
  }
  .brief-section-icon {
      font-size: 20px; margin-bottom: 6px; display: block;
  }
  .brief-section-title {
      font-size: 10px; font-weight: 800; text-transform: uppercase;
      letter-spacing: 0.7px; margin-bottom: 6px;
  }
  .brief-section-body {
      font-size: 14px; line-height: 1.7;
  }
  /* Highlight numeri nel briefing */
  .brief-num {
      background: rgba(26,86,219,0.1); color: var(--blue);
      border-radius: 6px; padding: 1px 5px;
      font-weight: 700; font-size: 13px;
  }

  /* ── BUTTONS ── */
  div[data-testid="stButton"] > button {
      min-height: 46px !important; font-size: 14px !important;
      border-radius: 12px !important; font-weight: 600 !important;
      transition: transform 0.1s, box-shadow 0.1s !important;
  }
  div[data-testid="stButton"] > button:active { transform: scale(0.97) !important; }
  div[data-testid="stButton"] > button[kind="primary"] {
      background: linear-gradient(135deg, var(--blue) 0%, var(--blue2) 100%) !important;
      border: none !important;
      box-shadow: 0 4px 12px rgba(26,86,219,0.30) !important;
  }

  /* ── INPUTS ── */
  div[data-testid="stTextInput"] input,
  div[data-testid="stChatInput"] textarea {
      font-size: 16px !important; min-height: 48px !important; border-radius: 12px !important;
  }
  div[data-testid="stSlider"] { padding: 0 12px; }

  /* ── MISC ── */
  .sec-pad { padding: 0 12px; }
  .mob-divider { height: 1px; background: var(--gray200); margin: 12px; }
  .status-badge {
      display: inline-block; border-radius: 20px;
      padding: 4px 14px; font-size: 13px; font-weight: 700;
  }
  .cal-day-act {
      background: #fff; border-radius: 10px;
      padding: 4px 6px; margin: 2px 0;
      font-size: 11px; display: flex; align-items: center; gap: 4px;
  }
  .cal-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)
# fine CSS injection

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
    # Fix: sci alpino su pista → dislivello sempre zero (funivia conta come salita GPS)
    if a_type == "AlpineSki":
        elev = 0
    hr_avg = row.get("average_heartrate")
    hr_max = row.get("max_heartrate")
    watts  = row.get("average_watts")
    cals   = row.get("kilojoules")  # Strava fornisce kJ per ciclismo
    if cals is None or not pd.notna(cals):
        # Stima calorie da FC se disponibile, altrimenti da MET
        if pd.notna(hr_avg) and hr_avg > 0:
            peso = 75  # default, verrà sovrascritto nel contesto
            cals_est = (time / 60) * (0.014 * float(hr_avg) - 0.05) * peso / 60 * 4.184
        else:
            cals_est = None
        cals_str = f"{cals_est:.0f} kcal" if cals_est else "—"
    else:
        cals_str = f"{float(cals)*0.239:.0f} kcal"  # kJ → kcal
    hrs    = int(time // 3600)
    mins   = int((time % 3600) // 60)
    secs   = int(time % 60)
    dur_str = f"{hrs}h {mins:02d}m" if hrs > 0 else f"{mins}m {secs:02d}s"
    if a_type in ("Ride", "VirtualRide", "MountainBikeRide"):
        speed   = dist / (time / 3600) if time > 0 else 0
        pace_str = f"{speed:.1f} km/h"
    elif a_type in ("AlpineSki", "BackcountrySki"):
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
        "cals":      cals_str,
        "dist_km":   dist,
        "time_sec":  time,
        "elev_raw":  elev,
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
def calc_tss_vectorized(df: pd.DataFrame, u: dict) -> pd.Series:
    """Calcola TSS per tutto il DataFrame in modo vettorizzato — molto più veloce di apply()."""
    fc_max = u["fc_max"]
    fc_min = u["fc_min"]
    ftp    = u.get("ftp", 200)
    dur    = df["moving_time"] / 60
    hr     = df["average_heartrate"].fillna(0)
    watts  = df["average_watts"].fillna(0)

    # TSS da FC
    has_hr  = (hr > 0) & (fc_max > fc_min)
    intens  = ((hr - fc_min) / (fc_max - fc_min)).clip(0, 1)
    tss_hr  = (dur * hr * intens) / (fc_max * 60) * 100

    # TSS da watt
    has_w   = (watts > 0) & (ftp > 0) & ~has_hr
    IF_     = watts / ftp
    tss_w   = (df["moving_time"] * watts * IF_) / (ftp * 3600) * 100

    # Fallback: stima da durata
    tss_fallback = dur * 0.4

    return pd.Series(
        np.where(has_hr, tss_hr,
        np.where(has_w,  tss_w,
                         tss_fallback)),
        index=df.index
    )

def assign_zones_vectorized(df: pd.DataFrame, fc_max: int):
    """Assegna zone FC in modo vettorizzato — evita 3 apply() separati."""
    hr_pct = df["average_heartrate"] / fc_max if fc_max > 0 else pd.Series(np.nan, index=df.index)

    conditions = [
        hr_pct < 0.60,
        hr_pct < 0.70,
        hr_pct < 0.80,
        hr_pct < 0.90,
        hr_pct >= 0.90,
    ]
    zone_nums   = [1, 2, 3, 4, 5]
    zone_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]
    zone_labels = ["Z1", "Z2", "Z3", "Z4", "Z5"]

    has_hr = df["average_heartrate"].notna() & (fc_max > 0)

    nums   = np.select(conditions, zone_nums,   default=0)
    colors = np.select(conditions, zone_colors, default="#9E9E9E")
    labels = np.select(conditions, zone_labels, default="N/A")

    # Dove non c'è FC → default
    return (
        pd.Series(np.where(has_hr, nums,   0),         index=df.index),
        pd.Series(np.where(has_hr, colors, "#9E9E9E"), index=df.index),
        pd.Series(np.where(has_hr, labels, "N/A"),     index=df.index),
    )

def compute_fitness(df):
    daily = df.groupby(df["start_date"].dt.date)["tss"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    ctl   = daily.ewm(span=42, adjust=False).mean()
    atl   = daily.ewm(span=7,  adjust=False).mean()
    tsb   = ctl - atl
    # Vettorizzato — evita il lambda per riga
    df_ts     = pd.to_datetime(df["start_date"].dt.date.astype(str))
    ctl_mapped = df_ts.map(ctl)
    atl_mapped = df_ts.map(atl)
    tsb_mapped = df_ts.map(tsb)
    return ctl_mapped, atl_mapped, tsb_mapped, ctl, atl, tsb, daily


def calc_vo2max_estimate(df_sorted, ftp=200, peso=75):
    """
    VO2max stimato — prende il massimo tra stima da corsa e stima da bici.
    
    Corsa: formula Daniels da velocità e durata.
    Bici: formula da FTP — VO2max ≈ (FTP_W/kg × 10.8 + 7) / 1.064
    Source: Coggan, "Training and Racing with a Power Meter"
    """
    best_run = 0.0
    best_bike = 0.0

    # ── Stima da corsa (Daniels) ──
    runs = df_sorted[
        (df_sorted["type"].isin(["Run","TrailRun"])) &
        (df_sorted["distance"] >= 5000)
    ]
    if not runs.empty:
        time_min = runs["moving_time"] / 60
        dist_m   = runs["distance"]
        valid    = time_min > 0
        if valid.any():
            time_min = time_min[valid]
            dist_m   = dist_m[valid]
            vel  = dist_m / time_min
            pct  = (0.8 + 0.1894393 * np.exp(-0.012778 * time_min)
                       + 0.2989558 * np.exp(-0.1932605 * time_min))
            vo2  = -4.60 + 0.182258 * vel + 0.000104 * vel**2
            arr  = np.where(pct > 0, vo2 / pct, 0)
            best_run = float(np.max(arr))

    # ── Stima da bici (FTP/peso) ──
    # Usa le uscite bici reali se hanno potenza, altrimenti usa FTP da profilo
    rides = df_sorted[
        df_sorted["type"].isin(["Ride","VirtualRide","MountainBikeRide"]) &
        df_sorted["average_watts"].notna()
    ]
    if not rides.empty and peso > 0:
        # Best 20-min power proxy: usa i top 10% per watt medi
        top_watts = rides["average_watts"].quantile(0.90)
        ftp_est   = top_watts * 0.95  # FTP ≈ 95% del best 60min ≈ 97% del best 20min
        wkg       = max(ftp_est, ftp) / peso  # prende il più alto tra stimato e dichiarato
        best_bike = (wkg * 10.8 + 7) / 1.064
    elif ftp > 0 and peso > 0:
        # Fallback: solo FTP dal profilo
        wkg       = ftp / peso
        best_bike = (wkg * 10.8 + 7) / 1.064

    best = max(best_run, best_bike)
    return round(best, 1) if best > 0 else None

# ============================================================
# ============================================================
# INTERVALS.ICU — Fetch CTL/ATL/TSB reali
# ============================================================
@st.cache_data(ttl=900)
def fetch_intervals_wellness(athlete_id: str, api_key: str, date_str: str):
    """Wellness intervals.icu per una data: ctl, atl, tsb, rampRate, weight, hrv..."""
    if not api_key:
        return None
    try:
        import base64 as _b64
        token = _b64.b64encode(f"API_KEY:{api_key}".encode()).decode()
        url   = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness/{date_str}"
        resp  = requests.get(url, headers={"Authorization": f"Basic {token}"}, timeout=8)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=900)
def fetch_intervals_wellness_range(athlete_id: str, api_key: str, oldest: str, newest: str):
    """Lista wellness per range di date — usata per sparkline storici."""
    if not api_key:
        return []
    try:
        import base64 as _b64
        token = _b64.b64encode(f"API_KEY:{api_key}".encode()).decode()
        url   = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness"
        resp  = requests.get(url, headers={"Authorization": f"Basic {token}"},
                             params={"oldest": oldest, "newest": newest}, timeout=10)
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=900)
def fetch_intervals_activities_page(athlete_id: str, api_key: str, oldest: str, newest: str):
    """Fetch una pagina di attività da intervals.icu."""
    if not api_key or not athlete_id:
        return []
    try:
        import base64 as _b64
        token = _b64.b64encode(f"API_KEY:{api_key}".encode()).decode()
        url   = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities"
        resp  = requests.get(url, headers={"Authorization": f"Basic {token}"},
                             params={"oldest": oldest, "newest": newest}, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        st.session_state["_icu_acts_error"] = f"HTTP {resp.status_code}: {resp.text[:120]}"
        return []
    except Exception as e:
        st.session_state["_icu_acts_error"] = str(e)
        return []


def load_all_from_intervals(athlete_id: str, api_key: str):
    """
    Scarica tutta la storia attività da intervals.icu a finestre di 6 mesi.
    Mappa i campi intervals.icu → struttura compatibile con il DataFrame esistente.
    Campi chiave intervals.icu:
      id, start_date_local, type, moving_time, distance, total_elevation_gain,
      average_heartrate, max_heartrate, average_watts, icu_weighted_avg_watts (NP),
      icu_training_load (TSS reale!), icu_ftp, name, calories, average_cadence,
      icu_power_zone (zona dominante), max_watts
    """
    from datetime import date
    all_acts = []
    # Parti da oggi e vai indietro a finestre da 6 mesi finché non trovi attività
    end   = datetime.now(timezone.utc).date()
    # Vai indietro fino a 10 anni (ma smette prima se non trova nulla)
    max_years = 10
    empty_windows = 0

    for _ in range(max_years * 2):  # finestre da 6 mesi
        start = end - timedelta(days=180)
        batch = fetch_intervals_activities_page(
            athlete_id, api_key,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d")
        )
        if batch:
            all_acts.extend(batch)
            empty_windows = 0
        else:
            empty_windows += 1
            if empty_windows >= 2:  # 2 finestre vuote consecutive → fine storia
                break
        end = start - timedelta(days=1)
        if end.year < 2010:
            break

    return all_acts


def normalize_intervals_activity(act: dict) -> dict:
    """
    Converte un'attività intervals.icu nel formato usato dal DataFrame.
    Preserva i campi Strava-compatibili e aggiunge quelli intervals.icu.
    """
    # TSS reale da intervals.icu (icu_training_load è il TSS calcolato da intervals)
    tss_real = act.get("icu_training_load") or act.get("training_load")

    # Normalized Power da intervals.icu
    np_watts = act.get("icu_weighted_avg_watts")

    # FTP usato per questa attività da intervals.icu
    act_ftp  = act.get("icu_ftp")

    # Mappa tipo sport: intervals.icu usa gli stessi tipi Strava
    sport_type = act.get("type", "Workout")

    return {
        # Campi base compatibili con il df esistente
        "id":                    act.get("id", ""),
        "name":                  act.get("name", ""),
        "type":                  sport_type,
        "start_date_local":      act.get("start_date_local", ""),
        "moving_time":           act.get("moving_time") or 0,
        "distance":              act.get("distance") or 0,
        "total_elevation_gain":  act.get("total_elevation_gain") or 0,
        "average_heartrate":     act.get("average_heartrate"),
        "max_heartrate":         act.get("max_heartrate"),
        "average_watts":         act.get("average_watts"),
        "max_watts":             act.get("max_watts"),
        "average_cadence":       act.get("average_cadence"),
        "calories":              act.get("calories"),
        "kilojoules":            act.get("kilojoules"),
        "suffer_score":          act.get("suffer_score"),
        # Campi extra intervals.icu
        "icu_training_load":     tss_real,       # TSS reale
        "icu_weighted_avg_watts": np_watts,       # NP
        "icu_ftp":               act_ftp,         # FTP al momento dell'attività
        "icu_power_zone":        act.get("icu_power_zone"),
        # Polyline GPS: non disponibile da intervals.icu → verrà fetchato da Strava on-demand
        "map":                   act.get("map", {}),
        # Strava ID per fetch GPS on-demand (intervals.icu lo espone come external_id o strava_id)
        "strava_id":             act.get("strava_id") or act.get("external_id", ""),
        # Flag sorgente
        "_source": "intervals.icu",
    }


def get_intervals_fitness(athlete_id: str, api_key: str):
    """
    Entry point principale. Ritorna dict con ctl, atl, tsb, ramp_rate, history (30gg).
    history: lista di dict {date, ctl, atl, tsb} per sparkline.
    Ritorna None se intervals.icu non è configurato o irraggiungibile.
    """
    if not api_key:
        return None
    today = datetime.now(timezone.utc).date()
    # Prova oggi, poi ieri (il dato di oggi può non essere ancora disponibile di mattina)
    for delta in (0, 1):
        d = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
        w = fetch_intervals_wellness(athlete_id, api_key, d)
        if w and (w.get("ctl") is not None or w.get("fitnessScore") is not None):
            break
    else:
        return None

    ctl = w.get("ctl") or w.get("fitnessScore")
    atl = w.get("atl") or w.get("fatigueScore")
    tsb = w.get("form") or ((ctl - atl) if (ctl and atl) else None)
    ramp = w.get("rampRate")

    # Storico 30 giorni per sparkline
    oldest_str = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    today_str  = today.strftime("%Y-%m-%d")
    raw_history = fetch_intervals_wellness_range(athlete_id, api_key, oldest_str, today_str)
    history = []
    for rec in raw_history:
        c = rec.get("ctl") or rec.get("fitnessScore")
        a = rec.get("atl") or rec.get("fatigueScore")
        t = rec.get("form") or ((c - a) if (c and a) else None)
        if c is not None:
            history.append({
                "date": rec.get("id", ""),
                "ctl":  float(c),
                "atl":  float(a) if a else 0.0,
                "tsb":  float(t) if t else 0.0,
            })
    return {
        "ctl":        float(ctl),
        "atl":        float(atl) if atl else 0.0,
        "tsb":        float(tsb) if tsb else 0.0,
        "ramp_rate":  float(ramp) if ramp else None,
        "wellness":   w,
        "history":    history,
        "source":     "intervals.icu",
    }


# GOOGLE SHEETS — Cache persistente
# ============================================================
def _get_gsheet_client():
    """Restituisce (client, sheet) se configurato — con cache in session_state per evitare
    riconnessioni OAuth ad ogni chiamata (era il collo di bottiglia principale GSheet)."""
    if not GSHEET_ID or not GSHEET_CREDS:
        st.session_state["_gsheet_err"] = "❌ GSHEET_ID o GSHEET_CREDENTIALS mancanti nei Secrets"
        return None, None

    # ── Cache del client nella sessione ──
    _cached = st.session_state.get("_gsheet_client_cache")
    if _cached is not None:
        return _cached  # già connesso, ritorna subito

    try:
        import gspread
        from google.oauth2.service_account import Credentials
        try:
            creds_dict = json.loads(GSHEET_CREDS)
        except json.JSONDecodeError as je:
            st.session_state["_gsheet_err"] = f"❌ GSHEET_CREDENTIALS non è un JSON valido: {je}"
            return None, None
        if creds_dict.get("type") != "service_account":
            st.session_state["_gsheet_err"] = f"❌ JSON non è un service_account (type={creds_dict.get('type')})"
            return None, None
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        try:
            sheet = client.open_by_key(GSHEET_ID)
        except gspread.exceptions.SpreadsheetNotFound:
            st.session_state["_gsheet_err"] = (
                f"❌ Sheet non trovato (ID: {GSHEET_ID[:12]}...). "
                f"Controlla GSHEET_ID e condividi con: {creds_dict.get('client_email','?')}"
            )
            return None, None
        except Exception as e:
            st.session_state["_gsheet_err"] = f"❌ Errore apertura sheet: {e}"
            return None, None
        st.session_state["_gsheet_err"]          = None
        st.session_state["_gsheet_email"]        = creds_dict.get("client_email","?")
        st.session_state["_gsheet_client_cache"] = (client, sheet)  # salva in cache
        return client, sheet
    except ImportError:
        st.session_state["_gsheet_err"] = "❌ Libreria 'gspread' non installata."
        return None, None
    except Exception as e:
        st.session_state["_gsheet_err"] = f"❌ Errore generico GSheet: {e}"
        return None, None

def gsheet_load_activities() -> list:
    """Carica attività dal Google Sheet. get_all_values è ~3x più veloce di get_all_records."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return []
    try:
        ws = sheet.worksheet("activities")
        rows = ws.get_all_values()
        if len(rows) < 2:
            return []
        headers = rows[0]
        return [dict(zip(headers, row)) for row in rows[1:] if any(row)]
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
        st.session_state["_gsheet_save_ok"]  = True
        st.session_state["_gsheet_save_rows"] = len(rows) - 1
        return True
    except Exception as e:
        st.session_state["_gsheet_save_ok"]  = False
        st.session_state["_gsheet_save_err"] = str(e)
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

def gsheet_save_profile(profile: dict):
    """Salva profilo utente (peso, FC, FTP, età) nel tab 'user_profile'."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return
    try:
        try:
            ws = sheet.worksheet("user_profile")
        except Exception:
            ws = sheet.add_worksheet(title="user_profile", rows=20, cols=5)
        rows = [["key","value"]] + [[k, str(v)] for k,v in profile.items()]
        ws.clear()
        ws.update(rows, "A1")
    except Exception:
        pass

def gsheet_load_profile() -> dict:
    """Carica profilo utente dal tab 'user_profile'. Ritorna {} se non trovato."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return {}
    try:
        ws = sheet.worksheet("user_profile")
        records = ws.get_all_records()
        profile = {}
        type_map = {"peso": float, "fc_min": int, "fc_max": int, "ftp": int, "eta": int}
        for r in records:
            k, v = r.get("key",""), r.get("value","")
            if k and v != "":
                try:
                    profile[k] = type_map.get(k, str)(v)
                except Exception:
                    profile[k] = v
        return profile
    except Exception:
        return {}

def gsheet_save_conversations(messages: list):
    """Salva le ultime 50 conversazioni nel tab 'conversations'."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return
    try:
        try:
            ws = sheet.worksheet("conversations")
        except Exception:
            ws = sheet.add_worksheet(title="conversations", rows=200, cols=4)
        rows = [["timestamp","role","content","session"]]
        session_id = datetime.now().strftime("%Y%m%d")
        for m in messages[-50:]:
            rows.append([
                datetime.now().isoformat(),
                m.get("role",""),
                str(m.get("content",""))[:2000],
                session_id,
            ])
        ws.clear()
        ws.update(rows, "A1")
    except Exception:
        pass

def gsheet_load_conversations() -> list:
    """Carica conversazioni salvate. Ritorna lista messaggi."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return []
    try:
        ws = sheet.worksheet("conversations")
        records = ws.get_all_records()
        return [{"role": r["role"], "content": r["content"]}
                for r in records if r.get("role") and r.get("content")]
    except Exception:
        return []

def gsheet_save_coach_memory(memory: dict):
    """Salva la memoria del coach (fatti chiave sull'atleta) nel tab 'coach_memory'."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return
    try:
        try:
            ws = sheet.worksheet("coach_memory")
        except Exception:
            ws = sheet.add_worksheet(title="coach_memory", rows=50, cols=3)
        rows = [["key", "value", "updated_at"]]
        for k, v in memory.items():
            rows.append([k, str(v), datetime.now().isoformat()])
        ws.clear()
        ws.update(rows, "A1")
    except Exception:
        pass

def gsheet_load_coach_memory() -> dict:
    """Carica la memoria del coach. Ritorna dict {key: value}."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return {}
    try:
        ws = sheet.worksheet("coach_memory")
        records = ws.get_all_records()
        return {r["key"]: r["value"] for r in records if r.get("key")}
    except Exception:
        return {}

def gsheet_save_weekly_plan(plan: str):
    """Salva il piano settimanale nel tab 'weekly_plan'."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return
    try:
        try:
            ws = sheet.worksheet("weekly_plan")
        except Exception:
            ws = sheet.add_worksheet(title="weekly_plan", rows=20, cols=3)
        ws.clear()
        ws.update([["generated_at","plan"],
                   [datetime.now().isoformat(), plan]], "A1")
    except Exception:
        pass

def gsheet_load_weekly_plan() -> tuple:
    """Carica piano settimanale. Ritorna (piano_str, data_generazione) o (None, None)."""
    _, sheet = _get_gsheet_client()
    if sheet is None:
        return None, None
    try:
        ws   = sheet.worksheet("weekly_plan")
        data = ws.get_all_values()
        if len(data) < 2:
            return None, None
        ts   = data[1][0] if data[1][0] else None
        plan = data[1][1] if len(data[1]) > 1 else None
        dt   = datetime.fromisoformat(ts) if ts else None
        return plan, dt
    except Exception:
        return None, None

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
# MAPPA 2D + 3D
# ============================================================
def build_map3d_html(encoded_polyline, mapbox_token, sport_type="", elev_gain=0,
                     dist_km=0, dur_str="", height=340) -> str:
    """Mappa Mapbox 3D con fullscreen, layer switcher, pitch, fly-to e stats overlay."""
    if not encoded_polyline or not mapbox_token:
        return None
    try:
        import json as _j
        pts    = polyline.decode(encoded_polyline)
        coords = [[lon, lat] for lat, lon in pts]
        if len(coords) < 2:
            return None
        clon  = sum(c[0] for c in coords) / len(coords)
        clat  = sum(c[1] for c in coords) / len(coords)
        geoj  = _j.dumps({"type":"Feature","properties":{},
                           "geometry":{"type":"LineString","coordinates":coords}})
        start_j = _j.dumps(coords[0])
        end_j   = _j.dumps(coords[-1])
        line_color = ("#FF4B4B" if sport_type in ("Run","TrailRun") else
                      "#4FC3F7" if sport_type in ("BackcountrySki","AlpineSki") else "#2196F3")
        _dist_str = f"{dist_km:.1f} km" if dist_km else ""
        _elev_str = f"{elev_gain:.0f} m D+" if elev_gain else ""

        html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  html,body{{width:100%;height:{height}px;background:#000;overflow:hidden;font-family:-apple-system,sans-serif}}
  #map{{width:100%;height:100%}}
  .mapboxgl-ctrl-group{{background:rgba(0,0,0,0.5)!important;border:none!important}}
  .mapboxgl-ctrl-group button{{background:rgba(255,255,255,0.12)!important;color:#fff!important}}

  /* Pannello controlli layer + pitch — in basso a sinistra */
  #ctrl-panel{{
    position:absolute;bottom:12px;left:12px;z-index:10;
    display:flex;flex-direction:column;gap:6px;
  }}
  .ctrl-row{{display:flex;gap:6px;align-items:center}}
  .ctrl-btn{{
    background:rgba(0,0,0,0.65);color:#fff;border:1px solid rgba(255,255,255,0.25);
    border-radius:8px;padding:6px 10px;font-size:12px;font-weight:600;
    cursor:pointer;backdrop-filter:blur(6px);transition:background 0.15s;white-space:nowrap;
  }}
  .ctrl-btn:hover,.ctrl-btn.active{{background:rgba(21,101,192,0.85);border-color:#42A5F5}}
  .ctrl-btn.sm{{padding:5px 8px;font-size:11px}}

  /* Bottone ✕ fullscreen — grande, in alto a destra, sempre visibile */
  #fs-btn{{
    position:absolute;top:12px;right:12px;z-index:20;
    background:rgba(0,0,0,0.7);color:#fff;
    border:2px solid rgba(255,255,255,0.4);
    border-radius:12px;padding:10px 18px;
    font-size:18px;font-weight:800;
    cursor:pointer;backdrop-filter:blur(8px);
    transition:background 0.15s;
    display:none;
  }}
  body.is-fullscreen #fs-btn{{display:block}}
  #fs-btn:hover{{background:rgba(198,40,40,0.85);border-color:#ef9a9a}}

  /* Bottone ⛶ entra fullscreen — piccolo, in alto a destra */
  #enter-fs-btn{{
    position:absolute;top:12px;right:12px;z-index:20;
    background:rgba(0,0,0,0.60);color:#fff;
    border:1px solid rgba(255,255,255,0.25);
    border-radius:10px;padding:7px 13px;font-size:14px;font-weight:700;
    cursor:pointer;backdrop-filter:blur(6px);transition:background 0.15s;
  }}
  body.is-fullscreen #enter-fs-btn{{display:none}}
  #enter-fs-btn:hover{{background:rgba(21,101,192,0.8)}}

  /* Stats overlay in alto a sinistra */
  #stats-overlay{{
    position:absolute;top:10px;left:10px;z-index:10;
    background:rgba(0,0,0,0.55);backdrop-filter:blur(6px);
    border-radius:10px;padding:8px 12px;color:#fff;
    font-size:12px;line-height:1.6;border:1px solid rgba(255,255,255,0.15);
    display:none;
  }}
  #stats-overlay.visible{{display:block}}
  #stats-overlay b{{color:#64B5F6}}

  /* Fullscreen */
  body.is-fullscreen{{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:99999}}
  body.is-fullscreen #map{{height:100vh}}
</style></head><body>
<div id="map"></div>

<!-- Stats overlay top-left -->
<div id="stats-overlay">
  {'<b>📏</b> ' + _dist_str + '<br>' if _dist_str else ''}{'<b>⛰</b> ' + _elev_str + '<br>' if _elev_str else ''}{'<b>⏱</b> ' + dur_str if dur_str else ''}
</div>

<!-- Bottone entra fullscreen (visibile quando NON siamo in FS) -->
<button id="enter-fs-btn" onclick="toggleFS()">⛶ Schermo intero</button>

<!-- Bottone ESCI fullscreen (visibile solo quando siamo in FS) -->
<button id="fs-btn" onclick="toggleFS()">✕ Esci</button>

<!-- Pannello controlli: layer + pitch + stats -->
<div id="ctrl-panel">
  <!-- Layer switcher -->
  <div class="ctrl-row">
    <button class="ctrl-btn sm active" id="l-sat"   onclick="setLayer('satellite-streets-v12','l-sat')">🛰 Satellite</button>
    <button class="ctrl-btn sm"        id="l-out"   onclick="setLayer('outdoors-v12','l-out')">🗺 Topo</button>
    <button class="ctrl-btn sm"        id="l-dark"  onclick="setLayer('dark-v11','l-dark')">🌑 Dark</button>
    <button class="ctrl-btn sm"        id="l-str"   onclick="setLayer('streets-v12','l-str')">🏙 Street</button>
  </div>
  <!-- Pitch + stats -->
  <div class="ctrl-row">
    <button class="ctrl-btn sm" onclick="setPitch(0)"  title="Vista piatta">📐 0°</button>
    <button class="ctrl-btn sm" onclick="setPitch(45)" title="Vista 45°">🏔 45°</button>
    <button class="ctrl-btn sm" onclick="setPitch(70)" title="Vista immersiva">🎮 70°</button>
    <button class="ctrl-btn sm" onclick="toggleStats()" title="Statistiche">📊 Stats</button>
  </div>
</div>

<script>
mapboxgl.accessToken = "{mapbox_token}";
const map = new mapboxgl.Map({{
  container:"map", style:"mapbox://styles/mapbox/satellite-streets-v12",
  center:[{clon},{clat}], zoom:12, pitch:55, bearing:0, antialias:true
}});
map.addControl(new mapboxgl.NavigationControl(),"top-right");

const routeCoords = {_j.dumps(coords)};
const bounds = routeCoords.reduce((b,c)=>b.extend(c), new mapboxgl.LngLatBounds(routeCoords[0],routeCoords[0]));

function addRoute(){{
  if(map.getSource("route")){{map.removeLayer("route-glow");map.removeLayer("route");map.removeSource("route");}}
  map.addSource("route",{{type:"geojson",data:{geoj}}});
  map.addLayer({{id:"route-glow",type:"line",source:"route",
    layout:{{"line-join":"round","line-cap":"round"}},
    paint:{{"line-color":"{line_color}","line-width":7,"line-opacity":0.3,"line-blur":5}}}});
  map.addLayer({{id:"route",type:"line",source:"route",
    layout:{{"line-join":"round","line-cap":"round"}},
    paint:{{"line-color":"{line_color}","line-width":3.5,"line-opacity":0.95}}}});
}}

map.on("load",()=>{{
  map.addSource("dem",{{type:"raster-dem",url:"mapbox://mapbox.mapbox-terrain-dem-v1",tileSize:512}});
  map.setTerrain({{"source":"dem","exaggeration":1.5}});
  map.addLayer({{id:"sky",type:"sky",paint:{{"sky-type":"atmosphere","sky-atmosphere-sun":[0,60],"sky-atmosphere-sun-intensity":15}}}});
  addRoute();
  new mapboxgl.Marker({{color:"#4CAF50",scale:0.9}}).setLngLat({start_j}).addTo(map);
  new mapboxgl.Marker({{color:"#F44336",scale:0.9}}).setLngLat({end_j}).addTo(map);
  map.fitBounds(bounds,{{padding:50,duration:800,pitch:55}});
}});

// Layer switcher
var _layerBtns={{'l-sat':'satellite-streets-v12','l-out':'outdoors-v12','l-dark':'dark-v11','l-str':'streets-v12'}};
var _curLayer='satellite-streets-v12';
function setLayer(styleId, btnId){{
  if(styleId===_curLayer) return;
  _curLayer=styleId;
  Object.keys(_layerBtns).forEach(id=>document.getElementById(id).classList.remove('active'));
  document.getElementById(btnId).classList.add('active');
  map.setStyle('mapbox://styles/mapbox/'+styleId);
  map.once('style.load',()=>{{
    try{{map.addSource("dem",{{type:"raster-dem",url:"mapbox://mapbox.mapbox-terrain-dem-v1",tileSize:512}});}}catch(e){{}}
    try{{map.setTerrain({{"source":"dem","exaggeration":1.5}});}}catch(e){{}}
    addRoute();
  }});
}}

// Pitch control
function setPitch(p){{map.easeTo({{pitch:p,duration:400}});}}

// Stats toggle
function toggleStats(){{
  document.getElementById('stats-overlay').classList.toggle('visible');
}}

// Fullscreen
var _isFS=false;
function toggleFS(){{
  _isFS=!_isFS;
  if(_isFS){{
    document.body.classList.add('is-fullscreen');
    if(document.documentElement.requestFullscreen) document.documentElement.requestFullscreen();
  }} else {{
    document.body.classList.remove('is-fullscreen');
    if(document.exitFullscreen) document.exitFullscreen();
  }}
  setTimeout(()=>map.resize(),200);
}}
document.addEventListener('fullscreenchange',()=>{{
  if(!document.fullscreenElement && _isFS){{
    _isFS=false;
    document.body.classList.remove('is-fullscreen');
    setTimeout(()=>map.resize(),200);
  }}
}});
</script></body></html>"""
        return html
    except Exception:
        return None

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
# Modelli flash in ordine di preferenza (più recente → più vecchio)
_FLASH_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-latest",
]
_PRO_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]
# Costruisce lista modelli disponibili via autodiscovery
def _discover_available_models() -> dict:
    """Scopre i modelli disponibili e ritorna dict {id: label}."""
    found = {"auto": "🤖 Auto (consigliato)"}
    if _ai_sdk_mode == "new":
        try:
            discovered = []
            for m in _ai_client.models.list():
                name = m.name.replace("models/", "")
                if "gemini" in name and "embedding" not in name and "aqa" not in name:
                    discovered.append(name)
        except Exception:
            discovered = []
        # Unisci con statica
        static = [
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.5-flash-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
        all_models = list(dict.fromkeys(static + discovered))  # dedup, mantieni ordine
        icons = {"3.1-flash": "🚀", "2.5-pro": "🧠", "2.5-flash": "⚡",
                 "2.0-flash-lite": "💨", "2.0-flash": "⚡", "1.5-pro": "📚", "1.5-flash": "💨"}
        for m in all_models:
            icon = next((v for k,v in icons.items() if k in m), "🔷")
            found[m] = f"{icon} {m}"
    elif _ai_sdk_mode == "old":
        try:
            for m in _ai_client.list_models():
                if "generateContent" in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if "gemini" in name:
                        found[name] = f"🔷 {name}"
        except Exception:
            for m in ["gemini-1.5-flash","gemini-1.5-pro","gemini-1.5-flash-8b"]:
                found[m] = f"🔷 {m}"
    elif _ai_sdk_mode == "grok":
        for m in ["grok-3","grok-3-fast","grok-3-mini","grok-2-1212"]:
            icons = {"grok-3-fast":"⚡","grok-3-mini":"🚀"}
            found[m] = f"{icons.get(m,'🧠')} {m}"
    return found

_ALL_MODELS_LABELS = _discover_available_models()

def _is_quota_error(e) -> bool:
    s = str(e).lower()
    return any(k in s for k in ["quota", "429", "resource_exhausted", "rate limit", "exceeded"])

def ai_generate(prompt: str, max_tokens: int = 1500) -> str:
    if _ai_sdk_mode is None:
        return "⚠️ Nessun provider AI configurato. Aggiungi GOOGLE_API_KEY nei Secrets."
    last_err = ""
    # Usa modello preferito se impostato
    _pref = st.session_state.get("ai_model_pref", "auto")
    _flash_list = (_FLASH_MODELS if _pref == "auto" or _pref not in _ALL_MODELS_LABELS
                   else [_pref] + [m for m in _FLASH_MODELS if m != _pref])
    if _ai_sdk_mode == "new":
        for model in _flash_list:
            try:
                resp = _ai_client.models.generate_content(model=model, contents=prompt)
                return resp.text
            except Exception as e:
                last_err = str(e)
                if _is_quota_error(e):
                    continue  # prova il prossimo modello
                break
        if _is_quota_error(Exception(last_err)):
            return ("⚠️ Quota AI esaurita per oggi su tutti i modelli flash. "
                    "Riprova domani oppure aggiungi una nuova GOOGLE_API_KEY "
                    "(puoi creare chiavi gratuite su aistudio.google.com).")
        return f"⚠️ Errore AI: {last_err}"
    elif _ai_sdk_mode == "old":
        for m in ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-8b"]:
            try:
                model = _ai_client.GenerativeModel(m)
                return model.generate_content(prompt).text
            except Exception as e:
                last_err = str(e)
                if _is_quota_error(e): continue
                break
        return f"⚠️ Errore AI: {last_err}"
    elif _ai_sdk_mode == "grok":
        try:
            resp = _ai_client.chat.completions.create(
                model="grok-3-fast",
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ Errore AI: {e}"
    return "⚠️ Provider AI non riconosciuto."

def ai_deep(prompt: str) -> str:
    """Analisi approfondita — tenta Pro poi fallback su Flash."""
    _pref = st.session_state.get("ai_model_pref", "auto")
    _pro_list = (_PRO_MODELS if _pref == "auto" or _pref not in _ALL_MODELS_LABELS
                 else [_pref] + [m for m in _PRO_MODELS if m != _pref])
    if _ai_sdk_mode == "new":
        for model in _pro_list:
            try:
                resp = _ai_client.models.generate_content(model=model, contents=prompt)
                return resp.text
            except Exception as e:
                if _is_quota_error(e): continue
                break
    return ai_generate(prompt, max_tokens=2000)

# ============================================================
# SPARKLINE SVG helper
# ============================================================
def make_sparkline_svg(values, color, width=80, height=32, show_zero_line=False) -> str:
    """Genera un mini SVG sparkline da una lista di valori."""
    if not values or len(values) < 2:
        return ""
    vals = list(values)
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    pad = 3
    def _x(i):  return round(pad + i * (width - 2*pad) / (len(vals)-1), 1)
    def _y(v):  return round(height - pad - (v - mn) / rng * (height - 2*pad), 1)
    pts = " ".join(f"{_x(i)},{_y(v)}" for i,v in enumerate(vals))
    zero_line = ""
    if show_zero_line and mn < 0 < mx:
        zy = _y(0)
        zero_line = f'<line x1="{pad}" y1="{zy}" x2="{width-pad}" y2="{zy}" stroke="#ccc" stroke-width="0.8" stroke-dasharray="2,2"/>'
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'{zero_line}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{_x(len(vals)-1)}" cy="{_y(vals[-1])}" r="2.5" fill="{color}"/>'
            f'</svg>')

# ============================================================
# CONTESTO AI — builder functions
# ============================================================

def build_activity_context(row, df, u, current_ctl, current_atl, current_tsb,
                            status_label, window_days=14) -> str:
    """Contesto ricco per analisi singola attività con storico 14gg."""
    s   = get_sport_info(row["type"], row.get("name",""))
    m   = format_metrics(row)
    act_date = row["start_date"]
    ftp      = u.get("ftp", 200)
    is_bike  = row["type"] in ("Ride","VirtualRide","MountainBikeRide")
    watts    = row.get("average_watts")

    # Attività nei window_days precedenti
    cutoff = act_date - pd.Timedelta(days=window_days)
    prev   = df[(df["start_date"] >= cutoff) & (df["start_date"] < act_date)]
    prev_lines = ""
    for _, pr in prev.tail(12).iterrows():
        ps = get_sport_info(pr["type"])
        pm = format_metrics(pr)
        prev_lines += ("  - " + pr["start_date"].strftime("%d/%m") + " " + ps["icon"] +
                       " " + str(pr["name"])[:30] + ": " +
                       pm["dist_str"] + " " + pm["dur_str"] +
                       " TSS=" + str(round(pr["tss"],0)) + "\n")

    act_ctl = float(row.get("ctl", current_ctl))
    act_atl = float(row.get("atl", current_atl))
    act_tsb = float(row.get("tsb", current_tsb))

    watt_line = ""
    if is_bike and pd.notna(watts) and watts and watts > 0 and ftp > 0:
        IF_  = watts / ftp
        tssp = (row["moving_time"] * watts * IF_) / (ftp * 3600) * 100
        est  = " (stimati Strava)" if not row.get("device_watts", False) else ""
        watt_line = "Watt: " + str(round(watts)) + "W IF=" + str(round(IF_,2)) + " TSS_pot=" + str(round(tssp)) + est + "\n"

    lines = [
        "=== ATTIVITÀ ===",
        "Data: " + act_date.strftime("%d/%m/%Y %H:%M"),
        "Sport: " + s["label"],
        "Nome: " + str(row["name"]),
        "Distanza: " + m["dist_str"],
        "Durata: " + m["dur_str"],
        "Ritmo/Vel: " + m["pace_str"],
        "Dislivello: " + m["elev"],
        "FC media: " + m["hr_avg"] + " bpm | FC max: " + m["hr_max"] + " bpm",
        watt_line.strip() if watt_line else None,
        "TSS: " + str(round(row["tss"],1)),
        "",
        "=== STATO AL MOMENTO ===",
        "CTL: " + str(round(act_ctl)) + " | ATL: " + str(round(act_atl)) + " | TSB: " + str(round(act_tsb,1)),
        "",
        "=== PROFILO ATLETA ===",
        "Eta: " + str(u.get("eta",33)) + " anni | Peso: " + str(u.get("peso",75)) + "kg",
        "FC max: " + str(u["fc_max"]) + " bpm | FC riposo: " + str(u["fc_min"]) + " bpm",
        "FTP: " + str(ftp) + " W",
        "",
        "=== ULTIME " + str(window_days) + " GIORNATE (sessioni precedenti) ===",
        "Sessioni: " + str(len(prev)) + " | TSS totale: " + str(round(prev["tss"].sum())),
    ]
    ctx = "\n".join(l for l in lines if l is not None)
    if prev_lines:
        ctx += "\nDettaglio:\n" + prev_lines
    return ctx


def build_chat_context(df, u, current_ctl, current_atl, current_tsb,
                       status_label, vo2max_val) -> str:
    """Contesto ricco per la chat: 6 mesi completi + storico mensile."""
    ftp = u.get("ftp", 200)
    now = df["start_date"].max()
    cut6m = now - pd.Timedelta(days=180)
    df6m  = df[df["start_date"] >= cut6m].copy()

    # Ultimi 6 mesi — ogni attività
    lines_6m = []
    for _, row in df6m.iterrows():
        s_ = get_sport_info(row["type"])
        m_ = format_metrics(row)
        wl = ""
        if pd.notna(row.get("average_watts")) and row.get("average_watts",0) > 0 and ftp > 0:
            w_  = row["average_watts"]
            IF_ = w_ / ftp
            wl  = " W=" + str(round(w_)) + "(IF=" + str(round(IF_,2)) + ")"
        hrl = ""
        if pd.notna(row.get("average_heartrate")):
            hrl = " FC=" + str(round(row["average_heartrate"]))
        lines_6m.append(
            row["start_date"].strftime("%d/%m/%y") + " " + s_["icon"] + " " +
            str(row["name"])[:35] + ": " + m_["dist_str"] + " " + m_["dur_str"] +
            " D+" + m_["elev"] + hrl + wl +
            " TSS=" + str(round(row["tss"])) +
            " CTL=" + str(round(float(row.get("ctl",0)))) +
            " TSB=" + str(round(float(row.get("tsb",0)),1))
        )

    # Storico precedente — mensile
    df_old = df[df["start_date"] < cut6m].copy()
    monthly = []
    if not df_old.empty:
        df_old["ym"] = df_old["start_date"].dt.to_period("M")
        for ym, grp in df_old.groupby("ym"):
            monthly.append(
                str(ym) + ": " + str(len(grp)) + " sess, " +
                "km=" + str(round(grp["distance"].sum()/1000)) + " " +
                "TSS=" + str(round(grp["tss"].sum())) + " " +
                "D+=" + str(round(grp["total_elevation_gain"].sum())) + "m"
            )

    df7  = df[df["start_date"] >= now - pd.Timedelta(days=7)]
    df28 = df[df["start_date"] >= now - pd.Timedelta(days=28)]

    header = "\n".join([
        "=== PROFILO ATLETA ===",
        "Eta: " + str(u.get("eta",33)) + " anni | Peso: " + str(u.get("peso",75)) + "kg",
        "FC max: " + str(u["fc_max"]) + " | FC riposo: " + str(u["fc_min"]),
        "FTP: " + str(ftp) + "W | VO2max: " + str(vo2max_val or "N/D") + " ml/kg/min",
        "",
        "=== STATO ATTUALE ===",
        "CTL=" + str(round(current_ctl)) + " ATL=" + str(round(current_atl)) + " TSB=" + str(round(current_tsb,1)),
        "Stato: " + status_label,
        "Ultimi 7gg: " + str(len(df7)) + " sess, TSS=" + str(round(df7["tss"].sum())),
        "Ultimi 28gg: " + str(len(df28)) + " sess, TSS=" + str(round(df28["tss"].sum())) +
        " km=" + str(round(df28["distance"].sum()/1000)),
        "",
        "=== ULTIMI 6 MESI — " + str(len(df6m)) + " ATTIVITA' ===",
    ])
    ctx = header + "\n" + "\n".join(lines_6m)
    if monthly:
        ctx += "\n\n=== STORICO MENSILE PRECEDENTE ===\n" + "\n".join(monthly)
    return ctx

def build_daily_briefing(df, u, current_ctl, current_atl, current_tsb,
                         status_label, vo2max_val) -> str:
    """
    Genera il briefing giornaliero — stato forma, commento ultime 3 attività, prossimi giorni.
    """
    ftp  = u.get("ftp", 200)
    now  = df["start_date"].max()
    df7  = df[df["start_date"] >= now - pd.Timedelta(days=7)]
    df14 = df[df["start_date"] >= now - pd.Timedelta(days=14)]
    df28 = df[df["start_date"] >= now - pd.Timedelta(days=28)]
    last = df.iloc[-1]
    s_last = get_sport_info(last["type"])
    m_last = format_metrics(last)
    days_since_last = (datetime.now() - last["start_date"]).days

    # Trend CTL ultimi 14gg
    _spark14 = df[df["start_date"] >= now - pd.Timedelta(days=14)]
    ctl_trend = "stabile"
    if len(_spark14) >= 2:
        _ctl_start = float(_spark14["ctl"].iloc[0])
        _ctl_end   = float(_spark14["ctl"].iloc[-1])
        _delta = _ctl_end - _ctl_start
        if _delta > 3:    ctl_trend = "in crescita (+" + str(round(_delta)) + " pts)"
        elif _delta < -3: ctl_trend = "in calo (" + str(round(_delta)) + " pts)"

    # TSB trend
    tsb_trend = "neutro"
    if current_tsb > 10:   tsb_trend = "fresco, pronto per caricare"
    elif current_tsb > 0:  tsb_trend = "leggermente fresco"
    elif current_tsb > -10: tsb_trend = "lieve accumulo fatica"
    elif current_tsb > -20: tsb_trend = "affaticato, gestire i carichi"
    else:                   tsb_trend = "sovraccarico, recupero necessario"

    # Sport mix ultimi 28gg
    sport_mix = df28["type"].value_counts().head(4)
    sport_str = " | ".join([str(k) + "=" + str(v) for k,v in sport_mix.items()])

    # Ultime 3 sessioni — dettaglio ricco
    last3_lines = []
    for _, r3 in df.iloc[-3:].iterrows():
        s3 = get_sport_info(r3["type"])
        m3 = format_metrics(r3)
        wl3 = ""
        if pd.notna(r3.get("average_watts")) and r3.get("average_watts", 0) > 0 and ftp > 0:
            IF3 = r3["average_watts"] / ftp
            wl3 = f" W={r3['average_watts']:.0f}(IF={IF3:.2f})"
        hr3 = f" FC={r3['average_heartrate']:.0f}bpm" if pd.notna(r3.get("average_heartrate")) else ""
        elev3 = f" D+{m3['elev']}" if float(r3.get("total_elevation_gain", 0) or 0) > 20 else ""
        last3_lines.append(
            "  " + r3["start_date"].strftime("%d/%m %a") +
            " " + s3["icon"] + " " + s3["label"] +
            " — " + m3["dist_str"] + " " + m3["dur_str"] +
            elev3 + hr3 + wl3 +
            " · TSS=" + str(round(r3["tss"])) +
            " · CTL=" + str(round(float(r3.get("ctl", current_ctl)))) +
            " TSB=" + str(round(float(r3.get("tsb", current_tsb)), 1))
        )

    # Prossimi giorni — analisi carico
    _ramp = (current_ctl - float(_spark14["ctl"].iloc[0])) / 14 if len(_spark14) >= 2 else 0
    ramp_str = f"{_ramp:+.1f} CTL/gg (14gg)"

    lines_b = [
        "Sei un coach sportivo d'elite specializzato in ciclismo, trail running e sci alpinismo.",
        "Rispondi in italiano. Tono diretto, concreto, professionale. Usa i numeri.",
        "",
        "=== PROFILO ATLETA ===",
        str(u.get("eta",33)) + " anni | " + str(u.get("peso",75)) + "kg | "
            "FTP=" + str(ftp) + "W | FCmax=" + str(u["fc_max"]) + "bpm | "
            "FCmin=" + str(u["fc_min"]) + "bpm | VO2max=" + str(vo2max_val or "N/D") + " ml/kg/min",
        "",
        "=== STATO FORMA ATTUALE ===",
        "CTL=" + str(round(current_ctl)) + " (" + ctl_trend + ") | "
            "ATL=" + str(round(current_atl)) + " | "
            "TSB=" + str(round(current_tsb, 1)) + " → " + tsb_trend,
        "Ramp rate: " + ramp_str,
        "Ultima uscita: " + str(days_since_last) + " giorni fa (" + s_last["label"] + " "
            + m_last["dist_str"] + " TSS=" + str(round(last["tss"])) + ")",
        "",
        "=== CARICO RECENTE ===",
        "7gg:  " + str(len(df7)) + " sessioni | TSS=" + str(round(df7["tss"].sum()))
            + " | km=" + str(round(df7["distance"].sum()/1000))
            + " | D+=" + str(round(df7["total_elevation_gain"].sum())) + "m",
        "14gg: " + str(len(df14)) + " sessioni | TSS=" + str(round(df14["tss"].sum()))
            + " | km=" + str(round(df14["distance"].sum()/1000)),
        "28gg: " + str(len(df28)) + " sessioni | TSS=" + str(round(df28["tss"].sum()))
            + " | km=" + str(round(df28["distance"].sum()/1000)),
        "Mix sport 28gg: " + sport_str,
        "",
        "=== ULTIME 3 SESSIONI (dettaglio) ===",
    ] + last3_lines + [
        "",
        "=== COMPITO ===",
        "Scrivi un briefing in 3 sezioni ben distinte (usa \\n tra sezioni, NO markdown con *):",
        "",
        "1. STATO FORMA (2-3 frasi): interpreta i numeri CTL/ATL/TSB con il ramp rate."
        " Spiega cosa significano per questo atleta in questo momento.",
        "",
        "2. ULTIME 3 SESSIONI (3-4 frasi): commenta qualitativamente ogni sessione."
        " Come sono state rispetto al profilo? Zone corrette? Carico adeguato?",
        "",
        "3. PROSSIMI 3-5 GIORNI (3-4 frasi): cosa dovrebbe fare, in quale ordine."
        " Tipo di sessione, durata indicativa, zona FC/potenza target, TSS target."
        " Se serve recupero, dillo chiaramente e spiega perché.",
        "",
        "Sii specifico con i numeri. Niente frasi generiche. Max 200 parole totali.",
    ]
    ctx = "\n".join(lines_b)
    return ai_deep(ctx)

def get_daily_briefing_key() -> str:
    return "daily_brief_" + datetime.now().strftime("%Y%m%d")

# ============================================================
# SESSION STATE
# ============================================================
for key, val in {
    "strava_token_info":  {},
    "messages":           [],
    "user_data":          {"peso": 75.0, "fc_min": 50, "fc_max": 190, "ftp": 200, "eta": 33},
    "activities_cache":   [],
    "activities_last_ts": 0,
    "activities_token":   "",
    "mob_menu":           "dashboard",
    "selected_act_id":    None,
    "gsheet_loaded":      False,
    "ai_model_pref":      "gemini-3.1-flash-lite-preview",
    "conv_loaded":        False,
    "weekly_plan":        None,
    "weekly_plan_date":   None,
    "_chat_pending":      False,
    "_nav_open":          False,
    "coach_memory":       {},
    "structured_plan":    None,
    "structured_plan_date": None,
    "_memory_loaded":     False,
    "_proactive_done":    False,
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

# Gestisci ?act=ID — apre dettaglio attività da card HTML
_act_param = st.query_params.get("act", "")
if _act_param:
    try:
        st.session_state.selected_act_id = int(_act_param)
    except Exception:
        st.session_state.selected_act_id = _act_param
    st.query_params.clear()
    # Non serve rerun — la pagina si renderizza con il nuovo selected_act_id

def compute_fitness_by_sport(df: pd.DataFrame) -> dict:
    """CTL/ATL/TSB per sport — cachata in session_state per df_cache_key."""
    _cache_key = f"_sport_fitness_{st.session_state.get('_df_cache_key','')}"
    if _cache_key in st.session_state:
        return st.session_state[_cache_key]
    sport_groups = {
        "run":      df["type"].isin(["Run", "TrailRun"]),
        "bike":     df["type"].isin(["Ride", "VirtualRide", "MountainBikeRide"]),
        "mountain": df["type"].isin(["BackcountrySki", "AlpineSki", "Hike"]),
    }
    icons = {"run": "🏃", "bike": "🚴", "mountain": "🎿"}
    labels = {"run": "Corsa", "bike": "Ciclismo", "mountain": "Montagna"}
    result = {}
    for key, mask in sport_groups.items():
        sub = df[mask]
        if sub.empty:
            result[key] = None
            continue
        ctl_s, atl_s, tsb_s, ctl_d, atl_d, tsb_d, _ = compute_fitness(sub)
        result[key] = {
            "ctl": float(ctl_s.iloc[-1]) if len(ctl_s) > 0 else 0,
            "atl": float(atl_s.iloc[-1]) if len(atl_s) > 0 else 0,
            "tsb": float(tsb_s.iloc[-1]) if len(tsb_s) > 0 else 0,
            "ctl_daily": ctl_d,
            "atl_daily": atl_d,
            "tsb_daily": tsb_d,
            "icon": icons[key],
            "label": labels[key],
            "n": len(sub),
        }
    st.session_state[_cache_key] = result
    return result


def extract_and_update_memory(messages: list, memory: dict) -> dict:
    """
    Analizza la conversazione e aggiorna la memoria con fatti chiave nuovi.
    Chiama l'AI solo se ci sono abbastanza messaggi nuovi.
    """
    if len(messages) < 4:
        return memory
    _conv_text = "\n".join([
        ("Atleta" if m["role"] == "user" else "Coach") + ": " + str(m["content"])
        for m in messages[-20:]
    ])
    _prompt = (
        "Analizza questa conversazione tra atleta e coach sportivo.\n"
        "Estrai i FATTI CHIAVE nuovi che il coach dovrebbe ricordare nelle sessioni future.\n"
        "Includi solo informazioni concrete e durature: obiettivi, infortuni, preferenze, "
        "punti di forza/debolezza emersi, eventi in programma, feedback su allenamenti.\n"
        "NON includere dati già visibili in Strava (CTL, TSS, ecc).\n\n"
        f"Memoria attuale: {json.dumps(memory, ensure_ascii=False)}\n\n"
        f"Conversazione:\n{_conv_text}\n\n"
        "Rispondi SOLO con un JSON valido {\"chiave\": \"valore\"} con max 10 fatti. "
        "Usa chiavi brevi in italiano (es: 'obiettivo_gara', 'infortunio', 'sport_preferito'). "
        "Mantieni i fatti esistenti e aggiungi/aggiorna quelli nuovi. "
        "Se non ci sono nuovi fatti rilevanti rispondi con {}."
    )
    try:
        _raw = ai_generate(_prompt, max_tokens=400)
        _raw = _raw.strip()
        # Estrai JSON dalla risposta
        _start = _raw.find("{")
        _end   = _raw.rfind("}") + 1
        if _start >= 0 and _end > _start:
            _new = json.loads(_raw[_start:_end])
            memory.update(_new)
    except Exception:
        pass
    return memory


def build_proactive_opener(df, u, current_ctl, current_atl, current_tsb,
                            status_label, memory: dict) -> str:
    """
    Genera un messaggio proattivo del coach basato su:
    - Stato forma attuale
    - Ultima attività
    - Memoria precedente
    - Alert attivi
    Restituisce stringa vuota se non c'è niente di rilevante.
    """
    last = df.iloc[-1]
    days_off = (datetime.now() - last["start_date"]).days
    m_last = format_metrics(last)
    s_last = get_sport_info(last["type"])

    memory_str = ""
    if memory:
        memory_str = "Cose che so sull'atleta: " + "; ".join(
            f"{k}={v}" for k, v in list(memory.items())[:5]
        )

    _prompt = (
        "Sei un coach sportivo. Stai aprendo una nuova sessione di coaching.\n"
        f"Atleta: CTL={current_ctl:.0f}, ATL={current_atl:.0f}, "
        f"TSB={current_tsb:+.0f} ({status_label})\n"
        f"Ultima uscita: {days_off} giorni fa — {s_last['label']} "
        f"{m_last['dist_str']} {m_last['dur_str']} TSS={last['tss']:.0f}\n"
        f"{memory_str}\n\n"
        "Scrivi UN messaggio di apertura breve (2-3 frasi) che:\n"
        "1. Commenta lo stato attuale con un fatto concreto\n"
        "2. Fa UNA domanda specifica o propone un'azione\n"
        "Non usare formule di saluto generiche. Sii diretto come un coach vero.\n"
        "Rispondi in italiano."
    )
    try:
        return ai_generate(_prompt, max_tokens=120)
    except Exception:
        return ""


def build_structured_weekly_plan(df, u, current_ctl, current_atl, current_tsb,
                                  status_label, vo2max_val, memory: dict) -> dict:
    """
    Genera un piano settimanale strutturato in JSON con 7 giorni.
    Formato: {"giorni": [{"giorno": "Lun", "data": "...", "tipo": "...",
              "durata": "...", "zona": "...", "tss": N, "note": "..."}],
              "tss_totale": N, "focus": "..."}
    """
    df7  = df[df["start_date"] >= df["start_date"].max() - pd.Timedelta(days=7)]
    df28 = df[df["start_date"] >= df["start_date"].max() - pd.Timedelta(days=28)]
    avg_tss = df28["tss"].sum() / 28 if not df28.empty else 50
    sport_mix = df28["type"].value_counts().head(3)
    sport_str = " / ".join([str(k) for k in sport_mix.index])
    memory_str = "; ".join(f"{k}={v}" for k, v in memory.items()) if memory else ""

    _today = datetime.now()
    _days_labels = []
    _giorni_ita  = ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"]
    for i in range(7):
        _d = _today + timedelta(days=i)
        _days_labels.append(f"{_giorni_ita[_d.weekday()]} {_d.strftime('%d/%m')}")

    _prompt = (
        "Sei un coach sportivo d'elite. Crea un piano allenamento per i prossimi 7 giorni.\n\n"
        f"ATLETA: {u.get('eta',33)} anni, {u.get('peso',75)}kg, "
        f"FTP={u.get('ftp',200)}W, FCmax={u['fc_max']}bpm\n"
        f"FORMA: CTL={current_ctl:.0f} ATL={current_atl:.0f} TSB={current_tsb:+.0f} ({status_label})\n"
        f"TSS medio/giorno (28gg): {avg_tss:.0f}\n"
        f"Sport praticati: {sport_str}\n"
        f"Sessioni ultima settimana: {len(df7)}, TSS={df7['tss'].sum():.0f}\n"
        + (f"Note atleta: {memory_str}\n" if memory_str else "") +
        f"\nGiorni: {', '.join(_days_labels)}\n\n"
        "Rispondi SOLO con JSON valido, nessun testo fuori:\n"
        '{"focus": "obiettivo settimana in 1 frase", '
        '"tss_totale": numero, '
        '"giorni": ['
        '{"giorno": "Lun 16/06", "tipo": "Riposo|Recupero|Aerobico|Soglia|Intervalli|Lungo|Gara", '
        '"durata": "45 min", "zona": "Z1-Z2", "tss": 40, "note": "breve nota specifica"}'
        "]}"
    )
    try:
        _raw = ai_deep(_prompt)
        _raw = _raw.strip()
        _start = _raw.find("{")
        _end   = _raw.rfind("}") + 1
        if _start >= 0 and _end > _start:
            return json.loads(_raw[_start:_end])
    except Exception:
        pass
    return {}


# ============================================================
# BOTTOM NAV BAR
# ============================================================
NAV_ITEMS = [
    ("dashboard", "🏠", "Home"),
    ("fitness",   "📈", "Fitness"),
    ("storico",   "📅", "Storico"),
    ("chat",      "💬", "Coach"),
    ("profilo",   "👤", "Profilo"),
]

def render_bottom_nav():
    """Nav radio orizzontale — fissa in basso."""
    cur = st.session_state.mob_menu

    _options = [icon for _, icon, _ in NAV_ITEMS]
    _keys    = [key  for key, _, _ in NAV_ITEMS]
    _cur_idx = _keys.index(cur) if cur in _keys else 0

    st.markdown("""
<style>
/* NAV FISSA — selettore globale, funziona su tutti i browser */
[data-testid="stRadio"] {
    position: fixed !important;
    bottom: 50px !important; left: 0 !important; right: 0 !important;
    z-index: 99999 !important;
    background: #ffffff !important;
    border-top: 1.5px solid #e8e8e8 !important;
    border-bottom: 1.5px solid #e8e8e8 !important;
    box-shadow: 0 -2px 12px rgba(0,0,0,0.08) !important;
    padding: 6px 8px !important;
    margin: 0 !important;
}
[data-testid="stRadio"] > div {
    gap: 4px !important; flex-wrap: nowrap !important; width: 100% !important;
}
[data-testid="stRadio"] label {
    flex: 1 !important; display: flex !important;
    align-items: center !important; justify-content: center !important;
    padding: 6px 0 !important; border-radius: 10px !important;
    cursor: pointer !important; min-width: 0 !important;
}
[data-testid="stRadio"] label:has(input:checked) { background: #E3F2FD !important; }
[data-testid="stRadio"] label p { font-size: 26px !important; line-height: 1 !important; margin: 0 !important; }
/* Nascondi pallino */
[data-testid="stRadio"] label > div:first-child {
    display: none !important; width: 0 !important; height: 0 !important;
    overflow: hidden !important; margin: 0 !important; padding: 0 !important;
}
[data-testid="stRadio"] input[type="radio"] { display: none !important; }
[data-testid="stRadio"] > label { display: none !important; }
/* Override: ripristina il radio Vista storico (non-nav) */
[data-testid="stRadio"]:has(label p:not(:empty)) + div,
.storico-view-toggle [data-testid="stRadio"] {
    position: static !important;
    box-shadow: none !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
/* Fascia bianca sotto */
.nav-cover-strip {
    position: fixed !important;
    bottom: 0 !important; left: 0 !important; right: 0 !important;
    height: 50px !important; background: #ffffff !important;
    z-index: 99998 !important;
}
.block-container { padding-bottom: 220px !important; }
</style>
<div class="nav-cover-strip"></div>
""", unsafe_allow_html=True)

    _sel = st.radio(
        "nav", options=_options, index=_cur_idx,
        horizontal=True, label_visibility="collapsed", key="nav_radio"
    )

    _sel_key = _keys[_options.index(_sel)]
    if _sel_key != cur:
        st.session_state.mob_menu = _sel_key
        st.session_state.selected_act_id = None
        st.rerun()


def get_act_micro_comment(row_data, metrics, sport_info) -> str:
    """Genera (e cacha) un commento AI specifico basato sui dati reali dell'attività."""
    _act_id = row_data.get("id", str(row_data.get("start_date", "")))
    _key = f"micro_ai_{_act_id}"
    if _key in st.session_state:
        return st.session_state[_key]
    if _ai_sdk_mode is None:
        return ""
    m = metrics
    s = sport_info

    # Raccoglie dati specifici — priorità a quelli meno ovvi
    _name      = str(row_data.get("name", ""))
    _type      = s["label"]
    _hr_avg    = m["hr_avg"]
    _hr_max    = m["hr_max"]
    _pace      = m["pace_str"]
    _watts_avg = row_data.get("average_watts")
    _watts_max = row_data.get("max_watts")
    _speed_max = row_data.get("max_speed")  # m/s
    _cadence   = row_data.get("average_cadence")
    _suffer    = row_data.get("suffer_score")
    _tss       = f"{row_data.get('tss', 0):.0f}"

    # Costruisce lista dati — NON include dislivello (troppo ripetitivo)
    _facts = []
    if pd.notna(_watts_max) and _watts_max and float(_watts_max) > 0:
        _facts.append(f"picco potenza {float(_watts_max):.0f}W")
    if pd.notna(_watts_avg) and _watts_avg and float(_watts_avg) > 0:
        _facts.append(f"potenza media {float(_watts_avg):.0f}W")
    if pd.notna(_speed_max) and _speed_max and float(_speed_max) > 0:
        _kmh_max = float(_speed_max) * 3.6
        if _kmh_max > 45:  # solo se notevole
            _facts.append(f"max {_kmh_max:.0f} km/h")
    if pd.notna(_hr_max) and _hr_max != "—":
        _facts.append(f"FC max {_hr_max} bpm")
    if pd.notna(_suffer) and _suffer and float(_suffer) > 60:
        _facts.append(f"suffer {_suffer:.0f}")
    if pd.notna(_cadence) and _cadence and float(_cadence) > 0:
        _facts.append(f"cadenza {float(_cadence):.0f}")
    _facts_str = ", ".join(_facts[:2]) if _facts else ""  # max 2 dati

    _prompt = (
        f"Dati uscita: sport={_type}, nome='{_name}', "
        f"distanza={m['dist_str']}, durata={m['dur_str']}, "
        f"passo/velocità={_pace}, FC media={_hr_avg} bpm, "
        f"dislivello={m['elev']}, TSS={_tss}"
        + (f", {_facts_str}" if _facts_str else "") + ".\n\n"
        "Scrivi UNA frase di massimo 8 parole su questa uscita.\n"
        "REGOLA FONDAMENTALE: usa SEMPRE i numeri reali dall'elenco sopra. "
        "Vietato usare aggettivi vaghi come 'ottimo', 'buono', 'bel'. "
        "Esempi corretti: '800m D+ in 28km a 5:10/km' — '42km a 26 km/h FC 142bpm' — "
        "'Giro del Velino 1400m D+' — 'Intervalli: FC max 178bpm TSS 85' — "
        "'Lungo Z2: 2h15 a 155bpm'.\n"
        "Scegli il dato che spicca di più (geografico, potenza, velocità, FC, dislivello se alto).\n"
        "Solo la frase, nessuna punteggiatura finale."
    )
    try:
        _res = ai_generate(_prompt, max_tokens=40)
        _res = _res.strip().strip('"').strip("'").rstrip(".")
        st.session_state[_key] = _res
        return _res
    except Exception:
        return ""


def render_act_card(row_data, metrics, sport_info, zone_color, zone_label,
                    act_id, header_label="", key_prefix="act"):
    """
    Renderizza una card attività completa in HTML puro.
    Il bottone 🔍 è sovrapposto in basso a destra via position:absolute.
    Il click setta ?act=ID nella URL — Streamlit lo legge e apre il dettaglio.
    """
    m = metrics
    s = sport_info
    _zc = zone_color
    _zl = zone_label
    _date = row_data["start_date"].strftime("%d %b · %H:%M")
    _color = s["color"]

    _header_html = (
        f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px">{header_label}</div>'
        if header_label else ""
    )

    # Pills
    _pills = (
        f'<span class="act-pill">📏 <b>{m["dist_str"]}</b></span>'
        f'<span class="act-pill">⏱ <b>{m["dur_str"]}</b></span>'
        f'<span class="act-pill">⚡ <b>{m["pace_str"]}</b></span>'
        f'<span class="act-pill">⛰ <b>{m["elev"]}</b></span>'
        f'<span class="act-pill">❤️ <b>{m["hr_avg"]} bpm</b></span>'
    )
    if m["hr_max"] != "—":
        _pills += f'<span class="act-pill">💓 <b>{m["hr_max"]} bpm</b></span>'
    if m["cals"] != "—":
        _pills += f'<span class="act-pill">🔥 <b>{m["cals"]}</b></span>'
    _pills += f'<span class="act-pill">TSS <b>{row_data["tss"]:.0f}</b></span>'

    # Tag descrittivo deterministico — calcolato dai dati, niente AI
    _atype    = row_data.get("type", "")
    _dur_sec  = float(row_data.get("moving_time", 0) or 0)
    _dur_h    = _dur_sec / 3600
    _elev_raw = float(row_data.get("total_elevation_gain", 0) or 0)
    _tss_val  = float(row_data.get("tss", 0) or 0)
    _hr_avg_v = row_data.get("average_heartrate")
    _fc_max   = u.get("fc_max", 190) if "u" in dir() else 190
    _watts_v  = row_data.get("average_watts")
    _ftp      = u.get("ftp", 200) if "u" in dir() else 200
    _hr_pct   = (float(_hr_avg_v) / _fc_max) if pd.notna(_hr_avg_v) and _fc_max > 0 else None
    _if_val   = (float(_watts_v) / _ftp) if pd.notna(_watts_v) and _watts_v and _ftp > 0 else None

    _tag = ""
    _tag_color = "#888"

    if _atype in ("AlpineSki",):
        _tag = "Sci pista"
        _tag_color = "#81D4FA"
    elif _atype in ("BackcountrySki",):
        if _elev_raw > 1500:  _tag, _tag_color = "Scialpinismo impegnativo", "#4FC3F7"
        elif _elev_raw > 800: _tag, _tag_color = "Scialpinismo medio", "#4FC3F7"
        else:                 _tag, _tag_color = "Scialpinismo facile", "#81D4FA"
    elif _atype in ("Run", "TrailRun"):
        if _hr_pct and _hr_pct < 0.65:   _tag, _tag_color = "Recupero", "#4CAF50"
        elif _hr_pct and _hr_pct < 0.75: _tag, _tag_color = "Fondo Z2", "#8BC34A"
        elif _tss_val > 120:              _tag, _tag_color = "Uscita dura", "#F44336"
        elif _elev_raw > 1000:            _tag, _tag_color = "Trail montagna", "#FF7043"
        elif _elev_raw > 400:             _tag, _tag_color = "Trail collinare", "#FF9800"
        elif _dur_h > 1.8:               _tag, _tag_color = "Lungo", "#2196F3"
        elif _hr_pct and _hr_pct > 0.88: _tag, _tag_color = "Ritmo soglia", "#FF5722"
        else:                             _tag, _tag_color = "Corsa", "#FF4B4B"
    elif _atype in ("Ride", "VirtualRide", "MountainBikeRide"):
        if _if_val and _if_val > 0.90:   _tag, _tag_color = "Sforzo soglia", "#FF5722"
        elif _if_val and _if_val < 0.65: _tag, _tag_color = "Recupero attivo", "#4CAF50"
        elif _elev_raw > 2000:           _tag, _tag_color = "Gran fondo", "#9C27B0"
        elif _elev_raw > 1000:           _tag, _tag_color = "Granfondo collinare", "#673AB7"
        elif _dur_h > 3:                 _tag, _tag_color = "Lungo endurance", "#2196F3"
        elif _tss_val > 100:             _tag, _tag_color = "Carico elevato", "#FF9800"
        else:                            _tag, _tag_color = "Uscita base", "#42A5F5"
    elif _atype in ("Hike",):
        if _elev_raw > 1000: _tag, _tag_color = "Escursione alpina", "#4FC3F7"
        elif _dur_h > 3:     _tag, _tag_color = "Escursione lunga", "#4CAF50"
        else:                _tag, _tag_color = "Escursione", "#8BC34A"
    elif _atype in ("Workout",):
        if _tss_val > 80:  _tag, _tag_color = "Allenamento intenso", "#FF5722"
        elif _dur_h > 1.2: _tag, _tag_color = "Allenamento lungo", "#FF9800"
        else:              _tag, _tag_color = "Allenamento", "#FF9800"

    # Sfondo sport — colori pastello solidi, più visibili di linear-gradient
    _sport_styles = {
        "Run":              ("#FFF5F5", "#EF4444"),
        "TrailRun":         ("#FFF4EE", "#F97316"),
        "Ride":             ("#EFF6FF", "#3B82F6"),
        "VirtualRide":      ("#F0F4FF", "#6366F1"),
        "MountainBikeRide": ("#EEF2FF", "#4F46E5"),
        "BackcountrySki":   ("#F0F9FF", "#0EA5E9"),
        "AlpineSki":        ("#E0F2FE", "#38BDF8"),
        "Hike":             ("#F0FDF4", "#22C55E"),
        "Workout":          ("#FFF7ED", "#F97316"),
    }
    _bg_color, _border_color = _sport_styles.get(_atype, ("#FAFAFA", "#94A3B8"))

    _tag_html = (
        f' &nbsp;<span style="background:{_tag_color}22;color:{_tag_color};'
        f'border-radius:20px;padding:2px 8px;font-size:10px;font-weight:700">{_tag}</span>'
        if _tag else ""
    )

    # Barra TSS inline
    _tss_pct = min(100, int(_tss_val / 150 * 100))
    _tss_bar = (
        f'<div style="margin-top:8px;height:4px;background:#E2E8F0;border-radius:4px;overflow:hidden">'
        f'<div style="width:{_tss_pct}%;height:4px;border-radius:4px;background:{_zc}"></div>'
        f'</div>'
    )

    card_html = (
        f'<div class="act-card" style="background:{_bg_color};border-left-color:{_border_color}">'
        f'{_header_html}'
        f'<div class="act-title">{s["icon"]} {str(row_data["name"])}</div>'
        f'<div class="act-meta">{_date} &middot;'
        f'<span class="zone-chip" style="background:{_zc}22;color:{_zc}">{_zl}</span>{_tag_html}'
        f'</div>'
        f'<div class="act-pills">{_pills}</div>'
        f'{_tss_bar}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
    # Bottone in colonna piccola a destra
    _, _cb2 = st.columns([7, 1])
    with _cb2:
        if st.button("🔍", key=f"{key_prefix}_{act_id}", help="Apri dettaglio"):
            st.session_state.selected_act_id = act_id
            st.rerun()

# ============================================================
# LOGIN PAGE
# ============================================================
token_ok = refresh_token_if_needed()

if not token_ok:
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .block-container {
        background: #0F2744 !important;
        padding: 0 !important;
    }
    </style>
    <div style="min-height:100vh;background:linear-gradient(160deg,#0F2744 0%,#1e3a5f 100%);
                display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:48px 24px;text-align:center">

      <!-- Logo -->
      <div style="width:80px;height:80px;background:rgba(255,255,255,0.1);
                  border-radius:24px;display:flex;align-items:center;justify-content:center;
                  font-size:40px;margin-bottom:24px;
                  box-shadow:0 8px 32px rgba(0,0,0,0.3)">🏆</div>

      <div style="font-size:28px;font-weight:900;color:#fff;letter-spacing:-0.5px;
                  margin-bottom:8px">Elite AI Coach</div>
      <div style="font-size:15px;color:rgba(255,255,255,0.55);margin-bottom:12px;
                  max-width:280px;line-height:1.5">
        Il tuo coach personale.<br>Analisi, piani e coaching basati sui tuoi dati reali.
      </div>

      <!-- Feature pills -->
      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-bottom:40px">
        <span style="background:rgba(255,255,255,0.1);color:rgba(255,255,255,0.7);
                     border-radius:20px;padding:4px 12px;font-size:12px">📊 CTL/ATL/TSB</span>
        <span style="background:rgba(255,255,255,0.1);color:rgba(255,255,255,0.7);
                     border-radius:20px;padding:4px 12px;font-size:12px">🤖 Coach AI</span>
        <span style="background:rgba(255,255,255,0.1);color:rgba(255,255,255,0.7);
                     border-radius:20px;padding:4px 12px;font-size:12px">📅 Piani 7gg</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if CLIENT_ID:
        auth_url = (
            f"https://www.strava.com/oauth/authorize"
            f"?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
            f"&response_type=code&scope=read,activity:read_all"
        )
        st.markdown(f"""
        <div style="position:fixed;bottom:40px;left:24px;right:24px;text-align:center">
            <a href="{auth_url}" style="
                display:block;
                background:linear-gradient(135deg,#FC4C02,#e84300);
                color:white;border-radius:16px;padding:18px;
                font-size:17px;font-weight:700;text-decoration:none;
                box-shadow:0 8px 24px rgba(252,76,2,0.45);
                letter-spacing:0.2px">
                🔗 Connetti Strava
            </a>
            <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:12px">
              Dati sicuri · Solo lettura · Nessuna modifica
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Configura STRAVA_CLIENT_ID e STRAVA_CLIENT_SECRET nei Secrets.")
    st.stop()

# ============================================================
# DATI: intervals.icu (attività) + Strava (GPS on-demand)
# ============================================================
access_token = st.session_state.strava_token_info["access_token"]
athlete      = fetch_athlete(access_token)

# ── Logica caricamento con Google Sheets ──
_gsheet_ok = bool(GSHEET_ID and GSHEET_CREDS)

# ── Carica profilo da GSheet (una volta per sessione) ──
if _gsheet_ok and not st.session_state.get("_profile_loaded"):
    _saved_profile = gsheet_load_profile()
    if _saved_profile:
        merged_profile = dict(st.session_state.user_data)
        merged_profile.update(_saved_profile)
        st.session_state.user_data = merged_profile
    st.session_state["_profile_loaded"] = True

if not st.session_state.activities_cache:
    if INTERVALS_API_KEY and INTERVALS_ATHLETE_ID:
        # ── Fonte primaria: intervals.icu ──
        # Prima prova dalla cache GSheet
        if _gsheet_ok and not st.session_state.gsheet_loaded:
            with st.spinner("📊 Carico storico dalla cache..."):
                sheet_data = gsheet_load_activities()
            if sheet_data:
                # Controlla se i dati sono già da intervals.icu o da Strava (vecchi)
                _sample = sheet_data[0] if sheet_data else {}
                _is_icu = _sample.get("_source") == "intervals.icu"
                if _is_icu:
                    st.session_state.activities_cache = sheet_data
                    st.session_state.gsheet_loaded    = True
                    # Aggiornamento incrementale: prendi le nuove da intervals.icu
                    if gsheet_needs_sync():
                        st.toast("🔄 Aggiorno con le ultime attività da intervals.icu...", icon="⏳")
                        _dates_icu = []
                        for a in sheet_data:
                            try:
                                d = datetime.fromisoformat(str(a.get("start_date_local","")).replace("Z",""))
                                _dates_icu.append(d.date())
                            except Exception:
                                pass
                        _last_date = max(_dates_icu) if _dates_icu else (datetime.now().date() - timedelta(days=365))
                        _newest    = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
                        _oldest    = (_last_date - timedelta(days=3)).strftime("%Y-%m-%d")  # 3gg overlap sicurezza
                        _new_raw   = fetch_intervals_activities_page(
                            INTERVALS_ATHLETE_ID, INTERVALS_API_KEY, _oldest, _newest)
                        if _new_raw:
                            _new_norm     = [normalize_intervals_activity(a) for a in _new_raw]
                            _existing_ids = {a["id"] for a in sheet_data}
                            _added        = [a for a in _new_norm if a["id"] not in _existing_ids]
                            if _added:
                                merged = sheet_data + _added
                                st.session_state.activities_cache = merged
                                gsheet_save_activities(merged)
                                st.toast(f"✅ {len(_added)} nuove attività sincronizzate da intervals.icu", icon="🏃")
                            else:
                                gsheet_save_activities(sheet_data)
                else:
                    # Cache contiene dati Strava → ricarica tutto da intervals.icu
                    st.toast("🔄 Migrazione dati a intervals.icu in corso...", icon="⏳")
                    with st.spinner("⏳ Carico storico completo da intervals.icu (prima volta, 30-60s)..."):
                        _icu_raw  = load_all_from_intervals(INTERVALS_ATHLETE_ID, INTERVALS_API_KEY)
                    if _icu_raw:
                        _icu_norm = [normalize_intervals_activity(a) for a in _icu_raw]
                        st.session_state.activities_cache = _icu_norm
                        st.session_state.gsheet_loaded    = True
                        if _gsheet_ok:
                            with st.spinner("💾 Salvo in cache persistente..."):
                                gsheet_save_activities(_icu_norm)
                        st.toast(f"✅ {len(_icu_norm)} attività migrate da intervals.icu", icon="🏃")
            else:
                # Nessuna cache → carica tutto da intervals.icu
                with st.spinner("⏳ Carico storico completo da intervals.icu (prima volta, 30-60s)..."):
                    _icu_raw = load_all_from_intervals(INTERVALS_ATHLETE_ID, INTERVALS_API_KEY)
                if _icu_raw:
                    _icu_norm = [normalize_intervals_activity(a) for a in _icu_raw]
                    st.session_state.activities_cache = _icu_norm
                    st.session_state.gsheet_loaded    = True
                    if _gsheet_ok:
                        with st.spinner("💾 Salvo in cache persistente..."):
                            gsheet_save_activities(_icu_norm)
                    st.toast(f"✅ {len(_icu_norm)} attività caricate da intervals.icu", icon="🏃")
        else:
            # Nessun GSheet → carica direttamente da intervals.icu
            with st.spinner("⏳ Carico storico da intervals.icu..."):
                _icu_raw = load_all_from_intervals(INTERVALS_ATHLETE_ID, INTERVALS_API_KEY)
            if _icu_raw:
                _icu_norm = [normalize_intervals_activity(a) for a in _icu_raw]
                st.session_state.activities_cache = _icu_norm
                st.toast(f"✅ {len(_icu_norm)} attività caricate da intervals.icu", icon="🏃")
    else:
        # ── Fallback: Strava (se intervals.icu non configurato) ──
        if _gsheet_ok and not st.session_state.gsheet_loaded:
            with st.spinner("📊 Carico storico dalla cache..."):
                sheet_data = gsheet_load_activities()
            if sheet_data:
                st.session_state.activities_cache = sheet_data
                st.session_state.gsheet_loaded    = True
                if gsheet_needs_sync():
                    st.toast("🔄 Aggiorno da Strava...", icon="⏳")
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
            else:
                with st.spinner("⏳ Primo caricamento da Strava (30-60 sec)..."):
                    raw = load_all_from_strava(access_token)
                st.session_state.activities_cache = raw
                st.session_state.gsheet_loaded    = True
                if _gsheet_ok and raw:
                    gsheet_save_activities(raw)
        else:
            with st.spinner("⏳ Carico da Strava..."):
                raw = load_all_from_strava(access_token)
            st.session_state.activities_cache = raw

raw = st.session_state.activities_cache
if not raw:
    st.error("Impossibile recuperare le attività. Verifica la configurazione di intervals.icu o Strava.")
    st.stop()

u = st.session_state.user_data

# ── Cache del DataFrame elaborato in session_state ──
# La chiave include il numero di attività + hash del profilo utente
# → si ricalcola solo se cambiano i dati o il profilo (FTP, FC max, ecc.)
_df_cache_key = f"df_built_{len(raw)}_{u.get('fc_max',190)}_{u.get('fc_min',50)}_{u.get('ftp',200)}"

if st.session_state.get("_df_cache_key") != _df_cache_key:
    # ── Build DataFrame ──
    df = pd.DataFrame(raw)
    df["start_date"] = pd.to_datetime(
        df.get("start_date_local", df.get("start_date",""))
    ).dt.tz_localize(None)
    df = df.sort_values("start_date").reset_index(drop=True)

    for col in ["average_heartrate","max_heartrate","average_watts",
                "total_elevation_gain","average_cadence","kilojoules",
                "calories","suffer_score","distance","moving_time"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # TSS: usa icu_training_load da intervals.icu se disponibile, altrimenti calcola
    if "icu_training_load" in df.columns:
        _icu_tss = pd.to_numeric(df["icu_training_load"], errors="coerce")
        _calc_tss = calc_tss_vectorized(df, u)
        df["tss"] = _icu_tss.where(_icu_tss.notna() & (_icu_tss > 0), _calc_tss)
    else:
        df["tss"] = calc_tss_vectorized(df, u)

    # NP (Normalized Power) da intervals.icu
    if "icu_weighted_avg_watts" in df.columns:
        df["normalized_power"] = pd.to_numeric(df["icu_weighted_avg_watts"], errors="coerce")

    # Fix: sci alpino su pista → dislivello zero (funivia conta come salita GPS)
    # Fatto sul DataFrame così l'AI riceve dati corretti in tutti i contesti
    df.loc[df["type"] == "AlpineSki", "total_elevation_gain"] = 0.0

    # Fitness (CTL/ATL/TSB)
    ctl_s, atl_s, tsb_s, ctl_daily, atl_daily, tsb_daily, tss_daily = compute_fitness(df)
    df["ctl"] = ctl_s.values
    df["atl"] = atl_s.values
    df["tsb"] = tsb_s.values

    # Zone FC vettorizzate (erano 3 apply() separati)
    df["zone_num"], df["zone_color"], df["zone_label"] = assign_zones_vectorized(df, u["fc_max"])

    # VO2max — usa sia dati corsa che bici, prende il massimo
    vo2max_val = calc_vo2max_estimate(df, ftp=u.get("ftp", 200), peso=u.get("peso", 75))

    # Salva tutto in cache
    st.session_state["_df_cache_key"]   = _df_cache_key
    st.session_state["_df_cached"]      = df
    st.session_state["_ctl_daily"]      = ctl_daily
    st.session_state["_atl_daily"]      = atl_daily
    st.session_state["_tsb_daily"]      = tsb_daily
    st.session_state["_tss_daily"]      = tss_daily
    st.session_state["_vo2max_val"]     = vo2max_val
else:
    # Recupera dalla cache — zero ricalcoli
    df         = st.session_state["_df_cached"]
    ctl_daily  = st.session_state["_ctl_daily"]
    atl_daily  = st.session_state["_atl_daily"]
    tsb_daily  = st.session_state["_tsb_daily"]
    tss_daily  = st.session_state["_tss_daily"]
    vo2max_val = st.session_state["_vo2max_val"]

# Carica conversazioni e piano (una sola volta per sessione, indipendente dal df)
if _gsheet_ok and not st.session_state.get("conv_loaded") and not st.session_state.messages:
    _saved_conv = gsheet_load_conversations()
    if _saved_conv:
        st.session_state.messages = _saved_conv
    st.session_state.conv_loaded = True

if _gsheet_ok and st.session_state.get("weekly_plan") is None:
    _saved_plan, _saved_plan_dt = gsheet_load_weekly_plan()
    if _saved_plan:
        st.session_state.weekly_plan      = _saved_plan
        st.session_state.weekly_plan_date = _saved_plan_dt

# ── CTL/ATL/TSB: intervals.icu (preciso) con fallback al calcolo locale ──
_icu_fitness = None
if INTERVALS_API_KEY:
    _icu_cache_key = f"_icu_fit_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')}"
    if st.session_state.get("_icu_fitness_key") != _icu_cache_key:
        with st.spinner("🔄 Aggiorno fitness da intervals.icu..."):
            _icu_fitness = get_intervals_fitness(INTERVALS_ATHLETE_ID, INTERVALS_API_KEY)
        st.session_state["_icu_fitness_key"]  = _icu_cache_key
        st.session_state["_icu_fitness_data"] = _icu_fitness
    else:
        _icu_fitness = st.session_state.get("_icu_fitness_data")

if _icu_fitness:
    current_ctl      = _icu_fitness["ctl"]
    current_atl      = _icu_fitness["atl"]
    current_tsb      = _icu_fitness["tsb"]
    _fitness_source  = "intervals.icu"
    _fitness_color   = "#10B981"
    # Aggiorna df per compatibilità con funzioni AI/chat
    df["ctl"] = current_ctl
    df["atl"] = current_atl
    df["tsb"] = current_tsb
    # Sparkline reali da intervals.icu
    _icu_hist = _icu_fitness.get("history", [])
    if len(_icu_hist) >= 2:
        _icu_idx  = pd.to_datetime([r["date"] for r in _icu_hist])
        ctl_daily = pd.Series([r["ctl"]  for r in _icu_hist], index=_icu_idx)
        atl_daily = pd.Series([r["atl"]  for r in _icu_hist], index=_icu_idx)
        tsb_daily = pd.Series([r["tsb"]  for r in _icu_hist], index=_icu_idx)
else:
    current_ctl      = float(df["ctl"].iloc[-1])
    current_atl      = float(df["atl"].iloc[-1])
    current_tsb      = float(df["tsb"].iloc[-1])
    _fitness_source  = "stimato"
    _fitness_color   = "#F97316"

if current_tsb > 10:   status_color, status_label = "#4CAF50", "🟢 In Forma"
elif current_tsb > -5: status_color, status_label = "#FF9800", "🟡 Stabile"
elif current_tsb > -20:status_color, status_label = "#FF5722", "🟠 Affaticato"
else:                   status_color, status_label = "#F44336", "🔴 Sovraccarico"

last_act = df.iloc[-1]

# ============================================================
# HEADER
# ============================================================
athlete_name   = athlete.get("firstname", "Atleta")
athlete_photo  = athlete.get("profile_medium") or athlete.get("profile", "")
_initials      = (athlete.get("firstname","?")[:1] + athlete.get("lastname","?")[:1]).upper()

# Foto o iniziali
if athlete_photo and "avatar/athlete" not in athlete_photo:
    _avatar_html = (
        f'<img src="{athlete_photo}" style="width:38px;height:38px;border-radius:50%;'
        f'object-fit:cover;border:2px solid rgba(255,255,255,0.4)">'
    )
else:
    _avatar_html = (
        f'<div style="width:38px;height:38px;border-radius:50%;background:rgba(255,255,255,0.2);'
        f'display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;'
        f'color:white;border:2px solid rgba(255,255,255,0.3)">{_initials}</div>'
    )

st.markdown(f"""
<div class="mob-header">
  <div style="display:flex;align-items:center;gap:12px">
    {_avatar_html}
    <div style="flex:1;min-width:0">
      <div style="font-size:11px;opacity:0.6;font-weight:600;letter-spacing:0.5px">ELITE COACH</div>
      <div style="font-size:16px;font-weight:800;margin:1px 0">{athlete_name}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:11px;opacity:0.6">Stato</div>
      <div style="font-size:13px;font-weight:700">{status_label}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Nav bar — chiamata subito dopo l'header così il CSS fixed è già iniettato
# I bottoni vengono posizionati in fondo via CSS position:fixed
render_bottom_nav()

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

        # Metriche compatte — grid densa con più dati
        _cals_row  = row.get("calories") or (float(row.get("kilojoules",0) or 0) * 0.239)
        _cals_disp = f"{_cals_row:.0f} kcal" if _cals_row and _cals_row > 0 else "—"
        _cadence   = row.get("average_cadence")
        _cad_disp  = f"{_cadence:.0f} rpm" if pd.notna(_cadence) else "—"
        _suffer    = row.get("suffer_score")
        _suffer_disp = f"{_suffer:.0f}" if pd.notna(_suffer) else "—"
        _hr_max_disp = m["hr_max"] + " bpm" if m["hr_max"] != "—" else "—"
        _elev_raw  = float(row.get("total_elevation_gain", 0) or 0)
        if row["type"] == "AlpineSki":
            _elev_raw = 0
        _elev_disp = f"{_elev_raw:.0f} m"
        _speed_raw = (row["distance"]/1000) / (row["moving_time"]/3600) if row["moving_time"] > 0 else 0
        _speed_disp = f"{_speed_raw:.1f} km/h"

        _stat_items = [
            ("📏", "Distanza",    m["dist_str"]),
            ("⏱",  "Durata",      m["dur_str"]),
            ("⚡",  "Vel. media",  m["pace_str"]),
            ("⛰",  "Dislivello",  _elev_disp),
            ("❤️", "FC media",    m["hr_avg"] + " bpm" if m["hr_avg"] != "—" else "—"),
            ("💓", "FC massima",  _hr_max_disp),
            ("⚡",  "Watt medi",   m["watts"]),
            ("🔄", "Cadenza",     _cad_disp),
            ("🔥", "Calorie",     _cals_disp),
            ("📊", "TSS",         f"{row['tss']:.0f}"),
            ("😓", "Suffer",      _suffer_disp),
            ("🏔",  "Velocità",   _speed_disp),
        ]
        # Filtra i "—" meno informativi tranne TSS e distanza
        _stat_shown = [x for x in _stat_items if x[2] != "—" or x[1] in ("TSS","Distanza","Durata")]

        _stats_html = (
            '<div class="mob-card">'
            '<div class="mob-card-title">📊 Statistiche</div>'
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:4px">'
        )
        for _ico, _lbl, _val in _stat_shown:
            _stats_html += (
                f'<div style="background:#f8f9fa;border-radius:8px;padding:7px 5px;text-align:center">'
                f'<div style="font-size:9px;color:#aaa;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.3px;margin-bottom:2px">{_ico} {_lbl}</div>'
                f'<div style="font-size:14px;font-weight:800;color:#1a1a1a;line-height:1">{_val}</div>'
                f'</div>'
            )
        _stats_html += '</div></div>'
        st.markdown(_stats_html, unsafe_allow_html=True)

        # ============================================================
        # MAPPA 2D / 3D — CON FETCH ON-DEMAND DA STRAVA
        # ============================================================
        import streamlit.components.v1 as _components
        def _get_polyline(row_data):
            """Estrae summary_polyline gestendo dict, stringa JSON o stringa diretta."""
            _map = row_data.get("map", {})
            if isinstance(_map, dict):
                return _map.get("summary_polyline") or None
            if isinstance(_map, str) and _map.strip():
                # Potrebbe essere JSON serializzato da GSheet
                try:
                    _parsed = json.loads(_map)
                    if isinstance(_parsed, dict):
                        return _parsed.get("summary_polyline") or None
                except Exception:
                    pass
                # O direttamente la polyline
                if len(_map) > 10:
                    return _map
            # Prova anche il campo diretto (alcuni dataset GSheet lo appiattiscono)
            _sp = row_data.get("map.summary_polyline") or row_data.get("summary_polyline")
            if isinstance(_sp, str) and len(_sp) > 10:
                return _sp
            return None

        # Primo tentativo: polyline dalle cache (GSheet)
        poly = _get_polyline(row)

        # Secondo tentativo: session_state cache (fetch precedente)
        _poly_key = f"poly_cache_{_sid}"
        if not poly and _poly_key in st.session_state:
            poly = st.session_state[_poly_key]

        # Terzo tentativo: auto-fetch da Strava alla prima apertura
        if not poly:
            _auto_fetch_key = f"poly_tried_{_sid}"
            if _auto_fetch_key not in st.session_state:
                st.session_state[_auto_fetch_key] = True
                with st.spinner("🗺️ Carico traccia GPS da Strava..."):
                    _det = fetch_activity_details_from_strava(_sid, access_token)
                    if _det and _det.get("map"):
                        _fp = _det["map"].get("summary_polyline")
                        if _fp:
                            st.session_state[_poly_key] = _fp
                            poly = _fp
                        else:
                            st.session_state[_poly_key] = "__no_poly__"
                    else:
                        st.session_state[_poly_key] = "__no_poly__"
                if poly:
                    st.rerun()
            else:
                # Fetch già tentato, nessuna traccia disponibile
                if st.session_state.get(_poly_key) != "__no_poly__":
                    poly = st.session_state.get(_poly_key)

        # Mostra mappa se polyline disponibile
        if poly and poly != "__no_poly__":
            st.markdown('<div class="mob-card"><div class="mob-card-title">🗺️ Traccia GPS</div>',
                        unsafe_allow_html=True)
            
            if MAPBOX_TOKEN:
                _t2d, _t3d = st.tabs(["🗺️ Mappa 2D", "🏔️ Mappa 3D"])
                with _t2d:
                    mobj = draw_map(poly, height=230)
                    if mobj:
                        st_folium(mobj, width=None, height=230, key=f"det_map_2d_{_sid}")
                    else:
                        st.info("❌ Errore nel rendering della mappa 2D.")
                with _t3d:
                    _eg  = float(row.get("total_elevation_gain") or 0)
                    if row.get("type") == "AlpineSki": _eg = 0
                    _m_det = format_metrics(row)
                    _h3d = build_map3d_html(poly, MAPBOX_TOKEN,
                                            sport_type=row.get("type",""),
                                            elev_gain=_eg,
                                            dist_km=_m_det["dist_km"],
                                            dur_str=_m_det["dur_str"],
                                            height=320)
                    if _h3d:
                        _components.html(_h3d, height=330, scrolling=False)
                    else:
                        st.info("Configura MAPBOX_TOKEN nei Secrets per la mappa 3D.")
            else:
                mobj = draw_map(poly, height=230)
                if mobj:
                    st_folium(mobj, width=None, height=230, key=f"det_map_{_sid}")
                else:
                    st.error("❌ Errore nel rendering della mappa.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        elif st.session_state.get(_poly_key) == "__no_poly__":
            st.markdown('<div class="mob-card">'
                        '<div class="mob-card-title">🗺️ Traccia GPS</div>'
                        '<div style="font-size:13px;color:#999;padding:4px 0">'
                        '⚠️ Traccia GPS non disponibile per questa attività.</div>'
                        '</div>', unsafe_allow_html=True)

        # Zone FC — stile Garmin con barre orizzontali
        hr_avg = row.get("average_heartrate")
        fc_max_u = u["fc_max"]
        _dur_fc = float(row.get("moving_time", 0))
        _fc_zones_def = [
            (5, "#F44336", "Zona 5 · Massima",  0.90, 1.00, "Sforzo massimo"),
            (4, "#FF9800", "Zona 4 · Soglia",    0.80, 0.90, "Soglia anaerobica"),
            (3, "#FFC107", "Zona 3 · Aerobico",  0.70, 0.80, "Resistenza aerobica"),
            (2, "#8BC34A", "Zona 2 · Facile",    0.60, 0.70, "Base aerobica"),
            (1, "#4CAF50", "Zona 1 · Riscaldamento", 0.00, 0.60, "Recupero attivo"),
        ]

        # Stima distribuzione FC
        def _fc_time_est(zlo, zhi, hr_pct, dur):
            if hr_pct is None: return 0.20
            mid = (zlo + zhi) / 2
            dist = abs(hr_pct - mid)
            if zlo <= hr_pct < zhi: return 0.60
            if dist < 0.08: return 0.18
            if dist < 0.15: return 0.10
            if dist < 0.25: return 0.05
            return 0.02

        _hr_pct = (hr_avg / fc_max_u) if pd.notna(hr_avg) and fc_max_u > 0 else None
        _fc_raw  = [_fc_time_est(zlo, zhi, _hr_pct, _dur_fc)
                    for (_,_,_,zlo,zhi,_) in _fc_zones_def]
        _fc_tot  = sum(_fc_raw) or 1
        _fc_pcts = [r / _fc_tot for r in _fc_raw]
        _fc_times= [p * _dur_fc for p in _fc_pcts]

        # Trova zona attiva
        _fc_cur_idx = None
        if _hr_pct is not None:
            for _i, (_,_,_,zlo,zhi,_) in enumerate(_fc_zones_def):
                if zlo <= _hr_pct < zhi:
                    _fc_cur_idx = _i
                    break

        st.markdown('<div class="mob-card"><div class="mob-card-title">&#10084;&#65039; Zone Frequenza Cardiaca</div>',
                    unsafe_allow_html=True)
        _fc_html = ""
        for _i, (_zn, _zc, _zlabel, _zlo, _zhi, _zdesc) in enumerate(_fc_zones_def):
            _blo = int(fc_max_u * _zlo)
            _bhi = int(fc_max_u * _zhi) if _zhi < 1.0 else fc_max_u
            _pct  = _fc_pcts[_i]
            _tsec = _fc_times[_i]
            _tmin = int(_tsec // 60)
            _tsec2= int(_tsec % 60)
            _tstr = f"{_tmin}:{_tsec2:02d}"
            _bar_w= max(2, int(_pct * 100))
            _is_cur = (_i == _fc_cur_idx)
            _bg   = f"{_zc}18" if _is_cur else "transparent"
            _brd  = f"border-left:3px solid {_zc};" if _is_cur else "border-left:3px solid transparent;"
            _fc_html += f"""
            <div style="padding:8px 4px;{_brd}background:{_bg};border-radius:0 8px 8px 0;margin:2px 0">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px">
                <div>
                  <span style="font-size:13px;font-weight:700;color:{_zc}">{_zlabel}</span>
                  <span style="font-size:10px;color:#aaa;margin-left:6px">{_blo}&#8211;{_bhi} bpm</span>
                </div>
                <div style="display:flex;gap:12px;align-items:baseline">
                  <span style="font-size:13px;font-weight:700;color:#333">{_tstr}</span>
                  <span style="font-size:12px;color:#888;min-width:32px;text-align:right">{_pct*100:.0f}%</span>
                </div>
              </div>
              <div style="background:#f0f0f0;border-radius:4px;height:6px;overflow:hidden">
                <div style="background:{_zc};width:{_bar_w}%;height:6px;border-radius:4px"></div>
              </div>
            </div>"""
        if pd.notna(hr_avg):
            _fc_html += (f'<div style="font-size:11px;color:#888;margin-top:6px;padding:0 4px">'
                         f'FC media: <b style="color:#333">{hr_avg:.0f} bpm</b> '
                         f'({_hr_pct*100:.0f}% FCmax)</div>' if _hr_pct else "")
        st.markdown(_fc_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Zone Potenza (stile Garmin — barre orizzontali) ──
        watts_avg = row.get("average_watts")
        is_bike   = row["type"] in ("Ride","VirtualRide","MountainBikeRide")
        ftp       = u.get("ftp", 200)
        if is_bike and pd.notna(watts_avg) and watts_avg and watts_avg > 0 and ftp > 0:
            is_estimated = not row.get("device_watts", False)
            _pwr_zones_def = [
                (1,"#9E9E9E","Z1 · Recupero Attivo",  0.00, 0.55, "Pedalata leggera, recupero"),
                (2,"#4CAF50","Z2 · Resistenza",        0.55, 0.75, "Base aerobica, lunghe uscite"),
                (3,"#8BC34A","Z3 · Tempo",             0.75, 0.90, "Sforzo sostenuto confortevole"),
                (4,"#FFC107","Z4 · Soglia",            0.90, 1.05, "Soglia lattato, duro ma gestibile"),
                (5,"#FF9800","Z5 · VO2max",            1.05, 1.20, "Sforzo intenso, breve durata"),
                (6,"#FF5722","Z6 · Anaerobico",        1.20, 1.50, "Scatti brevi, oltre soglia"),
                (7,"#F44336","Z7 · Neuromuscolare",    1.50, 9.99, "Sprint massimale, < 30 sec"),
            ]
            _wpct = watts_avg / ftp
            # Stima distribuzione tempo: usa curva trapezoidale centrata sulla zona attiva
            # (senza dati reali lap-by-lap, usiamo distribuzione approssimata dal avg watts)
            _dur_sec = float(row.get("moving_time", 0))
            def _zone_time_estimate(zlo, zhi, wpct, dur):
                """Stima % tempo in zona basata su distanza da zona media."""
                # La zona che contiene wpct prende ~60%, adiacenti ~20% e 10%
                mid = (zlo + zhi) / 2 if zhi < 9 else zlo + 0.25
                dist = abs(wpct - mid)
                if wpct >= zlo and wpct < zhi: return 0.60
                if dist < 0.15: return 0.18
                if dist < 0.30: return 0.10
                if dist < 0.45: return 0.05
                return 0.02
            _raw = [_zone_time_estimate(zlo, zhi, _wpct, _dur_sec)
                    for (_, _, _, zlo, zhi, _) in _pwr_zones_def]
            _tot = sum(_raw) or 1
            _pcts  = [r/_tot for r in _raw]
            _times = [p * _dur_sec for p in _pcts]

            st.markdown('<div class="mob-card">', unsafe_allow_html=True)
            if is_estimated:
                st.markdown('<div class="mob-card-title">⚡ Zone Potenza ⚠️ <span style="font-weight:400;color:#FF9800;font-size:10px">watt stimati Strava</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="mob-card-title">⚡ Zone Potenza · FTP {ftp} W</div>', unsafe_allow_html=True)

            _watt_cur_zone_idx = next(
                (i for i,(_, _,_,zlo,zhi,_) in enumerate(_pwr_zones_def) if zlo <= _wpct < zhi), 3)

            zones_html = ''
            for i, (zn, zc, zlabel, zlo, zhi, zdesc) in enumerate(_pwr_zones_def):
                _wlo = int(ftp * zlo)
                _whi = f"{int(ftp*zhi)}" if zhi < 9 else "∞"
                _pct = _pcts[i]
                _tsec = _times[i]
                _tmin = int(_tsec // 60)
                _tsec2 = int(_tsec % 60)
                _tstr = f"{_tmin}:{_tsec2:02d}"
                _is_cur = (i == _watt_cur_zone_idx)
                _bar_w  = max(2, int(_pct * 100))
                _bg     = f"{zc}18" if _is_cur else "transparent"
                _brd    = f"border-left:3px solid {zc};" if _is_cur else "border-left:3px solid transparent;"
                zones_html += f"""
                <div style="padding:8px 4px;{_brd}background:{_bg};border-radius:0 8px 8px 0;margin:2px 0">
                  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px">
                    <div>
                      <span style="font-size:13px;font-weight:700;color:{zc}">{zlabel}</span>
                      <span style="font-size:10px;color:#aaa;margin-left:6px">{_wlo}–{_whi} W</span>
                    </div>
                    <div style="display:flex;gap:12px;align-items:baseline">
                      <span style="font-size:13px;font-weight:700;color:#333">{_tstr}</span>
                      <span style="font-size:12px;color:#888;min-width:32px;text-align:right">{_pct*100:.0f}%</span>
                    </div>
                  </div>
                  <div style="background:#f0f0f0;border-radius:4px;height:6px;overflow:hidden">
                    <div style="background:{zc};width:{_bar_w}%;height:6px;border-radius:4px;
                         transition:width 0.3s"></div>
                  </div>
                </div>"""
            st.markdown(zones_html, unsafe_allow_html=True)

            # Metriche potenza
            _if_val = watts_avg / ftp
            _tss_p  = (row["moving_time"] * watts_avg * _if_val) / (ftp * 3600) * 100
            pw1, pw2, pw3 = st.columns(3)
            pw1.metric("⚡ Watt medi",  f"{watts_avg:.0f} W")
            pw2.metric("📊 IF",          f"{_if_val:.2f}")
            pw3.metric("📈 TSS pot.",    f"{_tss_p:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Analisi AI — auto con contesto ricco 14gg
        st.markdown('<div class="mob-card"><div class="mob-card-title">&#129302; Analisi Coach</div>',
                    unsafe_allow_html=True)
        _aid = str(row.get("id", str(row["start_date"])))
        _ck  = f"mob_ai_{_aid}"
        if _ck in st.session_state:
            st.markdown(f'<div class="ai-box">{st.session_state[_ck]}</div>',
                        unsafe_allow_html=True)
            if st.button("&#128260; Rigenera", key=f"regen_{_aid}", use_container_width=True):
                del st.session_state[_ck]
                st.rerun()
        else:
            # Genera automaticamente senza bisogno di pulsante
            with st.spinner("Il coach sta analizzando (14gg contesto)..."):
                _rich_ctx = build_activity_context(
                    row, df, u, current_ctl, current_atl, current_tsb, status_label,
                    window_days=14
                )
                _prompt = (
                    _rich_ctx +
                    "\n\n=== ISTRUZIONI COACH ===\n"
                    "Sei un coach sportivo d'élite specializzato in ciclismo e trail running. "
                    "Rispondi in italiano. Analizza questa sessione in 3 paragrafi:\n"
                    "1) Qualità dell'allenamento e zone di lavoro rispetto al profilo dell'atleta\n"
                    "2) Come questa sessione si inserisce nel contesto delle ultime 2 settimane\n"
                    "3) Un consiglio concreto e specifico per la prossima sessione\n"
                    "Sii diretto, usa i dati numerici, niente frasi generiche."
                )
                _res = ai_deep(_prompt)
                st.session_state[_ck] = _res
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.stop()

# ============================================================
# ── MENU: DASHBOARD ──────────────────────────────────────────
# ============================================================
if st.session_state.mob_menu == "dashboard":

    # ── Cache dei calcoli della dashboard — si ricalcola solo se il df cambia ──
    _dash_key = f"dash_computed_{_df_cache_key}"
    if st.session_state.get("_dash_computed_key") != _dash_key:
        _spark_days = 14
        _spark_data = pd.DataFrame({
            "ctl": ctl_daily, "atl": atl_daily, "tsb": tsb_daily,
        }).dropna().tail(_spark_days)

        _ctl_vals = list(_spark_data["ctl"]) if not _spark_data.empty else [current_ctl]
        _atl_vals = list(_spark_data["atl"]) if not _spark_data.empty else [current_atl]
        _tsb_vals = list(_spark_data["tsb"]) if not _spark_data.empty else [current_tsb]

        _7d_ago = df[df["start_date"] <= df["start_date"].max() - pd.Timedelta(days=7)]
        def _delta_html(cur, series_7d):
            if series_7d.empty: return ""
            old = float(series_7d.iloc[-1])
            d = cur - old
            arrow = "&#8593;" if d > 0.5 else "&#8595;" if d < -0.5 else "&#8594;"
            col = "#4CAF50" if d > 0.5 else "#F44336" if d < -0.5 else "#aaa"
            return f'<span style="color:{col};font-size:11px">{arrow}{abs(d):.0f}</span>'

        _dh_ctl = _delta_html(current_ctl, _7d_ago["ctl"] if not _7d_ago.empty else pd.Series())
        _dh_atl = _delta_html(current_atl, _7d_ago["atl"] if not _7d_ago.empty else pd.Series())
        _dh_tsb = _delta_html(current_tsb, _7d_ago["tsb"] if not _7d_ago.empty else pd.Series())

        ctl_color = "#4CAF50" if current_ctl > 60 else "#FF9800" if current_ctl > 40 else "#F44336"
        tsb_color = "#4CAF50" if current_tsb > 5 else "#FF9800" if current_tsb > -15 else "#F44336"
        atl_color = "#FF9800" if current_atl > current_ctl * 1.1 else "#4CAF50"

        _svg_ctl = make_sparkline_svg(_ctl_vals, ctl_color, width=88, height=30)
        _svg_atl = make_sparkline_svg(_atl_vals, atl_color, width=88, height=30)
        _svg_tsb = make_sparkline_svg(_tsb_vals, tsb_color, width=88, height=30, show_zero_line=True)

        _last7 = df[df["start_date"] >= df["start_date"].max() - pd.Timedelta(days=7)]
        _tss7  = str(round(_last7["tss"].sum()))
        _n7    = str(len(_last7))

        # Recap 7gg — calorie vettorizzate
        _w7 = _last7
        _w7_hrs  = _w7["moving_time"].sum() / 3600
        _w7_km   = _w7["distance"].sum() / 1000
        _w7_elev = _w7["total_elevation_gain"].sum()
        _w7_tss  = _w7["tss"].sum()
        _w7_n    = len(_w7)
        _cal_col  = _w7["calories"].fillna(0)
        _kj_col   = _w7["kilojoules"].fillna(0) * 0.239
        _fc_est   = (_w7["moving_time"] / 60 * (
            0.014 * _w7["average_heartrate"].fillna(0) - 0.05
        ) * float(u.get("peso",75)) / 60 * 4.184).fillna(0)
        _w7_kcal = float(
            np.where(_cal_col > 0, _cal_col,
            np.where(_kj_col  > 0, _kj_col, _fc_est))
        .sum())
        _w7_sports    = _w7["type"].value_counts()
        _sport_icons  = " ".join([SPORT_INFO.get(t, {"icon":"🏅"})["icon"] for t in _w7_sports.index[:3]])

        # ── Settimana precedente (7-14gg fa) per delta A ──
        _ref_date = df["start_date"].max()
        _pw7 = df[
            (df["start_date"] >= _ref_date - pd.Timedelta(days=14)) &
            (df["start_date"] <  _ref_date - pd.Timedelta(days=7))
        ]
        _pw7_hrs  = _pw7["moving_time"].sum() / 3600
        _pw7_km   = _pw7["distance"].sum() / 1000
        _pw7_elev = _pw7["total_elevation_gain"].sum()
        _pw7_tss  = _pw7["tss"].sum()
        _pw7_n    = len(_pw7)
        _cal_col_p = _pw7["calories"].fillna(0)
        _kj_col_p  = _pw7["kilojoules"].fillna(0) * 0.239
        _fc_est_p  = (_pw7["moving_time"] / 60 * (
            0.014 * _pw7["average_heartrate"].fillna(0) - 0.05
        ) * float(u.get("peso",75)) / 60 * 4.184).fillna(0)
        _pw7_kcal = float(
            np.where(_cal_col_p > 0, _cal_col_p,
            np.where(_kj_col_p  > 0, _kj_col_p, _fc_est_p))
        .sum())

        def _delta_pct(cur, prev):
            if prev == 0:
                return None
            return (cur - prev) / prev * 100

        _w7_deltas = [
            _delta_pct(_w7_hrs,  _pw7_hrs),
            _delta_pct(_w7_km,   _pw7_km),
            _delta_pct(_w7_elev, _pw7_elev),
            _delta_pct(_w7_kcal, _pw7_kcal),
            _delta_pct(_w7_tss,  _pw7_tss),
            _delta_pct(_w7_n,    _pw7_n),
        ]

        # ── Pallini giornalieri (D) — ultimi 7 giorni ──
        _today_date = datetime.now(timezone.utc).date()
        _daily_dots = []
        for _d in range(6, -1, -1):
            _day = _today_date - timedelta(days=_d)
            _day_mask = _w7["start_date"].dt.date == _day if len(_w7) > 0 else pd.Series([], dtype=bool)
            _day_acts = _w7[_day_mask] if len(_w7) > 0 else pd.DataFrame()
            if len(_day_acts) == 0:
                _daily_dots.append({"color": None, "icon": None, "label": _day.strftime("%a")[0].upper()})
            else:
                _top_type = _day_acts["type"].iloc[0]
                _sport    = SPORT_INFO.get(_top_type, {"icon": "🏅", "color": "#9E9E9E"})
                _daily_dots.append({"color": _sport["color"], "icon": _sport["icon"], "label": _day.strftime("%a")[0].upper()})

        st.session_state.update({
            "_dash_computed_key": _dash_key,
            "_dash_ctl_vals": _ctl_vals, "_dash_atl_vals": _atl_vals, "_dash_tsb_vals": _tsb_vals,
            "_dash_dh_ctl": _dh_ctl, "_dash_dh_atl": _dh_atl, "_dash_dh_tsb": _dh_tsb,
            "_dash_ctl_color": ctl_color, "_dash_tsb_color": tsb_color, "_dash_atl_color": atl_color,
            "_dash_svg_ctl": _svg_ctl, "_dash_svg_atl": _svg_atl, "_dash_svg_tsb": _svg_tsb,
            "_dash_tss7": _tss7, "_dash_n7": _n7,
            "_dash_w7_hrs": _w7_hrs, "_dash_w7_km": _w7_km, "_dash_w7_elev": _w7_elev,
            "_dash_w7_tss": _w7_tss, "_dash_w7_n": _w7_n, "_dash_w7_kcal": _w7_kcal,
            "_dash_sport_icons": _sport_icons,
            "_dash_w7_deltas": _w7_deltas,
            "_dash_daily_dots": _daily_dots,
        })

    # Leggi valori dalla cache
    ctl_color    = st.session_state["_dash_ctl_color"]
    tsb_color    = st.session_state["_dash_tsb_color"]
    atl_color    = st.session_state["_dash_atl_color"]
    _svg_ctl     = st.session_state["_dash_svg_ctl"]
    _svg_atl     = st.session_state["_dash_svg_atl"]
    _svg_tsb     = st.session_state["_dash_svg_tsb"]
    _dh_ctl      = st.session_state["_dash_dh_ctl"]
    _dh_atl      = st.session_state["_dash_dh_atl"]
    _dh_tsb      = st.session_state["_dash_dh_tsb"]
    _tss7        = st.session_state["_dash_tss7"]
    _n7          = st.session_state["_dash_n7"]
    _w7_hrs      = st.session_state["_dash_w7_hrs"]
    _w7_km       = st.session_state["_dash_w7_km"]
    _w7_elev     = st.session_state["_dash_w7_elev"]
    _w7_tss      = st.session_state["_dash_w7_tss"]
    _w7_n        = st.session_state["_dash_w7_n"]
    _w7_kcal     = st.session_state["_dash_w7_kcal"]
    _sport_icons = st.session_state["_dash_sport_icons"]
    _w7_deltas   = st.session_state.get("_dash_w7_deltas", [None]*6)
    _daily_dots  = st.session_state.get("_dash_daily_dots", [])

    def _spark_card(val_str, label, sub, color, delta_html, svg):
        return (
            f'<div style="flex:1;background:rgba(255,255,255,0.07);border-radius:14px;'
            f'padding:12px 10px;border:1px solid rgba(255,255,255,0.12);">'
            f'<div style="font-size:28px;font-weight:900;color:#fff;line-height:1">{val_str}</div>'
            f'<div style="font-size:11px;font-weight:700;color:rgba(255,255,255,0.7);margin:3px 0 0">{label}</div>'
            f'<div style="font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:6px">{sub}</div>'
            f'<div style="line-height:0;opacity:0.7">{svg}</div>'
            f'<div style="margin-top:4px">{delta_html} '
            f'<span style="font-size:10px;color:rgba(255,255,255,0.3)">vs 7gg</span></div>'
            f'</div>'
        )

    _card_ctl = _spark_card(f"{current_ctl:.0f}", "CTL", "Fitness", ctl_color, _dh_ctl, _svg_ctl)
    _card_tsb = _spark_card(f"{current_tsb:+.0f}", "TSB", "Forma", tsb_color, _dh_tsb, _svg_tsb)
    _card_atl = _spark_card(f"{current_atl:.0f}", "ATL", "Fatica", atl_color, _dh_atl, _svg_atl)

    _src_badge = (
        f'<span style="background:rgba(0,0,0,0.25);border-radius:8px;padding:2px 8px;'
        f'font-size:10px;font-weight:700;color:{_fitness_color};letter-spacing:0.3px">'
        f'● {_fitness_source}</span>'
    )
    st.markdown(
        f'<div class="hero-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
        f'<div style="font-size:11px;color:rgba(255,255,255,0.5);font-weight:700;'
        f'letter-spacing:0.6px;text-transform:uppercase">📈 Performance Management</div>'
        f'{_src_badge}'
        f'</div>'
        f'<div style="display:flex;gap:8px;margin-bottom:12px">'
        + _card_ctl + _card_tsb + _card_atl +
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div style="font-size:12px;color:rgba(255,255,255,0.45)">'
        f'TSS 7gg: <span style="color:rgba(255,255,255,0.8);font-weight:700">{_tss7}</span>'
        f' · sessioni: <span style="color:rgba(255,255,255,0.8);font-weight:700">{_n7}</span></div>'
        f'<div style="background:rgba(255,255,255,0.12);border-radius:20px;'
        f'padding:4px 12px;font-size:12px;font-weight:700;color:rgba(255,255,255,0.9)">'
        f'{status_label}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    _w7_metrics = [
        ("⏱", f"{_w7_hrs:.1f}h",         "ore",   "#3B82F6", "#EFF6FF"),
        ("📏", f"{_w7_km:.0f}",           "km",    "#10B981", "#ECFDF5"),
        ("⛰",  f"{_w7_elev/1000:.1f}k",  "D+",    "#8B5CF6", "#F5F3FF"),
        ("🔥", f"{_w7_kcal:.0f}",         "kcal",  "#F97316", "#FFF7ED"),
        ("📊", f"{_w7_tss:.0f}",          "TSS",   "#1A56DB", "#EFF6FF"),
        ("🏅", f"{_w7_n}",                "sess.", "#0F2744", "#F1F5F9"),
    ]

    def _delta_badge(pct):
        if pct is None:
            return '<span style="font-size:10px;color:#aaa">—</span>'
        arrow = "↑" if pct >= 0 else "↓"
        color = "#16A34A" if pct >= 0 else "#DC2626"
        bg    = "#DCFCE7" if pct >= 0 else "#FEE2E2"
        sign  = "+" if pct >= 0 else ""
        return (
            f'<span style="display:inline-block;background:{bg};color:{color};'
            f'border-radius:6px;padding:1px 5px;font-size:10px;font-weight:700;line-height:1.4">'
            f'{arrow}{sign}{pct:.0f}%</span>'
        )

    _recap_html = (
        '<div class="mob-card" style="margin-top:8px">'
        '<div class="mob-card-title">📆 Ultimi 7 giorni'
        ' · <span style="font-weight:400;text-transform:none;letter-spacing:0;font-size:10px">vs settimana precedente</span></div>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:4px">'
    )
    for _i, (_ico, _val, _lbl, _mc, _mbg) in enumerate(_w7_metrics):
        _badge = _delta_badge(_w7_deltas[_i] if _i < len(_w7_deltas) else None)
        _recap_html += (
            f'<div style="background:{_mbg};border-radius:12px;padding:10px 8px;text-align:center">'
            f'<div style="font-size:10px;color:{_mc};opacity:0.7;margin-bottom:2px;font-weight:600">{_ico} {_lbl}</div>'
            f'<div style="font-size:22px;font-weight:900;color:{_mc};line-height:1;margin-bottom:4px">{_val}</div>'
            f'{_badge}'
            f'</div>'
        )
    _recap_html += '</div>'

    if _daily_dots:
        _days_html = (
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'margin-top:12px;padding-top:10px;border-top:1px solid #f0f0f0">'
        )
        for _dot in _daily_dots:
            _is_today = (_dot == _daily_dots[-1])
            _outline  = "2px solid #1A56DB" if _is_today else "none"
            if _dot["color"]:
                _dot_html = (
                    f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px">'
                    f'<div style="width:30px;height:30px;border-radius:50%;background:{_dot["color"]};'
                    f'display:flex;align-items:center;justify-content:center;font-size:14px;'
                    f'outline:{_outline};outline-offset:1px">{_dot["icon"]}</div>'
                    f'<span style="font-size:9px;color:#94A3B8;font-weight:600">{_dot["label"]}</span>'
                    f'</div>'
                )
            else:
                _dot_html = (
                    f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px">'
                    f'<div style="width:30px;height:30px;border-radius:50%;background:#E2E8F0;'
                    f'outline:{_outline};outline-offset:1px"></div>'
                    f'<span style="font-size:9px;color:#CBD5E1;font-weight:600">{_dot["label"]}</span>'
                    f'</div>'
                )
            _days_html += _dot_html
        _days_html += '</div>'
        _recap_html += _days_html

    _recap_html += '</div>'
    st.markdown(_recap_html, unsafe_allow_html=True)

    # ── Briefing giornaliero ──
    if _ai_sdk_mode is not None:
        _bkey = get_daily_briefing_key()
        if _bkey not in st.session_state:
            with st.spinner("🤖 Briefing coach in preparazione..."):
                _brief = build_daily_briefing(
                    df, u, current_ctl, current_atl, current_tsb, status_label, vo2max_val)
                st.session_state[_bkey] = _brief
                st.rerun()
        if _bkey in st.session_state:
            _bt = st.session_state[_bkey]
            if not str(_bt).startswith("⚠️"):
                import re as _re

                # Configurazione sezioni
                _section_cfg = [
                    ("1.", "📈", "Stato Forma", "#EFF6FF", "#1D4ED8"),
                    ("2.", "🏅", "Ultime Sessioni", "#F0FDF4", "#15803D"),
                    ("3.", "🗓️", "Prossimi Giorni", "#FFF7ED", "#C2410C"),
                ]

                # Splitta il testo nelle 3 sezioni
                _raw_text = str(_bt)
                _sections_text = []
                _parts = _re.split(r'(?=\b[123]\.\s)', _raw_text)
                for _p in _parts:
                    _p = _p.strip()
                    if _p:
                        _sections_text.append(_p)

                # Se lo split non ha funzionato, metti tutto in sezione 1
                if len(_sections_text) < 2:
                    _sections_text = [_raw_text]

                def _highlight_nums(text):
                    """Evidenzia numeri CTL/ATL/TSB/TSS/FTP inline."""
                    text = _re.sub(
                        r'\b(CTL|ATL|TSB|TSS|FTP|VO2max?|W/kg|km/h|bpm)\s*[=:>]?\s*([+\-]?\d+[\.,]?\d*)',
                        lambda m: f'{m.group(1)} <span class="brief-num">{m.group(2)}</span>',
                        text
                    )
                    return text

                _brief_html = ''
                for _si, _st in enumerate(_sections_text):
                    if _si < len(_section_cfg):
                        _pfx, _ico, _ttl, _bg, _tc = _section_cfg[_si]
                        # Rimuovi il prefisso numerico
                        _body = _re.sub(r'^[123]\.\s*', '', _st).strip()
                        _body = _body.replace('\n', '<br>')
                        _body = _highlight_nums(_body)
                        _brief_html += (
                            f'<div class="brief-section" style="background:{_bg}">'
                            f'<span class="brief-section-icon">{_ico}</span>'
                            f'<div class="brief-section-title" style="color:{_tc}">{_ttl}</div>'
                            f'<div class="brief-section-body">{_body}</div>'
                            f'</div>'
                        )
                    else:
                        # Testo extra senza sezione
                        _brief_html += f'<div style="font-size:13px;color:#64748B;margin-top:4px">{_st}</div>'

                st.markdown(
                    '<div class="mob-card" style="margin-top:8px">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
                    '<div class="mob-card-title" style="margin:0">🤖 Briefing Coach</div>'
                    f'<div style="font-size:10px;color:#94A3B8">{datetime.now().strftime("%d %b")}</div>'
                    '</div>'
                    f'{_brief_html}'
                    '<div style="display:flex;align-items:center;gap:6px;margin-top:8px;'
                    'padding-top:8px;border-top:1px solid #F1F5F9">'
                    '<span style="font-size:14px">🏆</span>'
                    '<span style="font-size:10px;color:#94A3B8">Elite Coach AI</span>'
                    '<div style="flex:1"></div>',
                    unsafe_allow_html=True)
                if st.button("🔄 Rigenera", key="regen_brief", use_container_width=False):
                    if _bkey in st.session_state:
                        del st.session_state[_bkey]
                    st.rerun()
                st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Ultime 5 attività ──
    st.markdown('<div class="sec-pad"><h4 style="margin:16px 0 8px;color:#1a1a1a">🏅 Ultime attività</h4></div>',
                unsafe_allow_html=True)

    # CSS card-wrap non più necessario — ora tutto in HTML puro
    _last5_df = df.iloc[-5:][::-1]
    for _i5, (_, _row5) in enumerate(_last5_df.iterrows()):
        _s5  = get_sport_info(_row5["type"], _row5.get("name",""))
        _m5  = format_metrics(_row5)
        _id5 = _row5.get("id", _row5.name)
        _zn5, _zc5, _zl5 = get_zone_for_activity(_row5, u["fc_max"])
        _hdr = "⏱ Ultima Attività" if _i5 == 0 else ""
        render_act_card(_row5, _m5, _s5, _zc5, _zl5, _id5, header_label=_hdr,
                        key_prefix="dash5")

        # Solo prima attività: mappa + AI
        if _i5 == 0:
            _poly5 = _get_polyline(_row5) if "_get_polyline" in dir() else None
            if _poly5 is None:
                # Fallback inline se _get_polyline non ancora definita
                _pm = _row5.get("map", {})
                if isinstance(_pm, dict):
                    _poly5 = _pm.get("summary_polyline")
                elif isinstance(_pm, str) and len(_pm) > 10:
                    try:
                        _poly5 = json.loads(_pm).get("summary_polyline")
                    except Exception:
                        _poly5 = _pm
            if _poly5:
                _mobj5 = draw_map(_poly5)
                if _mobj5:
                    st_folium(_mobj5, width=None, height=200, key="dash_map_0")

            # AI automatico — si carica subito senza pulsante
            _ak5  = f"dash_ai_{_id5}"
            _tss5 = f"{_row5['tss']:.0f}"
            if _ak5 not in st.session_state and _ai_sdk_mode is not None:
                with st.spinner("&#129302; Coach analizza l'ultima uscita..."):
                    _ctx5 = (
                        f"Sei un coach d'élite. Atleta: {u.get('eta',33)} anni, "
                        f"FTP {u.get('ftp',200)}W, FCmax {u['fc_max']}bpm, "
                        f"CTL={current_ctl:.0f}, ATL={current_atl:.0f}, TSB={current_tsb:.0f} ({status_label}).\n"
                        f"Ultima sessione — Sport: {_s5['label']}, "
                        f"Distanza: {_m5['dist_str']}, Durata: {_m5['dur_str']}, "
                        f"Passo: {_m5['pace_str']}, D+: {_m5['elev']}, "
                        f"FC media: {_m5['hr_avg']} bpm, TSS: {_tss5}.\n\n"
                        "In 2 frasi concise: commenta la qualità della sessione e dai UN consiglio "
                        "pratico per la prossima. Sii diretto, niente preamboli."
                    )
                    _ai5 = ai_generate(_ctx5)
                    st.session_state[_ak5] = _ai5
                    st.rerun()

            if _ak5 in st.session_state and not str(st.session_state[_ak5]).startswith("⚠️"):
                st.markdown(
                    f'<div class="ai-box" style="margin:0 0 8px">{st.session_state[_ak5]}</div>',
                    unsafe_allow_html=True)

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
        name="CTL", line=dict(color="#34D399", width=2.5)))
    fig2.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df["ATL"],
        name="ATL", line=dict(color="#FB923C", width=2.5)))
    fig2.add_trace(go.Scatter(x=pmc_df.index, y=pmc_df["TSB"],
        name="TSB", line=dict(color="#60A5FA", width=2),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.10)"))
    fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
    fig2.update_layout(
        height=220, margin=dict(l=0,r=0,t=8,b=0),
        paper_bgcolor="#0F2744",
        plot_bgcolor="#0F2744",
        legend=dict(
            orientation="h", y=-0.28, font_size=11,
            font_color="rgba(255,255,255,0.6)",
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="rgba(255,255,255,0.4)"),
            linecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=10, color="rgba(255,255,255,0.4)"),
        ),
    )
    st.markdown(
        '<div style="background:#0F2744;border-radius:14px;overflow:hidden;margin:4px 0">',
        unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Fitness per sport ──
    _sport_fitness = compute_fitness_by_sport(df)
    _sport_color   = {"run": "#FF4B4B", "bike": "#2196F3", "mountain": "#4FC3F7"}

    st.markdown('<div class="mob-card"><div class="mob-card-title">🏅 CTL per Sport</div>',
                unsafe_allow_html=True)
    _sf_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0">'
    for _sk, _sv in _sport_fitness.items():
        if _sv is None:
            continue
        _col = _sport_color[_sk]
        _tsb_sign = "+" if _sv["tsb"] >= 0 else ""
        _sf_html += (
            f'<div style="background:#f8f9fa;border-radius:10px;padding:8px 6px;'
            f'text-align:center;border-top:3px solid {_col}">'
            f'<div style="font-size:18px">{_sv["icon"]}</div>'
            f'<div style="font-size:11px;color:#888;font-weight:600">{_sv["label"]}</div>'
            f'<div style="font-size:22px;font-weight:900;color:{_col};line-height:1.1">{_sv["ctl"]:.0f}</div>'
            f'<div style="font-size:10px;color:#aaa">CTL</div>'
            f'<div style="font-size:11px;color:#555;margin-top:3px">'
            f'TSB <b style="color:{"#4CAF50" if _sv["tsb"]>=0 else "#F44336"}">'
            f'{_tsb_sign}{_sv["tsb"]:.0f}</b></div>'
            f'<div style="font-size:10px;color:#bbb">{_sv["n"]} uscite</div>'
            f'</div>'
        )
    _sf_html += '</div>'
    st.markdown(_sf_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── PMC per sport (tab) ──
    _sport_tabs_avail = [(k, v) for k, v in _sport_fitness.items() if v is not None]
    if _sport_tabs_avail:
        st.markdown('<div class="mob-card"><div class="mob-card-title">📊 PMC per Sport — 60 giorni</div>',
                    unsafe_allow_html=True)
        _tab_labels = [f"{v['icon']} {v['label']}" for _, v in _sport_tabs_avail]
        _tabs = st.tabs(_tab_labels)
        for _ti, (_sk, _sv) in enumerate(_sport_tabs_avail):
            with _tabs[_ti]:
                _col = _sport_color[_sk]
                _pmc_s = pd.DataFrame({
                    "CTL": _sv["ctl_daily"],
                    "ATL": _sv["atl_daily"],
                    "TSB": _sv["tsb_daily"],
                }).dropna().tail(60)
                if not _pmc_s.empty:
                    _fig_s = go.Figure()
                    _fig_s.add_trace(go.Scatter(x=_pmc_s.index, y=_pmc_s["CTL"],
                        name="CTL", line=dict(color=_col, width=2)))
                    _fig_s.add_trace(go.Scatter(x=_pmc_s.index, y=_pmc_s["ATL"],
                        name="ATL", line=dict(color="#F44336", width=2, dash="dot")))
                    _fig_s.add_trace(go.Scatter(x=_pmc_s.index, y=_pmc_s["TSB"],
                        name="TSB", line=dict(color="#aaa", width=1.5),
                        fill="tozeroy", fillcolor="rgba(0,0,0,0.04)"))
                    _fig_s.add_hline(y=0, line_dash="dot", line_color="#ddd", line_width=1)
                    _fig_s.update_layout(
                        height=200, margin=dict(l=0,r=0,t=4,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=-0.3, font_size=10),
                        xaxis=dict(gridcolor="rgba(0,0,0,0.04)", tickfont_size=9),
                        yaxis=dict(gridcolor="rgba(0,0,0,0.04)", tickfont_size=9),
                    )
                    st.plotly_chart(_fig_s, use_container_width=True)
                else:
                    st.markdown('<div style="color:#aaa;font-size:13px;padding:8px">Dati insufficienti</div>',
                                unsafe_allow_html=True)
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

    # Volume settimanale — grafico HTML puro, leggibile
    st.markdown('<div class="mob-card"><div class="mob-card-title">📅 Volume settimanale (TSS)</div>',
                unsafe_allow_html=True)
    weekly = tss_daily.resample("W").sum().tail(8)
    _avg_w = float(weekly.mean()) if len(weekly) > 0 else 1
    _max_w = float(weekly.max()) if len(weekly) > 0 else 1

    _chart_html = '<div style="display:flex;flex-direction:column;gap:8px;margin-top:4px">'
    for _d, _v in weekly.items():
        _pct  = int(_v / _max_w * 100) if _max_w > 0 else 0
        _week = _d.strftime("%d/%m")
        # colore in base alla distanza dalla media
        if _v >= _avg_w * 1.15:   _c = "#1A56DB"
        elif _v >= _avg_w * 0.85: _c = "#6366F1"
        elif _v >= _avg_w * 0.5:  _c = "#94A3B8"
        else:                      _c = "#CBD5E1"
        _is_avg = abs(_v - _avg_w) < _avg_w * 0.05
        _chart_html += (
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="font-size:11px;color:#94A3B8;width:36px;flex-shrink:0;text-align:right">{_week}</div>'
            f'<div style="flex:1;background:#F1F5F9;border-radius:6px;height:22px;overflow:hidden">'
            f'<div style="width:{_pct}%;height:22px;background:{_c};border-radius:6px;'
            f'display:flex;align-items:center;justify-content:flex-end;padding-right:6px;'
            f'min-width:32px;transition:width 0.3s">'
            f'<span style="font-size:11px;font-weight:700;color:white">{int(_v)}</span>'
            f'</div></div>'
            f'</div>'
        )
    # Riga media
    _chart_html += (
        f'<div style="border-top:1px dashed #CBD5E1;padding-top:4px;'
        f'font-size:10px;color:#94A3B8;text-align:right">media settimana: {_avg_w:.0f} TSS</div>'
    )
    _chart_html += '</div>'
    st.markdown(_chart_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Piano settimanale AI ──
    if _ai_sdk_mode is not None:
        st.markdown('<div class="mob-card"><div class="mob-card-title">&#128197; Piano Settimana</div>',
                    unsafe_allow_html=True)
        _plan = st.session_state.get("weekly_plan")
        _plan_dt = st.session_state.get("weekly_plan_date")
        _plan_age = (datetime.now() - _plan_dt).days if _plan_dt else 999

        if _plan and _plan_age < 7:
            _plan_date_str = _plan_dt.strftime("%d/%m") if _plan_dt else ""
            st.markdown(
                f'<div style="font-size:10px;color:#aaa;margin-bottom:6px">'
                f'Generato il {_plan_date_str} &middot; valido fino a {(_plan_dt + timedelta(days=7)).strftime("%d/%m") if _plan_dt else "?"}</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="ai-box" style="border-left-color:#9C27B0">' + _plan + '</div>',
                unsafe_allow_html=True)
            if st.button("&#128260; Rigenera piano", use_container_width=True, key="regen_plan"):
                st.session_state.weekly_plan      = None
                st.session_state.weekly_plan_date = None
                st.rerun()
        else:
            if st.button("&#128197; Genera Piano Settimanale", use_container_width=True,
                         type="primary", key="gen_plan"):
                with st.spinner("Il coach costruisce il piano..."):
                    df7p  = df[df["start_date"] >= df["start_date"].max() - pd.Timedelta(days=7)]
                    df28p = df[df["start_date"] >= df["start_date"].max() - pd.Timedelta(days=28)]
                    avg_tss_day = df28p["tss"].sum() / 28 if not df28p.empty else 50

                    _plan_lines = [
                        "Sei un coach sportivo d'elite. Rispondi in italiano.",
                        "Crea un piano di allenamento per i prossimi 7 giorni.",
                        "",
                        "ATLETA: " + str(u.get("eta",33)) + " anni " + str(u.get("peso",75)) + "kg",
                        "FTP=" + str(u.get("ftp",200)) + "W FCmax=" + str(u["fc_max"]) + "bpm",
                        "CTL=" + str(round(current_ctl)) + " ATL=" + str(round(current_atl))
                            + " TSB=" + str(round(current_tsb,1)) + " stato=" + status_label,
                        "VO2max: " + str(vo2max_val or "N/D") + " ml/kg/min",
                        "TSS medio/giorno (28gg): " + str(round(avg_tss_day)),
                        "Ultima settimana: " + str(len(df7p)) + " sessioni TSS=" + str(round(df7p["tss"].sum())),
                        "Sport: ciclismo, trail running, sci alpinismo",
                        "",
                        "Piano 7 giorni (Lun-Dom). Per ogni giorno: tipo sessione o riposo, "
                        "durata, zona target, TSS stimato. "
                        "Formato bullet point. Nota finale su carico totale previsto.",
                    ]
                    plan_ctx = "\n".join(_plan_lines)
                    _new_plan = ai_deep(plan_ctx)
                    st.session_state.weekly_plan      = _new_plan
                    st.session_state.weekly_plan_date = datetime.now()
                    if _gsheet_ok:
                        gsheet_save_weekly_plan(_new_plan)
                    st.rerun()
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
    # Vista toggle con bottoni — evita conflitto CSS con nav radio
    _vt_col1, _vt_col2 = st.columns(2)
    with _vt_col1:
        _cal_btn = st.button("📅 Calendario", use_container_width=True,
                             type="primary" if st.session_state.get("_storico_view","cal") == "cal" else "secondary",
                             key="storico_cal_btn")
    with _vt_col2:
        _lst_btn = st.button("📋 Lista", use_container_width=True,
                             type="primary" if st.session_state.get("_storico_view","cal") == "lst" else "secondary",
                             key="storico_lst_btn")
    if _cal_btn:
        st.session_state["_storico_view"] = "cal"
        st.rerun()
    if _lst_btn:
        st.session_state["_storico_view"] = "lst"
        st.rerun()
    view_toggle = "📋 Lista" if st.session_state.get("_storico_view","cal") == "lst" else "📅 Calendario"

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
            render_act_card(row, m, s, z_c, z_l, _act_id, key_prefix="list")

    else:
        # ── Calendario mensile con navigazione rapida ──
        now = datetime.now()
        if "cal_year"  not in st.session_state: st.session_state.cal_year  = now.year
        if "cal_month" not in st.session_state: st.session_state.cal_month = now.month

        cy, cm = st.session_state.cal_year, st.session_state.cal_month

        # ── Selettori anno/mese compatti in due colonne ──
        _years_avail  = sorted(df["start_date"].dt.year.unique().tolist(), reverse=True)
        _month_labels_full = ["","Gennaio","Febbraio","Marzo","Aprile","Maggio","Giugno",
                               "Luglio","Agosto","Settembre","Ottobre","Novembre","Dicembre"]
        _months_with_acts = sorted(
            df[df["start_date"].dt.year == cy]["start_date"].dt.month.unique().tolist()
        )
        _sel_col1, _sel_col2 = st.columns(2)
        with _sel_col1:
            _yr_idx = _years_avail.index(cy) if cy in _years_avail else 0
            _new_yr = st.selectbox("Anno", _years_avail, index=_yr_idx,
                                   key="sel_year", label_visibility="collapsed")
            if _new_yr != cy:
                st.session_state.cal_year  = _new_yr
                st.session_state.cal_month = (_months_with_acts[-1]
                                               if _months_with_acts else 1)
                st.rerun()
        with _sel_col2:
            if _months_with_acts:
                # Se il mese corrente non esiste in questo anno → prendi l'ultimo disponibile
                if cm not in _months_with_acts:
                    cm = _months_with_acts[-1]
                    st.session_state.cal_month = cm
                _mo_options = _months_with_acts
                _mo_labels  = [_month_labels_full[m] for m in _mo_options]
                _mo_idx     = _mo_options.index(cm) if cm in _mo_options else len(_mo_options)-1
                _new_mo_lbl = st.selectbox("Mese", _mo_labels, index=_mo_idx,
                                            key="sel_month", label_visibility="collapsed")
                _new_mo = _mo_options[_mo_labels.index(_new_mo_lbl)]
                if _new_mo != cm:
                    st.session_state.cal_month = _new_mo
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

        # Lista attività del mese selezionato — schede cliccabili con overlay
        if not month_acts.empty:
            st.markdown(
                f'<div class="sec-pad" style="margin-top:8px">'
                f'<div style="font-size:12px;font-weight:700;color:#888;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:6px">'
                f'{len(month_acts)} attività · TSS {month_acts["tss"].sum():.0f} · '
                f'{month_acts["distance"].sum()/1000:.0f} km</div></div>',
                unsafe_allow_html=True)
            for _, row in month_acts.iloc[::-1].iterrows():
                s_   = get_sport_info(row["type"], row.get("name",""))
                m_   = format_metrics(row)
                _id  = row.get("id", row.name)
                _zn, _zc, _zl = get_zone_for_activity(row, u["fc_max"])
                render_act_card(row, m_, s_, _zc, _zl, _id, key_prefix="cal")

# ============================================================
# ── MENU: COACH CHAT ─────────────────────────────────────────
# ============================================================
elif st.session_state.mob_menu == "chat":

    st.markdown("""
    <style>
    .chat-messages-wrap { display:flex; flex-direction:column; gap:2px; padding-bottom:8px; }
    .typing-dots span {
        display:inline-block; width:7px; height:7px; margin:0 2px;
        background:#1565C0; border-radius:50%; animation:bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay:0.2s; }
    .typing-dots span:nth-child(3) { animation-delay:0.4s; }
    @keyframes bounce { 0%,80%,100%{transform:scale(0.7);opacity:0.5} 40%{transform:scale(1);opacity:1} }
    .qp-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:8px 12px 0; }
    div[data-testid="stChatInput"] {
        position:fixed !important; bottom:110px !important;
        left:0 !important; right:0 !important;
        background:#f0f2f6 !important; padding:8px 12px 4px !important;
        z-index:9999 !important;
        box-shadow:0 -2px 8px rgba(0,0,0,0.06) !important;
    }
    /* Piano strutturato */
    .plan-day {
        border-radius:12px; padding:10px 12px; margin:4px 0;
        border-left:4px solid #ccc;
    }
    .plan-day-rest  { background:#f8f9fa; border-left-color:#ccc; }
    .plan-day-easy  { background:#E8F5E9; border-left-color:#4CAF50; }
    .plan-day-base  { background:#E3F2FD; border-left-color:#2196F3; }
    .plan-day-hard  { background:#FFF3E0; border-left-color:#FF9800; }
    .plan-day-race  { background:#FCE4EC; border-left-color:#E91E63; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-pad"><h3 style="margin:12px 0 4px">💬 Coach AI</h3></div>',
                unsafe_allow_html=True)

    if _ai_sdk_mode is None:
        st.warning("⚠️ Aggiungi GOOGLE_API_KEY nei Secrets per abilitare il Coach AI.")
    else:
        # ── Carica memoria da GSheet (una volta per sessione) ──
        if not st.session_state.get("_memory_loaded") and _gsheet_ok:
            _loaded_mem = gsheet_load_coach_memory()
            if _loaded_mem:
                st.session_state["coach_memory"] = _loaded_mem
            st.session_state["_memory_loaded"] = True

        _memory = st.session_state.get("coach_memory", {})

        # ── Contesto sistema con memoria — cachato per df_cache_key ──
        _ctx_cache_key = f"chat_ctx_{st.session_state.get('_df_cache_key','')}"
        if st.session_state.get("_chat_ctx_key") != _ctx_cache_key or "chat_ctx_cache" not in st.session_state:
            with st.spinner("📊 Carico contesto atleta..."):
                st.session_state["chat_ctx_cache"] = build_chat_context(
                    df, u, current_ctl, current_atl, current_tsb, status_label, vo2max_val
                )
            st.session_state["_chat_ctx_key"] = _ctx_cache_key

        _memory_str = ""
        if _memory:
            _memory_str = "\n\n=== MEMORIA PERSISTENTE (sessioni precedenti) ===\n" + \
                "\n".join(f"• {k}: {v}" for k, v in _memory.items())

        _ctx_sys = (
            "Sei un coach sportivo d'elite specializzato in ciclismo, trail running e sci alpinismo. "
            "Sei sia ANALISTA (spieghi i dati, le cause, i trend) "
            "che PROGRAMMATORE (piani concreti, sessioni specifiche, carichi con numeri). "
            "Personalità: diretto, asciutto, professionale. Zero frasi motivazionali generiche. "
            "Rispondi sempre in italiano. Usa sempre i numeri disponibili. "
            "Se ti chiedono un piano: sessioni con tipo, durata, zona target, TSS stimato. "
            "Se ti chiedono un'analisi: usa CTL/ATL/TSB/TSS/watt/FC con valori precisi.\n\n"
            + st.session_state["chat_ctx_cache"]
            + _memory_str
        )

        # ── Stato forma rapido in cima ──
        _tsb_col = "#4CAF50" if current_tsb > 10 else ("#FF9800" if current_tsb > -5 else "#F44336")
        st.markdown(f"""
        <div style="display:flex;gap:8px;align-items:center;padding:4px 12px 8px;
                    font-size:12px;color:#666;flex-wrap:wrap">
            <span>CTL <b style="color:#4CAF50">{current_ctl:.0f}</b></span>
            <span>ATL <b style="color:#F44336">{current_atl:.0f}</b></span>
            <span>TSB <b style="color:{_tsb_col}">{current_tsb:+.0f}</b></span>
            <span style="background:{status_color}22;color:{status_color};
                         padding:2px 8px;border-radius:20px;font-weight:700">{status_label}</span>
            {"".join([f'<span style="background:#f0f0f0;padding:2px 6px;border-radius:8px;font-size:10px;color:#555">🧠 {k}: {v}</span>' for k,v in list(_memory.items())[:2]]) if _memory else ""}
        </div>
        """, unsafe_allow_html=True)

        # ── Messaggio proattivo del coach (una volta per sessione, se chat vuota) ──
        if (not st.session_state.messages and
                not st.session_state.get("_proactive_done")):
            st.session_state["_proactive_done"] = True
            with st.spinner("🤖 Il coach sta preparando un aggiornamento..."):
                _opener = build_proactive_opener(
                    df, u, current_ctl, current_atl, current_tsb,
                    status_label, _memory
                )
            if _opener:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": _opener
                })
                st.rerun()

        # ── Quick prompts (solo se chat ha ≤1 messaggio) ──
        if len(st.session_state.messages) <= 1:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center;padding:20px 12px 8px">
                    <div style="font-size:40px">🏆</div>
                    <div style="font-size:15px;font-weight:700;color:#1565C0;margin:8px 0 4px">Coach AI</div>
                    <div style="font-size:13px;color:#888">Chiedi qualsiasi cosa sul tuo allenamento</div>
                </div>
                """, unsafe_allow_html=True)

            quick_prompts = [
                ("💪", "Come sto fisicamente?", "CTL, TSB e stato forma"),
                ("🗓️", "Cosa fare oggi?",        "Sessione consigliata"),
                ("📋", "Piano questa settimana", "7 giorni strutturati"),
                ("📊", "Analizza gli ultimi 30gg","Trend e progressi"),
            ]
            _qp_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px 12px 0">'
            for _qi, (_icon, _title, _sub) in enumerate(quick_prompts):
                _qp_html += (
                    f'<div class="qp-card">'
                    f'<div class="qp-card-icon">{_icon}</div>'
                    f'<div class="qp-card-title">{_title}</div>'
                    f'<div class="qp-card-sub">{_sub}</div>'
                    f'</div>'
                )
            _qp_html += '</div>'
            st.markdown(_qp_html, unsafe_allow_html=True)
            # Bottoni invisibili sovrapposti per intercettare i click
            qc = st.columns(2)
            for i, (_icon, _title, _sub) in enumerate(quick_prompts):
                with qc[i % 2]:
                    if st.button(_title, use_container_width=True, key=f"qp_{i}",
                                 type="secondary"):
                        st.session_state.messages.append({"role": "user", "content": _title})
                        st.session_state["_chat_pending"] = True
                        st.rerun()

        # ── Messaggi chat ──
        _user_initials = (athlete.get("firstname","?")[:1] + athlete.get("lastname","?")[:1]).upper() if athlete else "IO"
        st.markdown('<div class="chat-messages-wrap">', unsafe_allow_html=True)
        for _mi, msg in enumerate(st.session_state.messages):
            _ts = datetime.now().strftime("%H:%M")  # approssimazione — non abbiamo ts reale
            if msg["role"] == "user":
                content = str(msg["content"]).replace("\n", "<br>")
                st.markdown(
                    f'<div class="chat-row user">'
                    f'<div class="chat-avatar user-av">{_user_initials}</div>'
                    f'<div class="chat-bubble-wrap user">'
                    f'<div class="chat-user">{content}</div>'
                    f'<div class="chat-ts">{_ts}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
            else:
                content = str(msg["content"]).replace("\n", "<br>")
                # Evidenzia numeri con pattern CTL/ATL/TSB/TSS + numero
                import re as _re
                content = _re.sub(
                    r'\b(CTL|ATL|TSB|TSS|FTP|VO2|W/kg|km/h|bpm)[\s=:]+(\d+[\.,]?\d*)',
                    lambda m: f'{m.group(1)} <span class="brief-num">{m.group(2)}</span>',
                    content
                )
                st.markdown(
                    f'<div class="chat-row">'
                    f'<div class="chat-avatar coach">🏆</div>'
                    f'<div class="chat-bubble-wrap">'
                    f'<div class="chat-ai">{content}</div>'
                    f'<div class="chat-ts">{_ts}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Risposta pendente ──
        if st.session_state.get("_chat_pending") and st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg["role"] == "user":
                st.markdown(
                    '<div class="chat-row">'
                    '<div class="chat-avatar coach">🏆</div>'
                    '<div class="chat-bubble-wrap">'
                    '<div class="chat-ai" style="padding:12px 16px">'
                    '<div class="typing-dots"><span></span><span></span><span></span></div>'
                    '</div>'
                    '<div class="chat-ts">Coach sta elaborando...</div>'
                    '</div></div>',
                    unsafe_allow_html=True)
                _hlines = [
                    ("Atleta" if _m["role"] == "user" else "Coach") + ": " + str(_m["content"])
                    for _m in st.session_state.messages[-12:]
                ]
                res = ai_deep(_ctx_sys + "\n\n=== CONVERSAZIONE ===\n" + "\n".join(_hlines))
                st.session_state.messages.append({"role": "assistant", "content": res})
                st.session_state["_chat_pending"] = False
                # Aggiorna memoria dopo ogni risposta
                if len(st.session_state.messages) % 6 == 0:
                    _new_mem = extract_and_update_memory(
                        st.session_state.messages,
                        st.session_state.get("coach_memory", {})
                    )
                    st.session_state["coach_memory"] = _new_mem
                    if _gsheet_ok:
                        gsheet_save_coach_memory(_new_mem)
                if _gsheet_ok:
                    gsheet_save_conversations(st.session_state.messages)
                st.rerun()

        # ── Input chat ──
        if prompt := st.chat_input("Scrivi al tuo coach..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state["_chat_pending"] = True
            st.rerun()

        # ── Piano strutturato ──
        st.markdown('<div class="mob-card" style="margin-top:8px">'
                    '<div class="mob-card-title">📅 Piano Settimanale Strutturato</div>',
                    unsafe_allow_html=True)

        _splan = st.session_state.get("structured_plan")
        _splan_dt = st.session_state.get("structured_plan_date")
        _splan_age = (datetime.now() - _splan_dt).days if _splan_dt else 999

        if _splan and _splan_age < 7 and isinstance(_splan, dict) and "giorni" in _splan:
            # Visualizza calendario 7 giorni
            st.markdown(
                f'<div style="font-size:11px;color:#888;margin-bottom:8px">'
                f'🎯 {_splan.get("focus","")}</div>',
                unsafe_allow_html=True)

            _tipo_class = {
                "Riposo": "plan-day-rest", "Recupero": "plan-day-easy",
                "Aerobico": "plan-day-easy", "Lungo": "plan-day-base",
                "Soglia": "plan-day-hard", "Intervalli": "plan-day-hard",
                "Gara": "plan-day-race",
            }
            for _day in _splan.get("giorni", []):
                _tipo = _day.get("tipo", "")
                _cls  = _tipo_class.get(_tipo, "plan-day-base")
                _tss_day = _day.get("tss", 0)
                _tss_str = f'<span style="font-size:10px;color:#888">TSS {_tss_day}</span>' if _tss_day else ""
                st.markdown(
                    f'<div class="plan-day {_cls}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div style="font-size:12px;font-weight:700;color:#555">{_day.get("giorno","")}</div>'
                    f'{_tss_str}</div>'
                    f'<div style="font-size:13px;font-weight:700;margin:2px 0">{_tipo} '
                    f'<span style="font-weight:400;color:#777">{_day.get("durata","")}</span></div>'
                    f'<div style="font-size:11px;color:#888">{_day.get("zona","")} · {_day.get("note","")}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            st.markdown(
                f'<div style="font-size:11px;color:#888;text-align:right;margin-top:4px">'
                f'TSS totale previsto: <b>{_splan.get("tss_totale", "?")}</b></div>',
                unsafe_allow_html=True)

            if st.button("🔄 Rigenera piano", use_container_width=True, key="regen_splan"):
                st.session_state["structured_plan"] = None
                st.session_state["structured_plan_date"] = None
                st.rerun()
        else:
            if st.button("📅 Genera Piano 7 Giorni", use_container_width=True,
                         type="primary", key="gen_splan"):
                with st.spinner("Il coach costruisce il piano..."):
                    _sp = build_structured_weekly_plan(
                        df, u, current_ctl, current_atl, current_tsb,
                        status_label, vo2max_val,
                        st.session_state.get("coach_memory", {})
                    )
                    if _sp:
                        st.session_state["structured_plan"] = _sp
                        st.session_state["structured_plan_date"] = datetime.now()
                        st.rerun()
                    else:
                        st.error("Errore nella generazione del piano. Riprova.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Memoria coach ──
        if _memory:
            with st.expander("🧠 Memoria Coach", expanded=False):
                for _mk, _mv in _memory.items():
                    st.markdown(
                        f'<div style="font-size:12px;padding:3px 0;border-bottom:1px solid #f0f0f0">'
                        f'<b style="color:#555">{_mk}</b>: {_mv}</div>',
                        unsafe_allow_html=True)
                if st.button("🗑️ Cancella memoria", key="clear_memory", use_container_width=True):
                    st.session_state["coach_memory"] = {}
                    if _gsheet_ok:
                        gsheet_save_coach_memory({})
                    st.rerun()

        # ── Azioni ──
        if st.session_state.messages:
            st.markdown('<div class="sec-pad" style="margin-top:4px">', unsafe_allow_html=True)
            c_clr1, c_clr2 = st.columns(2)
            with c_clr1:
                if st.button("🗑️ Nuova chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state["_chat_pending"] = False
                    st.session_state["_proactive_done"] = False
                    st.rerun()
            with c_clr2:
                if st.button("🔄 Aggiorna dati", use_container_width=True):
                    if "chat_ctx_cache" in st.session_state:
                        del st.session_state["chat_ctx_cache"]
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

    # ── Selettore modello AI ──
    # ── Card intervals.icu ──
    st.markdown('<div class="mob-card"><div class="mob-card-title">📊 Intervals.icu — Fitness reale</div>',
                unsafe_allow_html=True)
    if INTERVALS_API_KEY:
        _icu_d = st.session_state.get("_icu_fitness_data")
        if _icu_d:
            _ictl = _icu_d["ctl"]
            _iatl = _icu_d["atl"]
            _itsb = _icu_d["tsb"]
            _iramp = _icu_d.get("ramp_rate")
            _ramp_str = f"{_iramp:+.1f}/giorno" if _iramp else "—"
            _icu_w = _icu_d.get("wellness", {})
            _ihrv  = _icu_w.get("hrv") or _icu_w.get("avgHrv")
            _iweight = _icu_w.get("weight")
            _iresthr = _icu_w.get("restingHR")
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px">
              <div style="background:#ECFDF5;border-radius:12px;padding:12px 8px;text-align:center">
                <div style="font-size:10px;color:#16A34A;font-weight:700;margin-bottom:2px">CTL fitness</div>
                <div style="font-size:30px;font-weight:900;color:#15803D;line-height:1">{_ictl:.0f}</div>
              </div>
              <div style="background:#FFF7ED;border-radius:12px;padding:12px 8px;text-align:center">
                <div style="font-size:10px;color:#C2410C;font-weight:700;margin-bottom:2px">ATL fatica</div>
                <div style="font-size:30px;font-weight:900;color:#EA580C;line-height:1">{_iatl:.0f}</div>
              </div>
              <div style="background:#EFF6FF;border-radius:12px;padding:12px 8px;text-align:center">
                <div style="font-size:10px;color:#1D4ED8;font-weight:700;margin-bottom:2px">TSB forma</div>
                <div style="font-size:30px;font-weight:900;color:#1D4ED8;line-height:1">{_itsb:+.0f}</div>
              </div>
              <div style="background:#F5F3FF;border-radius:12px;padding:12px 8px;text-align:center">
                <div style="font-size:10px;color:#7C3AED;font-weight:700;margin-bottom:2px">Ramp rate</div>
                <div style="font-size:18px;font-weight:800;color:#7C3AED;line-height:1;margin-top:4px">{_ramp_str}</div>
              </div>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
              {"" if not _ihrv else f'<span style="background:#FDF4FF;border-radius:8px;padding:4px 10px;font-size:12px;font-weight:600;color:#7E22CE">HRV {_ihrv:.0f}</span>'}
              {"" if not _iweight else f'<span style="background:#F0FDF4;border-radius:8px;padding:4px 10px;font-size:12px;font-weight:600;color:#15803D">⚖️ {_iweight:.1f} kg</span>'}
              {"" if not _iresthr else f'<span style="background:#FEF2F2;border-radius:8px;padding:4px 10px;font-size:12px;font-weight:600;color:#DC2626">❤️ FC {_iresthr:.0f}</span>'}
            </div>
            <div style="font-size:11px;color:#10B981;font-weight:600">✅ Connesso a intervals.icu</div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Chiave configurata ma nessun dato. Verifica INTERVALS_API_KEY e INTERVALS_ATHLETE_ID nei Secrets.")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Aggiorna", use_container_width=True, key="refresh_icu"):
                for k in [k for k in st.session_state if k.startswith("_icu_")]:
                    del st.session_state[k]
                fetch_intervals_wellness.clear()
                fetch_intervals_wellness_range.clear()
                st.rerun()
        with col_r2:
            if st.button("🧹 Svuota cache", use_container_width=True, key="clear_icu"):
                for k in [k for k in st.session_state if k.startswith("_icu_")]:
                    del st.session_state[k]
                st.rerun()
    else:
        st.markdown("""
        <div style="font-size:12px;color:#64748B;line-height:1.8;padding:4px 0">
        ⚪ <b>Non configurato.</b> Per ottenere CTL/ATL/TSB identici a Garmin e Sunto:<br>
        1. Vai su <b>intervals.icu → Settings → Developer Settings</b><br>
        2. Genera la tua <b>API Key</b><br>
        3. Nei Secrets Streamlit aggiungi:<br>
        &nbsp;&nbsp;<code style="background:#f1f5f9;padding:1px 5px;border-radius:4px">INTERVALS_API_KEY = "la_tua_chiave"</code><br>
        &nbsp;&nbsp;<code style="background:#f1f5f9;padding:1px 5px;border-radius:4px">INTERVALS_ATHLETE_ID = "0"</code>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="mob-card"><div class="mob-card-title">🤖 Modello AI</div>',
                unsafe_allow_html=True)
    # Riscovery dinamica se richiesta
    if st.button("🔄 Aggiorna lista modelli", use_container_width=True, key="refresh_models"):
        _refreshed = _discover_available_models()
        st.session_state["_ai_models_cache"] = _refreshed
        st.rerun()
    # Usa cache se disponibile
    _models_to_use = st.session_state.get("_ai_models_cache", _ALL_MODELS_LABELS)
    _model_options = list(_models_to_use.keys())
    _model_labels  = list(_models_to_use.values())
    _cur_pref = st.session_state.get("ai_model_pref","auto")
    _cur_idx  = _model_options.index(_cur_pref) if _cur_pref in _model_options else 0
    _sel_idx  = st.selectbox(
        "Modello",
        options=range(len(_model_options)),
        format_func=lambda i: _model_labels[i],
        index=_cur_idx,
        label_visibility="collapsed",
    )
    _sel_model = _model_options[_sel_idx]
    if _sel_model != _cur_pref:
        st.session_state["ai_model_pref"] = _sel_model
        st.rerun()
    st.markdown(f'<div style="font-size:11px;color:#888;margin-top:4px">'
                f'{len(_model_options)-1} modelli disponibili</div>',
                unsafe_allow_html=True)
    # Nota info modello
    _notes = {
        "auto":                              "Sceglie automaticamente il modello migliore disponibile.",
        "gemini-3.1-flash-lite-preview":     "⭐ Default. Velocissimo e intelligente — ultima generazione.",
        "gemini-2.5-flash-preview-04-17":    "Veloce, intelligente, ottimo per il coaching.",
        "gemini-2.5-flash-preview-05-20":    "Versione aggiornata di Gemini 2.5 Flash.",
        "gemini-2.5-flash-preview":          "Gemini 2.5 Flash — veloce e potente.",
        "gemini-2.5-pro-preview-05-06":      "Migliore qualità, più lento. Ideale per analisi approfondite.",
        "gemini-2.0-flash":                  "Ottimo equilibrio velocità/qualità.",
        "gemini-2.0-flash-lite":             "Molto veloce, qualità leggermente inferiore.",
        "gemini-1.5-flash":                  "Fallback affidabile se i modelli 2.x sono esauriti.",
        "gemini-1.5-pro":                    "Buona qualità, compatibile con SDK vecchio.",
        "grok-3-fast":                       "Richiede XAI_API_KEY nei Secrets.",
        "grok-3":                            "Richiede XAI_API_KEY nei Secrets.",
    }
    st.markdown(f'<div style="font-size:12px;color:#888;margin-top:4px">ℹ️ {_notes.get(_sel_model,"")}</div>',
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Parametri fisiologici
    st.markdown('<div class="mob-card"><div class="mob-card-title">⚙️ Parametri Fisiologici</div>',
                unsafe_allow_html=True)

    with st.form("profilo_form"):
        peso  = st.number_input("Peso (kg)",    value=float(u["peso"]),      min_value=40.0, max_value=150.0, step=0.5)
        eta   = st.number_input("Età (anni)",   value=int(u.get("eta",33)),  min_value=15,   max_value=80)
        fc_min = st.number_input("FC riposo",   value=int(u["fc_min"]),      min_value=30,   max_value=80)
        fc_max = st.number_input("FC massima",  value=int(u["fc_max"]),      min_value=150,  max_value=230)
        ftp    = st.number_input("FTP (Watt)",  value=int(u.get("ftp",200)), min_value=50,   max_value=500)
        if st.form_submit_button("💾 Salva", use_container_width=True, type="primary"):
            _new_profile = {
                "peso": peso, "fc_min": fc_min, "fc_max": fc_max,
                "ftp": ftp, "eta": eta
            }
            st.session_state.user_data = _new_profile
            if _gsheet_ok:
                gsheet_save_profile(_new_profile)
                st.success("✅ Profilo aggiornato e salvato!")
            else:
                st.success("✅ Profilo aggiornato (solo sessione corrente)!")
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

    # ── Google Sheets status semplice ──
    if _gsheet_ok:
        st.markdown('<div class="mob-card"><div class="mob-card-title">💾 Cache Google Sheets</div>',
                    unsafe_allow_html=True)
        last_sync = gsheet_get_last_sync()
        sync_str  = last_sync.strftime("%d/%m/%Y %H:%M") if last_sync else "Mai sincronizzato"
        n_cached  = len(st.session_state.get("activities_cache", []))
        st.markdown(f"""
        <div style="font-size:13px;color:#555">
            🕐 Ultimo sync: <b>{sync_str}</b><br>
            📦 Attività in cache: <b>{n_cached}</b><br>
            <span style="font-size:11px;color:#888">Aggiornamento auto ogni 24h</span>
        </div>
        """, unsafe_allow_html=True)
        _sync_label = "🔄 Forza sync da intervals.icu" if (INTERVALS_API_KEY and INTERVALS_ATHLETE_ID) else "🔄 Forza sync da Strava"
        if st.button(_sync_label, use_container_width=True):
            with st.spinner("Sincronizzazione in corso..."):
                if INTERVALS_API_KEY and INTERVALS_ATHLETE_ID:
                    _raw_new = load_all_from_intervals(INTERVALS_ATHLETE_ID, INTERVALS_API_KEY)
                    raw_new  = [normalize_intervals_activity(a) for a in _raw_new] if _raw_new else []
                else:
                    raw_new = load_all_from_strava(access_token)
                if raw_new:
                    st.session_state.activities_cache = raw_new
                    # Invalida cache df
                    for k in ["_df_cache_key","_df_cached","_ctl_daily","_atl_daily",
                              "_tsb_daily","_tss_daily","_vo2max_val"]:
                        st.session_state.pop(k, None)
                    fetch_intervals_activities_page.clear()
                    gsheet_save_activities(raw_new)
                    st.success(f"✅ {len(raw_new)} attività sincronizzate!")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Logout
    st.markdown('<div class="sec-pad" style="margin-top:12px">', unsafe_allow_html=True)
    if st.button("🚪 Disconnetti da Strava", use_container_width=True):
        for k in ["strava_token_info","activities_cache","activities_last_ts",
                  "activities_token","gsheet_loaded"]:
            st.session_state[k] = {} if k == "strava_token_info" else ([] if "cache" in k else False if "loaded" in k else 0 if "ts" in k else "")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# fine app
