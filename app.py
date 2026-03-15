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
# CSS MOBILE — iniettato una sola volta per sessione
# ============================================================
if not st.session_state.get("_css_injected"):
    st.session_state["_css_injected"] = True
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

  /* Bottone dettaglio cucito sotto la card */
  .act-card + div[data-testid="stButton"] {
      margin: 0 12px 0 !important;
      padding: 0 !important;
  }
  .act-card + div[data-testid="stButton"] > button {
      border-radius: 0 0 14px 14px !important;
      border-top: 1px solid #f0f0f0 !important;
      background: #fafafa !important;
      color: #1565C0 !important;
      font-size: 13px !important;
      font-weight: 600 !important;
      min-height: 40px !important;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
      border-left: none !important;
      border-right: none !important;
      border-bottom: none !important;
  }
  .act-card + div[data-testid="stButton"] > button:hover {
      background: #E3F2FD !important;
  }  .act-title {
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
    df_dates   = df["start_date"].dt.date.map(lambda d: pd.Timestamp(d))
    ctl_mapped = df_dates.map(ctl)
    atl_mapped = df_dates.map(atl)
    tsb_mapped = df_dates.map(tsb)
    return ctl_mapped, atl_mapped, tsb_mapped, ctl, atl, tsb, daily

def calc_vo2max_estimate(df_sorted):
    """VO2max stimato — vettorizzato con numpy, non più iterrows()."""
    runs = df_sorted[
        (df_sorted["type"].isin(["Run","TrailRun"])) &
        (df_sorted["distance"] >= 5000)
    ]
    if runs.empty:
        return None
    time_min = runs["moving_time"] / 60
    dist_m   = runs["distance"]
    valid    = time_min > 0
    if not valid.any():
        return None
    time_min = time_min[valid]
    dist_m   = dist_m[valid]
    vel  = dist_m / time_min
    pct  = (0.8 + 0.1894393 * np.exp(-0.012778 * time_min)
               + 0.2989558 * np.exp(-0.1932605 * time_min))
    vo2  = -4.60 + 0.182258 * vel + 0.000104 * vel**2
    vo2max_arr = np.where(pct > 0, vo2 / pct, 0)
    best = float(np.max(vo2max_arr))
    return round(best, 1) if best > 0 else None

# ============================================================
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
def build_map3d_html(encoded_polyline, mapbox_token, sport_type="", elev_gain=0, height=340) -> str:
    """Mappa Mapbox 3D compatta per mobile con pulsante fullscreen."""
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
        html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  html,body{{width:100%;height:{height}px;background:#000;overflow:hidden}}
  #map{{width:100%;height:100%}}
  .mapboxgl-ctrl-group{{background:rgba(0,0,0,0.5)!important;border:none!important}}
  .mapboxgl-ctrl-group button{{background:rgba(255,255,255,0.15)!important;color:#fff!important}}
  #fs-btn{{
    position:absolute;bottom:12px;left:12px;z-index:10;
    background:rgba(0,0,0,0.6);color:#fff;border:1px solid rgba(255,255,255,0.3);
    border-radius:8px;padding:7px 12px;font-size:13px;font-weight:600;
    cursor:pointer;backdrop-filter:blur(4px);transition:background 0.2s;
  }}
  #fs-btn:hover{{background:rgba(21,101,192,0.8)}}
  body.is-fullscreen{{position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:99999}}
  body.is-fullscreen #map{{height:100vh}}
</style></head><body>
<div id="map"></div>
<button id="fs-btn" onclick="toggleFS()">⛶ Schermo intero</button>
<script>
mapboxgl.accessToken = "{mapbox_token}";
const map = new mapboxgl.Map({{
  container:"map", style:"mapbox://styles/mapbox/satellite-streets-v12",
  center:[{clon},{clat}], zoom:12, pitch:55, bearing:0, antialias:true
}});
map.addControl(new mapboxgl.NavigationControl(),"top-right");
map.on("load",()=>{{
  map.addSource("dem",{{type:"raster-dem",url:"mapbox://mapbox.mapbox-terrain-dem-v1",tileSize:512}});
  map.setTerrain({{"source":"dem","exaggeration":1.5}});
  map.addLayer({{id:"sky",type:"sky",paint:{{"sky-type":"atmosphere","sky-atmosphere-sun":[0,60],"sky-atmosphere-sun-intensity":15}}}});
  map.addSource("route",{{type:"geojson",data:{geoj}}});
  map.addLayer({{id:"route-glow",type:"line",source:"route",
    layout:{{"line-join":"round","line-cap":"round"}},
    paint:{{"line-color":"{line_color}","line-width":6,"line-opacity":0.35,"line-blur":4}}}});
  map.addLayer({{id:"route",type:"line",source:"route",
    layout:{{"line-join":"round","line-cap":"round"}},
    paint:{{"line-color":"{line_color}","line-width":3,"line-opacity":0.95}}}});
  new mapboxgl.Marker({{color:"#4CAF50",scale:0.8}}).setLngLat({start_j}).addTo(map);
  new mapboxgl.Marker({{color:"#F44336",scale:0.8}}).setLngLat({end_j}).addTo(map);
  const coords={_j.dumps(coords)};
  const bounds=coords.reduce((b,c)=>b.extend(c),new mapboxgl.LngLatBounds(coords[0],coords[0]));
  map.fitBounds(bounds,{{padding:40,duration:0}});
}});
var _isFS=false;
function toggleFS(){{
  _isFS=!_isFS;
  if(_isFS){{
    document.body.classList.add('is-fullscreen');
    document.getElementById('fs-btn').textContent='✕ Esci da schermo intero';
    // Prova Fullscreen API nativa
    if(document.documentElement.requestFullscreen) document.documentElement.requestFullscreen();
  }} else {{
    document.body.classList.remove('is-fullscreen');
    document.getElementById('fs-btn').textContent='⛶ Schermo intero';
    if(document.exitFullscreen) document.exitFullscreen();
  }}
  setTimeout(()=>map.resize(),200);
}}
document.addEventListener('fullscreenchange',()=>{{
  if(!document.fullscreenElement && _isFS){{
    _isFS=false;
    document.body.classList.remove('is-fullscreen');
    document.getElementById('fs-btn').textContent='⛶ Schermo intero';
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
    """
    Bottom nav fissa — i bottoni Streamlit vengono posizionati in fondo via CSS fixed.
    Va chiamata UNA SOLA VOLTA all'inizio, dopo l'header. Il CSS position:fixed li 
    mantiene visibili indipendentemente dallo scroll.
    """
    cur = st.session_state.mob_menu

    # CSS iniettato subito — position:fixed è già attivo prima del rendering dei bottoni
    st.markdown(f"""
    <style>
    /* Nav bar fissa — contenitore */
    div[data-testid="stBottom"] > div {{
        background: #ffffff !important;
        border-top: 1px solid #e0e0e0 !important;
        box-shadow: 0 -2px 16px rgba(0,0,0,0.10) !important;
        padding: 4px 4px 8px !important;
    }}
    /* Fallback: ultimo stHorizontalBlock se stBottom non disponibile */
    .nav-bar-fixed {{
        position: fixed !important;
        bottom: 0 !important; left: 0 !important; right: 0 !important;
        z-index: 1000 !important;
        background: #ffffff !important;
        border-top: 1px solid #e0e0e0 !important;
        box-shadow: 0 -2px 16px rgba(0,0,0,0.10) !important;
        padding: 4px 4px 8px !important;
        display: flex !important;
        gap: 0 !important;
    }}
    .nav-bar-fixed > div {{
        padding: 0 2px !important;
        flex: 1 !important;
    }}
    </style>
    <script>
    (function fixNav() {{
        function applyFix() {{
            // Cerca tutti gli stHorizontalBlock con esattamente 5 bottoni nav
            document.querySelectorAll('[data-testid="stHorizontalBlock"]').forEach(function(block) {{
                var btns = block.querySelectorAll('button');
                if (btns.length === 5 && btns[0] && btns[0].innerText.includes('\\n')) {{
                    block.classList.add('nav-bar-fixed');
                    // Stile bottoni
                    btns.forEach(function(b) {{
                        b.style.cssText = 'min-height:52px!important;height:52px!important;' +
                            'border-radius:10px!important;font-size:11px!important;' +
                            'font-weight:600!important;border:none!important;' +
                            'white-space:pre-line!important;line-height:1.2!important;';
                    }});
                }}
            }});
        }}
        // Applica subito e ogni volta che il DOM cambia
        applyFix();
        new MutationObserver(applyFix).observe(document.body, {{childList:true, subtree:true}});
    }})();
    </script>
    """, unsafe_allow_html=True)

    # Bottoni reali — l'unica cosa che Streamlit può cliccare
    _btn_cols = st.columns(5)
    for i, (key, icon, label) in enumerate(NAV_ITEMS):
        with _btn_cols[i]:
            _t = "primary" if cur == key else "secondary"
            if st.button(f"{icon}\n{label}", key=f"nav_btn_{key}",
                         use_container_width=True, type=_t):
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

# ── Carica profilo da GSheet (una volta per sessione) ──
if _gsheet_ok and not st.session_state.get("_profile_loaded"):
    _saved_profile = gsheet_load_profile()
    if _saved_profile:
        merged_profile = dict(st.session_state.user_data)
        merged_profile.update(_saved_profile)
        st.session_state.user_data = merged_profile
    st.session_state["_profile_loaded"] = True

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

    # TSS vettorizzato (era apply() riga per riga)
    df["tss"] = calc_tss_vectorized(df, u)

    # Fitness (CTL/ATL/TSB)
    ctl_s, atl_s, tsb_s, ctl_daily, atl_daily, tsb_daily, tss_daily = compute_fitness(df)
    df["ctl"] = ctl_s.values
    df["atl"] = atl_s.values
    df["tsb"] = tsb_s.values

    # Zone FC vettorizzate (erano 3 apply() separati)
    df["zone_num"], df["zone_color"], df["zone_label"] = assign_zones_vectorized(df, u["fc_max"])

    # VO2max (ora vettorizzato)
    vo2max_val = calc_vo2max_estimate(df)

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

current_ctl = float(df["ctl"].iloc[-1])
current_atl = float(df["atl"].iloc[-1])
current_tsb = float(df["tsb"].iloc[-1])

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
                    _h3d = build_map3d_html(poly, MAPBOX_TOKEN,
                                            sport_type=row.get("type",""), elev_gain=_eg, height=320)
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

    def _spark_card(val_str, label, sub, color, delta_html, svg):
        return (
            f'<div style="flex:1;background:#f8f9fa;border-radius:14px;padding:12px 10px;'
            f'border-top:3px solid {color};">' 
            f'<div style="font-size:30px;font-weight:900;color:{color};line-height:1">{val_str}</div>'
            f'<div style="font-size:11px;font-weight:700;color:#333;margin:3px 0 0">{label}</div>'
            f'<div style="font-size:10px;color:#aaa;margin-bottom:6px">{sub}</div>'
            f'<div style="line-height:0">{svg}</div>'
            f'<div style="margin-top:4px">{delta_html} <span style="font-size:10px;color:#ccc">vs 7gg</span></div>'
            f'</div>'
        )

    _card_ctl = _spark_card(f"{current_ctl:.0f}", "CTL", "Fitness cronico", ctl_color, _dh_ctl, _svg_ctl)
    _card_tsb = _spark_card(f"{current_tsb:+.0f}", "TSB", "&gt;5 fresco", tsb_color, _dh_tsb, _svg_tsb)
    _card_atl = _spark_card(f"{current_atl:.0f}", "ATL", "Fatica 7gg", atl_color, _dh_atl, _svg_atl)

    st.markdown(
        '<div class="mob-card">' +
        f'<div class="mob-card-title">&#128200; Stato Forma &middot; {status_label}</div>' +
        '<div style="display:flex;gap:8px;margin:8px 0">' +
        _card_ctl + _card_tsb + _card_atl +
        '</div>' +
        f'<div style="font-size:11px;color:#aaa;text-align:center;padding:2px 0">'
        f'TSS 7gg: <b style="color:#555">{_tss7}</b> &middot; sessioni: <b style="color:#555">{_n7}</b>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    _w7_metrics = [
        ("⏱", f"{_w7_hrs:.1f}h", "ore attività"),
        ("📏", f"{_w7_km:.0f}", "km totali"),
        ("⛰", f"{_w7_elev/1000:.1f}k", "metri D+"),
        ("🔥", f"{_w7_kcal:.0f}", "kcal stimate"),
        ("📊", f"{_w7_tss:.0f}", "TSS carico"),
        ("🏅", f"{_w7_n}", f"sessioni {_sport_icons}"),
    ]
    _recap_html = (
        '<div class="mob-card" style="margin-top:8px">'
        '<div class="mob-card-title">📆 Ultimi 7 giorni</div>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:4px">'
    )
    for _ico, _val, _lbl in _w7_metrics:
        _recap_html += (
            f'<div style="background:#f8f9fa;border-radius:10px;padding:8px 6px;text-align:center">'
            f'<div style="font-size:10px;color:#aaa;margin-bottom:2px">{_ico} {_lbl}</div>'
            f'<div style="font-size:20px;font-weight:900;color:#1565C0;line-height:1">{_val}</div>'
            f'</div>'
        )
    _recap_html += '</div></div>'
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
                # Formatta le 3 sezioni con separatori visivi
                _sections = str(_bt).split("\n")
                _formatted = ""
                for _line in _sections:
                    _l = _line.strip()
                    if not _l:
                        _formatted += "<br>"
                    elif any(_l.startswith(str(n) + ".") for n in [1, 2, 3]):
                        # Titolo sezione
                        _formatted += (
                            f'<div style="font-size:11px;font-weight:800;color:#1565C0;'
                            f'text-transform:uppercase;letter-spacing:0.5px;'
                            f'margin:10px 0 4px;border-bottom:1px solid #e3f2fd;padding-bottom:3px">'
                            f'{_l}</div>'
                        )
                    else:
                        _formatted += f'<span>{_l}</span><br>'
                st.markdown(
                    '<div class="mob-card" style="margin-top:8px">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
                    '<div class="mob-card-title" style="margin:0">🤖 BRIEFING COACH · OGGI</div>'
                    f'<div style="font-size:10px;color:#bbb">{datetime.now().strftime("%d/%m")}</div>'
                    '</div>'
                    f'<div style="font-size:14px;line-height:1.7;color:#212529">{_formatted}</div>'
                    '<div style="margin-top:10px;text-align:right">',
                    unsafe_allow_html=True)
                if st.button("🔄 Rigenera briefing", key="regen_brief", use_container_width=False):
                    if _bkey in st.session_state:
                        del st.session_state[_bkey]
                    st.rerun()
                st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Ultime 5 attività ──
    st.markdown('<div class="sec-pad"><h4 style="margin:16px 0 4px;color:#1a1a1a">&#127885; Ultime attività</h4></div>',
                unsafe_allow_html=True)

    _last5_df = df.iloc[-5:][::-1]
    for _i5, (_, _row5) in enumerate(_last5_df.iterrows()):
        _s5   = get_sport_info(_row5["type"], _row5.get("name",""))
        _m5   = format_metrics(_row5)
        _id5  = _row5.get("id", _row5.name)
        _zn5, _zc5, _zl5 = get_zone_for_activity(_row5, u["fc_max"])
        _is_first = (_i5 == 0)

        _title_html   = '<div class="mob-card-title">⏱ Ultima Attività</div>' if _is_first else ""
        _name5        = str(_row5["name"])
        _date5        = _row5["start_date"].strftime("%d %b %Y · %H:%M")
        _icon5        = _s5["icon"]
        _color5       = _s5["color"]
        _dist5        = _m5["dist_str"]
        _dur5         = _m5["dur_str"]
        _pace5        = _m5["pace_str"]
        _elev5        = _m5["elev"]
        _hr5          = _m5["hr_avg"]
        _tss5         = f"{_row5['tss']:.0f}"

        # Card con border-radius solo in alto — il bottone chiude in basso
        st.markdown(
            f'<div class="act-card" style="border-left-color:{_color5};'
            f'border-radius:14px 14px 0 0;margin-bottom:0">' +
            _title_html +
            f'<div class="act-title">{_icon5} {_name5}</div>'
            f'<div class="act-meta">{_date5} · '
            f'<span class="zone-chip" style="background:{_zc5}22;color:{_zc5}">{_zl5}</span></div>'
            f'<div class="act-pills" style="margin-top:6px">'
            f'<span class="act-pill">📏 <b>{_dist5}</b></span>'
            f'<span class="act-pill">⏱ <b>{_dur5}</b></span>'
            f'<span class="act-pill">⚡ <b>{_pace5}</b></span>'
            f'<span class="act-pill">⛰ <b>{_elev5}</b></span>'
            f'<span class="act-pill">❤️ <b>{_hr5} bpm</b></span>'
            f'<span class="act-pill">TSS <b>{_tss5}</b></span>'
            f'</div></div>',
            unsafe_allow_html=True)

        # Bottone "Apri dettaglio" — visivamente cucito sotto la card
        if st.button("Apri dettaglio →", key=f"dash5_{_id5}", use_container_width=True):
            st.session_state.selected_act_id = _id5
            st.rerun()
        st.markdown('<div style="margin-bottom:4px"></div>', unsafe_allow_html=True)

        # Solo prima attività: mappa + AI
        if _is_first:
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
            _ak5 = f"dash_ai_{_id5}"
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
        # ── Calendario mensile con navigazione rapida ──
        now = datetime.now()
        if "cal_year"  not in st.session_state: st.session_state.cal_year  = now.year
        if "cal_month" not in st.session_state: st.session_state.cal_month = now.month

        cy, cm = st.session_state.cal_year, st.session_state.cal_month

        # ── Selezione rapida anno ──
        _years_avail = sorted(df["start_date"].dt.year.unique().tolist(), reverse=True)
        st.markdown('<div class="sec-pad" style="margin-top:8px;margin-bottom:4px">'
                    '<div style="font-size:11px;font-weight:700;color:#888;margin-bottom:6px">ANNO</div>'
                    '<div style="display:flex;gap:6px;flex-wrap:wrap">',
                    unsafe_allow_html=True)
        _yr_cols = st.columns(len(_years_avail))
        for _yi, _yr in enumerate(_years_avail):
            with _yr_cols[_yi]:
                _yr_active = "primary" if _yr == cy else "secondary"
                if st.button(str(_yr), key=f"yr_{_yr}", use_container_width=True, type=_yr_active):
                    st.session_state.cal_year  = _yr
                    st.session_state.cal_month = 1
                    st.rerun()
        st.markdown('</div></div>', unsafe_allow_html=True)

        # ── Selezione rapida mese — solo mesi con attività per l'anno selezionato ──
        _months_with_acts = sorted(
            df[df["start_date"].dt.year == cy]["start_date"].dt.month.unique().tolist()
        )
        _month_labels = ["","Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"]
        if _months_with_acts:
            st.markdown('<div class="sec-pad" style="margin-bottom:8px">'
                        '<div style="font-size:11px;font-weight:700;color:#888;margin-bottom:6px">MESE</div>'
                        '<div style="display:flex;gap:6px;flex-wrap:wrap">',
                        unsafe_allow_html=True)
            _mo_cols = st.columns(len(_months_with_acts))
            for _mi, _mo in enumerate(_months_with_acts):
                with _mo_cols[_mi]:
                    _mo_active = "primary" if _mo == cm else "secondary"
                    if st.button(_month_labels[_mo], key=f"mo_{cy}_{_mo}",
                                 use_container_width=True, type=_mo_active):
                        st.session_state.cal_month = _mo
                        st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)

            # Se il mese corrente non ha attività in questo anno, vai al primo disponibile
            if cm not in _months_with_acts:
                cm = _months_with_acts[-1]
                st.session_state.cal_month = cm

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

        # Lista attività del mese selezionato — schede ricche cliccabili
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
                _hr  = m_["hr_avg"]
                _tss = f"{row['tss']:.0f}"
                _watts_str = f" · ⚡ {m_['watts']}" if m_["watts"] != "—" else ""

                st.markdown(
                    f'<div class="act-card" style="border-left-color:{s_["color"]};margin-bottom:0;border-radius:14px 14px 0 0">'
                    f'<div class="act-title">{s_["icon"]} {str(row["name"])}</div>'
                    f'<div class="act-meta">'
                    f'{row["start_date"].strftime("%d %b · %H:%M")} &middot; '
                    f'<span class="zone-chip" style="background:{_zc}22;color:{_zc}">{_zl}</span>'
                    f'</div>'
                    f'<div class="act-pills" style="margin-top:6px">'
                    f'<span class="act-pill">📏 <b>{m_["dist_str"]}</b></span>'
                    f'<span class="act-pill">⏱ <b>{m_["dur_str"]}</b></span>'
                    f'<span class="act-pill">⚡ <b>{m_["pace_str"]}</b></span>'
                    f'<span class="act-pill">⛰ <b>{m_["elev"]}</b></span>'
                    f'<span class="act-pill">❤️ <b>{_hr} bpm</b></span>'
                    f'<span class="act-pill">TSS <b>{_tss}</b></span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True)
                # Bottone che sembra la parte inferiore della card
                if st.button("Dettaglio →", key=f"cal_det_{_id}", use_container_width=True):
                    st.session_state.selected_act_id = _id
                    st.rerun()
                # Spacer
                st.markdown('<div style="margin-bottom:8px"></div>', unsafe_allow_html=True)

# ============================================================
# ── MENU: COACH CHAT ─────────────────────────────────────────
# ============================================================
elif st.session_state.mob_menu == "chat":

    st.markdown("""
    <style>
    /* Chat container scrollabile */
    .chat-messages-wrap {
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding-bottom: 8px;
    }
    /* Animazione typing */
    .typing-dots span {
        display: inline-block;
        width: 7px; height: 7px;
        margin: 0 2px;
        background: #1565C0;
        border-radius: 50%;
        animation: bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }
    /* Quick prompt buttons */
    .qp-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        padding: 8px 12px 0;
    }
    /* Streamlit chat input fix — sempre visibile sopra la nav */
    div[data-testid="stChatInput"] {
        position: sticky !important;
        bottom: 68px !important;
        background: #f0f2f6 !important;
        padding: 8px 0 4px !important;
        z-index: 998 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-pad"><h3 style="margin:12px 0 4px">💬 Coach AI</h3></div>',
                unsafe_allow_html=True)

    if _ai_sdk_mode is None:
        st.warning("⚠️ Aggiungi GOOGLE_API_KEY nei Secrets per abilitare il Coach AI.")
    else:
        # Contesto coach ricco — 6 mesi dati completi
        if "chat_ctx_cache" not in st.session_state:
            with st.spinner("📊 Carico contesto atleta..."):
                st.session_state["chat_ctx_cache"] = build_chat_context(
                    df, u, current_ctl, current_atl, current_tsb, status_label, vo2max_val
                )
        _ctx_sys = (
            "Sei un coach sportivo d'elite specializzato in ciclismo, trail running e sci alpinismo. "
            "Sei sia ANALISTA (spieghi i dati, le cause, i trend) "
            "che PROGRAMMATORE (piani concreti, sessioni specifiche, carichi con numeri). "
            "Personalita: diretto, asciutto, professionale. Zero frasi motivazionali generiche. "
            "Rispondi sempre in italiano. Usa sempre i numeri disponibili. "
            "Se ti chiedono un piano: sessioni con tipo, durata, zona target, TSS stimato. "
            "Se ti chiedono un'analisi: usa CTL/ATL/TSB/TSS/watt/FC con valori precisi.\n\n"
            + st.session_state["chat_ctx_cache"]
        )

        # Stato forma rapido in cima alla chat
        _tsb_col = "#4CAF50" if current_tsb > 10 else ("#FF9800" if current_tsb > -5 else "#F44336")
        st.markdown(f"""
        <div style="display:flex;gap:8px;align-items:center;padding:4px 12px 8px;
                    font-size:12px;color:#666;flex-wrap:wrap">
            <span>CTL <b style="color:#4CAF50">{current_ctl:.0f}</b></span>
            <span>ATL <b style="color:#F44336">{current_atl:.0f}</b></span>
            <span>TSB <b style="color:{_tsb_col}">{current_tsb:+.0f}</b></span>
            <span style="background:{status_color}22;color:{status_color};
                         padding:2px 8px;border-radius:20px;font-weight:700">{status_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # Quick prompts (solo se chat vuota o come suggerimenti)
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:20px 12px 8px">
                <div style="font-size:40px">🏆</div>
                <div style="font-size:15px;font-weight:700;color:#1565C0;margin:8px 0 4px">Coach AI</div>
                <div style="font-size:13px;color:#888">Chiedi qualsiasi cosa sul tuo allenamento</div>
            </div>
            """, unsafe_allow_html=True)

        quick_prompts = [
            "💪 Come sto fisicamente?",
            "🗓️ Cosa fare oggi?",
            "📋 Piano questa settimana",
            "📊 Analizza gli ultimi 30gg",
        ]
        st.markdown('<div class="qp-grid">', unsafe_allow_html=True)
        qc = st.columns(2)
        for i, qp in enumerate(quick_prompts):
            with qc[i % 2]:
                if st.button(qp, use_container_width=True, key=f"qp_{i}",
                             type="secondary"):
                    clean_qp = qp.split(" ", 1)[1] if qp[0] in "💪🗓📋📊" else qp
                    st.session_state.messages.append({"role": "user", "content": clean_qp})
                    st.session_state["_chat_pending"] = True
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Mostra messaggi con bubble pulite
        st.markdown('<div class="chat-messages-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-label" style="text-align:right;margin-right:14px">Tu</div>'
                    f'<div class="chat-user">{msg["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                content = str(msg["content"]).replace("\n", "<br>")
                st.markdown(
                    f'<div class="chat-label" style="margin-left:14px">🤖 Coach</div>'
                    f'<div class="chat-ai">{content}</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Risposta pendente (dopo quick prompt o invio)
        if st.session_state.get("_chat_pending") and st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg["role"] == "user":
                # Mostra indicatore typing
                st.markdown(
                    '<div class="chat-label" style="margin-left:14px">🤖 Coach</div>'
                    '<div class="chat-ai" style="padding:12px 16px">'
                    '<div class="typing-dots"><span></span><span></span><span></span></div>'
                    '</div>',
                    unsafe_allow_html=True)
                _hlines = [
                    ("Atleta" if _m["role"] == "user" else "Coach") + ": " + str(_m["content"])
                    for _m in st.session_state.messages[-12:]
                ]
                res = ai_deep(_ctx_sys + "\n\n=== CONVERSAZIONE ===\n" + "\n".join(_hlines))
                st.session_state.messages.append({"role": "assistant", "content": res})
                st.session_state["_chat_pending"] = False
                if _gsheet_ok:
                    gsheet_save_conversations(st.session_state.messages)
                st.rerun()

        # Input chat — sticky sopra la nav bar
        if prompt := st.chat_input("Scrivi al tuo coach..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state["_chat_pending"] = True
            st.rerun()

        # Azioni in fondo
        if st.session_state.messages:
            st.markdown('<div class="sec-pad" style="margin-top:4px">', unsafe_allow_html=True)
            c_clr1, c_clr2 = st.columns(2)
            with c_clr1:
                if st.button("🗑️ Nuova chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state["_chat_pending"] = False
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
        if st.button("🔄 Forza sync da Strava", use_container_width=True):
            with st.spinner("Sincronizzazione in corso..."):
                raw_new = load_all_from_strava(access_token)
                if raw_new:
                    st.session_state.activities_cache = raw_new
                    gsheet_save_activities(raw_new)
                    st.success(f"✅ {len(raw_new)} attività salvate!")
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
# BOTTOM NAV (sempre visibile)
# ============================================================
render_bottom_nav()
