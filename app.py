import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import polyline
import folium
from streamlit_folium import st_folium
from streamlit_calendar import calendar
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import calendar as cal_module

# ============================================================
# 1. CONFIGURAZIONE
# ============================================================
REDIRECT_URI = "https://elite-ai-coach-4lm2ecs6qfslfkkzaeacrd.streamlit.app"

def get_secret(key):
    return st.secrets.get(key) or os.getenv(key)

CLIENT_ID     = get_secret("STRAVA_CLIENT_ID")
CLIENT_SECRET = get_secret("STRAVA_CLIENT_SECRET")
GEMINI_KEY    = get_secret("GOOGLE_API_KEY")
GROK_KEY      = get_secret("GROK_API_KEY") or get_secret("XAI_API_KEY") or ""
ORS_API_KEY   = get_secret("ORS_API_KEY") or ""   # OpenRouteService — gratuito su openrouteservice.org
MAPBOX_TOKEN  = get_secret("MAPBOX_TOKEN") or ""
MAPBOX_MONTHLY_LIMIT  = 50_000   # Free tier ufficiale Mapbox GL JS
MAPBOX_DAILY_SOFT_CAP = 200      # Soglia soft giornaliera (50k/30gg ≈ 1666/gg, usiamo 200 come extra-safe)
MAPBOX_SESSION_CAP    = 200      # Max map load per sessione Streamlit

# ── Inizializzazione AI — Grok (xAI) + Gemini (Google) ──
_ai_client      = None   # client principale (2.0+)
_ai_client_v1a  = None   # client v1alpha per modelli 1.5
_ai_sdk_mode    = None   # "new" | "old" | "grok" | None

# Grok via OpenAI-compatible endpoint — ha priorità se la key è presente
if GROK_KEY:
    try:
        from openai import OpenAI as _OpenAI
        _ai_client   = _OpenAI(api_key=GROK_KEY, base_url="https://api.x.ai/v1")
        _ai_sdk_mode = "grok"
    except ImportError:
        pass  # openai non installato, prova Gemini

# Gemini — usato se Grok non disponibile
if _ai_sdk_mode is None and GEMINI_KEY:
    try:
        import google.genai as genai_new
        from google.genai import types as genai_types
        # Client principale: usa v1beta (default) — supporta 2.0+, 2.5, 3.x
        _ai_client = genai_new.Client(api_key=GEMINI_KEY)
        # Client secondario: usa v1alpha — supporta anche 1.5
        _ai_client_v1a = genai_new.Client(
            api_key=GEMINI_KEY,
            http_options=genai_types.HttpOptions(api_version="v1alpha")
        )
        _ai_sdk_mode = "new"
    except (ImportError, Exception):
        pass
    if _ai_sdk_mode is None:
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=GEMINI_KEY)
            _ai_client   = genai_old
            _ai_sdk_mode = "old"
        except ImportError:
            pass

@st.cache_data(ttl=3600, show_spinner=False)
def get_available_models(api_key: str) -> list[str]:
    """
    Scopre modelli disponibili. Per Grok usa lista statica aggiornata.
    Per Gemini fa auto-discovery + fallback statico completo.
    """
    # ── Grok (xAI) ──────────────────────────────────
    if _ai_sdk_mode == "grok":
        return [
            "grok-3", "grok-3-fast", "grok-3-mini", "grok-3-mini-fast",
            "grok-2-1212", "grok-2-vision-1212", "grok-beta",
        ]

    # ── Gemini — auto-discovery ───────────────────────────────────────────────
    models_found = []
    if _ai_sdk_mode == "new":
        try:
            for m in _ai_client.models.list():
                name = m.name.replace("models/", "")
                if "gemini" in name and "embedding" not in name and "aqa" not in name:
                    models_found.append(name)
        except Exception:
            pass
        # Prova anche il client v1alpha per i modelli 1.5
        if _ai_client_v1a is not None:
            try:
                for m in _ai_client_v1a.models.list():
                    name = m.name.replace("models/", "")
                    if "gemini" in name and "embedding" not in name and name not in models_found:
                        models_found.append(name)
            except Exception:
                pass
    elif _ai_sdk_mode == "old":
        try:
            for m in _ai_client.list_models():
                if "generateContent" in m.supported_generation_methods:
                    name = m.name.replace("models/", "")
                    if "gemini" in name:
                        models_found.append(name)
        except Exception:
            pass

    # Lista statica completa — ordine di preferenza (più capace/recente prima)
    # I modelli 1.5 vengono serviti via _ai_client_v1a (v1alpha)
    static_all = [
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-flash-preview",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    # Costruisci lista: prima quelli trovati in ordine preferito, poi aggiungi statici mancanti
    ordered = [m for m in static_all if m in models_found]
    ordered += [m for m in models_found if m not in ordered and "gemini" in m]
    # Aggiungi sempre la lista statica come fallback (così appaiono anche senza auto-discovery)
    for m in static_all:
        if m not in ordered:
            ordered.append(m)
    return ordered

st.set_page_config(page_title="Elite AI Coach Pro", page_icon="🏆", layout="wide")

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
<style>
    /* Card attività */
    .activity-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .activity-header {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .metric-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .metric-pill {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 13px;
        color: #444;
        font-weight: 500;
    }
    .metric-pill span { color: #e94560; font-weight: 700; }

    /* Zone badge */
    .zone-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 700;
    }

    /* Pulsanti sport filter */
    div[data-testid="stHorizontalBlock"] .stButton button {
        border-radius: 20px;
        font-size: 13px;
        padding: 4px 14px;
    }

    /* Sezione stato fisico */
    .fitness-indicator {
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    /* Heatmap cell */
    .hm-cell {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. DIZIONARIO SPORT COMPLETO
# ============================================================
# Sport principali dell'atleta
SPORT_INFO = {
    "Run":              {"icon": "🏃", "label": "Corsa",        "color": "#FF4B4B"},
    "TrailRun":         {"icon": "🏔️", "label": "Trail Run",    "color": "#FF7043"},
    "Ride":             {"icon": "🚴", "label": "Ciclismo",      "color": "#2196F3"},
    "VirtualRide":      {"icon": "🖥️", "label": "Ciclismo V.",  "color": "#42A5F5"},
    "MountainBikeRide": {"icon": "🚵", "label": "MTB",          "color": "#1565C0"},
    "BackcountrySki":   {"icon": "🎿", "label": "Sci Alpinismo","color": "#4FC3F7"},
    "AlpineSki":        {"icon": "⛷️", "label": "Sci Alpino",   "color": "#81D4FA"},
}
# Sport extra (tracciati ma raggruppati come "Altro" nei filtri)
SPORT_INFO_EXTRA = {
    "NordicSki":      {"icon": "⛷️", "label": "Sci di Fondo",  "color": "#B3E5FC"},
    "Snowboard":      {"icon": "🏂",  "label": "Snowboard",     "color": "#80DEEA"},
    "Hike":           {"icon": "🥾",  "label": "Escursionismo", "color": "#4CAF50"},
    "Walk":           {"icon": "🚶",  "label": "Camminata",     "color": "#8BC34A"},
    "Workout":        {"icon": "💪",  "label": "Allenamento",   "color": "#FF9800"},
    "WeightTraining": {"icon": "🏋️",  "label": "Pesi",          "color": "#FFA726"},
    "Yoga":           {"icon": "🧘",  "label": "Yoga",          "color": "#CE93D8"},
    "Rowing":         {"icon": "🚣",  "label": "Canottaggio",   "color": "#26C6DA"},
    "Kayaking":       {"icon": "🛶",  "label": "Kayak",         "color": "#00ACC1"},
    "Crossfit":       {"icon": "🔥",  "label": "CrossFit",      "color": "#EF5350"},
    "Soccer":         {"icon": "⚽",  "label": "Calcio",        "color": "#66BB6A"},
    "Tennis":         {"icon": "🎾",  "label": "Tennis",        "color": "#FFEE58"},
    "Swim":           {"icon": "🏊",  "label": "Nuoto",         "color": "#00BCD4"},
}
_ALL_SPORT_INFO = {**SPORT_INFO, **SPORT_INFO_EXTRA}
PRIMARY_SPORTS = ["Run", "TrailRun", "Ride", "MountainBikeRide",
                  "BackcountrySki", "AlpineSki", "VirtualRide"]
CALENDAR_FILTER_SPORTS = ["Run", "TrailRun", "Ride", "MountainBikeRide",
                           "BackcountrySki", "AlpineSki"]

def get_sport_info(a_type, name=""):
    if a_type == "Ride" and name:
        _n = name.lower()
        if any(k in _n for k in ["mtb","mountain","gravel","sterrato","trail","enduro","xc "]):
            a_type = "MountainBikeRide"
    return _ALL_SPORT_INFO.get(a_type, {"icon": "🏅", "label": a_type, "color": "#9E9E9E"})

# ============================================================
# 3. METRICHE FORMATTATE
# ============================================================
def format_metrics(row):
    a_type = row["type"]
    dist   = row["distance"] / 1000
    time   = row["moving_time"]
    elev   = row.get("total_elevation_gain", 0) or 0
    hr_avg = row.get("average_heartrate")
    hr_max = row.get("max_heartrate")
    cad    = row.get("average_cadence")
    watts  = row.get("average_watts")
    cal_   = row.get("kilojoules") or row.get("calories", 0) or 0
    suffer = row.get("suffer_score")
    hrs    = int(time // 3600)
    mins   = int((time % 3600) // 60)
    secs   = int(time % 60)
    dur_str = f"{hrs}h {mins:02d}m" if hrs > 0 else f"{mins}m {secs:02d}s"

    if a_type == "Swim":
        pace_raw = time / (dist * 10) if dist > 0 else 0
        pace_str = f"{int(pace_raw // 60)}:{int(pace_raw % 60):02d} /100m"
        speed_str = f"{dist / (time / 3600):.1f} km/h" if time > 0 else "N/A"
    elif a_type in ("Ride", "VirtualRide", "MountainBikeRide"):
        speed = dist / (time / 3600) if time > 0 else 0
        pace_str = f"{speed:.1f} km/h"
        speed_str = pace_str
    else:
        pace_raw = time / dist if dist > 0 else 0
        pace_str = f"{int(pace_raw // 60)}:{int(pace_raw % 60):02d} /km"
        speed_str = f"{dist / (time / 3600):.1f} km/h" if time > 0 else "N/A"

    return {
        "dist_str":  f"{dist:.2f} km",
        "pace_str":  pace_str,
        "speed_str": speed_str,
        "dur_str":   dur_str,
        "elev":      f"{elev:.0f} m",
        "hr_avg":    f"{hr_avg:.0f} bpm" if pd.notna(hr_avg) else "N/A",
        "hr_max":    f"{hr_max:.0f} bpm" if pd.notna(hr_max) else "N/A",
        "cadence":   f"{cad:.0f} rpm" if pd.notna(cad) else "N/A",
        "watts":     f"{watts:.0f} W"  if pd.notna(watts) else "N/A",
        "calories":  f"{cal_:.0f} kcal" if cal_ else "N/A",
        "suffer":    f"{suffer:.0f}" if pd.notna(suffer) else "N/A",
        "dist_km":   dist,
        "time_sec":  time,
    }

# ============================================================
# 4. ZONE FC
# ============================================================
def get_hr_zone(hr_pct):
    if hr_pct < 0.60: return 1, "#4CAF50", "Z1 Recupero"
    if hr_pct < 0.70: return 2, "#8BC34A", "Z2 Base"
    if hr_pct < 0.80: return 3, "#FFC107", "Z3 Aerobico"
    if hr_pct < 0.90: return 4, "#FF9800", "Z4 Soglia"
    return 5, "#F44336", "Z5 VO2max"

def get_zone_for_activity(row, fc_max):
    hr = row.get("average_heartrate")
    if pd.notna(hr) and fc_max > 0:
        pct = hr / fc_max
        z, color, label = get_hr_zone(pct)
        return z, color, label
    return 0, "#9E9E9E", "N/A"

# ============================================================
# 5. CALCOLO TSS
# ============================================================

def get_power_zone(watts_pct_ftp):
    """Zone di potenza Coggan 7 zone."""
    if watts_pct_ftp < 0.55: return 1, "#9E9E9E", "Z1 Recupero Attivo"
    if watts_pct_ftp < 0.75: return 2, "#4CAF50", "Z2 Resistenza"
    if watts_pct_ftp < 0.90: return 3, "#8BC34A", "Z3 Tempo"
    if watts_pct_ftp < 1.05: return 4, "#FFC107", "Z4 Soglia"
    if watts_pct_ftp < 1.20: return 5, "#FF9800", "Z5 VO2max"
    if watts_pct_ftp < 1.50: return 6, "#FF5722", "Z6 Anaerobico"
    return 7, "#F44336", "Z7 Neuromuscolare"


def open_activity_button(row, key_suffix=""):
    """Pulsante per aprire il dettaglio di un'attività."""
    _act_id = row.get("id") if pd.notna(row.get("id", None)) else row.name
    if st.button("🔍 Dettaglio", key=f"open_act_{_act_id}_{key_suffix}", use_container_width=True):
        st.session_state.selected_activity_id = _act_id
        st.rerun()


def render_activity_detail(row, u, MAPBOX_TOKEN, draw_map, build_inline_map3d,
                            mapbox_render_allowed, mapbox_register_load, ai_generate,
                            current_ctl, current_atl, current_tsb, status_label):
    """Pagina dettaglio completa per una singola attività."""
    import streamlit.components.v1 as _components
    s   = get_sport_info(row["type"], row.get("name",""))
    m   = format_metrics(row)
    z_n, z_c, z_l = get_zone_for_activity(row, u["fc_max"])
    ftp = u.get("ftp", 200)
    is_bike = row["type"] in ("Ride", "VirtualRide", "MountainBikeRide")
    is_estimated = is_bike and not row.get("device_watts", False)

    # Header con bottone indietro
    _bc, _tc = st.columns([1, 8])
    with _bc:
        if st.button("← Indietro", key="act_back", use_container_width=True):
            st.session_state.selected_activity_id = None
            st.rerun()
    with _tc:
        st.markdown(f"## {s['icon']} {row['name']}")
        st.caption(f"{row['start_date'].strftime('%A %d %B %Y · %H:%M')} — {s['label']}")

    if is_bike and is_estimated:
        st.warning("⚠️ Potenza stimata da Strava (no sensore) — errore indicativo 15-40% (più alto su MTB/sterrato).")

    st.divider()

    # Mappa 2D / 3D
    poly = row.get("map", {})
    poly = poly.get("summary_polyline") if isinstance(poly, dict) else None
    if poly:
        _t2d, _t3d = st.tabs(["🗺️ Mappa 2D", "🏔️ Mappa 3D"])
        with _t2d:
            _mobj = draw_map(poly)
            if _mobj:
                st_folium(_mobj, width=None, height=400, key="det_map_2d")
        with _t3d:
            if not MAPBOX_TOKEN:
                st.info("Configura MAPBOX_TOKEN nei Secrets per la mappa 3D.")
            else:
                _mb_ok, _ = mapbox_render_allowed()
                if not _mb_ok:
                    st.warning("Limite Mapbox raggiunto.")
                else:
                    _eg = float(row.get("total_elevation_gain") or 0)
                    _h3d = build_inline_map3d(poly, MAPBOX_TOKEN, sport_type=row.get("type",""), elev_gain=_eg, height=420)
                    if _h3d:
                        _components.html(_h3d, height=430, scrolling=False)
                        mapbox_register_load()
    else:
        st.info("Nessun tracciato GPS disponibile per questa attività.")

    st.divider()

    # KPI principali
    st.markdown("### 📊 Statistiche")
    _k = st.columns(6)
    _k[0].metric("📏 Distanza",   m["dist_str"])
    _k[1].metric("⏱️ Durata",     m["dur_str"])
    _k[2].metric("⚡ Passo/Vel",  m["pace_str"])
    _k[3].metric("⛰️ Dislivello", m["elev"])
    _k[4].metric("🔥 Calorie",    m["calories"])
    _k[5].metric("📊 TSS",        f"{row['tss']:.1f}")
    _k2 = st.columns(5)
    _k2[0].metric("❤️ FC Media",  m["hr_avg"])
    _k2[1].metric("💓 FC Max",    m["hr_max"])
    _k2[2].metric("🔄 Cadenza",   m["cadence"])
    _k2[3].metric("⚡ Watt avg",  m["watts"])
    _k2[4].metric("😓 Suffer",    m["suffer"])

    st.divider()

    # Zone FC
    hr_avg    = row.get("average_heartrate")
    hr_max_a  = row.get("max_heartrate")
    fc_max    = u["fc_max"]
    st.markdown("### ❤️ Zone Frequenza Cardiaca")
    _hr_zones = [(1,"#4CAF50","Z1 Recupero",0.00,0.60),(2,"#8BC34A","Z2 Base",0.60,0.70),
                 (3,"#FFC107","Z3 Aerobico",0.70,0.80),(4,"#FF9800","Z4 Soglia",0.80,0.90),
                 (5,"#F44336","Z5 VO2max",0.90,1.00)]
    _hz = st.columns(5)
    for _zi, (_zn,_zc,_zl,_zlo,_zhi) in enumerate(_hr_zones):
        _blo, _bhi = int(fc_max*_zlo), int(fc_max*_zhi)
        _cur = pd.notna(hr_avg) and fc_max>0 and _zlo <= hr_avg/fc_max < _zhi
        _active_html = ""
        if _cur:
            _pct_fc = hr_avg/fc_max*100
            _active_html = f"<div style=\'font-size:12px;font-weight:900;color:{_zc}\'>← qui<br>{_pct_fc:.0f}% FCmax</div>"
        _bg_a   = "20" if _cur else "0a"
        _brd_w  = "3" if _cur else "1"
        _brd_a  = "ff" if _cur else "33"
        _hz[_zi].markdown(
            f"<div style='background:{_zc}{_bg_a};border:{_brd_w}px solid {_zc}{_brd_a};"
            f"border-radius:10px;padding:10px;text-align:center'>"
            f"<div style='font-size:11px;font-weight:700;color:{_zc}'>{_zl}</div>"
            f"<div style='font-size:12px;color:#444'>{_blo}–{_bhi} bpm</div>"
            f"{_active_html}"
            f"</div>",
            unsafe_allow_html=True)
    if pd.notna(hr_avg) and fc_max > 0:
        st.caption(f"FC media: {hr_avg:.0f} bpm ({hr_avg/fc_max*100:.0f}% FCmax) → **{z_l}**" +
                   (f" | FC max attività: {hr_max_a:.0f} bpm ({hr_max_a/fc_max*100:.0f}% FCmax)" if pd.notna(hr_max_a) else ""))

    # Zone Potenza (solo bici)
    watts_avg = row.get("average_watts")
    if is_bike and pd.notna(watts_avg) and watts_avg and watts_avg > 0 and ftp > 0:
        st.divider()
        st.markdown(f"### ⚡ Zone di Potenza  *(FTP: {ftp} W)*")
        if is_estimated:
            st.caption("⚠️ Watt stimati da Strava — valori indicativi")
        _pwr_zones = [(1,"#9E9E9E","Z1 Recupero",0.00,0.55),(2,"#4CAF50","Z2 Resistenza",0.55,0.75),
                      (3,"#8BC34A","Z3 Tempo",0.75,0.90),(4,"#FFC107","Z4 Soglia",0.90,1.05),
                      (5,"#FF9800","Z5 VO2max",1.05,1.20),(6,"#FF5722","Z6 Anaerobico",1.20,1.50),
                      (7,"#F44336","Z7 Neuromuscolare",1.50,9.99)]
        _pz = st.columns(7)
        _wpct = watts_avg / ftp
        for _zi, (_zn,_zc,_zl_p,_zlo,_zhi) in enumerate(_pwr_zones):
            _wlo = int(ftp*_zlo)
            _whi = f"{int(ftp*_zhi)}" if _zhi < 9 else "+"
            _cur = _zlo <= _wpct < _zhi
            _act_html = f"<div style=\'font-size:10px;font-weight:900;color:{_zc}\'>← qui</div>" if _cur else ""
            _bg_a  = "20" if _cur else "0a"
            _brd_w = "3" if _cur else "1"
            _brd_a = "ff" if _cur else "33"
            _pz[_zi].markdown(
                f"<div style='background:{_zc}{_bg_a};border:{_brd_w}px solid {_zc}{_brd_a};"
                f"border-radius:10px;padding:8px;text-align:center'>"
                f"<div style='font-size:10px;font-weight:700;color:{_zc}'>{_zl_p}</div>"
                f"<div style='font-size:11px;color:#444'>{_wlo}–{_whi} W</div>"
                f"{_act_html}"
                f"</div>",
                unsafe_allow_html=True)
        _if_val = watts_avg / ftp
        _tss_p  = (row["moving_time"] * watts_avg * _if_val) / (ftp * 3600) * 100
        _pc1, _pc2, _pc3 = st.columns(3)
        _pc1.metric("Watt medi", f"{watts_avg:.0f} W")
        _pc2.metric("Intensity Factor (IF)", f"{_if_val:.2f}")
        _pc3.metric("TSS da potenza", f"{_tss_p:.0f}")

    st.divider()

    # Analisi AI
    st.markdown("### 🤖 Analisi Coach")
    _aid = str(row.get("id", str(row["start_date"])))
    _ck  = f"ai_analysis_{_aid}"
    if _ck in st.session_state:
        st.info(st.session_state[_ck])
    else:
        if st.button("🤖 Genera analisi AI", key="det_ai_btn"):
            with st.spinner("Analisi in corso..."):
                try:
                    _ctx = (f"Sport: {s['label']}. Data: {row['start_date'].strftime('%d/%m/%Y')}. "
                            f"Distanza: {m['dist_str']}. Durata: {m['dur_str']}. Passo/Vel: {m['pace_str']}. "
                            f"Dislivello: {m['elev']}. FC Media: {m['hr_avg']}, FC Max: {m['hr_max']}. "
                            f"Zona FC: {z_l}. Watt: {m['watts']}{'  (stimati Strava)' if is_estimated else ''}. "
                            f"TSS: {row['tss']:.1f}. CTL: {current_ctl:.1f}, TSB: {current_tsb:.1f}. Forma: {status_label}.")
                    _res = ai_generate(_ctx + "\n\nSei un coach d'élite (focus ciclismo e corsa). "
                        "Commenta questa sessione in 3-4 paragrafi: qualità allenamento, zone di lavoro, "
                        "impatto sul carico, suggerimento concreto per la prossima sessione.")
                    st.session_state[_ck] = _res
                    st.info(_res)
                except Exception as e:
                    st.error(f"Errore AI: {e}")

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
        duration_sec = row["moving_time"]
        IF = watts / ftp
        return (duration_sec * watts * IF) / (ftp * 3600) * 100

    return dur * 0.4

# ============================================================
# 6. CTL / ATL / TSB
# ============================================================
def compute_fitness(df):
    daily = df.groupby(df["start_date"].dt.date)["tss"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)

    ctl = daily.ewm(span=42, adjust=False).mean()
    atl = daily.ewm(span=7,  adjust=False).mean()
    tsb = ctl - atl

    df_dates   = df["start_date"].dt.date.map(lambda d: pd.Timestamp(d))
    ctl_mapped = df_dates.map(ctl)
    atl_mapped = df_dates.map(atl)
    tsb_mapped = df_dates.map(tsb)

    return ctl_mapped, atl_mapped, tsb_mapped, ctl, atl, tsb, daily

# ============================================================
# 6b. METRICHE AVANZATE — TIER 1 + TIER 3
# ============================================================

def calc_trimp(row, u):
    """
    TRIMP (Training Impulse) — Banister 1991.
    Formula: durata(min) × ΔHR × 0.64×e^(1.92×ΔHR_ratio)
    dove ΔHR = (HR_media - HR_riposo) / (HR_max - HR_riposo)
    """
    hr    = row.get("average_heartrate")
    dur   = row["moving_time"] / 60
    fc_r  = u["fc_min"]
    fc_m  = u["fc_max"]
    if pd.notna(hr) and fc_m > fc_r and hr > fc_r:
        delta_hr = (hr - fc_r) / (fc_m - fc_r)
        delta_hr = max(0.0, min(delta_hr, 1.0))
        return dur * delta_hr * 0.64 * np.exp(1.92 * delta_hr)
    return dur * 0.3  # fallback conservativo

def calc_acwr(df_sorted):
    """
    ACWR = TSS_7gg / TSS_28gg_media_rolling
    Restituisce il valore corrente (ultimo).
    Safe zone: 0.8–1.3. Danger zone: >1.5
    """
    daily = df_sorted.groupby(df_sorted["start_date"].dt.date)["tss"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)

    atl_7  = daily.rolling(7,  min_periods=1).mean()
    ctl_28 = daily.rolling(28, min_periods=1).mean()
    acwr   = atl_7 / ctl_28.replace(0, np.nan)
    return float(acwr.iloc[-1]) if not acwr.empty else 0.0, acwr

def calc_ramp_rate(ctl_daily):
    """
    Ramp Rate = variazione CTL negli ultimi 7 giorni.
    Ideale: +3/+7 per settimana. >+8 = rischio infortuni.
    """
    if len(ctl_daily) < 8:
        return 0.0
    return float(ctl_daily.iloc[-1] - ctl_daily.iloc[-8])

def calc_monotony(df_sorted, days=7):
    """
    Monotonia = media_TSS / std_TSS (ultimi N giorni).
    <1.5 = ottimo. 1.5–2 = attenzione. >2 = rischio overtraining.
    """
    daily = df_sorted.groupby(df_sorted["start_date"].dt.date)["tss"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    recent = daily.tail(days)
    std = recent.std()
    return float(recent.mean() / std) if std > 0 else 0.0

def calc_training_strain(df_sorted, days=7):
    """
    Training Strain = Monotonia × TSS_totale_7gg.
    Indice di stress cumulativo (Banister). >2000 = zona critica.
    """
    daily = df_sorted.groupby(df_sorted["start_date"].dt.date)["tss"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    recent = daily.tail(days)
    mono = calc_monotony(df_sorted, days)
    return float(recent.sum() * mono)

def calc_ef_series(df_sorted):
    """
    Efficiency Factor per attività aerobiche.
    EF = velocità_m_s / FC_media  (running) oppure watt / FC_media (bici).
    Trend crescente = miglioramento aerobico.
    """
    ef_list = []
    for _, row in df_sorted.iterrows():
        hr = row.get("average_heartrate")
        if not pd.notna(hr) or hr == 0:
            ef_list.append(np.nan)
            continue
        if row["type"] in ("Ride", "VirtualRide", "MountainBikeRide"):
            w = row.get("average_watts")
            ef_list.append(float(w) / float(hr) if pd.notna(w) and w > 0 else np.nan)
        else:
            dist = row["distance"]
            t    = row["moving_time"]
            speed_ms = dist / t if t > 0 else 0
            ef_list.append(speed_ms / hr if hr > 0 else np.nan)
    return ef_list

def calc_vo2max_estimate(df_sorted):
    """
    Stima VO2max con formula di Jack Daniels (VDOT approach) sulle ultime corse.
    VO2max ≈ (-4.60 + 0.182258*(dist_m/time_min) + 0.000104*(dist_m/time_min)^2) /
              (0.8 + 0.1894393*e^(-0.012778*time_min) + 0.2989558*e^(-0.1932605*time_min))
    Usa le attività Run con distanza ≥ 5 km e FC media disponibile.
    """
    runs = df_sorted[
        (df_sorted["type"].isin(["Run", "TrailRun"])) &
        (df_sorted["distance"] >= 5000)
    ].copy()
    if runs.empty:
        return None, None

    best_vo2 = 0
    best_row = None
    for _, row in runs.iterrows():
        dist_m   = row["distance"]
        time_min = row["moving_time"] / 60
        if time_min <= 0:
            continue
        vel = dist_m / time_min  # m/min
        pct_vo2 = 0.8 + 0.1894393 * np.exp(-0.012778 * time_min) + \
                  0.2989558 * np.exp(-0.1932605 * time_min)
        vo2  = (-4.60 + 0.182258 * vel + 0.000104 * vel**2)
        vo2max = vo2 / pct_vo2 if pct_vo2 > 0 else 0
        if vo2max > best_vo2:
            best_vo2 = vo2max
            best_row = row
    return round(best_vo2, 1) if best_vo2 > 0 else None, best_row

def predict_race_times(vo2max):
    """
    Stima tempi di gara da VO2max usando le tabelle VDOT di Daniels.
    Formula inversa approssimata per ogni distanza.
    """
    if not vo2max or vo2max <= 0:
        return {}
    # Formula approssimata: pace_min_km = a + b/vo2max
    races = {
        "5 km":     {"dist": 5.0,    "a": 1.60, "b": 220},
        "10 km":    {"dist": 10.0,   "a": 1.65, "b": 280},
        "Mezza":    {"dist": 21.097, "a": 1.70, "b": 310},
        "Maratona": {"dist": 42.195, "a": 1.80, "b": 380},
    }
    results = {}
    for label, p in races.items():
        pace_sec_km = (p["a"] + p["b"] / vo2max) * 60
        total_sec   = pace_sec_km * p["dist"]
        h = int(total_sec // 3600)
        m = int((total_sec % 3600) // 60)
        s = int(total_sec % 60)
        time_str  = f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}:{s:02d}"
        pace_str  = f"{int(pace_sec_km // 60)}:{int(pace_sec_km % 60):02d} /km"
        results[label] = {"time": time_str, "pace": pace_str}
    return results

def calc_variability_index(row):
    """
    Variability Index = NP / AP (Normalized Power / Average Power).
    Disponibile solo per bici con watt.
    <1.05 = costante; >1.15 = variabile/nervoso.
    """
    np_val  = row.get("normalized_power")
    ap_val  = row.get("average_watts")
    if pd.notna(np_val) and pd.notna(ap_val) and ap_val > 0:
        return round(float(np_val) / float(ap_val), 3)
    return None


# ============================================================
# 6d. METRICHE AVANZATE — Batch 2
# ============================================================

def calc_hrv_trend_slope(df_vitals, days=14) -> dict:
    """
    Calcola la pendenza lineare HRV negli ultimi N giorni.
    Pendenza negativa sostenuta = segnale di overreaching.
    """
    if df_vitals is None or df_vitals.empty:
        return None
    recent = df_vitals.dropna(subset=["hrv_avg"]).tail(days).copy()
    if len(recent) < 4:
        return None
    x = np.arange(len(recent), dtype=float)
    y = recent["hrv_avg"].values.astype(float)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()
    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
    y_pred = x * slope + (y_mean - slope * x_mean)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    baseline   = recent["hrv_avg"].median()
    latest_hrv = recent["hrv_avg"].iloc[-1]
    pct_change = (latest_hrv - baseline) / baseline * 100 if baseline > 0 else 0
    if slope > 0.5:    label, color, emoji = "In miglioramento", "#4CAF50", "📈"
    elif slope > -0.5: label, color, emoji = "Stabile", "#FF9800", "➡️"
    elif slope > -1.5: label, color, emoji = "In calo lieve", "#FF5722", "📉"
    else:              label, color, emoji = "Overreaching probabile", "#F44336", "⚠️"
    return {
        "slope": round(slope, 3), "r2": round(r2, 3),
        "label": label, "color": color, "emoji": emoji,
        "baseline": round(baseline, 1), "latest": round(latest_hrv, 1),
        "pct_change": round(pct_change, 1),
        "dates": recent["date"].tolist(),
        "values": y.tolist(), "trend_line": y_pred.tolist(),
    }


def calc_sleep_load_ratio(df_sleep, df_strava, days=14) -> dict:
    """
    Sleep/Load Ratio = minuti deep sleep / TSS medio giornaliero.
    Ideale: >= 0.8 minuti deep per ogni punto TSS.
    """
    if df_sleep is None or df_sleep.empty:
        return None
    recent_sleep  = df_sleep.tail(days).copy()
    avg_deep_min  = recent_sleep["deep_min"].dropna().mean()
    avg_total_min = recent_sleep["total_min"].dropna().mean()
    avg_eff       = recent_sleep["efficiency_pct"].dropna().mean()
    cutoff        = df_strava["start_date"].max() - timedelta(days=days)
    recent_strava = df_strava[df_strava["start_date"] >= cutoff]
    avg_daily_tss = recent_strava["tss"].sum() / days if days > 0 else 0
    ratio         = avg_deep_min / avg_daily_tss if avg_daily_tss > 0 else None
    if ratio is None:      status, color = "N/D", "#888"
    elif ratio >= 0.8:     status, color = "Ottimale", "#4CAF50"
    elif ratio >= 0.5:     status, color = "Sufficiente", "#FF9800"
    else:                  status, color = "Insufficiente", "#F44336"
    deep_target  = avg_daily_tss * 0.8
    deep_deficit = max(0.0, deep_target - avg_deep_min)
    tss_by_date  = df_strava.groupby(df_strava["start_date"].dt.date)["tss"].sum()
    daily_series = []
    for _, srow in recent_sleep.iterrows():
        d     = srow["date"].date() if hasattr(srow["date"], "date") else srow["date"]
        tss_d = float(tss_by_date.get(d, 0))
        daily_series.append({
            "date":     srow["date"],
            "deep_min": srow.get("deep_min", np.nan),
            "tss":      tss_d,
            "ratio":    srow.get("deep_min", 0) / tss_d if tss_d > 0 else None,
        })
    return {
        "ratio": round(ratio, 3) if ratio else None,
        "status": status, "color": color,
        "avg_deep_min": round(avg_deep_min, 1),
        "avg_daily_tss": round(avg_daily_tss, 1),
        "deep_target": round(deep_target, 1),
        "deep_deficit": round(deep_deficit, 1),
        "avg_total_min": round(avg_total_min, 1),
        "avg_efficiency": round(avg_eff, 1) if not np.isnan(avg_eff) else None,
        "daily_series": daily_series,
    }


def calc_circadian_performance(df) -> dict:
    """
    Raggruppa le attività per fascia oraria (slot 3h) e calcola
    efficienza media (TSS/ora). Identifica la finestra ottimale.
    """
    df_c = df.copy()
    df_c["hour"] = df_c["start_date"].dt.hour
    df_c["slot"] = (df_c["hour"] // 3) * 3
    slot_labels  = {0:"00-03",3:"03-06",6:"06-09",9:"09-12",
                    12:"12-15",15:"15-18",18:"18-21",21:"21-24"}
    results = []
    for slot, grp in df_c.groupby("slot"):
        if len(grp) < 2:
            continue
        avg_tss = grp["tss"].mean()
        avg_hrs = grp["moving_time"].mean() / 3600
        tss_hr  = avg_tss / avg_hrs if avg_hrs > 0 else 0
        runs    = grp[grp["type"].isin(["Run","TrailRun","VirtualRun"])]
        run_pace = None
        if not runs.empty and runs["distance"].sum() > 0:
            run_pace = runs["moving_time"].sum() / (runs["distance"].sum() / 1000)
        rides  = grp[grp["type"].isin(["Ride","VirtualRide","MountainBikeRide"])]
        r_watt = rides["average_watts"].dropna().mean() if not rides.empty else None
        results.append({
            "slot": slot, "label": slot_labels.get(slot, f"{slot:02d}h"),
            "n": len(grp), "avg_tss": round(avg_tss, 1),
            "tss_hr": round(tss_hr, 1),
            "run_pace": round(run_pace, 1) if run_pace else None,
            "ride_watts": round(r_watt, 1) if r_watt and not np.isnan(r_watt) else None,
        })
    if not results:
        return None
    results_df = pd.DataFrame(results)
    eligible   = results_df[results_df["n"] >= 3] if len(results_df[results_df["n"] >= 3]) > 0 else results_df
    best_row   = eligible.loc[eligible["tss_hr"].idxmax()]
    worst_row  = eligible.loc[eligible["tss_hr"].idxmin()]
    delta_pct  = ((best_row["tss_hr"] - worst_row["tss_hr"]) / worst_row["tss_hr"] * 100
                  if worst_row["tss_hr"] > 0 else 0)
    return {
        "results": results, "results_df": results_df,
        "best_slot": best_row["label"], "best_tss_hr": best_row["tss_hr"],
        "worst_slot": worst_row["label"], "delta_pct": round(delta_pct, 1),
    }


def calc_acwr_v2(df, df_vitals) -> dict:
    """ACWR pesato su HRV: se HRV basso, il rischio effettivo aumenta."""
    acwr_base, _ = calc_acwr(df)
    hrv_weight   = 1.0
    hrv_now = hrv_base_val = None
    if df_vitals is not None and not df_vitals.empty:
        valid_v = df_vitals.dropna(subset=["hrv_avg"])
        if not valid_v.empty:
            hrv_base_val = valid_v["hrv_avg"].median()
            hrv_now      = float(valid_v["hrv_avg"].iloc[-1])
            hrv_weight   = hrv_now / hrv_base_val if hrv_base_val > 0 else 1.0
    acwr_adj = acwr_base / hrv_weight if hrv_weight > 0 else acwr_base
    if acwr_adj < 0.8:    risk, color, label = 10,  "#2196F3", "Undertraining"
    elif acwr_adj < 1.0:  risk, color, label = 20,  "#4CAF50", "Zona sicura bassa"
    elif acwr_adj < 1.3:  risk, color, label = 35,  "#8BC34A", "Zona ottimale ✅"
    elif acwr_adj < 1.5:  risk, color, label = 60,  "#FF9800", "Attenzione ⚠️"
    elif acwr_adj < 1.8:  risk, color, label = 80,  "#FF5722", "Rischio elevato 🔴"
    else:                 risk, color, label = 95,  "#F44336", "Pericolo ⛔"
    if hrv_weight < 0.85 and acwr_base > 1.2:
        risk  = min(100, int(risk * 1.25))
        label += " + HRV basso"
        color  = "#F44336"
    return {
        "acwr_base": round(acwr_base, 3), "acwr_adj": round(acwr_adj, 3),
        "hrv_weight": round(hrv_weight, 3), "hrv_now": hrv_now,
        "hrv_base": hrv_base_val, "risk": risk, "color": color, "label": label,
    }


def calc_adaptive_tss_budget(readiness_score: float, df) -> dict:
    """Budget TSS giornaliero adattivo basato su Readiness Score."""
    recent_28    = df[df["start_date"] >= (df["start_date"].max() - timedelta(days=28))]
    avg_base_tss = recent_28["tss"].sum() / 28 if not recent_28.empty else 50
    if readiness_score >= 90:    mult = 1.40
    elif readiness_score >= 80:  mult = 1.20
    elif readiness_score >= 65:  mult = 1.00
    elif readiness_score >= 50:  mult = 0.80
    elif readiness_score >= 35:  mult = 0.55
    else:                        mult = 0.35
    budget     = round(avg_base_tss * mult)
    budget_max = round(budget * 1.30)
    if mult >= 1.2:    zone, color, advice = "Giornata HQ 🟢",  "#4CAF50", "Ottimo per sessione intensa o lungo."
    elif mult >= 0.9:  zone, color, advice = "Normale 🟡",      "#FF9800", "Allenamento standard ai tuoi ritmi."
    elif mult >= 0.6:  zone, color, advice = "Ridotto 🟠",      "#FF5722", "Privilegia Z1-Z2. Niente sopra soglia."
    else:              zone, color, advice = "Solo recupero 🔴","#F44336", "Riposo attivo. Il corpo ne ha bisogno."
    today_date  = datetime.now().date()
    today_acts  = df[df["start_date"].dt.date == today_date]
    tss_spent   = float(today_acts["tss"].sum())
    remaining   = max(0.0, budget - tss_spent)
    overspent   = max(0.0, tss_spent - budget_max)
    return {
        "budget": budget, "budget_max": budget_max, "multiplier": mult,
        "zone": zone, "color": color, "advice": advice,
        "avg_base": round(avg_base_tss, 1), "tss_spent": round(tss_spent, 1),
        "remaining": round(remaining, 1), "overspent": round(overspent, 1),
        "readiness": readiness_score,
    }


def calc_nutritional_window(df, days=7) -> list:
    """Calcola fabbisogno nutrizionale post-workout per ogni sessione recente."""
    cutoff  = df["start_date"].max() - timedelta(days=days)
    recent  = df[df["start_date"] >= cutoff].copy()
    recent  = recent.sort_values("start_date", ascending=False)
    def _safe_float(val, default=0.0):
        """Converte a float gestendo NaN, None e stringhe vuote."""
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    results = []
    for _, row in recent.iterrows():
        kj   = _safe_float(row.get("kilojoules"), 0.0)
        kcal = kj * 0.239 if kj > 0 else _safe_float(row.get("calories"), 0.0)
        if kcal < 100:
            continue
        hrs      = _safe_float(row.get("moving_time"), 0.0) / 3600
        tss_val  = _safe_float(row.get("tss"), 0.0)
        elev     = _safe_float(row.get("total_elevation_gain"), 0.0)
        carb_r   = 0.60 if tss_val > 80 else 0.50 if tss_val > 40 else 0.40
        carbs_g  = round(kcal * carb_r / 4)
        prot_g   = max(20, round(kcal * 0.15 / 4))
        water_ml = round(hrs * 500 + elev * 1.5)
        end_min  = int(_safe_float(row.get("moving_time"), 0.0) / 60)
        win_end  = row["start_date"] + timedelta(minutes=end_min + 45)
        results.append({
            "data": row["start_date"].strftime("%d/%m %H:%M"),
            "sport": row.get("type",""),
            "kcal": round(kcal), "tss": round(tss_val, 1),
            "carbs_g": carbs_g, "protein_g": prot_g, "water_ml": water_ml,
            "finestra": win_end.strftime("%H:%M"),
        })
    return results

# ============================================================
# 6c. DIZIONARIO TOOLTIP METRICHE
# ============================================================
METRIC_INFO = {
    "TSS": {
        "nome": "Training Stress Score",
        "desc": "Misura lo stress fisiologico di una singola sessione integrando durata e intensità. Sviluppato da Andrew Coggan.",
        "range": "0–50: recupero facile | 50–100: medio | 100–150: difficile | >150: molto duro (giorni di recupero necessari)",
        "fonte": "Coggan & Allen — Training and Racing with a Power Meter",
    },
    "CTL": {
        "nome": "Chronic Training Load (Fitness)",
        "desc": "Media esponenziale a 42 giorni del TSS giornaliero. Rappresenta il tuo livello di fitness cronico e capacità di lavoro.",
        "range": "<40: principiante/recupero | 40–60: buona base | 60–80: atleta allenato | 80–100: atleta avanzato | >100: elite",
        "fonte": "Modello PMC (Performance Management Chart) — Banister 1991",
    },
    "ATL": {
        "nome": "Acute Training Load (Fatica)",
        "desc": "Media esponenziale a 7 giorni del TSS. Rappresenta la fatica accumulata nell'ultima settimana. Valore alto = sei stanco.",
        "range": "Confronta con CTL: ATL > CTL = accumulo fatica. La differenza determina il TSB.",
        "fonte": "Modello PMC — Banister 1991",
    },
    "TSB": {
        "nome": "Training Stress Balance (Forma)",
        "desc": "TSB = CTL - ATL. Indica il bilanciamento tra fitness e fatica. Positivo = riposato. Negativo = affaticato.",
        "range": "> +25: detrain/troppo riposo | +10/+25: fresco per gara | -10/+10: zona ottimale allenamento | -20/-10: accumulo | < -20: rischio overtraining",
        "fonte": "Coggan — TrainingPeaks Performance Management",
    },
    "TRIMP": {
        "nome": "Training Impulse",
        "desc": "Metrica di carico allenamento basata su FC, durata e intensità relativa. Precede il TSS ed è indipendente da potenza o GPS.",
        "range": "Dipende dalla durata. 100 TRIMP ≈ sessione di 1h a Z3. Usa il trend storico come riferimento personale.",
        "fonte": "Banister et al. — A systems model of training, physical performance and retention (1975)",
    },
    "ACWR": {
        "nome": "Acute:Chronic Workload Ratio",
        "desc": "Rapporto tra carico degli ultimi 7 giorni e media degli ultimi 28. Indicatore di rischio infortuni (studi su atleti elite).",
        "range": "< 0.8: undertraining | 0.8–1.3: zona sicura ✅ | 1.3–1.5: attenzione ⚠️ | > 1.5: danger zone 🔴 rischio infortuni elevato",
        "fonte": "Gabbett TJ — British Journal of Sports Medicine 2016 | Malone et al. IJSPP 2017",
    },
    "RAMP_RATE": {
        "nome": "Ramp Rate (CTL settimanale)",
        "desc": "Variazione del CTL negli ultimi 7 giorni. Indica quanto velocemente sta crescendo il tuo fitness.",
        "range": "< 3: crescita lenta/riposo | 3–7: progressione ideale ✅ | > 8: rischio overtraining ⚠️ | > 10: pericolo 🔴",
        "fonte": "TrainingPeaks — Performance Management Chart guidelines",
    },
    "MONOTONIA": {
        "nome": "Monotonia dell'Allenamento",
        "desc": "Media TSS / Deviazione standard TSS degli ultimi 7 giorni. Misura quanto è vario il tuo allenamento. Troppa uniformità = rischio.",
        "range": "< 1.5: variazione sana ✅ | 1.5–2.0: attenzione ⚠️ | > 2.0: rischio overtraining anche con carichi moderati 🔴",
        "fonte": "Foster C. — Journal of Strength and Conditioning Research 1998",
    },
    "STRAIN": {
        "nome": "Training Strain",
        "desc": "Monotonia × TSS_totale_settimanale. Indice di stress cumulativo che combina volume e uniformità dell'allenamento.",
        "range": "< 1000: basso | 1000–2000: moderato | > 2000: elevato/critico 🔴",
        "fonte": "Foster C. — JSCR 1998 | Banister training model",
    },
    "EF": {
        "nome": "Efficiency Factor (Indice di Efficienza Aerobica)",
        "desc": "Velocità (m/s) diviso FC media (corsa) oppure Watt / FC (bici). Trend crescente = migliori adattamenti aerobici.",
        "range": "Corsa: EF ~0.010–0.017 m/s per bpm. Bici: EF ~1.5–2.5 W/bpm. Il valore assoluto dipende dall'atleta — conta il miglioramento nel tempo.",
        "fonte": "Joe Friel — Training Bible | Coggan normalized metrics",
    },
    "VO2MAX": {
        "nome": "VO2max Stimato",
        "desc": "Massimo consumo di ossigeno, indicatore fondamentale del fitness cardiovascolare. Stimato da pace e durata delle tue corse migliori (formula Daniels/VDOT).",
        "range": "< 35: sedentario | 35–45: nella media | 45–55: buono | 55–65: molto buono | > 65: atleta d'élite",
        "fonte": "Daniels J. — Daniels' Running Formula (VDOT tables) | Nes et al. 2011",
    },
    "VI": {
        "nome": "Variability Index (solo bici con potenza)",
        "desc": "Normalized Power / Average Power. Misura quanto è stato uniforme lo sforzo. Valore basso = corsa costante ed efficiente.",
        "range": "1.00–1.05: costante/pianura ✅ | 1.05–1.10: leggermente variabile | 1.10–1.15: misto | > 1.15: molto variabile/nervoso",
        "fonte": "Coggan & Allen — Training and Racing with a Power Meter",
    },
    "POL": {
        "nome": "Distribuzione Polarizzata",
        "desc": "% del tempo trascorso in bassa intensità (Z1-Z2). La ricerca mostra che gli atleti endurance d'élite trascorrono ~80% in Z1-Z2 e ~20% in Z4-Z5.",
        "range": "< 60%: troppo sforzo a media intensità (zona grigia) | 60–75%: accettabile | > 75%: distribuzione polarizzata ottimale ✅",
        "fonte": "Seiler S. — International Journal of Sports Physiology and Performance 2010",
    },
}

def metric_tooltip(key):
    """Renderizza un expander piccolo con le info sulla metrica."""
    info = METRIC_INFO.get(key)
    if not info:
        return
    with st.expander(f"ℹ️ Cos'è?", expanded=False):
        st.markdown(f"**{info['nome']}**")
        st.markdown(info["desc"])
        st.markdown(f"📊 **Range tipici:** {info['range']}")
        st.markdown(f"📚 *Fonte: {info['fonte']}*")

# ============================================================
# 7. MAPPA
# ============================================================
def draw_map(encoded_polyline, height=300):
    if not encoded_polyline:
        return None
    try:
        points = polyline.decode(encoded_polyline)
        m = folium.Map(
            location=points[0], zoom_start=13, tiles="CartoDB positron",
            scrollWheelZoom=True,   # ← abilita zoom con rotella mouse
        )
        folium.PolyLine(points, color="#e94560", weight=4, opacity=0.9).add_to(m)
        folium.CircleMarker(points[0],  radius=6, color="#4CAF50", fill=True).add_to(m)
        folium.CircleMarker(points[-1], radius=6, color="#F44336", fill=True).add_to(m)
        return m
    except Exception:
        return None


def build_map3d_html(encoded_polyline, mapbox_token, sport_type="", elev_gain=0,
                     map_style="satellite-streets-v12",
                     show_slope_map=False, compact=False) -> str:
    """
    Genera HTML completo per una mappa Mapbox GL JS 3D con:
    - Terrain 3D + sky layer
    - Tracciato GPS con glow
    - Slope colormap sul terreno (canvas 2D via hillshade + custom raster blend)
    - Controlli rotazione/inclinazione espliciti
    - compact=True → mappa ridotta per inline Dashboard/Calendario
    """
    if not encoded_polyline or not mapbox_token:
        return None
    try:
        import json as _j
        pts    = polyline.decode(encoded_polyline)
        coords = [[lon, lat] for lat, lon in pts]
        if len(coords) < 2:
            return None
        clon   = sum(c[0] for c in coords) / len(coords)
        clat   = sum(c[1] for c in coords) / len(coords)
        geoj   = _j.dumps({"type":"Feature","properties":{},
                            "geometry":{"type":"LineString","coordinates":coords}})
        start_j = _j.dumps(coords[0])
        end_j   = _j.dumps(coords[-1])

        # Hillshade nativo Mapbox (ombre rilievo)
        hillshade_layer = """
    map.addLayer({
        id: 'hillshade',
        type: 'hillshade',
        source: 'dem',
        paint: {
            'hillshade-shadow-color': '#2a1a0a',
            'hillshade-highlight-color': '#fff',
            'hillshade-accent-color': '#5a3e1b',
            'hillshade-exaggeration': 0.6,
            'hillshade-illumination-anchor': 'map'
        }
    }, 'waterway-label');"""

        # Slope colormap: usa mapbox-dem RGB tiles per calcolare pendenza
        # Il tile DEM codifica altitudine come: H = -10000 + (R*256*256 + G*256 + B) * 0.1
        # Calcoliamo il gradiente su 3x3 kernel (Sobel) in un canvas offscreen
        # e mappiamo gradi AINEVA → colore (tutto lato client JS, zero costi)
        slope_canvas_js = """
    // ── SLOPE COLORMAP (Sobel su DEM terrain-rgb) ──
    const SLOPE_COLORS = [
        {deg: 0,  color: [0,   0,   0,   0  ]},  // trasparente sotto 3°
        {deg: 3,  color: [21,  101, 192, 160 ]},  // blu    3-25°
        {deg: 25, color: [255, 235, 59,  190 ]},  // giallo 25-30°
        {deg: 30, color: [255, 152, 0,   210 ]},  // arancio 30-35°
        {deg: 35, color: [244, 67,  54,  230 ]},  // rosso  >35°
    ];

    function degToColor(deg) {
        if (deg < 3)  return [0,0,0,0];
        if (deg < 25) return [21,101,192,160];
        if (deg < 30) return [255,235,59,190];
        if (deg < 35) return [255,152,0,210];
        return [244,67,54,230];
    }

    function buildSlopeCanvas(imgData, tileSize) {
        const w = tileSize, h = tileSize;
        const src = imgData.data;
        const out  = new Uint8ClampedArray(w * h * 4);
        const cellSize = 30; // metri approssimativi per pixel a zoom 12

        function getH(x, y) {
            const idx = (y * w + x) * 4;
            const R = src[idx], G = src[idx+1], B = src[idx+2];
            return -10000 + (R*256*256 + G*256 + B) * 0.1;
        }

        for (let y = 1; y < h-1; y++) {
            for (let x = 1; x < w-1; x++) {
                // Sobel 3x3
                const dzdx = (getH(x+1,y-1) + 2*getH(x+1,y) + getH(x+1,y+1)
                            - getH(x-1,y-1) - 2*getH(x-1,y) - getH(x-1,y+1)) / (8*cellSize);
                const dzdy = (getH(x-1,y+1) + 2*getH(x,y+1) + getH(x+1,y+1)
                            - getH(x-1,y-1) - 2*getH(x,y-1) - getH(x+1,y-1)) / (8*cellSize);
                const slopeDeg = Math.atan(Math.sqrt(dzdx*dzdx + dzdy*dzdy)) * 180 / Math.PI;
                const c = degToColor(slopeDeg);
                const oi = (y * w + x) * 4;
                out[oi]=c[0]; out[oi+1]=c[1]; out[oi+2]=c[2]; out[oi+3]=c[3];
            }
        }
        return out;
    }

    // Aggiungi slope layer come custom raster via canvas
    const TILE_SIZE = 256;
    const slopeCanvas = document.createElement('canvas');
    slopeCanvas.width = slopeCanvas.height = 1; // placeholder

    map.addSource('slope-src', {
        type: 'canvas',
        canvas: slopeCanvas,
        coordinates: [
            [map.getBounds().getWest(), map.getBounds().getNorth()],
            [map.getBounds().getEast(), map.getBounds().getNorth()],
            [map.getBounds().getEast(), map.getBounds().getSouth()],
            [map.getBounds().getWest(), map.getBounds().getSouth()]
        ],
        animate: true
    });
    map.addLayer({
        id: 'slope-layer',
        type: 'raster',
        source: 'slope-src',
        paint: { 'raster-opacity': 0.75, 'raster-resampling': 'nearest' }
    });

    // Processa i tile DEM per costruire la slope map
    function processDemTile(url, bounds) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const c = document.createElement('canvas');
            c.width = c.height = TILE_SIZE;
            const ctx = c.getContext('2d');
            ctx.drawImage(img, 0, 0, TILE_SIZE, TILE_SIZE);
            const imgData = ctx.getImageData(0, 0, TILE_SIZE, TILE_SIZE);
            const slopeData = buildSlopeCanvas(imgData, TILE_SIZE);
            slopeCanvas.width  = TILE_SIZE;
            slopeCanvas.height = TILE_SIZE;
            const sCtx = slopeCanvas.getContext('2d');
            const sImg = sCtx.createImageData(TILE_SIZE, TILE_SIZE);
            sImg.data.set(slopeData);
            sCtx.putImageData(sImg, 0, 0);
            // Aggiorna coordinate del canvas source
            const src = map.getSource('slope-src');
            if (src) src.setCoordinates([
                [bounds.getWest(), bounds.getNorth()],
                [bounds.getEast(), bounds.getNorth()],
                [bounds.getEast(), bounds.getSouth()],
                [bounds.getWest(), bounds.getSouth()]
            ]);
        };
        img.src = url;
    }

    function updateSlopeLayer() {
        const zoom   = Math.floor(map.getZoom());
        const bounds = map.getBounds();
        const lat    = map.getCenter().lat;
        // Tile XY
        function lonToTile(lon, z) { return Math.floor((lon+180)/360 * Math.pow(2,z)); }
        function latToTile(lat, z) {
            const r = Math.PI/180;
            return Math.floor((1 - Math.log(Math.tan(lat*r)+1/Math.cos(lat*r))/Math.PI)/2 * Math.pow(2,z));
        }
        const z = Math.min(zoom, 14);
        const tx = lonToTile(map.getCenter().lng, z);
        const ty = latToTile(map.getCenter().lat, z);
        const url = `https://api.mapbox.com/v4/mapbox.terrain-rgb/${z}/${tx}/${ty}.pngraw?access_token=TOKEN_PLACEHOLDER`;
        processDemTile(url, bounds);
    }

    map.on('moveend', updateSlopeLayer);
    map.on('zoomend', updateSlopeLayer);
    setTimeout(updateSlopeLayer, 500);
"""
        # Sostituisce TOKEN_PLACEHOLDER con il token reale
        slope_canvas_js = slope_canvas_js.replace("TOKEN_PLACEHOLDER", mapbox_token)

        slope_legend = """
  <div id="slope-legend" style="position:absolute;bottom:30px;left:10px;
       background:rgba(14,17,23,0.88);color:#fff;padding:10px 14px;
       border-radius:10px;font-size:11px;font-family:sans-serif;z-index:10;line-height:1.9">
    <b>Pendenza terreno</b>
    <div style="display:flex;align-items:center;gap:7px"><div style="width:11px;height:11px;border-radius:50%;background:#1565C0;flex-shrink:0"></div>3-25° percorribile</div>
    <div style="display:flex;align-items:center;gap:7px"><div style="width:11px;height:11px;border-radius:50%;background:#FFEB3B;flex-shrink:0"></div>25-30° attenzione</div>
    <div style="display:flex;align-items:center;gap:7px"><div style="width:11px;height:11px;border-radius:50%;background:#FF9800;flex-shrink:0"></div>30-35° rischio</div>
    <div style="display:flex;align-items:center;gap:7px"><div style="width:11px;height:11px;border-radius:50%;background:#F44336;flex-shrink:0"></div>&gt;35° pericolo</div>
    <div style="color:#666;font-size:10px;margin-top:4px">Soglie AINEVA</div>
  </div>""" if show_slope_map else ""

        rotate_hint = "" if compact else """
  <div id="rotate-hint" style="position:absolute;bottom:40px;right:10px;
       background:rgba(14,17,23,0.75);color:#aaa;padding:5px 10px;
       border-radius:8px;font-size:10px;font-family:sans-serif;z-index:10;line-height:1.7">
    🖱️ <b>Drag</b> pan &nbsp;|&nbsp; <b>Ctrl+drag</b> inclina<br>
    <b>Alt+drag</b> ruota &nbsp;|&nbsp; <b>Scroll</b> zoom
  </div>"""

        pitch_val    = 55 if compact else 65
        zoom_val     = 12
        slope_init   = slope_canvas_js if show_slope_map else ""
        hillshade_init = hillshade_layer if not compact else ""

        return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet"/>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<style>
  html,body{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#0e1117;}}
  #mw{{position:relative;width:100%;height:100%;}}
  #m{{position:absolute;top:0;left:0;width:100%;height:100%;}}
</style></head><body>
<div id="mw">
  <div id="m"></div>
  {slope_legend}
  {rotate_hint}
</div>
<script>
mapboxgl.accessToken='{mapbox_token}';
const map=new mapboxgl.Map({{
  container:'m',
  style:'mapbox://styles/mapbox/{map_style}',
  center:[{clon},{clat}],
  zoom:{zoom_val}, pitch:{pitch_val}, bearing:-15, antialias:true
}});
map.addControl(new mapboxgl.NavigationControl({{'visualizePitch':true}}),'top-left');
map.addControl(new mapboxgl.FullscreenControl(),'top-left');
map.addControl(new mapboxgl.ScaleControl(),'bottom-right');
map.dragRotate.enable();
map.touchZoomRotate.enableRotation();

// ── Scroll zoom: attivo con prevenzione propagazione nell'iframe ──
map.scrollZoom.enable();
map.scrollZoom.setWheelZoomRate(1/450);  // sensibilità standard
const canvas = map.getCanvas();
canvas.addEventListener('wheel', (e) => {{
  e.stopPropagation();
}}, {{ passive: false }});

// ── Rotazione con tasto centrale del mouse (click rotella + trascina) ──
(function() {{
  let isMiddleDragging = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener('mousedown', (e) => {{
    if (e.button === 1) {{   // tasto centrale = rotella
      e.preventDefault();
      isMiddleDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.style.cursor = 'grab';
    }}
  }});

  window.addEventListener('mousemove', (e) => {{
    if (!isMiddleDragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    // dx → bearing (rotazione orizzontale), dy → pitch (inclinazione)
    const bearing = map.getBearing() + dx * 0.5;
    const pitch   = Math.min(85, Math.max(0, map.getPitch() - dy * 0.4));
    map.jumpTo({{ bearing, pitch }});
  }});

  window.addEventListener('mouseup', (e) => {{
    if (e.button === 1) {{
      isMiddleDragging = false;
      canvas.style.cursor = '';
    }}
  }});
}})();

map.on('load',()=>{{
  map.addSource('dem',{{'type':'raster-dem','url':'mapbox://mapbox.mapbox-terrain-dem-v1','tileSize':512,'maxzoom':14}});
  map.setTerrain({{'source':'dem','exaggeration':1.9}});
  map.addLayer({{'id':'sky','type':'sky','paint':{{'sky-type':'atmosphere','sky-atmosphere-sun':[0,45],'sky-atmosphere-sun-intensity':15}}}});
  {hillshade_init}
  {slope_init}
  map.addSource('r',{{'type':'geojson','data':{geoj}}});
  map.addLayer({{id:'rg',type:'line',source:'r',paint:{{'line-color':'#e94560','line-width':16,'line-opacity':0.12,'line-blur':6}}}});
  map.addLayer({{id:'rl',type:'line',source:'r',paint:{{'line-color':'#e94560','line-width':4,'line-opacity':1}},layout:{{'line-cap':'round','line-join':'round'}}}});
  new mapboxgl.Marker({{color:'#4CAF50',scale:0.9}}).setLngLat({start_j}).setPopup(new mapboxgl.Popup().setHTML('<b>Partenza</b>')).addTo(map);
  new mapboxgl.Marker({{color:'#F44336',scale:0.9}}).setLngLat({end_j}).setPopup(new mapboxgl.Popup().setHTML('<b>Arrivo</b>')).addTo(map);
  map.flyTo({{center:[{clon},{clat}],zoom:13,pitch:{pitch_val},duration:1800,essential:true}});
}});
</script></body></html>"""
    except Exception:
        return None

# Alias per compatibilità con chiamate esistenti
def build_inline_map3d(encoded_polyline, mapbox_token, sport_type="", elev_gain=0,
                       map_style="satellite-streets-v12", height=380) -> str:
    return build_map3d_html(encoded_polyline, mapbox_token, sport_type=sport_type,
                            elev_gain=elev_gain, map_style=map_style, compact=True)


# ============================================================
# 8. RINGCONN — Parser CSV + Readiness Score
# ============================================================

def parse_ringconn_vitals(file) -> pd.DataFrame:
    df_v = pd.read_csv(file)
    df_v.columns = [c.strip() for c in df_v.columns]
    df_v = df_v.rename(columns={
        "Date": "date",
        "Avg. Heart Rate(bpm)": "hr_avg",
        "Min. Heart Rate(bpm)": "hr_min",
        "Max. Heart Rate(bpm)": "hr_max",
        "Avg. Spo2(%)": "spo2_avg",
        "Min. Spo2(%)": "spo2_min",
        "Max. Spo2(%)": "spo2_max",
        "Avg. HRV(ms)": "hrv_avg",
        "Min. HRV(ms)": "hrv_min",
        "Max. HRV(ms)": "hrv_max",
    })
    df_v["date"] = pd.to_datetime(df_v["date"])
    for col in ["spo2_avg", "spo2_min", "spo2_max"]:
        if col in df_v.columns:
            df_v[col] = df_v[col].astype(str).str.replace("%","").str.strip()
    for col in ["hr_avg","hr_min","hr_max","spo2_avg","spo2_min","spo2_max",
                "hrv_avg","hrv_min","hrv_max"]:
        df_v[col] = pd.to_numeric(df_v[col], errors="coerce")
    return df_v.sort_values("date").reset_index(drop=True)


def parse_ringconn_sleep(file) -> pd.DataFrame:
    df_s = pd.read_csv(file)
    df_s.columns = [c.strip() for c in df_s.columns]
    df_s = df_s.rename(columns={
        "Start Time": "start_time",
        "End Time": "end_time",
        "Falling Asleep Time": "sleep_onset",
        "Wake-up time": "wakeup_time",
        "Sleep Time Ratio(%)": "efficiency_pct",
        "Time Asleep(min)": "total_min",
        "Sleep Stages - Awake(min)": "awake_min",
        "Sleep Stages - REM(min)": "rem_min",
        "Sleep Stages - Light Sleep(min)": "light_min",
        "Sleep Stages - Deep Sleep(min)": "deep_min",
    })
    df_s["start_time"] = pd.to_datetime(df_s["start_time"], errors="coerce")
    df_s["end_time"]   = pd.to_datetime(df_s["end_time"],   errors="coerce")
    df_s["efficiency_pct"] = df_s["efficiency_pct"].astype(str).str.replace("%","").str.strip()
    for col in ["efficiency_pct","total_min","awake_min","rem_min","light_min","deep_min"]:
        df_s[col] = pd.to_numeric(df_s[col], errors="coerce")
    df_s["date"]        = df_s["end_time"].dt.normalize()
    df_s["total_hours"] = df_s["total_min"] / 60
    df_s["deep_pct"]    = df_s["deep_min"]  / df_s["total_min"] * 100
    df_s["rem_pct"]     = df_s["rem_min"]   / df_s["total_min"] * 100
    return df_s.sort_values("date").reset_index(drop=True)



def parse_ringconn_activity(file) -> pd.DataFrame:
    """Parsa il CSV Activity RingConn: Date, Steps, Calories(kcal)."""
    df_a = pd.read_csv(file)
    df_a.columns = [c.strip() for c in df_a.columns]
    df_a["date"]     = pd.to_datetime(df_a["Date"], errors="coerce")
    df_a["steps"]    = pd.to_numeric(df_a.get("Steps", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    df_a["kcal_day"] = pd.to_numeric(df_a.get("Calories(kcal)", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    df_a = df_a.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df_a[["date","steps","kcal_day"]]



def parse_ringconn_zip(zip_file) -> tuple:
    """Parsa ZIP RingConn → (vitals_df, sleep_df, activity_df)"""
    import zipfile, io
    vitals = sleep = activity = None
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            for name in zf.namelist():
                nl = name.lower()
                buf = io.BytesIO(zf.read(name))
                if "vital" in nl:   vitals   = parse_ringconn_vitals(buf)
                elif "sleep" in nl: sleep    = parse_ringconn_sleep(buf)
                elif "activ" in nl: activity = parse_ringconn_activity(buf)
    except Exception as e:
        raise ValueError(f"Errore ZIP: {e}")
    return vitals, sleep, activity

def calc_readiness(vitals_row, sleep_row, tsb: float) -> dict:
    scores = {}
    hrv = vitals_row.get("hrv_avg") if vitals_row is not None else None
    if pd.notna(hrv) and hrv > 0:
        baseline  = vitals_row.get("hrv_baseline", hrv)
        ratio     = hrv / baseline if baseline > 0 else 1.0
        hrv_score = min(40, max(0, 40 * (0.5 + (ratio - 1) * 2)))
    else:
        hrv_score = 20
    scores["HRV"] = round(hrv_score, 1)

    if sleep_row is not None:
        eff   = sleep_row.get("efficiency_pct", 80) or 80
        hrs   = sleep_row.get("total_hours", 7) or 7
        deep  = sleep_row.get("deep_pct", 15) or 15
        eff_score  = min(1.0, eff / 90) * 12
        hrs_score  = min(1.0, max(0, 1 - abs(hrs - 8) / 4)) * 12
        deep_score = min(1.0, deep / 25) * 6
        sleep_score = eff_score + hrs_score + deep_score
    else:
        sleep_score = 15
    scores["Sonno"] = round(sleep_score, 1)

    if tsb > 5:       tsb_score = 20
    elif tsb > -5:    tsb_score = 16
    elif tsb > -15:   tsb_score = 10
    elif tsb > -25:   tsb_score = 5
    else:             tsb_score = 2
    scores["Forma"] = round(tsb_score, 1)

    spo2 = vitals_row.get("spo2_avg") if vitals_row is not None else None
    if pd.notna(spo2) and spo2 > 0:
        spo2_score = min(10, max(0, (spo2 - 90) / 8 * 10))
    else:
        spo2_score = 8
    scores["SpO2"] = round(spo2_score, 1)

    total = min(100, max(0, round(sum(scores.values()))))
    if total >= 80:   color, label, emoji = "#4CAF50", "Ottimo",               "🟢"
    elif total >= 65: color, label, emoji = "#8BC34A", "Buono",                "🟡"
    elif total >= 50: color, label, emoji = "#FF9800", "Discreto",             "🟠"
    elif total >= 35: color, label, emoji = "#FF5722", "Recupero necessario",  "🔴"
    else:             color, label, emoji = "#F44336", "Riposo completo",      "⛔"
    return {"score": total, "color": color, "label": label,
            "emoji": emoji, "breakdown": scores}


def get_ringconn_context(df_vitals, df_sleep, days=7) -> str:
    if df_vitals is None and df_sleep is None:
        return ""
    lines = ["\n--- DATI RECUPERO RINGCONN ---"]
    if df_vitals is not None and not df_vitals.empty:
        recent_v = df_vitals.tail(days)
        avg_hrv  = recent_v["hrv_avg"].dropna().mean()
        avg_hr   = recent_v["hr_avg"].dropna().mean()
        avg_spo2 = recent_v["spo2_avg"].dropna().mean()
        if pd.notna(avg_hrv):  lines.append(f"HRV medio {days}gg: {avg_hrv:.0f} ms")
        if pd.notna(avg_hr):   lines.append(f"FC riposo media {days}gg: {avg_hr:.0f} bpm")
        if pd.notna(avg_spo2): lines.append(f"SpO2 media {days}gg: {avg_spo2:.0f}%")
        last_v = df_vitals.iloc[-1]
        if pd.notna(last_v.get("hrv_avg")):
            lines.append(f"HRV ultima notte: {last_v['hrv_avg']:.0f} ms")
    if df_sleep is not None and not df_sleep.empty:
        recent_s = df_sleep.tail(days)
        avg_hrs  = recent_s["total_hours"].dropna().mean()
        avg_eff  = recent_s["efficiency_pct"].dropna().mean()
        avg_deep = recent_s["deep_pct"].dropna().mean()
        if pd.notna(avg_hrs):  lines.append(f"Ore sonno media {days}gg: {avg_hrs:.1f}h")
        if pd.notna(avg_eff):  lines.append(f"Efficienza sonno: {avg_eff:.0f}%")
        if pd.notna(avg_deep): lines.append(f"Deep sleep: {avg_deep:.0f}%")
    return "\n".join(lines)


def build_athlete_snapshot(df, u, current_ctl, current_atl, current_tsb, status_label,
                            ctl_daily, atl_daily, tsb_daily,
                            vo2max_val, acwr_v2, hrv_slope, slr, circadian,
                            tss_budget, readiness, rc_vitals, rc_sleep) -> str:
    """
    Contesto completo per il Coach AI (~3500 token):
    - Memoria storica riassuntiva per anno (dall'inizio)
    - Tutti i dettagli ultimi 6 mesi riga per riga
    - Metriche avanzate, RingConn, zone FC, record
    Focus: ciclismo e corsa per programmazione gare.
    Sci alpinismo/sci = sport liberi (no programmazione).
    """
    lines = ["=" * 60, "CONTESTO ATLETA — ELITE AI COACH", "=" * 60]
    _now_ts = df["start_date"].max()

    # ── 1. PROFILO ──────────────────────────────────────────────
    lines.append("\n[PROFILO ATLETA]")
    lines.append(f"Peso: {u['peso']}kg | FC riposo: {u['fc_min']}bpm | FC max: {u['fc_max']}bpm | FTP: {u['ftp']}W")
    lines.append("Sport principali con obiettivi gara: Ciclismo (Ride/MTB), Corsa (Run/TrailRun)")
    lines.append("Sport liberi/ludici (no programmazione necessaria): Sci Alpinismo, Sci Alpino")

    # ── 2. STATO FITNESS ────────────────────────────────────────
    lines.append("\n[FITNESS — PMC]")
    lines.append(f"CTL: {current_ctl:.1f} | ATL: {current_atl:.1f} | TSB: {current_tsb:.1f} | Stato: {status_label}")
    try:
        _ctl_4w = []
        for _wi in range(4, 0, -1):
            _d = (pd.Timestamp.now() - pd.Timedelta(weeks=_wi)).date()
            _v = ctl_daily.get(_d)
            if _v: _ctl_4w.append(f"{_v:.0f}")
        if _ctl_4w: lines.append(f"CTL trend 4 sett: {' → '.join(_ctl_4w)}")
    except Exception: pass

    # ── 3. METRICHE AVANZATE ────────────────────────────────────
    lines.append("\n[METRICHE AVANZATE]")
    if vo2max_val:
        lines.append(f"Indice forma aerobica (VO2max stimato): {vo2max_val:.1f} ml/kg/min")
    _df14 = df[df["start_date"] >= (_now_ts - pd.Timedelta(days=14))]
    _ef_vals = []
    for _, _r in _df14.iterrows():
        _hr = _r.get("average_heartrate")
        if not _hr or not pd.notna(_hr) or _hr <= 0: continue
        if _r["type"] in ["Run","TrailRun"] and _r["distance"] > 0:
            _ef_vals.append(1/( (_r["moving_time"]/60)/(_r["distance"]/1000) * _hr/100))
        elif _r["type"] in ["Ride","VirtualRide","MountainBikeRide"]:
            _w = _r.get("average_watts")
            if _w and pd.notna(_w): _ef_vals.append(float(_w)/_hr)
    if _ef_vals: lines.append(f"Efficienza aerobica (EF) 14gg: {float(np.mean(_ef_vals)):.3f}")
    _daily_tss = df.groupby(df["start_date"].dt.date)["tss"].sum()
    _last7 = [float(_daily_tss.get((_now_ts-pd.Timedelta(days=i)).date(), 0)) for i in range(7)]
    _std7 = float(np.std(_last7))
    _mono = float(np.mean(_last7))/_std7 if _std7>0 else 0
    lines.append(f"Monotonia Banister 7gg: {_mono:.2f} ({'alta' if _mono>2 else 'ok'})")
    _active21 = len(df[df["start_date"]>=(_now_ts-pd.Timedelta(days=21))]["start_date"].dt.date.unique())
    lines.append(f"Consistenza 21gg: {_active21}/21 giorni ({_active21/21*100:.0f}%)")
    lines.append(f"ACWR 2.0: {acwr_v2['acwr_adj']:.2f} | Risk: {acwr_v2['risk']}/100 | {acwr_v2['label']}")
    if hrv_slope and hrv_slope.get("slope") is not None:
        lines.append(f"HRV trend 14gg: {hrv_slope['slope']:+.2f}ms/gg | {hrv_slope['label']}")
    lines.append(f"TSS Budget oggi: {tss_budget['budget']} ({tss_budget['zone']})")
    lines.append(f"Readiness: {readiness['score']}/100 ({readiness['label']})")

    # ── 4. RINGCONN ─────────────────────────────────────────────
    lines.append("\n[RINGCONN — ULTIMI 7 GIORNI]")
    if rc_vitals is not None and not rc_vitals.empty:
        _rv7 = rc_vitals.tail(7)
        _hrv = _rv7["hrv_avg"].dropna().mean()
        _hr  = _rv7["hr_avg"].dropna().mean()
        _sp  = _rv7["spo2_avg"].dropna().mean()
        if pd.notna(_hrv): lines.append(f"HRV medio: {_hrv:.0f}ms")
        if pd.notna(_hr):  lines.append(f"FC riposo media: {_hr:.0f}bpm")
        if pd.notna(_sp):  lines.append(f"SpO2: {_sp:.0f}%")
        _lhrv = rc_vitals.iloc[-1].get("hrv_avg")
        _bhrv = rc_vitals.tail(30)["hrv_avg"].dropna().mean()
        if pd.notna(_lhrv) and pd.notna(_bhrv):
            lines.append(f"HRV ieri: {_lhrv:.0f}ms vs baseline 30gg {_bhrv:.0f}ms ({_lhrv/_bhrv*100:.0f}%)")
    else:
        lines.append("Dati RingConn non disponibili.")
    if rc_sleep is not None and not rc_sleep.empty:
        _rs = rc_sleep.tail(7)
        _sh = _rs["total_hours"].dropna().mean()
        _se = _rs["efficiency_pct"].dropna().mean()
        _sd = _rs["deep_pct"].dropna().mean()
        if pd.notna(_sh): lines.append(f"Sonno 7gg: {_sh:.1f}h | Eff: {_se:.0f}% | Deep: {_sd:.0f}%")

    # ── 5. ZONE FC 4 settimane ───────────────────────────────────
    lines.append("\n[ZONE FC — ultime 4 settimane]")
    _df28 = df[df["start_date"]>=(_now_ts-pd.Timedelta(days=28))]
    if not _df28.empty:
        _zc = {1:0,2:0,3:0,4:0,5:0}
        for _,_r in _df28.iterrows():
            _zn,_,_ = get_zone_for_activity(_r, u["fc_max"])
            if _zn in _zc: _zc[_zn]+=1
        _tz = sum(_zc.values()) or 1
        lines.append(" | ".join([f"Z{z}: {cnt/_tz*100:.0f}%" for z,cnt in _zc.items()]))

    # ── 6. MEMORIA STORICA (tutto il passato > 6 mesi fa) ────────
    _cutoff_6m = _now_ts - pd.Timedelta(days=183)
    _df_old = df[df["start_date"] < _cutoff_6m].copy()
    if not _df_old.empty:
        lines.append("\n[MEMORIA STORICA — riepilogo annuale prima dei 6 mesi recenti]")
        _df_old["year"] = _df_old["start_date"].dt.year
        for _yr in sorted(_df_old["year"].unique()):
            _ydf = _df_old[_df_old["year"]==_yr]
            _ykm = _ydf["distance"].sum()/1000
            _yh  = _ydf["moving_time"].sum()/3600
            _ytss= _ydf["tss"].sum()
            _yelv= float(_ydf["total_elevation_gain"].fillna(0).sum())
            _ysp = ", ".join([f"{get_sport_info(s)['label']} {len(_ydf[_ydf['type']==s])}sess"
                              for s in _ydf["type"].value_counts().index[:4]])
            lines.append(f"{_yr}: {len(_ydf)} attività | {_ykm:.0f}km | ↑{_yelv:.0f}m | {_yh:.0f}h | TSS {_ytss:.0f} | {_ysp}")

    # ── 7. RIEPILOGO SPORT 6 mesi ────────────────────────────────
    _df6m = df[df["start_date"] >= _cutoff_6m]
    lines.append("\n[RIEPILOGO SPORT — ultimi 6 mesi]")
    for _sp in _df6m["type"].value_counts().index:
        _si  = get_sport_info(_sp)
        _spd = _df6m[_df6m["type"]==_sp]
        _skm = _spd["distance"].sum()/1000
        _sel = float(_spd["total_elevation_gain"].fillna(0).sum())
        _sh  = _spd["moving_time"].sum()/3600
        _sts = _spd["tss"].sum()
        lines.append(f"{_si['label']}: {len(_spd)}sess | {_skm:.0f}km | ↑{_sel:.0f}m | {_sh:.0f}h | TSS {_sts:.0f}")

    # ── 8. TUTTE LE ATTIVITÀ ULTIMI 6 MESI riga per riga ─────────
    lines.append("\n[ATTIVITÀ ULTIMI 6 MESI — dalla più recente]")
    lines.append("Data       | Sport       |   Km | Durata  |  ↑m | FC  | Watt | TSS")
    lines.append("-" * 72)
    for _, _r in _df6m.sort_values("start_date", ascending=False).iterrows():
        _dt  = _r["start_date"].strftime("%Y-%m-%d")
        _sp  = _r["type"][:11]
        _km  = _r["distance"]/1000
        _h   = int(_r["moving_time"]//3600)
        _min = int((_r["moving_time"]%3600)//60)
        _el  = int(_r.get("total_elevation_gain") or 0)
        _fc  = f"{_r['average_heartrate']:.0f}" if pd.notna(_r.get("average_heartrate")) else "—"
        _w   = f"{_r['average_watts']:.0f}" if pd.notna(_r.get("average_watts")) else "—"
        _ts  = f"{_r['tss']:.0f}"
        lines.append(f"{_dt} | {_sp:<11} | {_km:5.1f} | {_h}h{_min:02d}m | {_el:4d} | {_fc:>3} | {_w:>4} | {_ts}")

    # ── 9. RECORD PERSONALI ─────────────────────────────────────
    lines.append("\n[RECORD PERSONALI — all time]")
    for _sp in df["type"].value_counts().index[:5]:
        _si  = get_sport_info(_sp)
        _spd = df[df["type"]==_sp]
        _mx  = _spd["distance"].max()/1000
        _mel = float(_spd["total_elevation_gain"].fillna(0).max())
        _mts = _spd["tss"].max()
        _ln  = f"{_si['label']}: max {_mx:.1f}km | ↑{_mel:.0f}m | TSS {_mts:.0f}"
        if _sp in ("Run","TrailRun"):
            _f5 = _spd[_spd["distance"]>=5000].copy()
            if not _f5.empty:
                _f5["psk"] = _f5["moving_time"]/(_f5["distance"]/1000)
                _bp = _f5["psk"].min()
                _ln += f" | best 5km {int(_bp//60)}:{int(_bp%60):02d}/km"
        elif _sp in ("Ride","VirtualRide","MountainBikeRide"):
            _fb = _spd[(_spd["distance"]>30000) & _spd["average_watts"].notna()].copy()
            if not _fb.empty:
                _bw = _fb["average_watts"].max()
                _ln += f" | best avg {_bw:.0f}W"
        lines.append(_ln)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ============================================================
# 9. TOKEN
# ============================================================
def build_athlete_snapshot(df, u, current_ctl, current_atl, current_tsb, status_label,
                            ctl_daily, atl_daily, tsb_daily, vo2max_val,
                            acwr_v2, hrv_slope, tss_budget, readiness,
                            rc_vitals, rc_sleep) -> str:
    """Contesto completo (~3500 token) per il Coach AI.
    - Memoria storica annuale (da sempre)
    - Dati completi ultimi 6 mesi riga per riga
    """
    lines = ["="*60, "CONTESTO ATLETA — ELITE AI COACH", "="*60]

    # Profilo
    lines.append("\n[PROFILO]")
    lines.append(f"Peso: {u['peso']}kg | FC riposo: {u['fc_min']} bpm | FC max: {u['fc_max']} bpm | FTP: {u['ftp']}W")
    lines.append("Focus: Ciclismo (gare) e Corsa (gare). Scialpinismo/Sci = sport ludici, no programmazione.")

    # Fitness PMC
    lines.append("\n[FITNESS — PMC]")
    lines.append(f"CTL (fitness 42gg): {current_ctl:.1f} | ATL (fatica 7gg): {current_atl:.1f} | TSB (forma): {current_tsb:.1f}")
    lines.append(f"Stato: {status_label}")
    try:
        _ctl_4w = []
        for _wi in range(4, 0, -1):
            _d = (pd.Timestamp.now() - pd.Timedelta(weeks=_wi)).date()
            _v = ctl_daily.get(_d)
            if _v: _ctl_4w.append(f"{_v:.0f}")
        if _ctl_4w: lines.append(f"CTL trend 4 sett: {' → '.join(_ctl_4w)}")
    except: pass

    # Metriche avanzate
    lines.append("\n[METRICHE AVANZATE]")
    if vo2max_val: lines.append(f"Indice forma aerobica (VO2max proxy): {vo2max_val:.1f} ml/kg/min")
    lines.append(f"ACWR 2.0: {acwr_v2['acwr_adj']:.2f} | Risk: {acwr_v2['risk']}/100 | {acwr_v2['label']}")
    if hrv_slope and hrv_slope.get("slope") is not None:
        lines.append(f"HRV trend 14gg: {hrv_slope['slope']:+.2f} ms/gg | {hrv_slope['label']}")
    lines.append(f"TSS Budget oggi: {tss_budget['budget']} | {tss_budget['zone']}")
    lines.append(f"Readiness: {readiness['score']}/100 ({readiness['label']})")
    _now = df["start_date"].max()
    _daily_tss = df.groupby(df["start_date"].dt.date)["tss"].sum()
    _last7 = [float(_daily_tss.get((_now - pd.Timedelta(days=i)).date(), 0)) for i in range(7)]
    _mono = np.mean(_last7) / np.std(_last7) if np.std(_last7) > 0 else 0
    lines.append(f"Monotonia Banister 7gg: {_mono:.2f} ({'⚠️ alta' if _mono > 2 else 'ok'})")
    _df21 = df[df["start_date"] >= (_now - pd.Timedelta(days=21))]
    lines.append(f"Consistenza 21gg: {len(_df21['start_date'].dt.date.unique())}/21 giorni")

    # RingConn
    lines.append("\n[RINGCONN — ULTIMI 7 GIORNI]")
    if rc_vitals is not None and not rc_vitals.empty:
        _rv7 = rc_vitals.tail(7)
        _hrv = _rv7["hrv_avg"].dropna().mean()
        _hr  = _rv7["hr_avg"].dropna().mean()
        _sp  = _rv7["spo2_avg"].dropna().mean()
        if pd.notna(_hrv): lines.append(f"HRV medio: {_hrv:.0f}ms")
        if pd.notna(_hr):  lines.append(f"FC riposo media: {_hr:.0f}bpm")
        if pd.notna(_sp):  lines.append(f"SpO2: {_sp:.0f}%")
        _last_hrv = rc_vitals.iloc[-1].get("hrv_avg")
        _base_hrv = rc_vitals.tail(30)["hrv_avg"].dropna().mean()
        if pd.notna(_last_hrv) and pd.notna(_base_hrv):
            lines.append(f"HRV ieri: {_last_hrv:.0f}ms (baseline 30gg: {_base_hrv:.0f}ms, ratio: {_last_hrv/_base_hrv*100:.0f}%)")
    else:
        lines.append("Dati RingConn non disponibili.")
    if rc_sleep is not None and not rc_sleep.empty:
        _rs7 = rc_sleep.tail(7)
        _sh = _rs7["total_hours"].dropna().mean()
        _se = _rs7["efficiency_pct"].dropna().mean()
        _sd = _rs7["deep_pct"].dropna().mean()
        if pd.notna(_sh): lines.append(f"Sonno medio: {_sh:.1f}h | Efficienza: {_se:.0f}% | Deep: {_sd:.0f}%")

    # Memoria storica annuale (da sempre, riassunto per anno/mese)
    lines.append("\n[MEMORIA STORICA — RIEPILOGO ANNUALE]")
    df["_year"] = df["start_date"].dt.year
    for _yr in sorted(df["_yr"].unique() if "_yr" in df.columns else df["start_date"].dt.year.unique()):
        _dfy = df[df["start_date"].dt.year == _yr]
        _by_sport = []
        for _sp in _dfy["type"].value_counts().index[:4]:
            _si = get_sport_info(_sp)
            _sd = _dfy[_dfy["type"]==_sp]
            _by_sport.append(f"{_si['label']} {_sd['distance'].sum()/1000:.0f}km/{len(_sd)}sess")
        lines.append(f"{_yr}: {_dfy['tss'].sum():.0f} TSS | {_dfy['distance'].sum()/1000:.0f}km | {' · '.join(_by_sport)}")
    df.drop(columns=["_year"], errors="ignore", inplace=True)

    # Riepilogo sport ultimi 6 mesi
    _df6m = df[df["start_date"] >= (_now - pd.Timedelta(days=180))]
    lines.append("\n[RIEPILOGO SPORT — ultimi 6 mesi]")
    for _sp in _df6m["type"].value_counts().index:
        _si = get_sport_info(_sp)
        _sd = _df6m[_df6m["type"]==_sp]
        lines.append(f"{_si['label']}: {len(_sd)} sess | {_sd['distance'].sum()/1000:.0f}km | "
                     f"↑{float(_sd['total_elevation_gain'].sum() or 0):.0f}m | {_sd['tss'].sum():.0f} TSS")

    # Tutte le attività ultimi 6 mesi riga per riga
    lines.append("\n[ATTIVITÀ ULTIMI 6 MESI — dalla più recente]")
    lines.append("Data       | Sport       | Km    | Durata  | ↑m   | FC avg/max | TSS  | Nome")
    lines.append("-" * 85)
    for _, _r in _df6m.sort_values("start_date", ascending=False).iterrows():
        _dt  = _r["start_date"].strftime("%Y-%m-%d")
        _sp  = _r["type"][:10]
        _km  = _r["distance"] / 1000
        _h   = int(_r["moving_time"] // 3600)
        _mn  = int((_r["moving_time"] % 3600) // 60)
        _el  = int(_r.get("total_elevation_gain") or 0)
        _ha  = f"{_r['average_heartrate']:.0f}" if pd.notna(_r.get("average_heartrate")) else "—"
        _hx  = f"{_r['max_heartrate']:.0f}" if pd.notna(_r.get("max_heartrate")) else "—"
        _ts  = f"{_r['tss']:.0f}"
        _nm  = str(_r.get("name",""))[:25]
        lines.append(f"{_dt} | {_sp:<11} | {_km:5.1f} | {_h}h{_mn:02d}m  | {_el:4d} | {_ha}/{_hx:>3} | {_ts:>4} | {_nm}")

    # Record personali
    lines.append("\n[RECORD PERSONALI]")
    for _sp in df["type"].value_counts().index[:4]:
        _si = get_sport_info(_sp)
        _sd = df[df["type"]==_sp]
        _md = _sd["distance"].max()/1000
        _me = float(_sd["total_elevation_gain"].fillna(0).max())
        _mt = _sd["tss"].max()
        _line = f"{_si['label']}: max {_md:.1f}km | ↑{_me:.0f}m | TSS {_mt:.0f}"
        if _sp in ("Run","TrailRun"):
            _f5 = _sd[_sd["distance"]>=5000].copy()
            if not _f5.empty:
                _f5["psk"] = _f5["moving_time"]/(_f5["distance"]/1000)
                _bp = _f5["psk"].min()
                _line += f" | best 5km {int(_bp//60)}:{int(_bp%60):02d}/km"
        elif _sp in ("Ride","MountainBikeRide"):
            _fsp = _sd[_sd["distance"]>=20000].copy()
            if not _fsp.empty:
                _fsp["kmh"] = _fsp["distance"]/1000 / (_fsp["moving_time"]/3600)
                _line += f" | best speed {_fsp['kmh'].max():.1f}km/h"
        lines.append(_line)

    lines.append("\n" + "="*60)
    return "\n".join(lines)


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

# ============================================================
# 9. FETCH — Sistema incrementale (scarica solo le attività nuove)
# ============================================================

def _fetch_page(access_token: str, page: int, after_ts: int = 0) -> list:
    """Scarica una singola pagina di attività, filtrando per timestamp se fornito."""
    params = f"per_page=200&page={page}"
    if after_ts:
        params += f"&after={after_ts}"
    r = requests.get(
        f"https://www.strava.com/api/v3/athlete/activities?{params}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if r.status_code == 200:
        return r.json()
    return []


def load_activities_incremental(access_token: str) -> list:
    """
    Sistema di fetch incrementale:
    - Prima esecuzione: scarica tutto lo storico (paginazione completa)
    - Esecuzioni successive: chiede solo le attività successive all'ultima già in cache
      usando il parametro ?after=<unix_timestamp>
    - Unisce le nuove attività alla cache esistente e rimuove eventuali duplicati per id
    - Nessuna chiamata API inutile se non ci sono attività nuove
    """
    cache_key  = "activities_cache"
    ts_key     = "activities_last_ts"
    token_key  = "activities_token"    # per invalidare cache se cambia utente

    existing   = st.session_state.get(cache_key, [])
    last_ts    = st.session_state.get(ts_key, 0)
    last_token = st.session_state.get(token_key, "")

    # Se il token è cambiato (nuovo login) azzera tutto
    if last_token != access_token[:20]:
        existing  = []
        last_ts   = 0

    headers = {"Authorization": f"Bearer {access_token}"}

    if not existing:
        # ── PRIMO CARICAMENTO: scarica tutto con paginazione ──
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
        st.session_state[cache_key]  = all_acts
        st.session_state[token_key]  = access_token[:20]
        # Salva il timestamp dell'attività più recente
        if all_acts:
            import time as _time
            # start_date delle attività Strava è ISO string → converti in unix
            dates = []
            for a in all_acts:
                try:
                    from datetime import datetime as _dt
                    d = _dt.fromisoformat(a.get("start_date","").replace("Z",""))
                    dates.append(int(d.timestamp()))
                except Exception:
                    pass
            st.session_state[ts_key] = max(dates) if dates else 0
        return st.session_state[cache_key]

    else:
        # ── AGGIORNAMENTO INCREMENTALE: solo le nuove ──
        new_acts = []
        page = 1
        while True:
            batch = _fetch_page(access_token, page, after_ts=last_ts)
            if not batch:
                break
            new_acts.extend(batch)
            if len(batch) < 200:
                break
            page += 1

        if new_acts:
            # Unisci evitando duplicati per id
            existing_ids = {a["id"] for a in existing}
            added = [a for a in new_acts if a["id"] not in existing_ids]
            if added:
                merged = existing + added
                # Aggiorna timestamp ultimo scaricato
                new_dates = []
                for a in added:
                    try:
                        from datetime import datetime as _dt
                        d = _dt.fromisoformat(a.get("start_date","").replace("Z",""))
                        new_dates.append(int(d.timestamp()))
                    except Exception:
                        pass
                if new_dates:
                    st.session_state[ts_key] = max(new_dates)
                st.session_state[cache_key] = merged
                st.session_state[token_key] = access_token[:20]
                return merged

        # Nessuna novità — ritorna la cache invariata
        return existing


@st.cache_data(ttl=300)
def fetch_athlete(access_token: str):
    r = requests.get(
        "https://www.strava.com/api/v3/athlete",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return r.json() if r.status_code == 200 else {}

# ============================================================
# 10. SESSION STATE
# ============================================================
for key, val in {
    "strava_token_info":  {},
    "messages":           [],
    "user_data":          {"peso": 75.0, "fc_min": 50, "fc_max": 190, "ftp": 200},
    "sport_filter":       None,
    "rc_vitals":          None,    # DataFrame RingConn Vital Signs
    "rc_sleep":           None,    # DataFrame RingConn Sleep
    "rc_activity":        None,    # DataFrame RingConn Activity (passi + kcal giornalieri)
    "activities_cache":   [],      # storico attività Strava (incrementale)
    "activities_last_ts": 0,       # unix timestamp ultima attività scaricata
    "activities_token":   "",      # fingerprint token per invalidare se cambia utente
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# 11. OAUTH
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
# 11b. MAPBOX — Contatori globali e helper render-safe
# ============================================================
def _mb_init_counters():
    """Inizializza/resetta i contatori Mapbox nella sessione."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    month_str = datetime.now().strftime("%Y-%m")
    if "mb_loads_session"  not in st.session_state: st.session_state.mb_loads_session  = 0
    if "mb_loads_daily"    not in st.session_state: st.session_state.mb_loads_daily    = 0
    if "mb_loads_monthly"  not in st.session_state: st.session_state.mb_loads_monthly  = 0
    if "mb_loads_daily_dt" not in st.session_state: st.session_state.mb_loads_daily_dt = today_str
    if "mb_loads_month_dt" not in st.session_state: st.session_state.mb_loads_month_dt = month_str
    if st.session_state.mb_loads_daily_dt != today_str:
        st.session_state.mb_loads_daily    = 0
        st.session_state.mb_loads_daily_dt = today_str
    if st.session_state.mb_loads_month_dt != month_str:
        st.session_state.mb_loads_monthly  = 0
        st.session_state.mb_loads_month_dt = month_str

def mapbox_render_allowed() -> tuple:
    """Controlla se è sicuro renderizzare una mappa. Ritorna (ok: bool, motivo: str)."""
    _mb_init_counters()
    if st.session_state.mb_loads_monthly >= MAPBOX_MONTHLY_LIMIT:
        return False, f"🛑 Limite mensile raggiunto ({MAPBOX_MONTHLY_LIMIT:,} map load)."
    if st.session_state.mb_loads_daily >= MAPBOX_DAILY_SOFT_CAP:
        return False, f"⚠️ Soglia giornaliera raggiunta ({MAPBOX_DAILY_SOFT_CAP}). Riprendi domani."
    if st.session_state.mb_loads_session >= MAPBOX_SESSION_CAP:
        return False, f"⚠️ Limite sessione ({MAPBOX_SESSION_CAP}). Ricarica la pagina."
    return True, ""

def mapbox_register_load():
    """Incrementa tutti i contatori dopo un render."""
    _mb_init_counters()
    st.session_state.mb_loads_session += 1
    st.session_state.mb_loads_daily   += 1
    st.session_state.mb_loads_monthly += 1

# ============================================================
# 12. CORE APP
# ============================================================
token_ok = refresh_token_if_needed()

if token_ok:
    access_token = st.session_state.strava_token_info["access_token"]
    athlete      = fetch_athlete(access_token)

    # Caricamento incrementale: primo accesso → tutto lo storico,
    # accessi successivi → solo le attività nuove dall'ultima scaricata.
    is_first_load = not st.session_state.get("activities_cache")
    if is_first_load:
        with st.spinner("⏳ Primo accesso: scaricamento storico completo da Strava (30-60 secondi)..."):
            raw = load_activities_incremental(access_token)
        n_loaded = len(raw)
        st.toast(f"✅ {n_loaded} attività caricate dallo storico Strava", icon="🏃")
    else:
        prev_count = len(st.session_state.activities_cache)
        raw        = load_activities_incremental(access_token)
        new_count  = len(raw) - prev_count
        if new_count > 0:
            st.toast(f"🆕 {new_count} nuova/e attività sincronizzate da Strava", icon="✅")

    if not raw:
        st.error("Impossibile recuperare le attività.")
        st.stop()

    df = pd.DataFrame(raw)
    df["start_date"] = pd.to_datetime(df["start_date_local"]).dt.tz_localize(None)
    df = df.sort_values("start_date").reset_index(drop=True)

    for col in ["average_heartrate", "max_heartrate", "average_watts", "total_elevation_gain",
                "average_cadence", "kilojoules", "calories", "suffer_score"]:
        if col not in df.columns:
            df[col] = np.nan

    u = st.session_state.user_data
    df["tss"] = df.apply(lambda row: calc_tss(row, u), axis=1)

    ctl_s, atl_s, tsb_s, ctl_daily, atl_daily, tsb_daily, tss_daily = compute_fitness(df)
    df["ctl"] = ctl_s.values
    df["atl"] = atl_s.values
    df["tsb"] = tsb_s.values

    current_ctl = df["ctl"].iloc[-1]
    current_atl = df["atl"].iloc[-1]
    current_tsb = df["tsb"].iloc[-1]

    # ---- METRICHE AVANZATE ----
    df["trimp"]      = df.apply(lambda row: calc_trimp(row, u), axis=1)
    acwr_val, acwr_series = calc_acwr(df)
    ramp_rate        = calc_ramp_rate(ctl_daily)
    monotonia        = calc_monotony(df)
    strain_val       = calc_training_strain(df)
    df["ef"]         = calc_ef_series(df)
    vo2max_val, _    = calc_vo2max_estimate(df)
    race_preds       = predict_race_times(vo2max_val)
    df["vi"]         = df.apply(calc_variability_index, axis=1)

    # Zone per ogni attività
    df["zone_num"]   = df.apply(lambda r: get_zone_for_activity(r, u["fc_max"])[0], axis=1)
    df["zone_color"] = df.apply(lambda r: get_zone_for_activity(r, u["fc_max"])[1], axis=1)
    df["zone_label"] = df.apply(lambda r: get_zone_for_activity(r, u["fc_max"])[2], axis=1)

    def tsb_status(v):
        if v > 20:  return "⚠️ Possibile detrain",  "#FF9800"
        if v > -10: return "✅ Forma ottimale",      "#4CAF50"
        if v > -20: return "🟡 Accumulo fatica",     "#FFC107"
        return "🔴 Sovraccarico", "#F44336"

    status_label, status_color = tsb_status(current_tsb)

    # ---- RINGCONN — shorthand + Readiness Score ----
    rc_vitals    = st.session_state.rc_vitals    # None o DataFrame
    rc_sleep     = st.session_state.rc_sleep     # None o DataFrame
    rc_activity  = st.session_state.rc_activity  # None o DataFrame (passi + kcal)

    # Readiness Score giornaliero (oggi o ultimo disponibile)
    today_str = datetime.now().strftime("%Y-%m-%d")
    _vrow, _srow = None, None
    if rc_vitals is not None and not rc_vitals.empty:
        hrv_baseline = rc_vitals["hrv_avg"].dropna().median()
        _match_v = rc_vitals[rc_vitals["date"].dt.strftime("%Y-%m-%d") == today_str]
        if _match_v.empty:
            _match_v = rc_vitals.dropna(subset=["hrv_avg"]).tail(1)
        if not _match_v.empty:
            _vrow = _match_v.iloc[0].to_dict()
            _vrow["hrv_baseline"] = hrv_baseline
    if rc_sleep is not None and not rc_sleep.empty:
        _match_s = rc_sleep[rc_sleep["date"].dt.strftime("%Y-%m-%d") == today_str]
        if _match_s.empty:
            _match_s = rc_sleep.tail(1)
        if not _match_s.empty:
            _srow = _match_s.iloc[0].to_dict()

    readiness = calc_readiness(_vrow, _srow, current_tsb)
    rc_ai_context = get_ringconn_context(rc_vitals, rc_sleep)

    # ---- METRICHE AVANZATE BATCH 2 ----
    hrv_slope    = calc_hrv_trend_slope(rc_vitals, days=14)
    slr          = calc_sleep_load_ratio(rc_sleep, df, days=14)
    circadian    = calc_circadian_performance(df)
    acwr_v2      = calc_acwr_v2(df, rc_vitals)
    tss_budget   = calc_adaptive_tss_budget(readiness["score"], df)
    nutrition    = calc_nutritional_window(df, days=7)

    # ---- SIDEBAR ----
    with st.sidebar:
        st.markdown(f"### 🏆 Elite AI Coach")
        if athlete:
            name = f"{athlete.get('firstname','')} {athlete.get('lastname','')}".strip()
            if name:
                st.markdown(f"**{name}**")
            if athlete.get("profile_medium"):
                st.image(athlete["profile_medium"], width=60)
        st.divider()

        menu = st.radio("", [
            "📊 Dashboard",
            "💪 Stato Fisico",
            "😴 Recupero & Sonno",
            "💬 Coach Chat",
            "🗺️ Planning Route",
            "🧬 Metriche Avanzate",
            "📅 Storico & Calendario",
            "🗺️ Mappe 3D",
            "👤 Profilo Fisico",
        ], label_visibility="collapsed")

        st.divider()
        # Auto-discovery modelli realmente disponibili per questa API key
        all_models = get_available_models(GEMINI_KEY or "")

        default_idx = 0
        if "sel_model" in st.session_state and st.session_state.sel_model in all_models:
            default_idx = all_models.index(st.session_state.sel_model)

        if _ai_sdk_mode == "grok":
            sdk_badge = "⚡ xAI Grok"
        elif _ai_sdk_mode == "new":
            sdk_badge = "🆕 Gemini (google-genai)"
        elif _ai_sdk_mode == "old":
            sdk_badge = "🔄 Gemini (generativeai)"
        else:
            sdk_badge = "⚠️ Nessun provider AI"

        sel_model = st.selectbox(
            "🧠 Modello AI:", all_models, index=default_idx,
            help=f"Provider: {sdk_badge}. Imposta GROK_API_KEY per Grok, GOOGLE_API_KEY per Gemini."
        )
        st.session_state.sel_model = sel_model
        st.caption(sdk_badge)

        def ai_generate(prompt: str) -> str:
            """Wrapper unificato — Grok (xAI) o Gemini (Google)."""
            if _ai_sdk_mode is None:
                raise RuntimeError("Nessun provider AI configurato. Aggiungi GROK_API_KEY o GOOGLE_API_KEY nei Secrets.")
            if _ai_sdk_mode == "grok":
                resp = _ai_client.chat.completions.create(
                    model=sel_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                )
                return resp.choices[0].message.content
            elif _ai_sdk_mode == "new":
                # Modelli 1.5 richiedono client v1alpha; 2.0+ usano il client standard
                _is_15 = "1.5" in sel_model
                _client_to_use = _ai_client_v1a if (_is_15 and _ai_client_v1a) else _ai_client
                # Il nome modello non deve avere il prefisso models/ per la nuova SDK
                _model_id = sel_model.replace("models/", "")
                response = _client_to_use.models.generate_content(
                    model=_model_id,
                    contents=prompt,
                )
                return response.text
            else:
                return _ai_client.GenerativeModel(sel_model).generate_content(prompt).text

        def ai_generate_model(prompt: str, model_id: str, max_tokens: int = 2000) -> str:
            """Come ai_generate ma con modello esplicito — per routing Gemini 2.5 Pro / Grok."""
            if _ai_sdk_mode is None:
                raise RuntimeError("Nessun provider AI configurato.")
            if _ai_sdk_mode == "grok":
                resp = _ai_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            elif _ai_sdk_mode == "new":
                _is_15 = "1.5" in model_id
                _cl = _ai_client_v1a if (_is_15 and _ai_client_v1a) else _ai_client
                response = _cl.models.generate_content(
                    model=model_id.replace("models/", ""),
                    contents=prompt,
                )
                return response.text
            else:
                return _ai_client.GenerativeModel(model_id).generate_content(prompt).text

        def ai_fast(prompt: str) -> str:
            """Risposta veloce — usa Grok se disponibile, altrimenti Flash."""
            _fast_model = "grok-3-beta" if _ai_sdk_mode == "grok" else "gemini-2.0-flash"
            return ai_generate_model(prompt, _fast_model, max_tokens=1000)

        def ai_deep(prompt: str) -> str:
            """Analisi approfondita — usa Gemini 2.5 Pro, fallback su modello selezionato."""
            try:
                return ai_generate_model(prompt, "gemini-2.5-pro-preview-03-25", max_tokens=2500)
            except Exception:
                return ai_generate(prompt)

        st.divider()
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("CTL", f"{current_ctl:.0f}")
        col_s2.metric("ATL", f"{current_atl:.0f}")
        col_s3.metric("TSB", f"{current_tsb:.0f}")
        st.markdown(f"<div style='text-align:center; color:{status_color}; font-size:13px'>{status_label}</div>", unsafe_allow_html=True)

        st.divider()
        # ── RingConn Upload ──
        st.markdown("##### 💍 RingConn")
        rc_has_data = st.session_state.rc_vitals is not None
        if rc_has_data:
            st.success(f"✅ {len(st.session_state.rc_vitals)} giorni caricati")
        with st.expander("📂 Carica dati" if not rc_has_data else "🔄 Aggiorna dati"):
            st.caption("Esporta dall'app RingConn: Profilo → Impostazioni → Data Export")
            f_zip = st.file_uploader("📦 ZIP RingConn (tutti i file)", type="zip", key="up_zip")
            if f_zip:
                try:
                    _v, _s, _a = parse_ringconn_zip(f_zip)
                    if _v is not None: st.session_state.rc_vitals   = _v
                    if _s is not None: st.session_state.rc_sleep    = _s
                    if _a is not None: st.session_state.rc_activity = _a
                    _loaded = [k for k,v in [("Vital Signs",_v),("Sleep",_s),("Activity",_a)] if v is not None]
                    st.success(f"✅ Caricati: {', '.join(_loaded)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore ZIP: {e}")
            with st.expander("📄 Oppure CSV singoli"):
                f_vitals   = st.file_uploader("Vital_Signs CSV", type="csv", key="up_vitals")
                f_sleep    = st.file_uploader("Sleep CSV",        type="csv", key="up_sleep")
                f_activity = st.file_uploader("Activity CSV",     type="csv", key="up_activity")
                if f_vitals:
                    try:
                        st.session_state.rc_vitals = parse_ringconn_vitals(f_vitals)
                        st.success(f"✅ Vital Signs: {len(st.session_state.rc_vitals)} giorni")
                    except Exception as e: st.error(f"Errore: {e}")
                if f_sleep:
                    try:
                        st.session_state.rc_sleep = parse_ringconn_sleep(f_sleep)
                        st.success(f"✅ Sleep: {len(st.session_state.rc_sleep)} sessioni")
                    except Exception as e: st.error(f"Errore: {e}")
                if f_activity:
                    try:
                        st.session_state.rc_activity = parse_ringconn_activity(f_activity)
                        st.success(f"✅ Activity: {len(st.session_state.rc_activity)} giorni")
                    except Exception as e: st.error(f"Errore: {e}")
            if any(st.session_state.get(k) is not None for k in ["rc_vitals","rc_sleep","rc_activity"]):
                if st.button("🗑️ Rimuovi dati RingConn", use_container_width=True):
                    st.session_state.rc_vitals = st.session_state.rc_sleep = st.session_state.rc_activity = None
                    st.rerun()

        st.divider()
        # Info sync + pulsante aggiornamento manuale
        n_cached = len(st.session_state.get("activities_cache", []))
        last_ts  = st.session_state.get("activities_last_ts", 0)
        if n_cached:
            st.caption(f"📦 {n_cached} attività in cache")
            if last_ts:
                from datetime import datetime as _dt2
                last_dt = _dt2.fromtimestamp(last_ts).strftime("%d/%m %H:%M")
                st.caption(f"🕐 Ultima sync: {last_dt}")
        col_sync, col_out = st.columns(2)
        with col_sync:
            if st.button("🔄 Sync", use_container_width=True,
                         help="Controlla se ci sono nuove attività su Strava"):
                prev = len(st.session_state.get("activities_cache", []))
                raw_new = load_activities_incremental(access_token)
                diff = len(raw_new) - prev
                if diff > 0:
                    st.toast(f"🆕 {diff} nuova/e attività!", icon="✅")
                else:
                    st.toast("Nessuna attività nuova", icon="✅")
                st.rerun()
        with col_out:
            if st.button("🚪 Esci", use_container_width=True):
                for k in ["strava_token_info","activities_cache",
                          "activities_last_ts","activities_token"]:
                    st.session_state[k] = {} if k == "strava_token_info" else []  if "cache" in k else 0 if "ts" in k else ""
                st.cache_data.clear()
                st.rerun()

    # ============================================================
    # DASHBOARD
    # ============================================================
    # ── DETTAGLIO ATTIVITÀ — intercetta qualunque sezione ──────
    if st.session_state.get("selected_activity_id") is not None:
        _sel_id = st.session_state.selected_activity_id
        _sel_row = df[df["id"] == _sel_id] if "id" in df.columns else pd.DataFrame()
        if _sel_row.empty:
            try:
                _sel_row = df.iloc[[int(_sel_id)]]
            except Exception:
                pass
        if not _sel_row.empty:
            render_activity_detail(
                row=_sel_row.iloc[0], u=u,
                MAPBOX_TOKEN=MAPBOX_TOKEN,
                draw_map=draw_map,
                build_inline_map3d=build_inline_map3d,
                mapbox_render_allowed=mapbox_render_allowed,
                mapbox_register_load=mapbox_register_load,
                ai_generate=ai_generate,
                current_ctl=current_ctl, current_atl=current_atl,
                current_tsb=current_tsb, status_label=status_label,
            )
            st.stop()

    # ── Auto-reset dettaglio se cambio menu ──────────────────
    _last_menu = st.session_state.get("_last_menu_for_detail", menu)
    if menu != _last_menu and st.session_state.get("selected_activity_id") is not None:
        st.session_state.selected_activity_id = None
        st.session_state["_last_menu_for_detail"] = menu
        st.rerun()
    st.session_state["_last_menu_for_detail"] = menu

    if menu == "📊 Dashboard":
        st.markdown("## 📊 Performance Hub")

        # ── Alert intelligenti ──
        alert_lines = []
        # Alert RingConn + TSB
        if rc_vitals is not None and _vrow is not None:
            hrv_now  = _vrow.get("hrv_avg")
            hrv_base = _vrow.get("hrv_baseline", hrv_now)
            if pd.notna(hrv_now) and pd.notna(hrv_base) and hrv_base > 0:
                hrv_ratio = hrv_now / hrv_base
                if hrv_ratio < 0.80 and current_tsb < -10:
                    alert_lines.append(("🔴", "ATTENZIONE", f"HRV al {hrv_ratio*100:.0f}% della baseline + TSB {current_tsb:.0f}: recupero prioritario oggi.", "#F44336"))
                elif hrv_ratio >= 1.05 and current_tsb > 0:
                    alert_lines.append(("🟢", "OTTIMO", f"HRV sopra baseline ({hrv_now:.0f} ms) e TSB positivo ({current_tsb:.0f}): giornata ideale per allenamento intenso.", "#4CAF50"))
                elif hrv_ratio < 0.85:
                    alert_lines.append(("🟡", "ATTENZIONE", f"HRV ridotto ({hrv_now:.0f} ms, -{100-hrv_ratio*100:.0f}% dalla baseline): considera intensità ridotta.", "#FF9800"))
        if acwr_v2["risk"] >= 80:
            alert_lines.append(("🔴", "RISCHIO INFORTUNI", f"ACWR 2.0: {acwr_v2['label']} (score {acwr_v2['risk']}/100). {acwr_v2['acwr_adj']:.2f} pesato su HRV.", "#F44336"))
        elif acwr_v2["risk"] >= 60:
            alert_lines.append(("🟠", "CARICO ALTO", f"ACWR 2.0: {acwr_v2['label']} — riduci l'intensità.", "#FF5722"))
        if hrv_slope is not None and "Overreaching" in hrv_slope["label"]:
            alert_lines.append(("⚠️", "HRV IN CALO", f"Pendenza HRV: {hrv_slope['slope']:+.2f} ms/gg negli ultimi 14gg. Possibile overreaching.", "#FF5722"))

        for emoji, title, msg, color in alert_lines:
            st.markdown(f"""
            <div style="background:{color}18;border-left:4px solid {color};border-radius:0 10px 10px 0;
                        padding:10px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px">
                <span style="font-size:20px">{emoji}</span>
                <div>
                    <span style="color:{color};font-weight:700;font-size:12px">{title}</span>
                    <span style="color:#ccc;font-size:13px;margin-left:8px">{msg}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── PERFORMANCE HUB ──────────────────────────────────────────
        st.markdown("### 🏆 Performance Hub — Stato Attuale")

        # Calcoli Performance Hub (ultimi 21 giorni)
        _now   = df["start_date"].max()
        _df21  = df[df["start_date"] >= (_now - timedelta(days=21))]
        _df14  = df[df["start_date"] >= (_now - timedelta(days=14))]
        _df7   = df[df["start_date"] >= (_now - timedelta(days=7))]

        # VO2max stima
        _vo2max, _ = calc_vo2max_estimate(df)

        # EF trend (ultimi 14gg): avg pace o watt / HR
        _ef_vals = []
        for _, _r in _df14.iterrows():
            _hr = _r.get("average_heartrate")
            if not _hr or not pd.notna(_hr) or _hr <= 0: continue
            if _r["type"] in ["Run","TrailRun"] and _r["distance"] > 0:
                _pace_mpm = (_r["moving_time"]/60) / (_r["distance"]/1000)
                _ef_vals.append(1 / (_pace_mpm * _hr / 100))   # higher = better
            elif _r["type"] in ["Ride","VirtualRide","MountainBikeRide"]:
                _w = _r.get("average_watts")
                if _w and pd.notna(_w) and _w > 0:
                    _ef_vals.append(_w / _hr)
        _ef_avg = float(np.mean(_ef_vals)) if _ef_vals else None

        # Monotonia (Banister)
        _daily_tss = df.groupby(df["start_date"].dt.date)["tss"].sum()
        _last7_tss = []
        for _i in range(7):
            _dd = (_now - timedelta(days=_i)).date()
            _last7_tss.append(float(_daily_tss.get(_dd, 0)))
        _mono_mean = np.mean(_last7_tss) if _last7_tss else 0
        _mono_std  = np.std(_last7_tss)  if _last7_tss else 1
        _monotonia = _mono_mean / _mono_std if _mono_std > 0 else 0
        _strain    = _mono_mean * _mono_std * 7

        # Consistenza 21gg
        _active_21 = len(_df21["start_date"].dt.date.unique())
        _consist_pct = _active_21 / 21 * 100

        # Weekly load trend (ultimi 4 settimane)
        _w_loads = []
        for _wi in range(4):
            _ws = _now - timedelta(days=7*(_wi+1))
            _we = _now - timedelta(days=7*_wi)
            _w_loads.append(float(df[(df["start_date"] >= _ws) & (df["start_date"] < _we)]["tss"].sum()))
        _load_trend = _w_loads[0] - _w_loads[1] if len(_w_loads) >= 2 else 0

        # Colori stato
        _ctl_col = "#4CAF50" if current_ctl > 50 else "#FF9800" if current_ctl > 30 else "#888"
        _tsb_col = "#4CAF50" if -10 <= current_tsb <= 20 else "#FF9800" if current_tsb > 20 else "#F44336"
        _atl_col = "#FF9800" if current_atl > current_ctl * 1.3 else "#4CAF50"

        # Riga 1: CTL / ATL / TSB / VO2max / FTP
        ph1, ph2, ph3, ph4, ph5 = st.columns(5)
        ph1.metric("🏋️ Fitness (CTL)", f"{current_ctl:.1f}",
                   delta=f"{'↑' if _load_trend>0 else '↓'} {abs(_load_trend):.0f} TSS/sett",
                   help="Chronic Training Load 42gg — la tua capacità di lavoro costruita")
        ph2.metric("⚡ Fatica (ATL)", f"{current_atl:.1f}",
                   help="Acute Training Load 7gg — stanchezza accumulata recente")
        ph3.metric("✨ Forma (TSB)", f"{current_tsb:.1f}",
                   delta=status_label, delta_color="off",
                   help="Training Stress Balance — positivo=fresco, negativo=affaticato")
        ph4.metric("🫁 VO2max stimato", f"{_vo2max:.1f} ml/kg/min" if _vo2max else "N/D",
                   help="Stima da migliore performance in corsa (formula Daniels)")
        ph5.metric("⚙️ FTP", f"{u['ftp']} W",
                   help="Functional Threshold Power — impostato in Profilo Fisico")

        # Riga 2: EF / Monotonia / Consistenza / Readiness / Budget TSS
        ph6, ph7, ph8, ph9, ph10 = st.columns(5)
        ph6.metric("📈 Efficienza (EF)", f"{_ef_avg:.3f}" if _ef_avg else "N/D",
                   help="Watt/FC (bici) o 1/(passo×FC/100) (corsa) — ultimi 14gg. Più alto = meglio")
        ph7.metric("🔄 Monotonia", f"{_monotonia:.2f}",
                   delta="⚠️ alta" if _monotonia > 2.0 else "✅ ok",
                   delta_color="off",
                   help="TSS_medio / TSS_std ultimi 7gg. >2.0 = troppo uniforme, rischio overtraining")
        ph8.metric("📅 Consistenza 21gg", f"{_consist_pct:.0f}%",
                   delta=f"{_active_21}/21 giorni attivi",
                   help="% giorni con almeno 1 attività negli ultimi 21 giorni")
        if rc_vitals is not None:
            rs = readiness
            ph9.metric("💍 Readiness", f"{rs['score']}/100",
                       delta=f"{rs['emoji']} {rs['label']}", delta_color="off")
        else:
            ph9.metric("💍 Readiness", "N/D", help="Carica i dati RingConn per abilitare")
        if rc_vitals is not None or rc_sleep is not None:
            b = tss_budget
            ph10.metric("💰 TSS Budget oggi", f"{b['budget']}",
                        delta=f"{b['zone']} · rem. {b['remaining']:.0f}", delta_color="off")
        else:
            _tss7_avg = _df7["tss"].sum() / 7
            ph10.metric("📊 TSS medio/gg 7gg", f"{_tss7_avg:.0f}")

        # Banner stato + budget (se RingConn)
        if rc_vitals is not None or rc_sleep is not None:
            b = tss_budget
            st.markdown(f"""
            <div style="background:{b['color']}10;border:1px solid {b['color']}33;
                        border-radius:10px;padding:10px 18px;margin:8px 0;
                        display:flex;align-items:center;gap:16px">
                <div style="text-align:center;min-width:55px">
                    <div style="font-size:26px;font-weight:900;color:{b['color']}">{b['budget']}</div>
                    <div style="font-size:10px;color:#888">TSS budget</div>
                </div>
                <div style="flex:1">
                    <div style="font-size:13px;font-weight:700;color:{b['color']}">{b['zone']}</div>
                    <div style="font-size:11px;color:#aaa">{b['advice']}</div>
                </div>
                <div style="font-size:12px;color:#666">
                    Speso: <b style="color:#fff">{b['tss_spent']:.0f}</b> &nbsp;|&nbsp;
                    Rimanente: <b style="color:{b['color']}">{b['remaining']:.0f}</b>
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- Recap ultimi 7 giorni ---
        st.markdown("### 📅 Recap Ultimi 7 Giorni")
        df7 = df[df["start_date"] >= (df["start_date"].max() - timedelta(days=7))]

        r7c1, r7c2, r7c3, r7c4, r7c5, r7c6 = st.columns(6)
        r7c1.metric("Sessioni",    len(df7))
        r7c2.metric("Km totali",   f"{df7['distance'].sum()/1000:.1f}")
        r7c3.metric("Ore totali",  f"{df7['moving_time'].sum()/3600:.1f}")
        r7c4.metric("TSS totale",  f"{df7['tss'].sum():.0f}")
        r7c5.metric("Dislivello",  f"{(df7['total_elevation_gain'].sum() or 0):.0f} m")
        avg_hr_7 = df7["average_heartrate"].dropna()
        r7c6.metric("FC media",    f"{avg_hr_7.mean():.0f} bpm" if not avg_hr_7.empty else "N/A")

        # Lista sport settimana + confronto settimana prec.
        if not df7.empty:
            df_prev7 = df[
                (df["start_date"] >= (df["start_date"].max() - timedelta(days=14))) &
                (df["start_date"] <  (df["start_date"].max() - timedelta(days=7)))
            ]
            col_sports7, col_days7 = st.columns([3, 1])
            with col_sports7:
                for sp in df7["type"].value_counts().index:
                    si       = get_sport_info(sp)
                    df7_sp   = df7[df7["type"] == sp]
                    prev_sp  = df_prev7[df_prev7["type"] == sp]
                    n_sess   = len(df7_sp)
                    km       = df7_sp["distance"].sum() / 1000
                    ore      = df7_sp["moving_time"].sum() / 3600
                    tss_sp   = df7_sp["tss"].sum()
                    elev_sp  = df7_sp["total_elevation_gain"].sum() or 0
                    # delta km vs settimana prec.
                    prev_km  = prev_sp["distance"].sum() / 1000 if not prev_sp.empty else 0
                    delta_km_sp = km - prev_km
                    dkm_col  = "#4CAF50" if delta_km_sp >= 0 else "#F44336"
                    dkm_str  = f"<span style='color:{dkm_col};font-size:11px'>{delta_km_sp:+.1f} km vs prec.</span>"
                    st.markdown(f"""
                    <div style="background:{si['color']}18;border-left:4px solid {si['color']};
                                border-radius:0 10px 10px 0;padding:10px 16px;margin:5px 0;
                                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
                        <div style="display:flex;align-items:center;gap:8px;min-width:160px">
                            <span style="font-size:22px">{si['icon']}</span>
                            <div>
                                <div style="font-weight:700;color:{si['color']};font-size:14px">{si['label']}</div>
                                <div style="font-size:11px;color:#444">{n_sess} {'sessione' if n_sess==1 else 'sessioni'}</div>
                            </div>
                        </div>
                        <div style="display:flex;gap:16px;flex-wrap:wrap">
                            <div style="text-align:center">
                                <div style="font-size:16px;font-weight:700;color:#111">{km:.1f}</div>
                                <div style="font-size:10px;color:#555">km</div>
                            </div>
                            <div style="text-align:center">
                                <div style="font-size:16px;font-weight:700;color:#111">{int(ore)}h {int((ore%1)*60)}m</div>
                                <div style="font-size:10px;color:#555">durata</div>
                            </div>
                            <div style="text-align:center">
                                <div style="font-size:16px;font-weight:700;color:#111">{tss_sp:.0f}</div>
                                <div style="font-size:10px;color:#555">TSS</div>
                            </div>
                            <div style="text-align:center">
                                <div style="font-size:16px;font-weight:700;color:#111">{elev_sp:.0f}</div>
                                <div style="font-size:10px;color:#555">↑ m</div>
                            </div>
                        </div>
                        <div>{dkm_str}</div>
                    </div>""", unsafe_allow_html=True)

            with col_days7:
                st.markdown("<div style='font-size:13px;color:#888;margin-bottom:6px'>Giorni attivi</div>", unsafe_allow_html=True)
                today = datetime.now().date()
                week_html = "<div style='display:flex;gap:4px;flex-wrap:wrap'>"
                giorni = ["L","M","M","G","V","S","D"]
                for i in range(6, -1, -1):
                    d = today - timedelta(days=i)
                    acts_d = df7[df7["start_date"].dt.date == d]
                    active = not acts_d.empty
                    if active:
                        top_sp = acts_d["type"].value_counts().index[0]
                        bg = get_sport_info(top_sp)["color"]
                        brd = bg
                    else:
                        bg = "rgba(255,255,255,0.07)"
                        brd = "rgba(255,255,255,0.1)"
                    week_html += (
                        f"<div style='width:30px;height:30px;border-radius:8px;"
                        f"background:{bg};border:1px solid {brd};"
                        f"display:flex;align-items:center;justify-content:center;"
                        f"font-size:11px;color:{'#fff' if active else '#555'}'>"
                        f"{giorni[d.weekday()]}</div>"
                    )
                week_html += "</div>"
                st.markdown(week_html, unsafe_allow_html=True)

                delta_tss = df7["tss"].sum() - df_prev7["tss"].sum()
                delta_km_tot = df7["distance"].sum()/1000 - df_prev7["distance"].sum()/1000
                tss_color = "#4CAF50" if delta_tss >= 0 else "#F44336"
                km_color  = "#4CAF50" if delta_km_tot >= 0 else "#F44336"
                st.markdown(
                    f"<div style='font-size:11px;color:#888;margin-top:10px'>vs settimana prec.</div>"
                    f"<div style='margin-top:4px'>"
                    f"<span style='color:{km_color};font-size:13px'>{delta_km_tot:+.1f} km</span>&nbsp;&nbsp;"
                    f"<span style='color:{tss_color};font-size:13px'>{delta_tss:+.0f} TSS</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.divider()

        # --- Ultime 5 attività ---
        st.markdown("### 🕐 Ultime 5 Attività")
        last3 = df.iloc[-5:][::-1]

        for idx, (_, row) in enumerate(last3.iterrows()):
            s   = get_sport_info(row["type"], row.get("name",""))
            m   = format_metrics(row)
            z_n, z_c, z_l = get_zone_for_activity(row, u["fc_max"])

            with st.container():
                _cc, _cb = st.columns([11, 1])
                with _cc:
                    st.markdown(f"""
                    <div class="activity-card" style="border-color: {s['color']}40;">
                        <div class="activity-header" style="color:{s['color']}">
                            {s['icon']} {row['name']}
                            <span style="font-size:12px; color:#666; font-weight:400; margin-left:8px">
                                {row['start_date'].strftime('%d %b %Y · %H:%M')}
                            </span>
                            <span class="zone-badge" style="background:{z_c}18; color:{z_c}; border:1px solid {z_c}55; float:right; font-size:12px; font-weight:700">
                                {z_l}
                            </span>
                        </div>
                        <div class="metric-row">
                            <div class="metric-pill">📏 Distanza <span>{m['dist_str']}</span></div>
                            <div class="metric-pill">⏱️ Durata <span>{m['dur_str']}</span></div>
                            <div class="metric-pill">⚡ {('Passo' if row['type'] not in ('Ride','VirtualRide','MountainBikeRide') else 'Velocità')} <span>{m['pace_str']}</span></div>
                            <div class="metric-pill">⛰️ Dislivello <span>{m['elev']}</span></div>
                            <div class="metric-pill">❤️ FC Media <span>{m['hr_avg']}</span></div>
                            <div class="metric-pill">💓 FC Max <span>{m['hr_max']}</span></div>
                            <div class="metric-pill">🔄 Cadenza <span>{m['cadence']}</span></div>
                            <div class="metric-pill">⚡ Watt <span>{m['watts']}</span></div>
                            <div class="metric-pill">🔥 Calorie <span>{m['calories']}</span></div>
                            <div class="metric-pill">📊 TSS <span>{row['tss']:.1f}</span></div>
                            <div class="metric-pill">😓 Suffer Score <span>{m['suffer']}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with _cb:
                    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                    open_activity_button(row, key_suffix=f"dash_{idx}")

            # Mappa per tutte le attività
            _poly = row.get("map", {})
            _poly = _poly.get("summary_polyline") if isinstance(_poly, dict) else None
            if _poly:
                _mb_ok, _ = mapbox_render_allowed()
                _tab2d, _tab3d = st.tabs(["🗺️ Mappa 2D", "🏔️ Mappa 3D"])
                with _tab2d:
                    _mobj = draw_map(_poly)
                    if _mobj:
                        _mc, _minfo = st.columns([3, 1])
                        with _mc:
                            st_folium(_mobj, width=None, height=260, key=f"map_{idx}")
                        with _minfo:
                            st.markdown(f"**Sport:** {s['label']}")
                            st.markdown(f"**Data:** {row['start_date'].strftime('%d %b')}")
                            st.markdown(f"**Distanza:** {m['dist_str']}")
                            st.markdown(f"**Durata:** {m['dur_str']}")
                            st.markdown(f"**Zona:** {z_l}")
                            _pct = (row.get('average_heartrate',0) or 0) / u['fc_max'] * 100
                            st.markdown(f"**%FC:** {_pct:.0f}%")
                with _tab3d:
                    if not MAPBOX_TOKEN:
                        st.info("Configura MAPBOX_TOKEN per la mappa 3D.")
                    elif not _mb_ok:
                        st.warning("Limite Mapbox raggiunto.")
                    else:
                        _eg = float(row.get("total_elevation_gain") or 0)
                        _h3d = build_inline_map3d(_poly, MAPBOX_TOKEN,
                                  sport_type=row.get("type",""), elev_gain=_eg, height=340)
                        if _h3d:
                            import streamlit.components.v1 as components
                            components.html(_h3d, height=350, scrolling=False)
                            mapbox_register_load()

            # --- AI Analisi: solo per la prima attività (la più recente) ---
            if idx == 0:
                with st.expander("🤖 Analisi Coach", expanded=True):
                    # Cache in session_state — non ricalcolare se stessa attività
                    _act_id = str(row.get("id", row["start_date"]))
                    _cache_key = f"ai_analysis_{_act_id}"
                    if _cache_key in st.session_state:
                        st.markdown(
                            f'<div style="background:#f8f9fa;border-left:4px solid {s["color"]};'
                            f'border-radius:8px;padding:16px 20px;color:#212529;font-size:15px;line-height:1.8">'
                            f'{st.session_state[_cache_key]}</div>',
                            unsafe_allow_html=True)
                    else:
                        with st.spinner("Il coach sta analizzando..."):
                            try:
                                _ctx = (
                                    f"Sport: {row['type']} ({s['label']}). "
                                    f"Distanza: {m['dist_str']}. Durata: {m['dur_str']}. "
                                    f"Passo/Vel: {m['pace_str']}. Dislivello: {m['elev']}. "
                                    f"FC Media: {m['hr_avg']}, FC Max: {m['hr_max']}. "
                                    f"Watt: {m['watts']}. TSS: {row['tss']:.1f}. "
                                    f"CTL attuale: {current_ctl:.1f}, TSB: {current_tsb:.1f}, ATL: {current_atl:.1f}. "
                                    f"Stato forma: {status_label}."
                                )
                                _prompt = (
                                    "Sei un coach sportivo di alto livello. "
                                    "Commenta questa sessione: qualità dell'allenamento, punti di forza e debolezze, "
                                    "come influisce sul carico settimanale, e suggerisci cosa fare nella prossima sessione "
                                    "in base allo stato di forma attuale. Sii specifico e pratico. Usa massimo 4 paragrafi."
                                )
                                _result = ai_generate(f"{_ctx}\n\n{_prompt}")
                                st.session_state[_cache_key] = _result
                                st.markdown(
                                    f'<div style="background:#f8f9fa;border-left:4px solid {s["color"]};'
                                    f'border-radius:8px;padding:16px 20px;color:#212529;font-size:15px;line-height:1.8">'
                                    f'{_result}</div>',
                                    unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Errore AI: {e}")

        # --- Grafico CTL/ATL/TSB ---
        st.divider()
        st.markdown("### 📈 Andamento Fitness")
        chart_df = pd.DataFrame({
            "Fitness (CTL)": ctl_daily,
            "Fatica (ATL)":  atl_daily,
            "Forma (TSB)":   tsb_daily,
        }).dropna().tail(120)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Fitness (CTL)"],
                                  name="CTL", line=dict(color="#2196F3", width=2.5), fill="tozeroy",
                                  fillcolor="rgba(33,150,243,0.08)"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Fatica (ATL)"],
                                  name="ATL", line=dict(color="#FF9800", width=2)))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Forma (TSB)"],
                                  name="TSB", line=dict(color="#4CAF50", width=2, dash="dot")))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=30, b=0), height=300,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # STATO FISICO
    # ============================================================

    # ============================================================
    # RECUPERO & SONNO
    # ============================================================
    elif menu == "😴 Recupero & Sonno":
        st.markdown("## 😴 Recupero & Sonno")

        rc_v = st.session_state.rc_vitals
        rc_s = st.session_state.rc_sleep

        if rc_v is None and rc_s is None:
            st.info("Nessun dato RingConn caricato. Usa il pannello **💍 RingConn** nella sidebar per caricare i tuoi CSV.")
            st.markdown("""
            **Come esportare i dati da RingConn:**
            1. Apri l'app RingConn sul telefono
            2. Vai su **Profilo → Impostazioni → Data Export**
            3. Seleziona il range di date (consigliato: ultimi 30-90 giorni)
            4. Scarica i file **Vital_Signs** e **Sleep**
            5. Caricali qui tramite la sidebar
            
            > 💡 **Consiglio:** fai l'upload ogni settimana per mantenere i dati aggiornati.
            """)
            st.stop()

        # ── Readiness Score ──
        rs = readiness
        st.markdown("### 🎯 Readiness Score")
        col_rs, col_rs_break = st.columns([1, 2])
        with col_rs:
            st.markdown(f"""
            <div style="background:{rs['color']}18;border:2px solid {rs['color']}55;
                        border-radius:16px;padding:24px;text-align:center">
                <div style="font-size:56px;font-weight:900;color:{rs['color']}">{rs['score']}</div>
                <div style="font-size:13px;color:#888;margin-top:2px">/ 100</div>
                <div style="font-size:18px;font-weight:700;color:{rs['color']};margin-top:8px">
                    {rs['emoji']} {rs['label']}
                </div>
                <div style="font-size:11px;color:#666;margin-top:6px">Aggiornato oggi</div>
            </div>""", unsafe_allow_html=True)
        with col_rs_break:
            st.markdown("**Componenti del punteggio:**")
            bd = rs["breakdown"]
            component_colors = {"HRV": "#e94560", "Sonno": "#2196F3", "Forma": "#4CAF50", "SpO2": "#9C27B0"}
            maxes = {"HRV": 40, "Sonno": 30, "Forma": 20, "SpO2": 10}
            for comp, val in bd.items():
                pct  = val / maxes[comp] * 100
                col  = component_colors.get(comp, "#888")
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
                        <span style="color:#ccc">{comp}</span>
                        <span style="color:{col};font-weight:700">{val:.0f}/{maxes[comp]}</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.07);border-radius:6px;height:8px">
                        <div style="background:{col};width:{pct:.0f}%;height:8px;border-radius:6px"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Tabs principali ──
        tab_hrv, tab_sleep, tab_steps, tab_corr = st.tabs([
            "❤️ HRV & Vitali", "💤 Sonno", "👟 Passi & Attività", "🔗 Correlazione Performance"
        ])

        # ────────────────────────────────────────────
        # TAB HRV & VITALI
        # ────────────────────────────────────────────
        with tab_hrv:
            if rc_v is None or rc_v.empty:
                st.info("Carica il file Vital_Signs CSV nella sidebar.")
            else:
                rc_v_valid = rc_v.dropna(subset=["hrv_avg"])
                hrv_baseline_val = rc_v_valid["hrv_avg"].median()

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("HRV medio", f"{rc_v_valid['hrv_avg'].mean():.0f} ms")
                k2.metric("HRV baseline (mediana)", f"{hrv_baseline_val:.0f} ms")
                k3.metric("FC riposo media", f"{rc_v['hr_avg'].dropna().mean():.0f} bpm")
                k4.metric("SpO2 media", f"{rc_v['spo2_avg'].dropna().mean():.0f}%")

                # HRV + TSB — ultimi 14gg, dual-line pulito
                st.markdown("##### HRV giornaliero vs TSB — ultimi 14 giorni")
                _rc14 = rc_v_valid.tail(14)
                _tsb14 = tsb_daily.reindex(_rc14["date"]).ffill()
                fig_hrv = make_subplots(specs=[[{"secondary_y": True}]])
                # Banda min-max HRV
                fig_hrv.add_trace(go.Scatter(
                    x=_rc14["date"], y=_rc14["hrv_max"],
                    mode="lines", line=dict(width=0), showlegend=False,
                ), secondary_y=False)
                fig_hrv.add_trace(go.Scatter(
                    x=_rc14["date"], y=_rc14["hrv_min"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(233,69,96,0.12)",
                    name="Range HRV",
                ), secondary_y=False)
                # Linea HRV medio
                fig_hrv.add_trace(go.Scatter(
                    x=_rc14["date"], y=_rc14["hrv_avg"],
                    name="HRV medio", line=dict(color="#e94560", width=2.5),
                    mode="lines+markers", marker=dict(size=7, color="#e94560",
                        line=dict(color="#fff", width=1.5)),
                ), secondary_y=False)
                # Linea baseline
                fig_hrv.add_hline(y=hrv_baseline_val, line_dash="dash",
                    line_color="rgba(233,69,96,0.45)",
                    annotation_text=f"Baseline {hrv_baseline_val:.0f}ms",
                    annotation_font_color="#e94560", annotation_position="top left")
                # TSB asse secondario
                fig_hrv.add_trace(go.Scatter(
                    x=_rc14["date"], y=_tsb14.values,
                    name="TSB", line=dict(color="#2196F3", width=2, dash="dot"),
                    mode="lines+markers", marker=dict(size=5),
                ), secondary_y=True)
                fig_hrv.add_trace(go.Scatter(
                    x=[_rc14["date"].min(), _rc14["date"].max()], y=[0, 0],
                    mode="lines", line=dict(color="rgba(33,150,243,0.25)", width=1),
                    showlegend=False,
                ), secondary_y=True)
                fig_hrv.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=300, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h", y=1.12, font=dict(size=12)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                )
                fig_hrv.update_yaxes(title_text="HRV (ms)", secondary_y=False,
                    gridcolor="rgba(255,255,255,0.05)", title_font=dict(color="#e94560"))
                fig_hrv.update_yaxes(title_text="TSB (forma)", secondary_y=True,
                    gridcolor="rgba(0,0,0,0)", title_font=dict(color="#2196F3"))
                st.plotly_chart(fig_hrv, use_container_width=True)
                st.caption("HRV alto + TSB positivo = giornata ideale per allenamento intenso · HRV basso + TSB negativo = recupero prioritario")

                # FC riposo + SpO2
                col_fc, col_spo2 = st.columns(2)
                with col_fc:
                    st.markdown("##### FC riposo (min giornaliero)")
                    rc_fc = rc_v.dropna(subset=["hr_min"])
                    fig_fc = go.Figure(go.Scatter(
                        x=rc_fc["date"], y=rc_fc["hr_min"],
                        fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
                        line=dict(color="#2196F3", width=2), mode="lines",
                    ))
                    fig_fc.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=200, margin=dict(l=0,r=0,t=10,b=0),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="bpm"),
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                with col_spo2:
                    st.markdown("##### SpO2 media — ultimi 30gg")
                    rc_sp = rc_v.dropna(subset=["spo2_avg"]).tail(30)
                    fig_sp = go.Figure()
                    # Zone colorate
                    fig_sp.add_hrect(y0=95, y1=101, fillcolor="rgba(76,175,80,0.07)", line_width=0)
                    fig_sp.add_hrect(y0=92, y1=95,  fillcolor="rgba(255,152,0,0.07)",  line_width=0)
                    fig_sp.add_hrect(y0=88, y1=92,  fillcolor="rgba(244,67,54,0.07)",  line_width=0)
                    # Linea principale
                    fig_sp.add_trace(go.Scatter(
                        x=rc_sp["date"], y=rc_sp["spo2_avg"],
                        mode="lines+markers",
                        line=dict(color="#7B1FA2", width=2),
                        marker=dict(size=6,
                            color=["#F44336" if v < 95 else "#4CAF50" for v in rc_sp["spo2_avg"]],
                            line=dict(color="#fff", width=1)),
                        name="SpO2",
                    ))
                    fig_sp.add_hline(y=95, line_dash="dash", line_color="rgba(244,67,54,0.5)",
                                     annotation_text="95%", annotation_position="top right")
                    _below95 = int((rc_sp["spo2_avg"] < 95).sum())
                    fig_sp.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=200, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="%", range=[88,101]),
                    )
                    st.plotly_chart(fig_sp, use_container_width=True)
                    if _below95 > 0:
                        st.caption(f"⚠️ {_below95} giorni con SpO2 < 95% negli ultimi 30gg")

        # ────────────────────────────────────────────
        # TAB SONNO
        # ────────────────────────────────────────────
        with tab_sleep:
            if rc_s is None or rc_s.empty:
                st.info("Carica il file Sleep CSV nella sidebar.")
            else:
                s1, s2, s3, s4, s5 = st.columns(5)
                avg_hrs_s = rc_s["total_hours"].dropna().mean()
                avg_eff_s = rc_s["efficiency_pct"].dropna().mean()
                avg_rem_s = rc_s["rem_pct"].dropna().mean()
                avg_deep_s= rc_s["deep_pct"].dropna().mean()
                avg_awk_s = rc_s["awake_min"].dropna().mean()
                s1.metric("Ore medie",      f"{avg_hrs_s:.1f}h")
                s2.metric("Efficienza",     f"{avg_eff_s:.0f}%")
                s3.metric("REM medio",      f"{avg_rem_s:.0f}%")
                s4.metric("Deep medio",     f"{avg_deep_s:.0f}%")
                s5.metric("Veglia media",   f"{avg_awk_s:.0f} min")

                # ── Fasi del sonno — ultimi 14gg stacked orizzontale ──
                st.markdown("##### Fasi del sonno — ultimi 14 notti")
                _rc_s14 = rc_s.dropna(subset=["total_min"]).tail(14).copy()
                _rc_s14["label"] = _rc_s14["date"].dt.strftime("%d/%m")
                fig_stages = go.Figure()
                fig_stages.add_trace(go.Bar(
                    name="Deep 🔵", y=_rc_s14["label"], x=_rc_s14["deep_min"],
                    orientation="h", marker_color="#1565C0",
                    text=_rc_s14["deep_min"].apply(lambda v: f"{int(v)}m"),
                    textposition="inside", insidetextanchor="middle",
                ))
                fig_stages.add_trace(go.Bar(
                    name="REM 🟣", y=_rc_s14["label"], x=_rc_s14["rem_min"],
                    orientation="h", marker_color="#7B1FA2",
                    text=_rc_s14["rem_min"].apply(lambda v: f"{int(v)}m"),
                    textposition="inside", insidetextanchor="middle",
                ))
                fig_stages.add_trace(go.Bar(
                    name="Light 🩵", y=_rc_s14["label"], x=_rc_s14["light_min"],
                    orientation="h", marker_color="#42A5F5", opacity=0.8,
                ))
                fig_stages.add_trace(go.Bar(
                    name="Veglia", y=_rc_s14["label"], x=_rc_s14["awake_min"],
                    orientation="h", marker_color="rgba(180,180,180,0.35)",
                ))
                fig_stages.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="stack", height=max(280, len(_rc_s14)*22+60),
                    margin=dict(l=10,r=10,t=10,b=0),
                    legend=dict(orientation="h", y=1.05, font=dict(size=12)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="minuti"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.0)", autorange="reversed"),
                    font=dict(size=12),
                )
                st.plotly_chart(fig_stages, use_container_width=True)

                # ── Efficienza + distribuzione ora sonno ──
                col_eff, col_time = st.columns(2)
                with col_eff:
                    st.markdown("##### Efficienza sonno — ultimi 14gg")
                    _eff14 = rc_s.dropna(subset=["efficiency_pct"]).tail(14)
                    eff_colors = ["#4CAF50" if v >= 85 else "#FF9800" if v >= 75 else "#F44336"
                                  for v in _eff14["efficiency_pct"]]
                    fig_eff = go.Figure(go.Bar(
                        x=_eff14["date"], y=_eff14["efficiency_pct"],
                        marker_color=eff_colors, opacity=0.85,
                        text=[f"{v:.0f}%" for v in _eff14["efficiency_pct"]],
                        textposition="outside",
                    ))
                    fig_eff.add_hline(y=85, line_dash="dash",
                                      line_color="rgba(76,175,80,0.5)",
                                      annotation_text="85% ottimale")
                    fig_eff.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=220, margin=dict(l=0,r=0,t=20,b=0),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="%", range=[0,105]),
                    )
                    st.plotly_chart(fig_eff, use_container_width=True)

                with col_time:
                    st.markdown("##### Quando vai a dormire?")
                    _sh = rc_s["start_time"].dropna()
                    _hr = _sh.dt.hour + _sh.dt.minute / 60
                    # Fasce orarie
                    _bands = [("< 22h", "#4CAF50", 0, 22),
                               ("22–23h", "#8BC34A", 22, 23),
                               ("23–0h",  "#FF9800", 23, 24),
                               ("0–1h",   "#FF5722", 24, 25),
                               ("> 1h",   "#F44336", 25, 30)]
                    _hr_adj = _hr.apply(lambda h: h + 24 if h < 4 else h)
                    _counts = {b[0]: int(((_hr_adj >= b[2]) & (_hr_adj < b[3])).sum())
                                for b in _bands}
                    _tot = max(sum(_counts.values()), 1)
                    _pct = {k: v/_tot*100 for k,v in _counts.items()}
                    fig_sh = go.Figure(go.Bar(
                        x=list(_pct.keys()),
                        y=list(_pct.values()),
                        marker_color=[b[1] for b in _bands],
                        text=[f"{v:.0f}%" for v in _pct.values()],
                        textposition="outside",
                    ))
                    fig_sh.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=220, margin=dict(l=0,r=0,t=20,b=0),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Fascia oraria"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="%", range=[0,100]),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_sh, use_container_width=True)
                    _best = max(_pct, key=_pct.get)
                    st.caption(f"Fascia più frequente: **{_best}** ({_pct[_best]:.0f}% delle notti) · Target ideale: 22–23h")

        # ────────────────────────────────────────────
        # TAB PASSI & ATTIVITÀ GIORNALIERA
        # ────────────────────────────────────────────
        with tab_steps:
            st.markdown("### 👟 Passi & Attività Giornaliera")
            rc_act = st.session_state.rc_activity
            if rc_act is None:
                st.info("💍 Carica il file **Activity CSV** di RingConn (Profilo → Impostazioni → Data Export → Activity) per vedere i dati passi.")
            else:
                today_d = pd.Timestamp(datetime.now().date())
                rc7     = rc_act[rc_act["date"] >= today_d - timedelta(days=7)]
                rc30    = rc_act[rc_act["date"] >= today_d - timedelta(days=30)]
                last_day = rc_act.dropna(subset=["steps"]).iloc[-1]

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Passi ieri",      f"{int(last_day['steps']):,}",
                          delta="✅ ok" if last_day["steps"] >= 8000 else "📉 sotto target")
                k2.metric("Media 7gg",       f"{int(rc7['steps'].mean()):,}")
                k3.metric("Media 30gg",      f"{int(rc30['steps'].mean()):,}")
                k4.metric("Giorno top",      f"{int(rc_act['steps'].max()):,}",
                          delta=rc_act.loc[rc_act["steps"].idxmax(), "date"].strftime("%d/%m"))
                k5.metric("TDEE medio 7gg",  f"{int(rc7['kcal_day'].mean()):,} kcal")

                step_target = 8000
                pct_ok = int((rc30["steps"] >= step_target).sum() / max(len(rc30), 1) * 100)
                lbl = ("✅ Ottima mobilità di base." if pct_ok >= 70 else
                       "🟡 Cerca di muoverti di più nei giorni di recupero." if pct_ok >= 40 else
                       "🔴 Attività di base insufficiente. Aggiungi passeggiate nei giorni di riposo.")
                st.markdown(f"""
                <div style="background:rgba(76,175,80,0.08);border-left:3px solid #4CAF50;
                            border-radius:0 10px 10px 0;padding:10px 16px;margin:8px 0">
                    <b>{pct_ok}%</b> dei giorni (ultimi 30) hai raggiunto i <b>{step_target:,} passi target</b>. {lbl}
                </div>""", unsafe_allow_html=True)

                st.markdown("##### Passi giornalieri — ultimi 30 giorni")
                rc30_p = rc30.copy()
                rc30_p["color"] = rc30_p["steps"].apply(
                    lambda s: "#4CAF50" if s >= 10000 else "#FF9800" if s >= 6000 else "#F44336"
                )
                fig_steps = go.Figure(go.Bar(
                    x=rc30_p["date"], y=rc30_p["steps"],
                    marker_color=rc30_p["color"].tolist(), opacity=0.85,
                    text=rc30_p["steps"].apply(lambda x: f"{int(x/1000):.0f}k"),
                    textposition="outside",
                ))
                fig_steps.add_hline(y=step_target, line_dash="dash",
                                     line_color="rgba(255,255,255,0.4)",
                                     annotation_text=f"Target {step_target:,}")
                fig_steps.add_hline(y=10000, line_dash="dot",
                                     line_color="rgba(76,175,80,0.5)",
                                     annotation_text="10k")
                fig_steps.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=0,r=0,t=20,b=0),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Passi"),
                    showlegend=False,
                )
                st.plotly_chart(fig_steps, use_container_width=True)

                # Correlazione passi-riposo → HRV giorno dopo
                if rc_vitals is not None and not rc_vitals.empty:
                    st.markdown("##### 🔗 Passi nei giorni di riposo → HRV del giorno dopo")
                    strava_days = set(df["start_date"].dt.strftime("%Y-%m-%d").tolist())
                    rest_rows   = []
                    for _, srow in rc_act.iterrows():
                        d_str  = srow["date"].strftime("%Y-%m-%d")
                        if d_str in strava_days or srow["steps"] <= 0:
                            continue
                        next_d = (srow["date"] + timedelta(days=1)).strftime("%Y-%m-%d")
                        match  = rc_vitals[rc_vitals["date"].dt.strftime("%Y-%m-%d") == next_d]
                        if not match.empty and pd.notna(match.iloc[0]["hrv_avg"]):
                            rest_rows.append({"date": d_str,
                                              "steps": srow["steps"],
                                              "hrv_next": match.iloc[0]["hrv_avg"]})
                    if len(rest_rows) >= 5:
                        cdf    = pd.DataFrame(rest_rows)
                        xarr   = cdf["steps"].values.astype(float)
                        yarr   = cdf["hrv_next"].values.astype(float)
                        xm, ym = xarr.mean(), yarr.mean()
                        sl     = ((xarr-xm)*(yarr-ym)).sum() / max(((xarr-xm)**2).sum(), 1e-9)
                        yfit   = xarr * sl + (ym - sl*xm)
                        r_val  = float(np.corrcoef(xarr, yarr)[0, 1])
                        fig_cs = go.Figure()
                        fig_cs.add_trace(go.Scatter(
                            x=xarr, y=yarr, mode="markers",
                            marker=dict(color="#4CAF50", size=8, opacity=0.75),
                            text=cdf["date"],
                            hovertemplate="%{text}<br>Passi: %{x}<br>HRV+1gg: %{y} ms",
                            name="Giorni di riposo",
                        ))
                        fig_cs.add_trace(go.Scatter(
                            x=xarr, y=yfit, mode="lines",
                            line=dict(color="#FF9800", dash="dash"),
                            name=f"Trend (r={r_val:.2f})",
                        ))
                        fig_cs.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=240, margin=dict(l=0,r=0,t=10,b=0),
                            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Passi giorno di riposo"),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="HRV giorno dopo (ms)"),
                            legend=dict(orientation="h", y=1.1),
                        )
                        st.plotly_chart(fig_cs, use_container_width=True)
                        insight = ("camminare nei giorni di riposo tende a migliorare il recupero HRV." if r_val > 0.3 else
                                   "i passi nei giorni di riposo non mostrano correlazione chiara con l'HRV successivo.")
                        st.caption(f"r={r_val:.2f} — {insight}")
                    else:
                        st.info("Servono almeno 5 giorni di riposo con HRV per questa analisi.")

                st.markdown("##### 🔥 TDEE stimato (kcal totali giornata)")
                fig_tdee = go.Figure(go.Scatter(
                    x=rc30_p["date"], y=rc30_p["kcal_day"],
                    fill="tozeroy", line=dict(color="#e94560", width=2),
                    fillcolor="rgba(233,69,96,0.1)",
                ))
                fig_tdee.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=200, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="kcal"),
                    showlegend=False,
                )
                st.plotly_chart(fig_tdee, use_container_width=True)

        # ────────────────────────────────────────────
        # TAB CORRELAZIONE SONNO → PERFORMANCE
        # ────────────────────────────────────────────
        with tab_corr:
            if rc_s is None or rc_s.empty:
                st.info("Carica il file Sleep CSV (o il ZIP) per vedere le correlazioni.")
            else:
                st.markdown("##### 🔗 Correlazione Sonno → Performance")
                st.caption("Analisi basata su allenamenti il giorno successivo alla sessione di sonno.")

                df_corr = df[df["distance"] > 1000].copy()
                df_corr["date_key"] = df_corr["start_date"].dt.normalize()
                rc_s_key = rc_s.copy()
                rc_s_key["next_day"] = rc_s_key["date"] + pd.Timedelta(days=1)
                merged = df_corr.merge(
                    rc_s_key[["next_day","total_hours","efficiency_pct","deep_pct","rem_pct"]],
                    left_on="date_key", right_on="next_day", how="inner"
                )

                if merged.empty:
                    st.info("Dati sovrapposti insufficienti tra Strava e Sleep.")
                else:
                    # Aggiunge HRV dello stesso giorno
                    if rc_v is not None and not rc_v.empty:
                        merged = merged.merge(rc_v[["date","hrv_avg"]],
                            left_on="date_key", right_on="date", how="left")

                    def _r_comment(r, flip_sign=False):
                        """Converte r in interpretazione e colore."""
                        _r = -r if flip_sign else r
                        if   _r >  0.5: return "💚 Forte positiva", "#4CAF50"
                        elif _r >  0.2: return "🟢 Debole positiva", "#8BC34A"
                        elif _r < -0.5: return "🔴 Forte negativa", "#F44336"
                        elif _r < -0.2: return "🟠 Debole negativa", "#FF9800"
                        else:           return "⬜ Nessuna correlazione", "#888"

                    # Coppie (sonno/hrv → performance) da analizzare
                    pairs = []
                    perf_metrics = [("tss","TSS allenamento",False),
                                    ("average_heartrate","FC media",False)]
                    # Pace per corse
                    merged_run = merged[merged["type"].isin(["Run","TrailRun"])].copy()
                    merged_run = merged_run[merged_run["distance"] > 3000]
                    if not merged_run.empty:
                        merged_run["pace"] = merged_run["moving_time"] / (merged_run["distance"]/1000)
                        perf_metrics.append(("pace","Passo corsa (sec/km)",True))  # flip: basso=meglio

                    sleep_metrics = [("total_hours","Ore di sonno"),
                                     ("efficiency_pct","Efficienza sonno %"),
                                     ("deep_pct","Deep sleep %")]
                    if "hrv_avg" in merged.columns:
                        sleep_metrics.append(("hrv_avg","HRV giornaliero (ms)"))

                    for sm, sl in sleep_metrics:
                        for pm, pl, flip in perf_metrics:
                            _df = (merged_run if pm == "pace" else merged)[[sm, pm]].dropna()
                            if len(_df) < 4:
                                continue
                            _r = float(np.corrcoef(_df[sm], _df[pm])[0,1])
                            _label, _col = _r_comment(_r, flip_sign=flip)
                            # Suggerimento pratico
                            _tip = ""
                            if sm == "total_hours" and pm == "tss" and _r > 0.25:
                                _tip = "Dormi di più → tendenza ad allenarti di più/meglio"
                            elif sm == "efficiency_pct" and "heartrate" in pm and _r < -0.2:
                                _tip = "Sonno efficiente → FC più bassa a parità di sforzo"
                            elif sm == "hrv_avg" and pm == "pace" and _r < -0.2:
                                _tip = "HRV alto → passo migliore nelle corse"
                            elif sm == "deep_pct" and pm == "tss" and _r > 0.2:
                                _tip = "Più sonno profondo → sessioni più cariche"
                            pairs.append({
                                "Indicatore sonno": sl,
                                "Performance": pl,
                                "r": round(_r, 2),
                                "Correlazione": _label,
                                "N": len(_df),
                                "_color": _col,
                                "_tip": _tip,
                            })

                    if not pairs:
                        st.info("Dati insufficienti per calcolare le correlazioni.")
                    else:
                        # Visualizzazione come card per ogni coppia con r significativo
                        sig_pairs = [p for p in pairs if abs(p["r"]) > 0.2]
                        all_pairs = pairs

                        st.markdown("**Correlazioni significative (|r| > 0.2):**")
                        if sig_pairs:
                            for p in sig_pairs:
                                _bar_w = int(abs(p["r"]) * 100)
                                _bar_c = p["_color"]
                                _tip_html = f'<div style="font-size:12px;color:#888;margin-top:3px">💡 {p["_tip"]}</div>' if p["_tip"] else ""
                                st.markdown(f"""
                                <div style="background:rgba(255,255,255,0.03);border-left:3px solid {_bar_c};
                                     border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px">
                                  <div style="display:flex;justify-content:space-between;align-items:center">
                                    <span style="color:#ddd;font-size:13px;font-weight:600">
                                      {p["Indicatore sonno"]} → {p["Performance"]}
                                    </span>
                                    <span style="color:{_bar_c};font-size:18px;font-weight:800">r = {p["r"]:+.2f}</span>
                                  </div>
                                  <div style="display:flex;align-items:center;gap:8px;margin-top:5px">
                                    <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:6px;flex:1">
                                      <div style="background:{_bar_c};height:6px;border-radius:4px;width:{_bar_w}%"></div>
                                    </div>
                                    <span style="font-size:12px;color:{_bar_c}">{p["Correlazione"]}</span>
                                  </div>
                                  {_tip_html}
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.info("Nessuna correlazione significativa rilevata — servono più dati.")

                        with st.expander("📋 Tabella completa correlazioni"):
                            df_tab = pd.DataFrame([{k:v for k,v in p.items() if not k.startswith("_")}
                                                    for p in all_pairs])
                            st.dataframe(df_tab, use_container_width=True, hide_index=True)
                        st.caption("r = coefficiente di Pearson. Valori: |r|>0.5 = forte, 0.2–0.5 = debole, <0.2 = trascurabile. N = campioni. Correlazione ≠ causalità.")



    elif menu == "💪 Stato Fisico":
        st.markdown("## 💪 Analisi Stato Fisico Attuale")

        # ── Stato forma banner ──
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{status_color}22,{status_color}08);
                    border:1px solid {status_color}55; border-radius:16px;
                    padding:18px 24px; margin-bottom:20px; display:flex; align-items:center; gap:16px">
            <div style="font-size:36px">{status_label.split()[0]}</div>
            <div>
                <div style="font-size:22px; font-weight:800; color:{status_color}">{' '.join(status_label.split()[1:])}</div>
                <div style="color:#aaa; font-size:14px; margin-top:2px">
                    CTL {current_ctl:.1f} &nbsp;·&nbsp; ATL {current_atl:.1f} &nbsp;·&nbsp; TSB {current_tsb:.1f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── BLOCCO 1: Metriche PMC classiche ──
        # ── Commento AI automatico ──
        with st.spinner("🤖 Il coach sta valutando il tuo stato..."):
            try:
                ctl_30ago  = ctl_daily.iloc[-30] if len(ctl_daily) >= 30 else ctl_daily.iloc[0]
                trend_ctl  = "in crescita" if current_ctl > ctl_30ago else "in calo"
                df_sport_r = df.tail(14)["type"].value_counts().to_dict()
                # Prepara dati RingConn per contesto AI
                _rc_ctx_extra = ""
                if rc_vitals is not None and not rc_vitals.empty:
                    _rv7 = rc_vitals.tail(7)
                    _hrv_now  = float(rc_vitals.iloc[-1].get("hrv_avg") or 0)
                    _hrv_base = float(rc_vitals.tail(30)["hrv_avg"].dropna().mean() or 0)
                    _spo2_avg = float(_rv7["spo2_avg"].dropna().mean() or 0)
                    _fc_rest  = float(_rv7["hr_min"].dropna().mean() or 0)
                    _rc_ctx_extra += (
                        f" HRV ora={_hrv_now:.0f}ms (baseline 30gg={_hrv_base:.0f}ms, ratio={_hrv_now/_hrv_base:.2f})."
                        f" SpO2 media 7gg={_spo2_avg:.0f}%."
                        f" FC riposo media 7gg={_fc_rest:.0f}bpm."
                    ) if _hrv_base > 0 else ""
                if rc_sleep is not None and not rc_sleep.empty:
                    _rs7 = rc_sleep.tail(7)
                    _s_hrs  = float(_rs7["total_hours"].dropna().mean() or 0)
                    _s_eff  = float(_rs7["efficiency_pct"].dropna().mean() or 0)
                    _s_deep = float(_rs7["deep_pct"].dropna().mean() or 0)
                    _s_rem  = float(_rs7["rem_pct"].dropna().mean() or 0)
                    _rc_ctx_extra += (
                        f" Sonno 7gg: media={_s_hrs:.1f}h, efficienza={_s_eff:.0f}%,"
                        f" deep={_s_deep:.0f}%, REM={_s_rem:.0f}%."
                    )
                ctx_quick  = (
                    f"CTL={current_ctl:.1f} ({trend_ctl} rispetto a 30gg fa: {ctl_30ago:.1f}), "
                    f"ATL={current_atl:.1f}, TSB={current_tsb:.1f}, ACWR={acwr_val:.2f}, "
                    f"Ramp Rate={ramp_rate:+.1f}, Monotonia={monotonia:.2f}, Strain={strain_val:.0f}. "
                    f"VO2max stimato: {vo2max_val if vo2max_val else 'N/D'} ml/kg/min. "
                    f"Sport ultimi 14gg: {df_sport_r}. "
                    f"Readiness RingConn: {readiness['score']}/100 ({readiness['label']})."
                    + rc_ai_context + _rc_ctx_extra
                )
                prompt_quick = (
                    "Sei un coach sportivo d'élite. In 3-4 frasi dirette e concrete, "
                    "dammi un giudizio immediato sullo stato fisico attuale di questo atleta: "
                    "cosa sta andando bene, cosa richiede attenzione, e il consiglio più importante per questa settimana. "
                    "Tono professionale ma diretto. No elenchi puntati, solo testo fluido."
                )
                quick_summary = ai_generate(f"{ctx_quick}\n\n{prompt_quick}")
                st.markdown(f"""
                <div style="background:#f8f9fa;border:1px solid #dee2e6;border-left:4px solid #2196F3;
                             border-radius:8px; padding:16px 20px; margin-bottom:16px;">
                    <div style="font-size:12px;color:#2196F3;font-weight:700;margin-bottom:8px;letter-spacing:0.5px">🤖 VALUTAZIONE COACH</div>
                    <div style="color:#212529;font-size:15px;line-height:1.75">{quick_summary}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass  # Se fallisce non blocchiamo la pagina

        st.markdown("### 📊 Metriche PMC — Performance Management Chart")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        with r1c1:
            st.metric("CTL — Fitness", f"{current_ctl:.1f}")
            metric_tooltip("CTL")
        with r1c2:
            st.metric("ATL — Fatica", f"{current_atl:.1f}")
            metric_tooltip("ATL")
        with r1c3:
            st.metric("TSB — Forma", f"{current_tsb:.1f}")
            metric_tooltip("TSB")
        with r1c4:
            trimp_7 = df["trimp"].tail(7).sum()
            st.metric("TRIMP (7gg)", f"{trimp_7:.0f}")
            metric_tooltip("TRIMP")

        st.divider()



        # ── BLOCCO 3: Intensità, Volume & EF — spostato prima del PMC ──
        st.markdown("### ❤️ Intensità, Volume & Efficienza Aerobica")
        col_z, col_vol, col_ef = st.columns(3)

        with col_z:
            st.markdown("**Zone FC — ultimi 30gg**")
            metric_tooltip("POL")
            df30 = df[df["start_date"] >= (df["start_date"].max() - timedelta(days=30))]
            df30 = df30[df30["zone_num"] > 0]
            z12, z45, pol = 0, 0, 0
            if not df30.empty:
                zone_counts = df30.groupby(["zone_num", "zone_label", "zone_color"]).apply(
                    lambda x: x["moving_time"].sum() / 3600
                ).reset_index(name="ore")
                zone_counts = zone_counts.sort_values("zone_num")
                fig_z = go.Figure(go.Bar(
                    x=zone_counts["ore"], y=zone_counts["zone_label"],
                    orientation="h", marker_color=zone_counts["zone_color"],
                    text=[f"{v:.1f}h" for v in zone_counts["ore"]], textposition="outside",
                ))
                fig_z.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     height=200, margin=dict(l=0, r=60, t=0, b=0),
                                     xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                     showlegend=False)
                st.plotly_chart(fig_z, use_container_width=True)
                total_z = zone_counts["ore"].sum()
                z12 = zone_counts[zone_counts["zone_num"] <= 2]["ore"].sum()
                z45 = zone_counts[zone_counts["zone_num"] >= 4]["ore"].sum()
                pol = z12 / total_z * 100 if total_z > 0 else 0
                pol_color = "#4CAF50" if pol >= 75 else "#FF9800" if pol >= 60 else "#F44336"
                st.markdown(f"<span style='color:{pol_color};font-weight:700'>Bassa intensità: {pol:.0f}%</span> (target >75%)", unsafe_allow_html=True)
            else:
                st.info("Nessun dato FC disponibile")

        with col_vol:
            st.markdown("**Volume settimanale (km)**")
            df_weekly = df.copy()
            df_weekly["week"] = df_weekly["start_date"].dt.to_period("W").dt.start_time
            weekly_km = df_weekly.groupby("week")["distance"].sum() / 1000
            weekly_km = weekly_km.tail(12)
            avg_vol = weekly_km.mean()
            w_colors = ["#e94560" if v > avg_vol * 1.3 else "#2196F3" for v in weekly_km.values]
            fig_w = go.Figure(go.Bar(x=weekly_km.index, y=weekly_km.values,
                                      marker_color=w_colors, opacity=0.85))
            fig_w.add_hline(y=avg_vol, line_dash="dot", line_color="rgba(255,255,255,0.27)", line_width=1,
                             annotation_text=f"media {avg_vol:.0f}km", annotation_font_color="#aaa")
            fig_w.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  height=200, margin=dict(l=0, r=0, t=0, b=0),
                                  xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                                  yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_w, use_container_width=True)

        with col_ef:
            st.markdown("**Efficiency Factor — trend**")
            metric_tooltip("EF")
            ef_data = df[df["ef"].notna()][["start_date", "ef", "type"]].tail(30)
            if not ef_data.empty:
                fig_ef = go.Figure()
                for sport_type in ef_data["type"].unique():
                    sub = ef_data[ef_data["type"] == sport_type]
                    fig_ef.add_trace(go.Scatter(
                        x=sub["start_date"], y=sub["ef"],
                        mode="markers+lines",
                        name=f"{get_sport_info(sport_type)['icon']} {get_sport_info(sport_type)['label']}",
                        line=dict(color=get_sport_info(sport_type)["color"], width=1.5),
                        marker=dict(size=6),
                    ))
                fig_ef.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      height=200, margin=dict(l=0, r=0, t=0, b=0),
                                      legend=dict(font=dict(size=10), orientation="h", y=1.15),
                                      xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                      yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
                st.plotly_chart(fig_ef, use_container_width=True)
                ef_trend = ef_data["ef"].iloc[-1] - ef_data["ef"].iloc[0] if len(ef_data) > 1 else 0
                ef_emoji = "📈" if ef_trend > 0 else "📉"
                st.markdown(f"{ef_emoji} Trend EF: **{ef_trend:+.4f}** negli ultimi {len(ef_data)} allenamenti")
            else:
                st.info("Dati FC non sufficienti per EF")

        st.divider()

        # ── PMC grafico linee (senza barre) ──
        st.markdown("### 📈 Andamento Fitness — PMC 90 giorni")
        chart_df = pd.DataFrame({
            "CTL": ctl_daily, "ATL": atl_daily, "TSB": tsb_daily
        }).dropna().tail(90)
        fig_pmc = go.Figure()
        fig_pmc.add_trace(go.Scatter(x=chart_df.index, y=chart_df["CTL"],
            name="CTL — Fitness", line=dict(color="#2196F3", width=2.5),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.07)"))
        fig_pmc.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ATL"],
            name="ATL — Fatica", line=dict(color="#FF9800", width=2)))
        fig_pmc.add_trace(go.Scatter(x=chart_df.index, y=chart_df["TSB"],
            name="TSB — Forma", line=dict(color="#4CAF50", width=2, dash="dot")))
        fig_pmc.add_hrect(y0=-10, y1=10, fillcolor="rgba(76,175,80,0.05)", line_width=0)
        fig_pmc.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)
        fig_pmc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_pmc, use_container_width=True)
        st.caption("CTL=Fitness cronico (42gg) · ATL=Fatica acuta (7gg) · TSB=Forma (CTL−ATL) · zona verde = forma ottimale")

        st.divider()

        # ── BLOCCO 5: VO2max + Race Predictor + VI ──
        st.markdown("### 🔬 Capacità Aerobica & Performance")
        col_vo2, col_race, col_vi = st.columns(3)

        def _vo2_card(val, label_method, color_override=None):
            if val >= 65:   vc, vl = "#9C27B0", "🏆 Élite"
            elif val >= 55: vc, vl = "#4CAF50", "🥇 Molto Buono"
            elif val >= 45: vc, vl = "#2196F3", "🥈 Buono"
            elif val >= 35: vc, vl = "#FF9800", "🥉 Media"
            else:            vc, vl = "#F44336", "📈 Da Migliorare"
            if color_override: vc = color_override
            return (f"<div style='background:{vc}15;border:1px solid {vc}44;border-radius:10px;"
                    f"padding:10px 14px;margin-bottom:8px;display:flex;align-items:center;gap:12px'>"
                    f"<div style='font-size:32px;font-weight:900;color:{vc};min-width:54px;text-align:center'>{val:.0f}</div>"
                    f"<div>"
                    f"<div style='font-size:11px;color:#888;font-weight:600'>{label_method}</div>"
                    f"<div style='font-size:13px;color:{vc};font-weight:700'>{vl}</div>"
                    f"<div style='font-size:10px;color:#666'>ml/kg/min</div>"
                    f"</div></div>")

        with col_vo2:
            st.markdown("**VO2max — Corsa & Bici**")
            metric_tooltip("VO2MAX")
            _has_vo2 = False
            # ── Stima da corsa (Daniels VDOT) ──
            if vo2max_val:
                st.markdown(_vo2_card(vo2max_val, "🏃 Corsa — formula Daniels"), unsafe_allow_html=True)
                _has_vo2 = True

            # ── Stima da bici (FTP → Coggan/Hawley) ──
            _ftp = u.get("ftp", 0)
            _peso = u.get("weight", 0)
            if _ftp > 0 and _peso > 0:
                _ftp_per_kg = _ftp / _peso
                _vo2_bike   = round(_ftp_per_kg * 10.8 + 7, 1)
                st.markdown(_vo2_card(_vo2_bike, "🚴 Bici — FTP/kg × 10.8 + 7", "#2196F3"), unsafe_allow_html=True)
                _has_vo2 = True
                st.caption(f"FTP={_ftp}W · Peso={_peso}kg · FTP/kg={_ftp_per_kg:.2f}")
            elif _ftp > 0:
                st.caption("⚠️ Imposta il **peso** in Profilo Fisico per la stima da bici")

            if not _has_vo2:
                st.info("Serve almeno 1 corsa ≥5km per la stima da corsa. Imposta FTP e peso per la stima da bici.")

            st.caption("ℹ️ Garmin usa algoritmo Firstbeat (più accurato). Queste sono stime semplici da usare come trend.")

        with col_race:
            st.markdown("**🏁 Race Time Predictor**")
            if race_preds:
                for dist_label, pred in race_preds.items():
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                 background:rgba(255,255,255,0.03); border-radius:8px;
                                 padding:8px 12px; margin-bottom:6px;">
                        <span style="color:#aaa; font-size:14px">🏃 {dist_label}</span>
                        <span style="color:#e94560; font-weight:700; font-size:15px">{pred['time']}</span>
                        <span style="color:#666; font-size:12px">{pred['pace']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption("⚠️ Stime basate su VO2max stimato (formula Daniels VDOT). Usa come riferimento indicativo.")
            else:
                st.info("Calcola il VO2max per ottenere le stime.")

        with col_vi:
            st.markdown("**Variability Index (bici)**")
            metric_tooltip("VI")
            vi_data = df[df["vi"].notna()][["start_date", "vi", "name"]].tail(20)
            if not vi_data.empty:
                vi_colors = ["#4CAF50" if v <= 1.05 else "#FF9800" if v <= 1.10 else "#F44336"
                              for v in vi_data["vi"]]
                fig_vi = go.Figure(go.Bar(
                    x=vi_data["start_date"], y=vi_data["vi"],
                    marker_color=vi_colors, opacity=0.85,
                    text=[f"{v:.3f}" for v in vi_data["vi"]], textposition="outside",
                ))
                fig_vi.add_hline(y=1.05, line_dash="dot", line_color="rgba(76,175,80,0.33)")
                fig_vi.add_hline(y=1.10, line_dash="dot", line_color="rgba(255,152,0,0.33)")
                fig_vi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      height=200, margin=dict(l=0, r=0, t=20, b=0),
                                      xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                                      yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0.95, max(1.25, vi_data["vi"].max()+0.05)]))
                st.plotly_chart(fig_vi, use_container_width=True)
                avg_vi = vi_data["vi"].mean()
                vi_label = "Costante ✅" if avg_vi <= 1.05 else "Variabile ⚠️" if avg_vi <= 1.10 else "Molto variabile 🔴"
                st.markdown(f"Media VI: **{avg_vi:.3f}** — {vi_label}")
            else:
                st.info("Nessuna attività in bici con dati di potenza normalizzata.")

        st.divider()

        # ── BLOCCO 6: AI Analisi completa + Piano 7gg ──
        st.markdown("### 🤖 Coaching AI Avanzato")
        col_ai1, col_ai2 = st.columns(2)

        with col_ai1:
            st.markdown("#### Analisi Fisiologica Completa")
            if st.button("🔍 Genera Analisi", key="btn_analisi", use_container_width=True):
                with st.spinner("Analisi in corso..."):
                    try:
                        ctl_30ago = ctl_daily.iloc[-30] if len(ctl_daily) >= 30 else ctl_daily.iloc[0]
                        trend_ctl = "crescente" if current_ctl > ctl_30ago else "decrescente"
                        df_sport  = df.tail(20)["type"].value_counts().to_dict()
                        ctx_fitness = (
                            f"DATI ATLETA: CTL={current_ctl:.1f} (trend {trend_ctl}), "
                            f"ATL={current_atl:.1f}, TSB={current_tsb:.1f}. "
                            f"ACWR={acwr_val:.2f}, Ramp Rate={ramp_rate:+.1f}/settimana. "
                            f"Monotonia={monotonia:.2f}, Training Strain={strain_val:.0f}. "
                            f"VO2max stimato={vo2max_val if vo2max_val else 'N/D'} ml/kg/min. "
                            f"% allenamento bassa intensità (Z1-Z2)={pol:.0f}%. "
                            f"TRIMP ultimi 7gg={trimp_7:.0f}. "
                            f"Sport praticati={df_sport}."
                        )
                        prompt_fitness = (
                            "Sei un fisiolo dello sport e coach d'élite. "
                            "Fornisci un'analisi DETTAGLIATA e PROFESSIONALE dello stato fisico di questo atleta. "
                            "Struttura la risposta così: "
                            "1) Stato di forma attuale (interpreta CTL/ATL/TSB/ACWR), "
                            "2) Rischio overtraining/undertraining (monotonia, strain, ramp rate), "
                            "3) Qualità dell'allenamento (distribuzione intensità, EF), "
                            "4) Capacità aerobica (VO2max, implicazioni), "
                            "5) Raccomandazioni concrete per le prossime 2 settimane. "
                            "Usa terminologia tecnica. Sii diretto e specifico."
                        )
                        result_fit = ai_generate(f"{ctx_fitness}\n\n{prompt_fitness}")
                        st.session_state["analisi_fisica"] = result_fit
                    except Exception as e:
                        st.error(f"Errore AI: {e}")
            if "analisi_fisica" in st.session_state:
                st.markdown(f'<div style="background:#f8f9fa;border-left:4px solid #2196F3;border-radius:8px;padding:16px 20px;color:#212529;font-size:15px;line-height:1.8">{st.session_state["analisi_fisica"]}</div>', unsafe_allow_html=True)

        with col_ai2:
            st.markdown("#### 🗓️ Piano Allenamento — Prossimi 7 giorni")
            goal = st.selectbox("Obiettivo:", [
                "Mantenimento forma", "Aumentare il fitness (CTL)",
                "Recupero / scarico", "Preparazione gara (entro 2 settimane)", "Base aerobica"
            ], key="goal_select")
            if st.button("🔄 Genera Piano", use_container_width=True, key="btn_piano"):
                with st.spinner("Il coach sta pianificando..."):
                    try:
                        ctx_plan = (
                            f"CTL={current_ctl:.1f}, ATL={current_atl:.1f}, TSB={current_tsb:.1f}. "
                            f"ACWR={acwr_val:.2f}, Ramp Rate={ramp_rate:+.1f}. "
                            f"Monotonia={monotonia:.2f}, Strain={strain_val:.0f}. "
                            f"FC max={u['fc_max']}, FTP={u['ftp']}W. "
                            f"Sport principale: {df['type'].value_counts().index[0]}. "
                            f"% bassa intensità attuale: {pol:.0f}%. "
                            f"Obiettivo: {goal}."
                        )
                        prompt_plan = (
                            "Crea un piano di allenamento dettagliato per i prossimi 7 giorni. "
                            "Per ogni giorno: tipo sessione, durata precisa, intensità (zona FC o %FTP), "
                            "obiettivo fisiologico, note pratiche. "
                            "Calibra il carico considerando ACWR e TSB attuali. "
                            "Se ACWR > 1.3 riduci il volume. Se TSB < -20 inserisci più recupero. "
                            "Formato: Giorno N — [tipo]: descrizione dettagliata."
                        )
                        result_plan = ai_generate(f"{ctx_plan}\n\n{prompt_plan}")
                        st.session_state["piano_7gg"] = result_plan
                    except Exception as e:
                        st.error(f"Errore AI: {e}")
            if "piano_7gg" in st.session_state:
                st.markdown(f'<div style="background:#f0fff4;border-left:4px solid #4CAF50;border-radius:8px;padding:16px 20px;color:#1b5e20;font-size:15px;line-height:1.8">{st.session_state["piano_7gg"]}</div>', unsafe_allow_html=True)

    # ============================================================
    # RECAP
    # ============================================================

    # ============================================================
    # METRICHE AVANZATE
    # ============================================================

        st.divider()
        # ── ACWR & Rischio Infortuni (riepilogo) ──
        st.markdown("### ⚡ Carico & Rischio Infortuni")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)

        # Colori ACWR
        if acwr_val < 0.8:    acwr_color, acwr_emoji = "#2196F3", "🔵"
        elif acwr_val <= 1.3: acwr_color, acwr_emoji = "#4CAF50", "🟢"
        elif acwr_val <= 1.5: acwr_color, acwr_emoji = "#FF9800", "🟡"
        else:                  acwr_color, acwr_emoji = "#F44336", "🔴"

        # Colori Ramp Rate
        if abs(ramp_rate) <= 3:   rr_color = "#2196F3"
        elif abs(ramp_rate) <= 7: rr_color = "#4CAF50"
        elif abs(ramp_rate) <= 10: rr_color = "#FF9800"
        else:                      rr_color = "#F44336"

        # Colori Monotonia
        if monotonia < 1.5:   mono_color = "#4CAF50"
        elif monotonia < 2.0: mono_color = "#FF9800"
        else:                  mono_color = "#F44336"

        # Colori Strain
        if strain_val < 1000:   strain_color = "#4CAF50"
        elif strain_val < 2000: strain_color = "#FF9800"
        else:                    strain_color = "#F44336"

        with r2c1:
            st.markdown(f"<div style='font-size:13px;color:#888'>ACWR</div>"
                        f"<div style='font-size:28px;font-weight:800;color:{acwr_color}'>{acwr_emoji} {acwr_val:.2f}</div>",
                        unsafe_allow_html=True)
            metric_tooltip("ACWR")
        with r2c2:
            arrow = "↑" if ramp_rate > 0 else "↓"
            st.markdown(f"<div style='font-size:13px;color:#888'>Ramp Rate (7gg)</div>"
                        f"<div style='font-size:28px;font-weight:800;color:{rr_color}'>{arrow} {ramp_rate:+.1f} CTL</div>",
                        unsafe_allow_html=True)
            metric_tooltip("RAMP_RATE")
        with r2c3:
            st.markdown(f"<div style='font-size:13px;color:#888'>Monotonia</div>"
                        f"<div style='font-size:28px;font-weight:800;color:{mono_color}'>{monotonia:.2f}</div>",
                        unsafe_allow_html=True)
            metric_tooltip("MONOTONIA")
        with r2c4:
            st.markdown(f"<div style='font-size:13px;color:#888'>Training Strain</div>"
                        f"<div style='font-size:28px;font-weight:800;color:{strain_color}'>{strain_val:.0f}</div>",
                        unsafe_allow_html=True)
            metric_tooltip("STRAIN")

        # Gauge ACWR
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=acwr_val,
            title={"text": "ACWR — Rischio Infortuni", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 2.0], "tickwidth": 1},
                "bar":  {"color": acwr_color, "thickness": 0.25},
                "steps": [
                    {"range": [0,   0.8], "color": "rgba(33,150,243,0.15)"},
                    {"range": [0.8, 1.3], "color": "rgba(76,175,80,0.15)"},
                    {"range": [1.3, 1.5], "color": "rgba(255,152,0,0.15)"},
                    {"range": [1.5, 2.0], "color": "rgba(244,67,54,0.15)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": acwr_val},
            },
            number={"font": {"size": 32}, "suffix": ""},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220,
                                 margin=dict(l=20, r=20, t=40, b=10),
                                 font={"color": "#ccc"})

        g_col, acwr_hist_col = st.columns([1, 2])
        with g_col:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with acwr_hist_col:
            acwr_plot = acwr_series.tail(90).dropna()
            fig_acwr = go.Figure()
            fig_acwr.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(76,175,80,0.08)", line_width=0)
            fig_acwr.add_hrect(y0=1.3, y1=1.5, fillcolor="rgba(255,152,0,0.08)", line_width=0)
            fig_acwr.add_hrect(y0=1.5, y1=3.0, fillcolor="rgba(244,67,54,0.08)", line_width=0)
            acwr_fill = {"#4CAF50": "rgba(76,175,80,0.12)", "#FF9800": "rgba(255,152,0,0.12)",
                         "#F44336": "rgba(244,67,54,0.12)", "#2196F3": "rgba(33,150,243,0.12)"}.get(acwr_color, "rgba(150,150,150,0.12)")
            fig_acwr.add_trace(go.Scatter(x=acwr_plot.index, y=acwr_plot.values,
                                           line=dict(color=acwr_color, width=2),
                                           fill="tozeroy", fillcolor=acwr_fill,
                                           name="ACWR"))
            fig_acwr.add_hline(y=0.8, line_dash="dot", line_color="rgba(76,175,80,0.5)",  line_width=1)
            fig_acwr.add_hline(y=1.3, line_dash="dot", line_color="rgba(255,152,0,0.5)",  line_width=1)
            fig_acwr.add_hline(y=1.5, line_dash="dot", line_color="rgba(244,67,54,0.5)",  line_width=1)
            fig_acwr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=220, margin=dict(l=0, r=0, t=10, b=0),
                                    showlegend=False,
                                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, max(2.0, acwr_plot.max()+0.2)]))
            st.markdown("<div style='font-size:13px;color:#888;margin-top:8px'>Storico ACWR — 90 giorni</div>", unsafe_allow_html=True)
            st.plotly_chart(fig_acwr, use_container_width=True)

        st.divider()
    elif menu == "🧬 Metriche Avanzate":
        st.markdown("## 🧬 Metriche Avanzate")

        tab_budget, tab_acwr2, tab_hrv_slope = st.tabs([
            "💰 TSS Budget", "⚖️ ACWR 2.0", "📉 HRV Trend",
        ])

        # ─────────────────────────────────────────────
        # TAB 1 — ADAPTIVE TSS BUDGET
        # ─────────────────────────────────────────────
        with tab_budget:
            st.markdown("### 💰 Budget TSS Adattivo Giornaliero")
            st.caption("Il tuo 'credito energetico' per oggi, calcolato da Readiness Score + storico 28gg.")

            b = tss_budget
            # Card principale budget
            prog_pct = min(1.0, b["tss_spent"] / b["budget"]) if b["budget"] > 0 else 0
            over_pct = min(1.0, b["tss_spent"] / b["budget_max"]) if b["budget_max"] > 0 else 0
            bar_color = "#4CAF50" if prog_pct < 0.8 else "#FF9800" if prog_pct < 1.0 else "#F44336"

            col_card, col_detail = st.columns([1, 2])
            with col_card:
                st.markdown(f"""
                <div style="background:{b['color']}15;border:2px solid {b['color']}55;
                            border-radius:16px;padding:20px;text-align:center">
                    <div style="font-size:13px;color:#888;margin-bottom:4px">Budget oggi</div>
                    <div style="font-size:52px;font-weight:900;color:{b['color']}">{b['budget']}</div>
                    <div style="font-size:12px;color:#666">TSS</div>
                    <div style="font-size:15px;font-weight:700;color:{b['color']};margin-top:8px">{b['zone']}</div>
                    <div style="font-size:11px;color:#888;margin-top:4px">Moltiplicatore: {b['multiplier']:.2f}x</div>
                </div>""", unsafe_allow_html=True)

            with col_detail:
                st.markdown(f"**💡 {b['advice']}**")
                st.markdown(f"""
                <div style="margin:12px 0 6px;font-size:13px;color:#888">
                    Speso oggi: <b style="color:#fff">{b['tss_spent']:.0f}</b> /
                    Budget: <b style="color:{b['color']}">{b['budget']}</b> /
                    Limite allerta: <b style="color:#FF9800">{b['budget_max']}</b>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.06);border-radius:8px;height:14px;margin:6px 0">
                    <div style="background:{bar_color};width:{prog_pct*100:.0f}%;height:14px;
                                border-radius:8px;transition:width 0.4s"></div>
                </div>""", unsafe_allow_html=True)

                if b["overspent"] > 0:
                    st.warning(f"⚠️ Hai sforato il limite di allerta di **{b['overspent']:.0f} TSS**. "
                               f"Domani il rischio infortuni sale. Priorità al recupero.")
                elif b["remaining"] > 0:
                    st.success(f"✅ Hai ancora **{b['remaining']:.0f} TSS** disponibili oggi.")
                else:
                    st.info("Budget giornaliero raggiunto.")

                ba1, ba2, ba3 = st.columns(3)
                ba1.metric("Base media 28gg", f"{b['avg_base']:.0f} TSS/gg")
                ba2.metric("Readiness",        f"{b['readiness']:.0f}/100")
                ba3.metric("Rimanente",         f"{b['remaining']:.0f} TSS")

            # Grafico storico budget vs speso (ultimi 14 giorni)
            st.markdown("##### Storico budget vs TSS reale (ultimi 14 giorni)")
            df_14 = df[df["start_date"] >= (df["start_date"].max() - timedelta(days=14))]
            daily_spent = df_14.groupby(df_14["start_date"].dt.date)["tss"].sum().reset_index()
            daily_spent.columns = ["date", "tss_spent"]
            daily_spent["date"] = pd.to_datetime(daily_spent["date"])

            fig_bud = go.Figure()
            fig_bud.add_trace(go.Bar(
                x=daily_spent["date"], y=daily_spent["tss_spent"],
                name="TSS speso", marker_color="#2196F3", opacity=0.8,
            ))
            fig_bud.add_hline(y=b["budget"], line_dash="dash", line_color="#4CAF50",
                               annotation_text=f"Budget oggi {b['budget']}", annotation_position="top left")
            fig_bud.add_hline(y=b["budget_max"], line_dash="dot", line_color="#FF9800",
                               annotation_text=f"Limite allerta {b['budget_max']}", annotation_position="top right")
            fig_bud.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="TSS"),
                showlegend=False,
            )
            st.plotly_chart(fig_bud, use_container_width=True)

        # ─────────────────────────────────────────────
        # TAB 2 — ACWR 2.0
        # ─────────────────────────────────────────────
        with tab_acwr2:
            st.markdown("### ⚖️ ACWR 2.0 — Rischio Pesato su Readiness Biologica")
            st.caption("ACWR classico amplificato dall'HRV: a parità di carico, se il tuo sistema nervoso è affaticato il rischio è più alto.")

            av = acwr_v2
            col_gauge2, col_info2 = st.columns([1, 2])

            with col_gauge2:
                fig_av = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=av["risk"],
                    title={"text": "Risk Score", "font": {"size": 14, "color": "#aaa"}},
                    number={"suffix": "/100", "font": {"size": 22}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": av["color"], "thickness": 0.3},
                        "steps": [
                            {"range": [0,  20], "color": "rgba(33,150,243,0.15)"},
                            {"range": [20, 40], "color": "rgba(76,175,80,0.15)"},
                            {"range": [40, 65], "color": "rgba(255,152,0,0.15)"},
                            {"range": [65, 85], "color": "rgba(255,87,34,0.15)"},
                            {"range": [85,100], "color": "rgba(244,67,54,0.15)"},
                        ],
                        "threshold": {"line": {"color": av["color"], "width": 3}, "value": av["risk"]},
                    },
                ))
                fig_av.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200,
                                     margin=dict(l=10,r=10,t=30,b=0), font={"color":"#ccc"})
                st.plotly_chart(fig_av, use_container_width=True)

            with col_info2:
                st.markdown(f"""
                <div style="background:{av['color']}15;border-left:4px solid {av['color']};
                            border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:12px">
                    <div style="font-size:20px;font-weight:900;color:{av['color']}">{av['label']}</div>
                </div>""", unsafe_allow_html=True)

                ma1, ma2, ma3 = st.columns(3)
                ma1.metric("ACWR classico",  f"{av['acwr_base']:.2f}")
                ma2.metric("ACWR pesato",    f"{av['acwr_adj']:.2f}",
                           delta=f"{av['acwr_adj']-av['acwr_base']:+.2f} da HRV")
                ma3.metric("Peso HRV",       f"{av['hrv_weight']:.2f}",
                           delta=f"{'🔴 basso' if av['hrv_weight'] < 0.85 else '🟢 ok'}")

                if av["hrv_now"] and av["hrv_base"]:
                    st.info(f"HRV attuale: **{av['hrv_now']:.0f} ms** | Baseline: **{av['hrv_base']:.0f} ms** | "
                            f"Rapporto: **{av['hrv_weight']:.2f}** ({'🔴 sotto baseline' if av['hrv_weight'] < 0.9 else '✅ nella norma'})")
                elif rc_vitals is None:
                    st.info("💍 Carica i dati RingConn per abilitare la pesatura HRV. Ora stai vedendo l'ACWR classico.")

            # Storico ACWR classico vs pesato
            st.markdown("##### Confronto ACWR classico vs pesato nel tempo")
            _, acwr_series_data = calc_acwr(df)
            if acwr_series_data is not None and not acwr_series_data.empty:
                fig_acwr2 = go.Figure()
                fig_acwr2.add_trace(go.Scatter(
                    x=acwr_series_data.index, y=acwr_series_data.values,
                    name="ACWR classico", line=dict(color="#2196F3", width=2),
                ))
                # Zona sicura
                fig_acwr2.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(76,175,80,0.08)",
                                     line_width=0, annotation_text="Zona sicura")
                fig_acwr2.add_hline(y=1.5, line_dash="dash", line_color="rgba(244,67,54,0.5)",
                                     annotation_text="Soglia rischio")
                fig_acwr2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=220, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="ACWR"),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_acwr2, use_container_width=True)

        # ─────────────────────────────────────────────
        # TAB 3 — CIRCADIAN PERFORMANCE
        # ─────────────────────────────────────────────
        # ─────────────────────────────────────────────
        # TAB 4 — HRV TREND SLOPE
        # ─────────────────────────────────────────────
        with tab_hrv_slope:
            st.markdown("### 📉 HRV Trend Slope — Rilevatore di Overreaching")
            st.caption("La pendenza della retta di regressione HRV negli ultimi 14 giorni. Calo sostenuto = segnale precoce di overtraining.")

            if rc_vitals is None:
                st.info("💍 Carica i dati RingConn (Vital Signs CSV) per abilitare questa analisi.")
            elif hrv_slope is None:
                st.info("Servono almeno 4 giorni di dati HRV validi.")
            else:
                hs = hrv_slope
                col_hs1, col_hs2, col_hs3, col_hs4 = st.columns(4)
                col_hs1.metric("Pendenza HRV", f"{hs['slope']:+.2f} ms/gg",
                               delta=hs["label"], delta_color="off")
                col_hs2.metric("HRV attuale",  f"{hs['latest']:.0f} ms")
                col_hs3.metric("Baseline",     f"{hs['baseline']:.0f} ms")
                col_hs4.metric("Variazione",   f"{hs['pct_change']:+.1f}%",
                               delta_color="normal")

                # Alert overreaching
                if "Overreaching" in hs["label"]:
                    st.error(f"⚠️ **Overreaching probabile**: HRV in calo di {abs(hs['slope']):.2f} ms/giorno "
                             f"negli ultimi 14 giorni. Considera 2-3 giorni di recupero attivo.")
                elif "calo" in hs["label"].lower():
                    st.warning(f"📉 HRV in lieve calo ({hs['slope']:+.2f} ms/gg). Monitora i prossimi giorni.")
                else:
                    st.success(f"✅ {hs['label']} — HRV stabile o in miglioramento.")

                # Grafico HRV con retta di tendenza
                st.markdown("##### HRV giornaliero + trend lineare (14 giorni)")
                fig_hs = go.Figure()
                fig_hs.add_trace(go.Scatter(
                    x=hs["dates"], y=hs["values"],
                    name="HRV", mode="lines+markers",
                    line=dict(color="#e94560", width=2), marker=dict(size=6),
                ))
                fig_hs.add_trace(go.Scatter(
                    x=hs["dates"], y=hs["trend_line"],
                    name=f"Trend ({hs['slope']:+.2f} ms/gg)",
                    mode="lines", line=dict(color=hs["color"], width=2, dash="dash"),
                ))
                fig_hs.add_hline(y=hs["baseline"], line_dash="dot",
                                  line_color="rgba(255,255,255,0.3)",
                                  annotation_text=f"Baseline {hs['baseline']:.0f}ms")
                fig_hs.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="HRV (ms)"),
                )
                st.plotly_chart(fig_hs, use_container_width=True)

                # TSS sovrapposto per contestualizzare il calo
                st.markdown("##### TSS degli ultimi 14 giorni (confronto)")
                df_14_hs = df[df["start_date"] >= (df["start_date"].max() - timedelta(days=14))]
                daily_tss_hs = df_14_hs.groupby(df_14_hs["start_date"].dt.date)["tss"].sum()
                fig_tss_hs = go.Figure(go.Bar(
                    x=pd.to_datetime(daily_tss_hs.index), y=daily_tss_hs.values,
                    marker_color="#2196F3", opacity=0.7, name="TSS",
                ))
                fig_tss_hs.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=160, margin=dict(l=0,r=0,t=5,b=0),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="TSS"),
                    showlegend=False,
                )
                st.plotly_chart(fig_tss_hs, use_container_width=True)


    elif menu == "🗺️ Mappe 3D":
        st.markdown("## 🗺️ Mappe 3D")

        if not MAPBOX_TOKEN:
            st.warning("""
            **Token Mapbox non configurato.**

            Per abilitare le mappe 3D:
            1. Vai su [mapbox.com](https://mapbox.com) → Sign up gratuito
            2. Account → Tokens → copia il **Default public token**
            3. Aggiungi `MAPBOX_TOKEN = "pk...."` nei **Secrets** di Streamlit Cloud
            4. Ricarica l'app

            Il piano gratuito include 50.000 map load/mese.
            """)
            st.stop()

        # ── Widget contatori in sidebar ──
        with st.sidebar:
            monthly_pct = st.session_state.mb_loads_monthly / MAPBOX_MONTHLY_LIMIT * 100
            daily_pct   = st.session_state.mb_loads_daily   / MAPBOX_DAILY_SOFT_CAP  * 100
            bar_color_mb = "#4CAF50" if monthly_pct < 60 else "#FF9800" if monthly_pct < 85 else "#F44336"
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:10px 14px;margin:4px 0'>
                <div style='font-size:11px;color:#888;margin-bottom:4px'>🗺️ Mapbox GL JS — Free Tier</div>
                <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px'>
                    <span>Sessione</span>
                    <b style='color:{bar_color_mb}'>{st.session_state.mb_loads_session}/{MAPBOX_SESSION_CAP}</b>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px'>
                    <span>Oggi</span>
                    <b style='color:{bar_color_mb}'>{st.session_state.mb_loads_daily}/{MAPBOX_DAILY_SOFT_CAP}</b>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px'>
                    <span>Mese</span>
                    <b style='color:{bar_color_mb}'>{st.session_state.mb_loads_monthly:,}/{MAPBOX_MONTHLY_LIMIT:,}</b>
                </div>
                <div style='background:rgba(255,255,255,0.08);border-radius:4px;height:5px;margin-top:4px'>
                    <div style='background:{bar_color_mb};width:{min(monthly_pct,100):.1f}%;height:5px;border-radius:4px'></div>
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button("🔄 Reset contatore sessione", key="mb_reset_session"):
                st.session_state.mb_loads_session = 0
                st.rerun()

        # ── Filtro attività con tracciato (solo quelle con polyline) ──
        df_with_map = df[df["map"].apply(
            lambda x: bool(x.get("summary_polyline") if isinstance(x, dict) else False)
        )].copy() if "map" in df.columns else pd.DataFrame()

        if df_with_map.empty:
            st.info("Nessuna attività con tracciato GPS disponibile.")
            st.stop()

        # ── Filtro sport ──
        all_sports_map = sorted(df_with_map["type"].unique().tolist())
        if "map3d_sport_filter" not in st.session_state:
            st.session_state.map3d_sport_filter = set(all_sports_map)

        st.markdown("**Filtra sport:**")
        map_btn_cols = st.columns(min(len(all_sports_map), 9))
        for i, sport in enumerate(all_sports_map):
            si = get_sport_info(sport)
            is_active = sport in st.session_state.map3d_sport_filter
            with map_btn_cols[i % len(map_btn_cols)]:
                if st.button(f"{si['icon']} {si['label']}", key=f"map3d_btn_{sport}",
                             type="primary" if is_active else "secondary",
                             use_container_width=True):
                    sf = st.session_state.map3d_sport_filter
                    sf.discard(sport) if sport in sf else sf.add(sport)
                    st.rerun()

        mc1, mc2, mc3 = st.columns([1, 1, 7])
        with mc1:
            if st.button("✅ Tutti", key="map3d_all", use_container_width=True):
                st.session_state.map3d_sport_filter = set(all_sports_map)
                st.rerun()
        with mc2:
            if st.button("❌ Nessuno", key="map3d_none", use_container_width=True):
                st.session_state.map3d_sport_filter = set()
                st.rerun()

        df_map_filtered = df_with_map[df_with_map["type"].isin(st.session_state.map3d_sport_filter)]

        st.divider()

        # ── Modalità visualizzazione ──
        tab_single, tab_compare, tab_avalanche, tab_explore = st.tabs([
            "📍 Tracciato singolo", "🔀 Confronto tracciati", "⛷️ Valanghe (Scialpinismo)", "🧭 Esplora & Disegna"
        ])

        # ─────────────────────────────────────────────────────
        # Helper: costruisce array GeoJSON LineString da polyline
        # ─────────────────────────────────────────────────────
        def polyline_to_coords(encoded):
            try:
                pts = polyline.decode(encoded)
                return [[lon, lat] for lat, lon in pts]
            except Exception:
                return []

        def get_elevation_profile(coords):
            """Stima profilo quota da coordinate (lat differenziale) — placeholder."""
            # In assenza di DEM, usiamo la variazione lat come proxy di quota
            # In produzione si userebbe Open-Elevation API
            return list(range(len(coords)))

        def haversine_m(lon1, lat1, lon2, lat2):
            import math
            R  = 6371000
            d1 = math.radians(lat2 - lat1)
            d2 = math.radians(lon2 - lon1)
            a  = math.sin(d1/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(d2/2)**2
            return R * 2 * math.asin(math.sqrt(max(0.0, a)))

        @st.cache_data(ttl=3600, show_spinner="Recupero quote da Open-Elevation...")
        def fetch_elevations(coords_tuple, max_points=150):
            import urllib.request, json as _j
            coords = list(coords_tuple)
            n = len(coords)
            if n == 0:
                return None
            if n > max_points:
                step    = n / max_points
                indices = [int(i * step) for i in range(max_points)]
                indices[-1] = n - 1
                sampled = [coords[i] for i in indices]
            else:
                indices = list(range(n))
                sampled = coords
            locations = [{"latitude": lat, "longitude": lon} for lon, lat in sampled]
            payload   = _j.dumps({"locations": locations}).encode()
            try:
                req = urllib.request.Request(
                    "https://api.open-elevation.com/api/v1/lookup",
                    data=payload, headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _j.loads(resp.read())
                elevs_s = [r["elevation"] for r in data["results"]]
                if len(indices) == n:
                    return elevs_s
                elevs = [0.0] * n
                for k in range(len(indices)):
                    elevs[indices[k]] = elevs_s[k]
                for k in range(1, len(indices)):
                    i0, i1 = indices[k-1], indices[k]
                    e0, e1 = elevs_s[k-1], elevs_s[k]
                    for j in range(i0, i1):
                        t = (j - i0) / max(1, i1 - i0)
                        elevs[j] = e0 + t * (e1 - e0)
                elevs[n-1] = elevs_s[-1]
                return elevs
            except Exception:
                return None

        def build_slope_colors(coords, elevs, total_elev_gain=0):
            """
            Pendenza reale (gradi) da coordinate + quota Open-Elevation.
            AINEVA: <3 grigio | 3-25 blu | 25-30 giallo | 30-35 arancio | >35 rosso
            Adattiva (dislivello < 300m): soglie sui percentili del tracciato.
            """
            import math
            NEUTRAL = "#607D8B"
            BLUE    = "#1565C0"
            YELLOW  = "#FFEB3B"
            ORANGE  = "#FF9800"
            RED     = "#F44336"
            n = len(coords)
            if n < 2 or not elevs or len(elevs) < n:
                return [NEUTRAL] * n
            # Smoothing quota
            elev_arr = list(elevs[:n])
            if n >= 5:
                sm = elev_arr[:]
                for i in range(2, n - 2):
                    sm[i] = (elev_arr[i-2]*0.1 + elev_arr[i-1]*0.2 +
                             elev_arr[i]*0.4   + elev_arr[i+1]*0.2 +
                             elev_arr[i+2]*0.1)
                elev_arr = sm
            slopes = [0.0]
            for i in range(1, n):
                lon1, lat1 = coords[i-1]
                lon2, lat2 = coords[i]
                dh  = elev_arr[i] - elev_arr[i-1]
                dxy = haversine_m(lon1, lat1, lon2, lat2)
                if dxy < 2 or dxy > 300:
                    slopes.append(slopes[-1])
                    continue
                slopes.append(math.degrees(math.atan2(abs(dh), dxy)))
            # Soglie
            is_mountain = total_elev_gain >= 300
            if is_mountain:
                t_min, t_low, t_mid, t_high = 3.0, 25.0, 30.0, 35.0
            else:
                valid = sorted([s for s in slopes if s > 1.0])
                if len(valid) < 4:
                    return [NEUTRAL] * n
                p50 = valid[int(len(valid) * 0.50)]
                p70 = valid[int(len(valid) * 0.70)]
                p90 = valid[int(len(valid) * 0.90)]
                t_min, t_low, t_mid, t_high = 1.0, p50, p70, p90
            colors = []
            for s in slopes:
                if s < t_min:    colors.append(NEUTRAL)
                elif s < t_low:  colors.append(BLUE)
                elif s < t_mid:  colors.append(YELLOW)
                elif s < t_high: colors.append(ORANGE)
                else:            colors.append(RED)
            return colors

        # ─────────────────────────────────────────────────────
        # TAB 1 — TRACCIATO SINGOLO 3D
        # ─────────────────────────────────────────────────────
        with tab_single:
            if df_map_filtered.empty:
                st.info("Nessuna attività disponibile con il filtro attuale.")
            else:
                # Selettore attività
                act_options = {}
                for _, row in df_map_filtered.sort_values("start_date", ascending=False).head(100).iterrows():
                    si = get_sport_info(row["type"], row.get("name",""))
                    label = f"{si['icon']} {row['start_date'].strftime('%d/%m/%Y')} — {row.get('name','N/A')} ({row['distance']/1000:.1f} km)"
                    act_options[label] = row

                selected_label = st.selectbox("Seleziona attività:", list(act_options.keys()), key="map3d_sel")
                sel_row = act_options[selected_label]
                sel_si  = get_sport_info(sel_row["type"])

                coords = polyline_to_coords(sel_row["map"]["summary_polyline"])

                if not coords:
                    st.warning("Tracciato GPS non disponibile per questa attività.")
                else:
                    # Opzioni visualizzazione
                    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
                    with col_opt1:
                        map_style = st.selectbox("Stile mappa:", [
                            "satellite-streets-v12",
                            "outdoors-v12",
                            "dark-v11",
                            "light-v11",
                        ], key="map3d_style")
                    with col_opt2:
                        show_terrain = st.toggle("🏔️ Terrain 3D", value=True, key="map3d_terrain")
                    with col_opt3:
                        is_ski_act = sel_row.get("type","") in ["BackcountrySki","NordicSki","AlpineSki","Snowboard"]
                        show_slope = st.toggle("📐 Colore pendenza", value=is_ski_act, key="map3d_slope")
                    with col_opt4:
                        extrude_height = st.slider("Altezza estruso", 0, 500, 100, key="map3d_extrude",
                                                    help="Amplificazione visiva del percorso in 3D")

                    # Fetch quota reale e calcola colori pendenza
                    elev_gain_act = float(sel_row.get("total_elevation_gain") or 0)
                    if show_slope:
                        with st.spinner("⛰️ Recupero quote da Open-Elevation..."):
                            elevs = fetch_elevations(tuple(coords))
                        if elevs is None:
                            st.caption("⚠️ Open-Elevation non raggiungibile — pendenza non disponibile.")
                            slope_colors = [sel_si["color"]] * len(coords)
                        else:
                            slope_colors = build_slope_colors(coords, elevs, elev_gain_act)
                    else:
                        slope_colors = [sel_si["color"]] * len(coords)

                    # Coordinate centro
                    center_lon = sum(c[0] for c in coords) / len(coords)
                    center_lat = sum(c[1] for c in coords) / len(coords)

                    # Costruisci GeoJSON con proprietà colore per ogni segmento
                    segments_geojson = {"type": "FeatureCollection", "features": []}
                    for i in range(len(coords) - 1):
                        segments_geojson["features"].append({
                            "type": "Feature",
                            "properties": {"color": slope_colors[i], "index": i},
                            "geometry": {"type": "LineString", "coordinates": [coords[i], coords[i+1]]}
                        })

                    import json as _json
                    segments_json = _json.dumps(segments_geojson)
                    start_json    = _json.dumps(coords[0])
                    end_json      = _json.dumps(coords[-1])

                    # Statistiche attività
                    km_val   = float(sel_row["distance"]) / 1000
                    elev_val = float(sel_row.get("total_elevation_gain") or 0)
                    dur_val  = int(sel_row.get("moving_time") or 0)
                    tss_val  = float(sel_row.get("tss") or 0)
                    ms1,ms2,ms3,ms4 = st.columns(4)
                    ms1.metric("Distanza",   f"{km_val:.1f} km")
                    ms2.metric("Dislivello", f"{elev_val:.0f} m")
                    ms3.metric("Durata",     f"{dur_val//3600}h {(dur_val%3600)//60:02d}m")
                    ms4.metric("TSS",        f"{tss_val:.0f}")

                    _mb_ok, _mb_reason = mapbox_render_allowed()
                    if not _mb_ok:
                        st.error(_mb_reason)
                    else:
                        _poly = sel_row["map"]["summary_polyline"]
                        _html = build_map3d_html(
                            _poly, MAPBOX_TOKEN,
                            sport_type=sel_row.get("type",""),
                            elev_gain=elev_val,
                            map_style=map_style,
                            show_slope_map=show_slope,
                            compact=False,
                        )
                        if _html:
                            import streamlit.components.v1 as components
                            components.html(_html, height=600, scrolling=False)
                            mapbox_register_load()

        # ─────────────────────────────────────────────────────
        # TAB 2 — CONFRONTO TRACCIATI
        # ─────────────────────────────────────────────────────
        with tab_compare:
            st.markdown("### 🔀 Confronto tracciati sovrapposti")
            st.caption("Sovrapponi fino a 5 attività sulla stessa mappa 3D per confrontare percorsi.")

            if df_map_filtered.empty:
                st.info("Nessuna attività disponibile.")
            else:
                act_list = []
                for _, row in df_map_filtered.sort_values("start_date", ascending=False).head(50).iterrows():
                    si = get_sport_info(row["type"], row.get("name",""))
                    label = f"{si['icon']} {row['start_date'].strftime('%d/%m/%Y')} — {row.get('name','N/A')} ({row['distance']/1000:.1f} km)"
                    act_list.append((label, row))

                selected_labels = st.multiselect(
                    "Seleziona attività da confrontare (max 5):",
                    [a[0] for a in act_list],
                    default=[a[0] for a in act_list[:2]],
                    max_selections=5,
                    key="map3d_compare_sel"
                )

                comp_style = st.selectbox("Stile:", ["satellite-streets-v12", "outdoors-v12", "dark-v11"],
                                           key="map3d_comp_style")
                comp_terrain = st.toggle("🏔️ Terrain 3D", value=True, key="map3d_comp_terrain")

                if selected_labels:
                    selected_rows = [row for label, row in act_list if label in selected_labels]

                    # Palette colori per tracciati multipli
                    palette = ["#e94560", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

                    # Costruisci GeoJSON multi-tracciato
                    all_coords_center = []
                    layers_js = ""
                    sources_js = ""

                    import json as _json2
                    for idx, row in enumerate(selected_rows):
                        coords_c = polyline_to_coords(row["map"]["summary_polyline"])
                        if not coords_c:
                            continue
                        all_coords_center.extend(coords_c)
                        color_c = palette[idx % len(palette)]
                        geoj_c  = _json2.dumps({"type": "Feature", "properties": {},
                                                "geometry": {"type": "LineString", "coordinates": coords_c}})
                        start_c = _json2.dumps(coords_c[0])
                        end_c   = _json2.dumps(coords_c[-1])
                        si_c    = get_sport_info(row["type"], row.get("name",""))
                        name_c  = row.get("name", "Attività")[:20]
                        dist_c  = row["distance"] / 1000

                        sources_js += f"""
    map.addSource('route-{idx}', {{ type: 'geojson',
        data: {geoj_c} }});
"""
                        layers_js += f"""
    map.addLayer({{ id: 'route-line-{idx}', type: 'line',
        source: 'route-{idx}',
        paint: {{ 'line-color': '{color_c}', 'line-width': 4, 'line-opacity': 0.9 }},
        layout: {{ 'line-cap': 'round', 'line-join': 'round' }} }});
    map.addLayer({{ id: 'route-glow-{idx}', type: 'line',
        source: 'route-{idx}',
        paint: {{ 'line-color': '{color_c}', 'line-width': 14,
                  'line-opacity': 0.12, 'line-blur': 4 }} }});
    new mapboxgl.Marker({{ color: '{color_c}', scale: 0.9 }})
        .setLngLat({start_c})
        .setPopup(new mapboxgl.Popup().setHTML('<b style="color:{color_c}">{si_c["icon"]} {name_c}</b><br>{dist_c:.1f} km'))
        .addTo(map);
    new mapboxgl.Marker({{ color: '{color_c}', scale: 0.6 }})
        .setLngLat({end_c}).addTo(map);
"""

                    if all_coords_center:
                        clon = sum(c[0] for c in all_coords_center) / len(all_coords_center)
                        clat = sum(c[1] for c in all_coords_center) / len(all_coords_center)

                        terrain_comp_js = """
    map.addSource('mapbox-dem', { 'type': 'raster-dem',
        'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
        'tileSize': 512, 'maxzoom': 14 });
    map.setTerrain({'source': 'mapbox-dem', 'exaggeration': 1.8});
    map.addLayer({ 'id': 'sky', 'type': 'sky',
        'paint': { 'sky-type': 'atmosphere',
                   'sky-atmosphere-sun': [0.0, 90.0],
                   'sky-atmosphere-sun-intensity': 15 }});
""" if comp_terrain else ""

                        # Legenda tracciati
                        legend_html = ""
                        for idx, row in enumerate(selected_rows):
                            si_l = get_sport_info(row["type"], row.get("name",""))
                            legend_html += f"<div class='legend-row'><div class='legend-dot' style='background:{palette[idx]}'></div>{si_l['icon']} {row.get('name','')[:20]} ({row['distance']/1000:.1f}km)</div>"

                        html_compare = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet"/>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#0e1117; }}
  #map-wrap {{ position:relative; width:100%; height:100%; }}
  #map {{ position:absolute; top:0; left:0; width:100%; height:100%; }}
  .legend {{ position:absolute; bottom:30px; left:10px; background:rgba(14,17,23,0.88);
             color:#fff; padding:10px 14px; border-radius:10px; font-size:12px;
             font-family:sans-serif; line-height:2; z-index:10; }}
  .legend-row {{ display:flex; align-items:center; gap:8px; }}
  .legend-dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
</style>
</head><body>
<div id="map-wrap">
  <div id="map"></div>
  <div class="legend"><b>Tracciati</b>{legend_html}</div>
</div>
<script>
mapboxgl.accessToken = '{MAPBOX_TOKEN}';
const map = new mapboxgl.Map({{
    container: 'map',
    style: 'mapbox://styles/mapbox/{comp_style}',
    center: [{clon}, {clat}],
    zoom: 11, pitch: 55, bearing: -15, antialias: true
}});
map.addControl(new mapboxgl.NavigationControl({{'visualizePitch':true}}), 'top-left');
map.dragRotate.enable();
map.touchZoomRotate.enableRotation();
map.scrollZoom.enable();
map.scrollZoom.setWheelZoomRate(1/450);
(function() {{
  const cv = map.getCanvas();
  cv.addEventListener('wheel', (e) => {{ e.stopPropagation(); }}, {{ passive: false }});
  let mid = false, lx = 0, ly = 0;
  cv.addEventListener('mousedown', (e) => {{
    if (e.button === 1) {{ e.preventDefault(); mid = true; lx = e.clientX; ly = e.clientY; cv.style.cursor = 'grab'; }}
  }});
  window.addEventListener('mousemove', (e) => {{
    if (!mid) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    map.jumpTo({{ bearing: map.getBearing() + dx * 0.5,
                  pitch: Math.min(85, Math.max(0, map.getPitch() - dy * 0.4)) }});
  }});
  window.addEventListener('mouseup', (e) => {{ if (e.button === 1) {{ mid = false; cv.style.cursor = ''; }} }});
}})();
map.addControl(new mapboxgl.FullscreenControl(), 'top-left');
map.addControl(new mapboxgl.ScaleControl(), 'bottom-right');
map.on('load', () => {{
    {terrain_comp_js}
    {sources_js}
    {layers_js}
    map.flyTo({{ center: [{clon},{clat}], zoom:12, pitch:55, duration:2000 }});
}});
</script></body></html>"""

                        import streamlit.components.v1 as components
                        _mb_ok2, _mb_reason2 = mapbox_render_allowed()
                        if not _mb_ok2:
                            st.error(_mb_reason2)
                        else:
                            components.html(html_compare, height=580, scrolling=False)
                            mapbox_register_load()

        # ─────────────────────────────────────────────────────
        # TAB 3 — VALANGHE (SCIALPINISMO)
        # ─────────────────────────────────────────────────────
        with tab_avalanche:
            st.markdown("### ⛷️ Layer Valanghe — Scialpinismo")
            st.info("""
            **Layer disponibili in questo tab:**
            - 🏔️ Terrain 3D Mapbox (DEM ad alta risoluzione)
            - 🎿 OpenSnowMap (piste e itinerari sci)
            - ⚠️ Bollettino Valanghe AINEVA (WMS ufficiale — zone di pericolo)
            - 📐 Pendenza stimata sul tracciato (colori AINEVA standard)
            """)

            # Filtra solo scialpinismo/ski/trail in montagna
            ski_sports   = ["BackcountrySki", "NordicSki", "AlpineSki", "Snowboard",
                            "Hike", "TrailRun", "Walk"]
            df_ski       = df_map_filtered[df_map_filtered["type"].isin(ski_sports)]
            df_ski_all   = df_with_map[df_with_map["type"].isin(ski_sports)]

            if df_ski.empty and df_ski_all.empty:
                st.warning("Nessuna attività di scialpinismo/montagna trovata. "
                           "Assicurati di avere attività di tipo BackcountrySki, Hike o TrailRun con GPS.")
                df_ski_use = df_map_filtered.head(20)  # fallback
            else:
                df_ski_use = df_ski if not df_ski.empty else df_ski_all

            act_ski_opts = {}
            for _, row in df_ski_use.sort_values("start_date", ascending=False).head(50).iterrows():
                si = get_sport_info(row["type"], row.get("name",""))
                label = f"{si['icon']} {row['start_date'].strftime('%d/%m/%Y')} — {row.get('name','N/A')} ({row['distance']/1000:.1f} km)"
                act_ski_opts[label] = row

            if not act_ski_opts:
                st.info("Nessuna attività disponibile.")
            else:
                sel_ski_label = st.selectbox("Seleziona uscita:", list(act_ski_opts.keys()), key="map3d_ski_sel")
                sel_ski       = act_ski_opts[sel_ski_label]
                coords_ski    = polyline_to_coords(sel_ski["map"]["summary_polyline"])
                elev_gain_ski = float(sel_ski.get("total_elevation_gain") or 0)
                with st.spinner("⛰️ Recupero quote..."):
                    elevs_ski = fetch_elevations(tuple(coords_ski))
                if elevs_ski is None:
                    st.caption("⚠️ Open-Elevation non raggiungibile — pendenza stimata non disponibile.")
                    slope_cols = ["#00BCD4"] * len(coords_ski)
                else:
                    slope_cols = build_slope_colors(coords_ski, elevs_ski, elev_gain_ski)

                col_av1, col_av2, col_av3 = st.columns(3)
                with col_av1:
                    show_aineva   = st.toggle("⚠️ Bollettino AINEVA", value=True, key="av_aineva",
                                              help="Layer WMS ufficiale zone di pericolo valanghe")
                with col_av2:
                    show_snowmap  = st.toggle("🎿 OpenSnowMap", value=True, key="av_snow",
                                              help="Itinerari e piste sciistiche")
                with col_av3:
                    show_slope_av = st.toggle("📐 Pendenza tracciato", value=True, key="av_slope")

                if not coords_ski:
                    st.warning("Tracciato GPS non disponibile.")
                else:
                    import json as _json3
                    seg_ski_geoj = {"type": "FeatureCollection", "features": []}
                    for i in range(len(coords_ski)-1):
                        color_use = slope_cols[i] if show_slope_av else "#00BCD4"
                        seg_ski_geoj["features"].append({
                            "type": "Feature",
                            "properties": {"color": color_use},
                            "geometry": {"type": "LineString", "coordinates": [coords_ski[i], coords_ski[i+1]]}
                        })

                    clon_ski = sum(c[0] for c in coords_ski) / len(coords_ski)
                    clat_ski = sum(c[1] for c in coords_ski) / len(coords_ski)
                    seg_ski_json  = _json3.dumps(seg_ski_geoj)
                    start_ski_j   = _json3.dumps(coords_ski[0])
                    end_ski_j     = _json3.dumps(coords_ski[-1])

                    aineva_js = """
    // AINEVA WMS — Bollettino valanghe (Italia)
    map.addSource('aineva-source', {
        type: 'raster',
        tiles: ['https://bollettino.aineva.it/geoserver/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=true&LAYERS=cargis:eaws_bulletins&WIDTH=256&HEIGHT=256&SRS=EPSG%3A3857&BBOX={bbox-epsg-3857}'],
        tileSize: 256,
        attribution: '© AINEVA'
    });
    map.addLayer({ id: 'aineva-layer', type: 'raster',
        source: 'aineva-source', paint: { 'raster-opacity': 0.55 } });
""" if show_aineva else ""

                    snowmap_js = """
    // OpenSnowMap — itinerari sci
    map.addSource('snowmap-source', {
        type: 'raster',
        tiles: ['https://tiles.opensnowmap.org/pistes/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© OpenSnowMap'
    });
    map.addLayer({ id: 'snowmap-layer', type: 'raster',
        source: 'snowmap-source', paint: { 'raster-opacity': 0.7 } });
""" if show_snowmap else ""

                    si_ski = get_sport_info(sel_ski["type"])
                    dist_ski  = sel_ski["distance"]/1000
                    elev_ski  = sel_ski.get("total_elevation_gain") or 0
                    dur_ski   = int(sel_ski.get("moving_time") or 0)
                    ski_name  = (sel_ski.get('name','') or '')[:25]
                    ski_icon  = si_ski['icon']

                    aineva_row   = '<div class="legend-row"><div class="legend-dot" style="background:#FF9800"></div>AINEVA Valanghe</div>' if show_aineva else ''
                    snow_row     = '<div class="legend-row"><div class="legend-dot" style="background:#2196F3"></div>OpenSnowMap</div>' if show_snowmap else ''
                    slope_rows   = '<div style="margin-top:4px"><b>Pendenza</b></div><div class="legend-row"><div class="legend-dot" style="background:#4CAF50"></div>&lt;25° ok</div><div class="legend-row"><div class="legend-dot" style="background:#FFEB3B"></div>25-30°</div><div class="legend-row"><div class="legend-dot" style="background:#FF9800"></div>30-35°</div><div class="legend-row"><div class="legend-dot" style="background:#F44336"></div>&gt;35°</div>' if show_slope_av else ''

                    html_avalanche = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet"/>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#0e1117; }}
  #map-wrap {{ position:relative; width:100%; height:100%; }}
  #map {{ position:absolute; top:0; left:0; width:100%; height:100%; }}
  .legend-av {{
    position:absolute; bottom:30px; left:10px; background:rgba(14,17,23,0.88);
    color:#fff; padding:10px 14px; border-radius:10px; font-size:11px;
    font-family:sans-serif; line-height:1.9; z-index:10;
  }}
  .legend-row {{ display:flex; align-items:center; gap:7px; }}
  .legend-dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  #info-ski {{
    position:absolute; top:10px; right:10px; background:rgba(14,17,23,0.88);
    color:#fff; padding:12px 16px; border-radius:10px; font-size:12px;
    font-family:sans-serif; z-index:10;
  }}
</style>
</head><body>
<div id="map-wrap">
  <div id="map"></div>
  <div class="legend-av">
    <b>Layers attivi</b>
    {aineva_row}{snow_row}{slope_rows}
    <div style="margin-top:6px;color:#666;font-size:10px">Pendenza = stima GPS</div>
  </div>
  <div id="info-ski">
    <div style="font-weight:700;margin-bottom:4px">{ski_icon} {ski_name}</div>
    <div>{dist_ski:.1f} km | {elev_ski:.0f}m D+ | {dur_ski//3600}h {(dur_ski%3600)//60:02d}m</div>
  </div>
</div>
<script>
mapboxgl.accessToken = '{MAPBOX_TOKEN}';
const map = new mapboxgl.Map({{
    container: 'map',
    style: 'mapbox://styles/mapbox/satellite-streets-v12',
    center: [{clon_ski}, {clat_ski}],
    zoom: 12, pitch: 70, bearing: -10, antialias: true
}});
map.addControl(new mapboxgl.NavigationControl({{'visualizePitch':true}}), 'top-left');
map.dragRotate.enable();
map.touchZoomRotate.enableRotation();
map.scrollZoom.enable();
map.scrollZoom.setWheelZoomRate(1/450);
(function() {{
  const cv = map.getCanvas();
  cv.addEventListener('wheel', (e) => {{ e.stopPropagation(); }}, {{ passive: false }});
  let mid = false, lx = 0, ly = 0;
  cv.addEventListener('mousedown', (e) => {{
    if (e.button === 1) {{ e.preventDefault(); mid = true; lx = e.clientX; ly = e.clientY; cv.style.cursor = 'grab'; }}
  }});
  window.addEventListener('mousemove', (e) => {{
    if (!mid) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    map.jumpTo({{ bearing: map.getBearing() + dx * 0.5,
                  pitch: Math.min(85, Math.max(0, map.getPitch() - dy * 0.4)) }});
  }});
  window.addEventListener('mouseup', (e) => {{ if (e.button === 1) {{ mid = false; cv.style.cursor = ''; }} }});
}})();
map.addControl(new mapboxgl.FullscreenControl(), 'top-left');
map.addControl(new mapboxgl.ScaleControl(), 'bottom-right');

map.on('load', () => {{
    map.addSource('mapbox-dem', {{
        'type': 'raster-dem',
        'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
        'tileSize': 512, 'maxzoom': 14
    }});
    map.setTerrain({{'source': 'mapbox-dem', 'exaggeration': 2.0}});
    map.addLayer({{ 'id': 'sky', 'type': 'sky',
        'paint': {{ 'sky-type': 'atmosphere',
                   'sky-atmosphere-sun': [0.0, 45.0],
                   'sky-atmosphere-sun-intensity': 15 }} }});

    {aineva_js}
    {snowmap_js}

    map.addSource('ski-route', {{ type: 'geojson', data: {seg_ski_json} }});
    map.addLayer({{
        id: 'ski-shadow', type: 'line', source: 'ski-route',
        paint: {{ 'line-color': 'rgba(0,0,0,0.5)', 'line-width': 9,
                  'line-blur': 5, 'line-translate': [2, 2] }}
    }});
    map.addLayer({{
        id: 'ski-glow', type: 'line', source: 'ski-route',
        paint: {{ 'line-color': ['get','color'], 'line-width': 16,
                  'line-opacity': 0.15, 'line-blur': 5 }}
    }});
    map.addLayer({{
        id: 'ski-line', type: 'line', source: 'ski-route',
        paint: {{ 'line-color': ['get','color'], 'line-width': 4,
                  'line-opacity': 1.0 }},
        layout: {{ 'line-cap': 'round', 'line-join': 'round' }}
    }});

    new mapboxgl.Marker({{ color: '#4CAF50', scale: 1.3 }})
        .setLngLat({start_ski_j})
        .setPopup(new mapboxgl.Popup().setHTML('<b>Partenza</b>'))
        .addTo(map);
    new mapboxgl.Marker({{ color: '#F44336', scale: 1.3 }})
        .setLngLat({end_ski_j})
        .setPopup(new mapboxgl.Popup().setHTML('<b>Arrivo</b>'))
        .addTo(map);

    map.flyTo({{
        center: [{clon_ski}, {clat_ski}],
        zoom: 13, pitch: 70, duration: 2000
    }});
}});
</script></body></html>"""

                    import streamlit.components.v1 as components
                    _mb_ok3, _mb_reason3 = mapbox_render_allowed()
                    if not _mb_ok3:
                        st.error(_mb_reason3)
                    else:
                        components.html(html_avalanche, height=640, scrolling=False)
                        mapbox_register_load()

                    st.caption("""
                    **Layer attivi:**
                    - 🏔️ **Terrain 3D** — DEM Mapbox ad alta risoluzione (esagerazione 2×)
                    - ⚠️ **AINEVA** — Bollettino valanghe ufficiale italiano (WMS geoserver.aineva.it)
                    - 🎿 **OpenSnowMap** — Piste e itinerari sciistici
                    - 📐 **Pendenza** — Stima da coordinate GPS (approssimativa, non sostituisce la cartografia ufficiale)

                    > ⚠️ **Avviso sicurezza**: i dati di pendenza mostrati sono una stima approssimativa basata sulle coordinate GPS planari. Per valutare il rischio valanga usa sempre il bollettino ufficiale AINEVA e cartografia specializzata.
                    """)


        # ─────────────────────────────────────────────────────
        # TAB 4 — ESPLORA & DISEGNA
        # ─────────────────────────────────────────────────────
        with tab_explore:
            st.markdown("### 🧭 Esplora & Disegna Percorso")
            st.caption("Naviga liberamente la mappa 3D con terrain e layer valanghe. Clicca per aggiungere punti e disegnare un percorso ipotetico. I waypoint vengono salvati nella sessione.")

            col_ex1, col_ex2, col_ex3 = st.columns(3)
            with col_ex1:
                explore_style = st.selectbox("Stile:", [
                    "satellite-streets-v12", "outdoors-v12", "dark-v11"
                ], key="explore_style")
            with col_ex2:
                explore_aineva = st.toggle("⚠️ AINEVA Valanghe", value=True, key="explore_aineva")
            with col_ex3:
                explore_snow   = st.toggle("🎿 OpenSnowMap", value=True, key="explore_snow")

            # Zona di partenza: centro sull'ultima attività disponibile, o Aosta come default montagna
            default_lon, default_lat = 7.3153, 45.7369  # Valle d'Aosta
            if not df_with_map.empty:
                last_act = df_with_map.sort_values("start_date", ascending=False).iloc[0]
                last_coords = polyline_to_coords(last_act["map"]["summary_polyline"])
                if last_coords:
                    default_lon = sum(c[0] for c in last_coords) / len(last_coords)
                    default_lat = sum(c[1] for c in last_coords) / len(last_coords)

            aineva_ex_js = """
    map.addSource('aineva-ex', {
        type: 'raster',
        tiles: ['https://bollettino.aineva.it/geoserver/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=true&LAYERS=cargis:eaws_bulletins&WIDTH=256&HEIGHT=256&SRS=EPSG%3A3857&BBOX={bbox-epsg-3857}'],
        tileSize: 256, attribution: 'AINEVA'
    });
    map.addLayer({ id: 'aineva-ex-layer', type: 'raster',
        source: 'aineva-ex', paint: { 'raster-opacity': 0.5 } });
""" if explore_aineva else ""

            snow_ex_js = """
    map.addSource('snow-ex', {
        type: 'raster',
        tiles: ['https://tiles.opensnowmap.org/pistes/{z}/{x}/{y}.png'],
        tileSize: 256, attribution: 'OpenSnowMap'
    });
    map.addLayer({ id: 'snow-ex-layer', type: 'raster',
        source: 'snow-ex', paint: { 'raster-opacity': 0.7 } });
""" if explore_snow else ""

            html_explore = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet"/>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link rel="stylesheet" href="https://api.mapbox.com/mapbox-gl-js/plugins/mapbox-gl-draw/v1.4.3/mapbox-gl-draw.css"/>
<script src="https://api.mapbox.com/mapbox-gl-js/plugins/mapbox-gl-draw/v1.4.3/mapbox-gl-draw.js"></script>
<style>
  html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#0e1117; }}
  #map-wrap {{ position:relative; width:100%; height:100%; }}
  #map {{ position:absolute; top:0; left:0; width:100%; height:100%; }}
  #controls {{
    position:absolute; top:10px; right:10px; background:rgba(14,17,23,0.90);
    color:#fff; padding:12px 16px; border-radius:10px; font-size:12px;
    font-family:sans-serif; z-index:10; min-width:200px;
  }}
  #controls b {{ display:block; margin-bottom:6px; font-size:13px; }}
  #stats {{ margin-top:8px; line-height:1.9; color:#ccc; }}
  .stat-val {{ color:#fff; font-weight:700; }}
  #hint {{
    position:absolute; bottom:40px; left:50%; transform:translateX(-50%);
    background:rgba(14,17,23,0.85); color:#aaa; padding:6px 14px;
    border-radius:20px; font-size:11px; font-family:sans-serif; z-index:10;
    pointer-events:none;
  }}
  button.clear-btn {{
    margin-top:8px; width:100%; padding:6px; border-radius:6px;
    background:#e94560; color:#fff; border:none; cursor:pointer; font-size:12px;
  }}
  button.clear-btn:hover {{ background:#c73652; }}
</style>
</head><body>
<div id="map-wrap">
  <div id="map"></div>
  <div id="controls">
    <b>✏️ Disegna percorso</b>
    <div style="color:#aaa;font-size:11px;margin-bottom:6px">
      Clicca sulla mappa per aggiungere punti.<br>
      Doppio click per terminare il tracciato.
    </div>
    <div id="stats">
      <div>Punti: <span class="stat-val" id="pt-count">0</span></div>
      <div>Distanza: <span class="stat-val" id="dist-val">0.0 km</span></div>
      <div>D+ stimato: <span class="stat-val" id="elev-val">—</span></div>
    </div>
    <button class="clear-btn" onclick="clearDraw()">🗑️ Cancella tutto</button>
  </div>
  <div id="hint">🖱️ Clicca per aggiungere waypoint • Doppio click per terminare</div>
</div>
<script>
mapboxgl.accessToken = '{MAPBOX_TOKEN}';
const map = new mapboxgl.Map({{
    container: 'map',
    style: 'mapbox://styles/mapbox/{explore_style}',
    center: [{default_lon}, {default_lat}],
    zoom: 11, pitch: 60, bearing: -10, antialias: true
}});
map.addControl(new mapboxgl.NavigationControl({{'visualizePitch':true}}), 'top-left');
map.addControl(new mapboxgl.FullscreenControl(), 'top-left');
map.addControl(new mapboxgl.ScaleControl(), 'bottom-right');
map.dragRotate.enable();
map.touchZoomRotate.enableRotation();
map.scrollZoom.enable();
map.scrollZoom.setWheelZoomRate(1/450);
(function() {{
  const cv = map.getCanvas();
  cv.addEventListener('wheel', (e) => {{ e.stopPropagation(); }}, {{ passive: false }});
  let mid = false, lx = 0, ly = 0;
  cv.addEventListener('mousedown', (e) => {{
    if (e.button === 1) {{ e.preventDefault(); mid = true; lx = e.clientX; ly = e.clientY; cv.style.cursor = 'grab'; }}
  }});
  window.addEventListener('mousemove', (e) => {{
    if (!mid) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    map.jumpTo({{ bearing: map.getBearing() + dx * 0.5,
                  pitch: Math.min(85, Math.max(0, map.getPitch() - dy * 0.4)) }});
  }});
  window.addEventListener('mouseup', (e) => {{ if (e.button === 1) {{ mid = false; cv.style.cursor = ''; }} }});
}})();

const draw = new MapboxDraw({{
    displayControlsDefault: false,
    controls: {{ line_string: true, trash: true }},
    styles: [
        {{ id: 'gl-draw-line', type: 'line', filter: ['all',['==','$type','LineString'],['!=','mode','static']],
           paint: {{ 'line-color': '#e94560', 'line-width': 4, 'line-opacity': 0.9 }} }},
        {{ id: 'gl-draw-line-static', type: 'line', filter: ['all',['==','$type','LineString'],['==','mode','static']],
           paint: {{ 'line-color': '#e94560', 'line-width': 3, 'line-opacity': 0.8 }} }},
        {{ id: 'gl-draw-point-active', type: 'circle', filter: ['all',['==','$type','Point'],['!=','meta','midpoint']],
           paint: {{ 'circle-radius': 5, 'circle-color': '#fff', 'circle-stroke-color': '#e94560', 'circle-stroke-width': 2 }} }},
    ]
}});
map.addControl(draw, 'top-left');

map.on('load', () => {{
    map.addSource('mapbox-dem', {{
        'type': 'raster-dem', 'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
        'tileSize': 512, 'maxzoom': 14
    }});
    map.setTerrain({{'source': 'mapbox-dem', 'exaggeration': 2.0}});
    map.addLayer({{ 'id': 'sky', 'type': 'sky',
        'paint': {{ 'sky-type': 'atmosphere', 'sky-atmosphere-sun': [0.0, 45.0],
                   'sky-atmosphere-sun-intensity': 15 }} }});
    {aineva_ex_js}
    {snow_ex_js}
}});

function haversineKm(lon1, lat1, lon2, lat2) {{
    const R = 6371;
    const dLat = (lat2-lat1)*Math.PI/180;
    const dLon = (lon2-lon1)*Math.PI/180;
    const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
    return R * 2 * Math.asin(Math.sqrt(a));
}}

function updateStats() {{
    const data = draw.getAll();
    let totalKm = 0, points = 0;
    data.features.forEach(f => {{
        if (f.geometry.type === 'LineString') {{
            const coords = f.geometry.coordinates;
            points += coords.length;
            for (let i = 1; i < coords.length; i++) {{
                totalKm += haversineKm(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]);
            }}
        }}
    }});
    document.getElementById('pt-count').textContent = points;
    document.getElementById('dist-val').textContent = totalKm.toFixed(2) + ' km';
    // Nascondi hint dopo primo click
    if (points > 0) document.getElementById('hint').style.display = 'none';
}}

function clearDraw() {{
    draw.deleteAll();
    document.getElementById('pt-count').textContent = '0';
    document.getElementById('dist-val').textContent = '0.0 km';
    document.getElementById('elev-val').textContent = '—';
    document.getElementById('hint').style.display = 'block';
}}

map.on('draw.create', updateStats);
map.on('draw.update', updateStats);
map.on('draw.delete', updateStats);
</script></body></html>"""

            _mb_ex_ok, _mb_ex_reason = mapbox_render_allowed()
            if not _mb_ex_ok:
                st.error(_mb_ex_reason)
            else:
                import streamlit.components.v1 as components
                components.html(html_explore, height=640, scrolling=False)
                mapbox_register_load()

            st.caption("""
            **Come usare la mappa esplorativa:**
            - 🖊️ Clicca il pulsante **Linea** (toolbar sinistra) per iniziare a disegnare
            - Clicca sulla mappa per aggiungere waypoint uno per uno
            - **Doppio click** per terminare il tracciato
            - 🗑️ Usa il pulsante cestino o il bottone rosso per cancellare
            - I layer **AINEVA** e **OpenSnowMap** aiutano a valutare il percorso

            > ⚠️ Il percorso disegnato non viene salvato tra sessioni. Per salvarlo, usa il tasto 🗑️ per cancellarlo e disegnarne uno nuovo, oppure annota le coordinate manualmente.
            """)

    elif menu == "📅 Storico & Calendario":
        st.markdown("## 📅 Storico & Calendario")

        # ── Barra di ricerca ──────────────────────────────────────
        _sc, _scc = st.columns([5, 1])
        with _sc:
            _search_q = st.text_input("🔍 Cerca attività", value="",
                placeholder='Es. "Sirente", "Terminillo", "Giro"...',
                label_visibility="collapsed", key="storico_search_input")
        with _scc:
            if st.button("✕", key="search_clear", use_container_width=True):
                st.rerun()

        if _search_q.strip():
            _q = _search_q.strip().lower()
            _df_found = df[df["name"].str.lower().str.contains(_q, na=False)].copy()
            st.markdown(f"#### 🔍 Risultati per «{_search_q}» — {len(_df_found)} attività trovate")
            if _df_found.empty:
                st.info("Nessuna attività trovata.")
            else:
                for _sri, (_sidx, _sr) in enumerate(_df_found.sort_values("start_date", ascending=False).iterrows()):
                    _ssi = get_sport_info(_sr["type"])
                    _sm  = format_metrics(_sr)
                    _zn_s, _zc_s, _zl_s = get_zone_for_activity(_sr, u["fc_max"])
                    _sc2, _sbc = st.columns([11, 1])
                    with _sc2:
                        st.markdown(f"""
                        <div style="background:{_ssi['color']}10;border-left:4px solid {_ssi['color']};
                                    border-radius:0 10px 10px 0;padding:10px 16px;margin:4px 0;
                                    display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
                            <div style="display:flex;align-items:center;gap:8px">
                                <span style="font-size:20px">{_ssi['icon']}</span>
                                <div>
                                    <div style="font-weight:700;color:#111;font-size:14px">{_sr['name']}</div>
                                    <div style="font-size:11px;color:#555">{_sr['start_date'].strftime('%d %b %Y · %H:%M')} — {_ssi['label']}</div>
                                </div>
                            </div>
                            <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">
                                <span style="font-size:13px;color:#111;font-weight:600">📏 {_sm['dist_str']}</span>
                                <span style="font-size:12px;color:#111">⏱️ {_sm['dur_str']}</span>
                                <span style="font-size:12px;color:#111">⛰️ {_sm['elev']}</span>
                                <span style="font-size:12px;color:#111">❤️ {_sm['hr_avg']}</span>
                                <span style="font-size:12px;color:#111">📊 {_sr['tss']:.0f} TSS</span>
                                <span style="background:{_zc_s}22;color:{_zc_s};border:1px solid {_zc_s}55;
                                             border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700">{_zl_s}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with _sbc:
                        if st.button("🔍", key=f"srch_det_{_sri}", use_container_width=True,
                                     help="Apri dettaglio"):
                            st.session_state.selected_activity_id = _sr.get("id", _sidx)
                            st.rerun()
            st.divider()

        # ── Filtro sport (solo primari + Altro) ──────────────────
        _primary_in_df = [s for s in CALENDAR_FILTER_SPORTS if s in df["type"].values]
        _extra_in_df   = [s for s in df["type"].unique() if s not in CALENDAR_FILTER_SPORTS]
        _filter_opts   = _primary_in_df + (["_altro_"] if _extra_in_df else [])

        if "recap_sport_filter" not in st.session_state:
            st.session_state.recap_sport_filter = set(_filter_opts)

        st.markdown("**Filtra sport:**")
        _ncols = min(len(_filter_opts), 9)
        btn_cols_r = st.columns(_ncols)
        for i, sport in enumerate(_filter_opts):
            si = get_sport_info(sport) if sport != "_altro_" else {"icon": "🏅", "label": f"Altro ({len(_extra_in_df)})", "color": "#9E9E9E"}
            is_active = sport in st.session_state.recap_sport_filter
            with btn_cols_r[i % _ncols]:
                if st.button(f"{si['icon']} {si['label']}", key=f"recap_btn_{sport}",
                             type="primary" if is_active else "secondary", use_container_width=True):
                    sf = st.session_state.recap_sport_filter
                    sf.discard(sport) if sport in sf else sf.add(sport)
                    st.rerun()

        col_all_r, col_none_r, _ = st.columns([1, 1, 7])
        with col_all_r:
            if st.button("✅ Tutti", key="recap_all", use_container_width=True):
                st.session_state.recap_sport_filter = set(_filter_opts)
                st.rerun()
        with col_none_r:
            if st.button("❌ Nessuno", key="recap_none", use_container_width=True):
                st.session_state.recap_sport_filter = set()
                st.rerun()

        _active_primary = [s for s in _primary_in_df if s in st.session_state.recap_sport_filter]
        _include_extra  = "_altro_" in st.session_state.recap_sport_filter
        df_r = df[df["type"].isin(_active_primary + (_extra_in_df if _include_extra else []))].copy()

        st.divider()

        # ── Selezione periodo ──
        tab_cal_main, tab_week, tab_month, tab_year, tab_yoy = st.tabs([
            "📅 Calendario", "📊 Settimana", "🗓️ Mese", "📆 Anno", "↔️ Anno su Anno"
        ])

        with tab_cal_main:
            st.markdown("#### 📅 Calendario Allenamenti")

            # ── State ──
            for _k, _v in [("cal_selected_day", None), ("cal_view", "Mese"),
                            ("cal_month", datetime.now().month), ("cal_year", datetime.now().year)]:
                if _k not in st.session_state:
                    st.session_state[_k] = _v

            # ── Filtri sport ──
            all_sports = sorted(df["type"].unique().tolist())
            if st.session_state.sport_filter is None:
                st.session_state.sport_filter = set(all_sports)

            st.markdown("**Filtra per sport:**")
            _sport_cols = st.columns(min(len(all_sports) + 2, 10))
            for _si, _sp in enumerate(all_sports):
                _sinfo = get_sport_info(_sp)
                with _sport_cols[_si % (len(_sport_cols))]:
                    if st.button(f"{_sinfo['icon']} {_sinfo['label']}", key=f"calsport_{_sp}",
                                 type="primary" if _sp in st.session_state.sport_filter else "secondary",
                                 use_container_width=True):
                        _sf = st.session_state.sport_filter
                        if _sp in _sf: _sf.discard(_sp)
                        else:          _sf.add(_sp)
                        st.rerun()
            _ca, _cn, _ = st.columns([1, 1, 6])
            with _ca:
                if st.button("✅ Tutti",   key="calall",  use_container_width=True):
                    st.session_state.sport_filter = set(all_sports); st.rerun()
            with _cn:
                if st.button("❌ Nessuno", key="calnone", use_container_width=True):
                    st.session_state.sport_filter = set(); st.rerun()

            df_cal = df[df["type"].isin(st.session_state.sport_filter)].copy()
            df_cal["_ds"] = df_cal["start_date"].dt.strftime("%Y-%m-%d")
            acts_by_day = {ds: grp for ds, grp in df_cal.groupby("_ds")}

            st.divider()

            # ── Vista switcher + navigazione ──
            _vc1, _vc2, _vc3, _nav1, _nav2, _nav3 = st.columns([1,1,1,1,2,1])
            for _col, _lbl, _view in [(_vc1,"📅 Settimana","Settimana"),
                                       (_vc2,"🗓️ Mese","Mese"),
                                       (_vc3,"📆 Anno","Anno")]:
                with _col:
                    if st.button(_lbl, key=f"calview_{_view}",
                                 type="primary" if st.session_state.cal_view == _view else "secondary",
                                 use_container_width=True):
                        st.session_state.cal_view = _view
                        st.session_state.cal_selected_day = None
                        st.rerun()

            with _nav1:
                if st.button("◀", key="calprev", use_container_width=True):
                    if st.session_state.cal_view == "Anno":
                        st.session_state.cal_year -= 1
                    elif st.session_state.cal_month == 1:
                        st.session_state.cal_month = 12; st.session_state.cal_year -= 1
                    else:
                        st.session_state.cal_month -= 1
                    st.session_state.cal_selected_day = None; st.rerun()
            with _nav2:
                import calendar as _calmod
                if st.session_state.cal_view == "Anno":
                    _nav_label = str(st.session_state.cal_year)
                else:
                    _nav_label = f"{_calmod.month_name[st.session_state.cal_month]} {st.session_state.cal_year}"
                st.markdown(f"<div style='text-align:center;font-size:17px;font-weight:700;color:#111;padding:7px'>{_nav_label}</div>",
                            unsafe_allow_html=True)
            with _nav3:
                if st.button("▶", key="calnext", use_container_width=True):
                    if st.session_state.cal_view == "Anno":
                        st.session_state.cal_year += 1
                    elif st.session_state.cal_month == 12:
                        st.session_state.cal_month = 1; st.session_state.cal_year += 1
                    else:
                        st.session_state.cal_month += 1
                    st.session_state.cal_selected_day = None; st.rerun()

            st.divider()

            import calendar as _cm

            # ════════════════════════════════════
            # VISTA MESE e SETTIMANA
            # ════════════════════════════════════
            if st.session_state.cal_view in ("Mese", "Settimana"):
                _y = st.session_state.cal_year
                _m = st.session_state.cal_month

                if st.session_state.cal_view == "Settimana":
                    _td  = datetime.now().date()
                    _ws  = _td - timedelta(days=_td.weekday())
                    _weeks = [[_ws + timedelta(days=i) for i in range(7)]]
                    _y, _m = _td.year, _td.month
                else:
                    _weeks = _cm.Calendar(firstweekday=0).monthdatescalendar(_y, _m)

                # Header giorni
                _DAY_NAMES = ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"]
                _hcols = st.columns(7)
                for _di, _dn in enumerate(_DAY_NAMES):
                    _hcols[_di].markdown(
                        f"<div style='text-align:center;font-size:11px;color:#444;font-weight:700'>{_dn}</div>",
                        unsafe_allow_html=True)

                # Griglia settimane
                for _week in _weeks:
                    _wcols = st.columns(7)
                    for _ci, _day in enumerate(_week):
                        with _wcols[_ci]:
                            _ds      = _day.strftime("%Y-%m-%d")
                            _in_m    = (_day.month == _m)
                            _is_tod  = (_day == datetime.now().date())
                            _sel     = (st.session_state.cal_selected_day == _ds)
                            _dacts   = acts_by_day.get(_ds)
                            _has_act = _dacts is not None and not _dacts.empty

                            # Colore bordo/sfondo cella (light theme)
                            if _sel:
                                _bg, _brd = "rgba(233,69,96,0.12)", "2px solid #e94560"
                            elif _is_tod:
                                _bg, _brd = "rgba(33,150,243,0.10)", "2px solid #2196F3"
                            elif not _in_m:
                                _bg, _brd = "transparent", "1px solid transparent"
                            elif _has_act:
                                _bg, _brd = "rgba(0,0,0,0.03)", "1px solid rgba(0,0,0,0.10)"
                            else:
                                _bg, _brd = "rgba(0,0,0,0.01)", "1px solid rgba(0,0,0,0.05)"

                            _num_col = "#e94560" if _is_tod else "#111" if _in_m else "#bbb"

                            # Cella stile Suunto: numero grande + cerchi proporzionali TSS
                            if _has_act and _in_m:
                                if _sel:
                                    _bg2, _brd2 = "rgba(233,69,96,0.10)", "2px solid #e94560"
                                elif _is_tod:
                                    _bg2, _brd2 = "rgba(33,150,243,0.10)", "2px solid #2196F3"
                                else:
                                    _bg2, _brd2 = "transparent", "1px solid rgba(0,0,0,0.06)"
                                _circles = ""
                                for _, _ar in _dacts.head(3).iterrows():
                                    _asi2 = get_sport_info(_ar["type"], _ar.get("name",""))
                                    _tss2 = float(_ar.get("tss") or 0)
                                    _km2  = _ar["distance"] / 1000
                                    _sec2 = _ar["moving_time"]
                                    _h2, _m2 = int(_sec2//3600), int((_sec2%3600)//60)
                                    _d2  = f"{_h2}h{_m2:02d}" if _h2 > 0 else f"{_m2}m"
                                    _sz2 = max(12, min(28, int(_tss2 / 6) + 12))
                                    _circles += (
                                        f'<div title="{_asi2["label"]} · {_km2:.1f}km · {_d2} · TSS {_tss2:.0f}" '
                                        f'style="display:flex;align-items:center;gap:3px;margin-bottom:2px">'
                                        f'<div style="width:{_sz2}px;height:{_sz2}px;border-radius:50%;'
                                        f'background:{_asi2["color"]};flex-shrink:0"></div>'
                                        f'<div style="font-size:10px;color:#333;line-height:1.1;font-weight:500">'
                                        f'{_asi2["icon"]} {_km2:.0f}km'
                                        f'<br><span style="color:#777;font-size:9px">TSS {_tss2:.0f}</span></div>'
                                        f'</div>'
                                    )
                                if len(_dacts) > 3:
                                    _circles += f'<div style="font-size:10px;color:#777">+{len(_dacts)-3}</div>'
                                st.markdown(
                                    f'<div style="background:{_bg2};border:{_brd2};border-radius:12px;'
                                    f'padding:6px 6px 5px;min-height:88px;pointer-events:none;overflow:hidden">'
                                    f'<div style="font-size:15px;font-weight:700;color:{_num_col};line-height:1;padding:0 2px 4px">{_day.day}</div>'
                                    f'<div style="display:flex;flex-direction:column;padding:0 2px">{_circles}</div>'
                                    f'</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<div style="background:{_bg};border:{_brd};border-radius:12px;'
                                    f'padding:6px 4px 5px;min-height:76px;pointer-events:none">'
                                    f'<div style="font-size:16px;font-weight:{"700" if _in_m else "400"};'
                                    f'     color:{_num_col};line-height:1;padding:0 4px">{_day.day if _in_m else ""}</div>'
                                    f'</div>', unsafe_allow_html=True)

                            # Bottone click — solo sui giorni con attività
                            if _has_act and _in_m:
                                _btn_lbl = f"{'●' if not _sel else '○'}"
                                if st.button(_btn_lbl, key=f"cd_{_ds}",
                                             help=f"{len(_dacts)} {'attività' if len(_dacts)>1 else 'attività'} — clicca per dettaglio",
                                             use_container_width=True):
                                    st.session_state.cal_selected_day = None if _sel else _ds
                                    st.rerun()

            # ════════════════════════════════════
            # VISTA ANNO
            # ════════════════════════════════════
            elif st.session_state.cal_view == "Anno":
                _y = st.session_state.cal_year
                for _row_s in range(0, 12, 3):
                    _mcols = st.columns(3)
                    for _mi, _mn in enumerate(range(_row_s+1, _row_s+4)):
                        with _mcols[_mi]:
                            st.markdown(
                                f"<div style='font-size:13px;font-weight:700;color:#333;margin-bottom:4px'>"
                                f"{_cm.month_name[_mn]}</div>",
                                unsafe_allow_html=True)
                            _yr_weeks = _cm.Calendar(firstweekday=0).monthdatescalendar(_y, _mn)
                            for _yw in _yr_weeks:
                                _yc = st.columns(7)
                                for _yci, _yd in enumerate(_yw):
                                    with _yc[_yci]:
                                        if _yd.month != _mn:
                                            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
                                            continue
                                        _yds   = _yd.strftime("%Y-%m-%d")
                                        _ydact = acts_by_day.get(_yds)
                                        _yhas  = _ydact is not None and not _ydact.empty
                                        _ysel  = (st.session_state.cal_selected_day == _yds)
                                        if _yhas:
                                            _ysp   = _ydact["type"].value_counts().index[0]
                                            _ysi   = get_sport_info(_ysp)
                                            _ysz   = 11 if len(_ydact) > 1 else 9
                                            _ybrd  = "2px solid #e94560" if _ysel else "none"
                                            st.markdown(
                                                f'<div title="{_yds}: {len(_ydact)} att." style="width:{_ysz}px;height:{_ysz}px;'
                                                f'border-radius:50%;background:{_ysi["color"]};border:{_ybrd};margin:auto;cursor:pointer"></div>',
                                                unsafe_allow_html=True)
                                            if st.button("·", key=f"yrd_{_yds}",
                                                         help=f"{len(_ydact)} att. — click per dettaglio"):
                                                st.session_state.cal_selected_day = _yds
                                                st.session_state.cal_view  = "Mese"
                                                st.session_state.cal_month = _yd.month
                                                st.session_state.cal_year  = _yd.year
                                                st.rerun()
                                        else:
                                            st.markdown(
                                                f'<div title="{_yds}" style="width:7px;height:7px;border-radius:50%;'
                                                f'background:rgba(0,0,0,0.08);border:1px solid rgba(0,0,0,0.15);margin:auto"></div>',
                                                unsafe_allow_html=True)

            st.divider()

            # ════════════════════════════════════
            # DETTAGLIO GIORNO SELEZIONATO
            # ════════════════════════════════════
            _sel_day = st.session_state.cal_selected_day
            if _sel_day and _sel_day in acts_by_day:
                _sel_acts = acts_by_day[_sel_day]
                st.markdown(f"### 📋 {_sel_day} — {len(_sel_acts)} attività")
                for _cai, (_, _ar) in enumerate(_sel_acts.iterrows()):
                    _asi     = get_sport_info(_ar["type"], _ar.get("name",""))
                    _dist_km = _ar["distance"] / 1000
                    _hrs     = _ar["moving_time"] / 3600
                    _tss_v   = float(_ar.get("tss") or 0)
                    _elev    = float(_ar.get("total_elevation_gain") or 0)
                    _hr_avg  = _ar.get("average_heartrate")
                    _name_a  = _ar.get("name") or _ar["type"]
                    _pace_str = ""
                    if _ar["type"] in ["Run","TrailRun","VirtualRun"] and _dist_km > 0:
                        _ps = _ar["moving_time"] / _dist_km
                        _pace_str = f"🏃 <b>{int(_ps//60)}:{int(_ps%60):02d} /km</b>"
                    _hr_str  = f"❤️ <b>{int(_hr_avg)} bpm</b>" if _hr_avg and pd.notna(_hr_avg) else ""
                    _elev_str = f"⛰️ <b>{int(_elev)} m</b>" if _elev > 0 else ""
                    _pills = "".join([
                        f'<span style="background:rgba(0,0,0,0.06);border-radius:20px;padding:3px 12px;font-size:12px;margin:2px;color:#111">📏 <b>{_dist_km:.1f} km</b></span>',
                        f'<span style="background:rgba(0,0,0,0.06);border-radius:20px;padding:3px 12px;font-size:12px;margin:2px;color:#111">⏱️ <b>{int(_hrs)}h {int((_hrs%1)*60)}m</b></span>',
                        f'<span style="background:rgba(0,0,0,0.06);border-radius:20px;padding:3px 12px;font-size:12px;margin:2px;color:#111">{_pace_str}</span>' if _pace_str else "",
                        f'<span style="background:rgba(0,0,0,0.06);border-radius:20px;padding:3px 12px;font-size:12px;margin:2px;color:#111">{_hr_str}</span>' if _hr_str else "",
                        f'<span style="background:rgba(0,0,0,0.06);border-radius:20px;padding:3px 12px;font-size:12px;margin:2px;color:#111">⚡ <b>{_tss_v:.0f} TSS</b></span>',
                        f'<span style="background:rgba(0,0,0,0.06);border-radius:20px;padding:3px 12px;font-size:12px;margin:2px;color:#111">{_elev_str}</span>' if _elev_str else "",
                    ])
                    _calcc, _calcb = st.columns([10, 1])
                    with _calcc:
                        st.markdown(
                            f'<div style="background:{_asi["color"]}0a;border:1px solid {_asi["color"]}33;'
                            f'border-left:4px solid {_asi["color"]};border-radius:12px;padding:14px 18px;margin:6px 0">'
                            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
                            f'<span style="font-size:22px">{_asi["icon"]}</span>'
                            f'<span style="font-size:16px;font-weight:700;color:#111">{_name_a}</span>'
                            f'<span style="font-size:12px;color:#555">{_ar["start_date"].strftime("%H:%M")}</span>'
                            f'</div><div style="display:flex;flex-wrap:wrap;gap:4px">{_pills}</div></div>',
                            unsafe_allow_html=True)
                    with _calcb:
                        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                        open_activity_button(_ar, key_suffix=f"cal_{_cai}_{_sel_day}")
            elif _sel_day:
                st.info(f"Nessuna attività il {_sel_day} (controlla i filtri sport).")

            st.divider()

            # ── Heatmap consistenza annuale ──
            st.markdown("#### 🟩 Consistenza Annuale")
            _today   = datetime.now().date()
            _ys      = _today.replace(month=1, day=1)
            _wks     = []
            _cw      = []
            _d       = _ys
            _act_set = set(acts_by_day.keys())
            while _d <= _today:
                _dstr = _d.strftime("%Y-%m-%d")
                _dact = acts_by_day.get(_dstr)
                if _dact is not None and not _dact.empty:
                    _top = _dact["type"].value_counts().index[0]
                    _clr = get_sport_info(_top)["color"]
                else:
                    _clr = "rgba(0,0,0,0.07)"
                _brd = "rgba(0,0,0,0.12)"
                _cw.append(f'<div title="{_dstr}" style="width:14px;height:14px;border-radius:3px;'
                           f'background:{_clr};border:1px solid {_brd}"></div>')
                if _d.weekday() == 6:
                    _wks.append("".join(_cw)); _cw = []
                _d += timedelta(days=1)
            if _cw:
                _wks.append("".join(_cw))

            _heat = '<div style="display:flex;gap:3px">'
            for _w in _wks:
                _heat += f'<div style="display:flex;flex-direction:column;gap:2px">{_w}</div>'
            _heat += "</div>"

            _streak = 0
            _d2 = _today
            while _d2.strftime("%Y-%m-%d") in _act_set:
                _streak += 1; _d2 -= timedelta(days=1)
            _total  = (_today - _ys).days + 1
            _active = sum(1 for _dd in _act_set if str(_ys) <= _dd <= str(_today))

            _h1, _h2, _h3 = st.columns(3)
            _h1.metric("🔥 Streak attuale",      f"{_streak} giorni")
            _h2.metric("📅 Giorni attivi (anno)", _active)
            _h3.metric("📊 % Consistenza",        f"{_active/_total*100:.0f}%")
            st.markdown(_heat, unsafe_allow_html=True)

            # Legenda
            _legs = []
            for _sp in sorted(df["type"].unique()):
                _sil = get_sport_info(_sp)
                _legs.append(f'<span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;color:#888;margin:2px 6px">'
                             f'<div style="width:10px;height:10px;border-radius:50%;background:{_sil["color"]}"></div>'
                             f'{_sil["icon"]} {_sil["label"]}</span>')
            st.markdown('<div style="display:flex;flex-wrap:wrap;margin-top:6px">' + "".join(_legs) + "</div>",
                        unsafe_allow_html=True)



        # ──────────────────────────────────────────────
        # HELPER: genera KPI cards + grafici per un df
        # ──────────────────────────────────────────────
        def recap_kpi_row(df_sub, label_periodo):
            """Mostra 6 KPI + delta vs periodo precedente."""
            n_sess  = len(df_sub)
            km      = df_sub["distance"].sum() / 1000
            ore     = df_sub["moving_time"].sum() / 3600
            tss_tot = df_sub["tss"].sum()
            elev    = df_sub["total_elevation_gain"].sum() or 0
            hr_vals = df_sub["average_heartrate"].dropna()
            fc_med  = hr_vals.mean() if not hr_vals.empty else None
            calorie = df_sub["kilojoules"].sum() or 0

            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("Sessioni",   n_sess)
            c2.metric("Km totali",  f"{km:.1f}")
            c3.metric("Ore",        f"{ore:.1f}")
            c4.metric("TSS",        f"{tss_tot:.0f}")
            c5.metric("Dislivello", f"{elev:.0f} m")
            c6.metric("FC media",   f"{fc_med:.0f} bpm" if fc_med else "N/A")
            c7.metric("Calorie",    f"{calorie:.0f} kcal" if calorie else "N/A")

        def sport_breakdown_chart(df_sub, height=200, compare_df=None):
            """Card lista per sport — stesso stile Dashboard."""
            if df_sub.empty:
                st.info("Nessuna attività nel periodo selezionato.")
                return
            for sp in df_sub["type"].value_counts().index:
                si      = get_sport_info(sp)
                sp_df   = df_sub[df_sub["type"] == sp]
                n_s     = len(sp_df)
                km_s    = sp_df["distance"].sum() / 1000
                ore_s   = sp_df["moving_time"].sum() / 3600
                tss_s   = sp_df["tss"].sum()
                elev_s  = float(sp_df["total_elevation_gain"].sum() or 0)
                hr_s    = sp_df["average_heartrate"].dropna()
                fc_s    = f"{hr_s.mean():.0f}" if not hr_s.empty else "—"
                # delta vs compare_df
                dkm_html = ""
                if compare_df is not None and not compare_df.empty:
                    prev_sp = compare_df[compare_df["type"] == sp]
                    prev_km = prev_sp["distance"].sum() / 1000
                    dkm     = km_s - prev_km
                    dcol    = "#2e7d32" if dkm >= 0 else "#c62828"
                    dkm_html = f'<span style="font-size:11px;color:{dcol}">{dkm:+.1f} km vs prec.</span>'
                st.markdown(f"""
                <div style="background:{si['color']}14;border-left:4px solid {si['color']};
                            border-radius:0 12px 12px 0;padding:12px 18px;margin:5px 0;
                            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
                    <div style="display:flex;align-items:center;gap:10px;min-width:150px">
                        <span style="font-size:24px">{si['icon']}</span>
                        <div>
                            <div style="font-weight:700;color:{si['color']};font-size:14px">{si['label']}</div>
                            <div style="font-size:11px;color:#444">{n_s} {'sessione' if n_s==1 else 'sessioni'}</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center">
                        <div style="text-align:center">
                            <div style="font-size:17px;font-weight:700;color:#111">{km_s:.1f}</div>
                            <div style="font-size:10px;color:#555">km</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:17px;font-weight:700;color:#111">{int(ore_s)}h {int((ore_s%1)*60)}m</div>
                            <div style="font-size:10px;color:#555">durata</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:17px;font-weight:700;color:#111">{tss_s:.0f}</div>
                            <div style="font-size:10px;color:#555">TSS</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:17px;font-weight:700;color:#111">{int(elev_s)}</div>
                            <div style="font-size:10px;color:#555">↑ m</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-size:17px;font-weight:700;color:#111">{fc_s}</div>
                            <div style="font-size:10px;color:#555">FC avg</div>
                        </div>
                    </div>
                    <div>{dkm_html}</div>
                </div>""", unsafe_allow_html=True)

        def weekly_bars_chart(df_sub, n_weeks=12, height=220):
            """Barre volume settimanale con colori per sport."""
            if df_sub.empty: return
            df_w = df_sub.copy()
            df_w["week"] = df_w["start_date"].dt.to_period("W").dt.start_time
            sports_in_sub = df_w["type"].unique()
            fig = go.Figure()
            for sport in sports_in_sub:
                si  = get_sport_info(sport)
                sub = df_w[df_w["type"] == sport].groupby("week")["distance"].sum() / 1000
                sub = sub.reindex(
                    pd.date_range(df_w["week"].min(), df_w["week"].max(), freq="W-MON"),
                    fill_value=0
                ).tail(n_weeks)
                fig.add_trace(go.Bar(
                    name=f"{si['icon']} {si['label']}",
                    x=sub.index, y=sub.values,
                    marker_color=si["color"], opacity=0.85,
                ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                barmode="stack", height=height, margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h", y=1.1, font=dict(size=11)),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d/%m"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="km"),
            )
            st.plotly_chart(fig, use_container_width=True)

        def activity_list_table(df_sub, n=10, sort_by="distance"):
            """Lista attività cliccabili con pulsante dettaglio."""
            if df_sub.empty:
                st.info("Nessuna attività nel periodo selezionato.")
                return
            top = df_sub.nlargest(n, sort_by).copy()
            for _ti, (_, _tr) in enumerate(top.iterrows()):
                _si = get_sport_info(_tr["type"])
                _mi = format_metrics(_tr)
                _zn, _zc, _zl = get_zone_for_activity(_tr, u["fc_max"])
                _is_bike = _tr["type"] in ("Ride","VirtualRide","MountainBikeRide")
                _w_badge = (f" · ⚡{_mi['watts']}" + (" <span style='font-size:9px;color:#FF9800'>est.</span>"
                    if not _tr.get("device_watts",False) else "")) if _mi["watts"] != "N/A" else ""
                _cc, _cb = st.columns([11, 1])
                with _cc:
                    st.markdown(f"""
                    <div style="background:{_si['color']}08;border-left:4px solid {_si['color']};
                                border-radius:0 8px 8px 0;padding:8px 14px;margin:3px 0;
                                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
                        <div style="display:flex;align-items:center;gap:8px;min-width:180px">
                            <span style="font-size:18px">{_si['icon']}</span>
                            <div>
                                <div style="font-weight:700;color:#111;font-size:13px">{_tr['name']}</div>
                                <div style="font-size:11px;color:#555">{_tr['start_date'].strftime('%d %b %Y')} · {_si['label']}</div>
                            </div>
                        </div>
                        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
                            <span style="font-size:12px;color:#111;font-weight:600">📏 {_mi['dist_str']}</span>
                            <span style="font-size:12px;color:#111">⏱️ {_mi['dur_str']}</span>
                            <span style="font-size:12px;color:#111">⛰️ {_mi['elev']}</span>
                            <span style="font-size:12px;color:#111">❤️ {_mi['hr_avg']}</span>
                            <span style="font-size:12px;color:#111">{_w_badge}</span>
                            <span style="font-size:12px;color:#111">📊 {_tr['tss']:.0f} TSS</span>
                            <span style="background:{_zc}22;color:{_zc};border:1px solid {_zc}55;
                                         border-radius:4px;padding:1px 6px;font-size:10px;font-weight:700">{_zl}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with _cb:
                    open_activity_button(_tr, key_suffix=f"tbl_{_ti}")

        # ────────────────────────────────────
        # TAB 1 — SETTIMANA CORRENTE
        # ────────────────────────────────────
        with tab_week:
            now = datetime.now()
            # Lunedì della settimana corrente
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            df_week = df_r[df_r["start_date"] >= week_start]

            # Settimana precedente per delta
            prev_week_start = week_start - timedelta(weeks=1)
            df_prev_week    = df_r[(df_r["start_date"] >= prev_week_start) & (df_r["start_date"] < week_start)]

            st.markdown(f"#### Settimana {week_start.strftime('%d %b')} — {now.strftime('%d %b %Y')}")

            # KPI con delta
            n1,n2,n3,n4,n5,n6 = st.columns(6)
            def delta_str(curr, prev):
                d = curr - prev
                return f"{d:+.1f}" if abs(d) > 0.05 else None

            n1.metric("Sessioni",   len(df_week),
                      delta=str(len(df_week)-len(df_prev_week)) if len(df_prev_week) else None)
            n2.metric("Km",         f"{df_week['distance'].sum()/1000:.1f}",
                      delta=delta_str(df_week['distance'].sum()/1000, df_prev_week['distance'].sum()/1000))
            n3.metric("Ore",        f"{df_week['moving_time'].sum()/3600:.1f}",
                      delta=delta_str(df_week['moving_time'].sum()/3600, df_prev_week['moving_time'].sum()/3600))
            n4.metric("TSS",        f"{df_week['tss'].sum():.0f}",
                      delta=delta_str(df_week['tss'].sum(), df_prev_week['tss'].sum()))
            n5.metric("Dislivello", f"{(df_week['total_elevation_gain'].sum() or 0):.0f} m")
            hr_w = df_week["average_heartrate"].dropna()
            n6.metric("FC media",   f"{hr_w.mean():.0f} bpm" if not hr_w.empty else "N/A")

            if not df_week.empty:
                st.markdown("##### Distribuzione sport")
                sport_breakdown_chart(df_week, compare_df=df_prev_week)

                # Progress bar verso obiettivo settimanale (km)
                target_km = st.number_input("🎯 Obiettivo km settimana", value=50, min_value=0, step=5, key="target_km_w")
                curr_km   = df_week["distance"].sum() / 1000
                prog      = min(curr_km / target_km, 1.0) if target_km > 0 else 0
                prog_color = "#4CAF50" if prog >= 1 else "#FF9800" if prog >= 0.6 else "#2196F3"
                st.markdown(f"""
                <div style='margin:8px 0 4px'>
                    <div style='font-size:13px;color:#888'>Progressione km: {curr_km:.1f} / {target_km} km</div>
                    <div style='background:rgba(255,255,255,0.08);border-radius:8px;height:12px;margin-top:6px'>
                        <div style='background:{prog_color};width:{prog*100:.1f}%;height:12px;border-radius:8px;transition:width 0.4s'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("##### Attività della settimana")
                activity_list_table(df_week, n=20)
            else:
                st.info("Nessuna attività questa settimana.")

        # ────────────────────────────────────
        # TAB 2 — MESE CORRENTE
        # ────────────────────────────────────
        with tab_month:
            # Selezione mese
            available_months = df_r["start_date"].dt.to_period("M").unique()
            available_months_str = [str(m) for m in sorted(available_months, reverse=True)]
            sel_month = st.selectbox("Mese:", available_months_str, index=0, key="sel_month")
            sel_period = pd.Period(sel_month, "M")

            df_month_r = df_r[df_r["start_date"].dt.to_period("M") == sel_period]

            # Stesso mese anno precedente per confronto
            prev_year_period = pd.Period(f"{sel_period.year - 1}-{sel_period.month:02d}", "M")
            df_month_py      = df_r[df_r["start_date"].dt.to_period("M") == prev_year_period]

            st.markdown(f"#### {sel_period.strftime('%B %Y')}")

            m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
            km_m   = df_month_r["distance"].sum()/1000
            km_mpy = df_month_py["distance"].sum()/1000
            ore_m  = df_month_r["moving_time"].sum()/3600
            ore_mpy= df_month_py["moving_time"].sum()/3600
            tss_m  = df_month_r["tss"].sum()
            tss_mpy= df_month_py["tss"].sum()

            m1.metric("Sessioni",   len(df_month_r), delta=str(len(df_month_r)-len(df_month_py)) if not df_month_py.empty else None)
            m2.metric("Km",         f"{km_m:.1f}",   delta=f"{km_m-km_mpy:+.1f}" if not df_month_py.empty else None)
            m3.metric("Ore",        f"{ore_m:.1f}",  delta=f"{ore_m-ore_mpy:+.1f}" if not df_month_py.empty else None)
            m4.metric("TSS",        f"{tss_m:.0f}",  delta=f"{tss_m-tss_mpy:+.0f}" if not df_month_py.empty else None)
            m5.metric("Dislivello", f"{(df_month_r['total_elevation_gain'].sum() or 0):.0f} m")
            hr_mo = df_month_r["average_heartrate"].dropna()
            m6.metric("FC media",   f"{hr_mo.mean():.0f}" if not hr_mo.empty else "N/A")
            cal_mo = df_month_r["kilojoules"].sum() or 0
            m7.metric("Calorie",    f"{cal_mo:.0f}" if cal_mo else "N/A")

            if not df_month_r.empty:
                st.markdown("##### Distribuzione sport")
                sport_breakdown_chart(df_month_r, compare_df=df_month_py)

                # Grafico giornaliero del mese
                st.markdown("##### Volume giornaliero (km)")
                df_daily_m = df_month_r.copy()
                df_daily_m["day"] = df_daily_m["start_date"].dt.date
                daily_km = df_daily_m.groupby(["day","type"])["distance"].sum().reset_index()
                daily_km["km"] = daily_km["distance"] / 1000
                fig_dm = go.Figure()
                for sport in daily_km["type"].unique():
                    si  = get_sport_info(sport)
                    sub = daily_km[daily_km["type"] == sport]
                    fig_dm.add_trace(go.Bar(
                        x=sub["day"], y=sub["km"],
                        name=f"{si['icon']} {si['label']}",
                        marker_color=si["color"], opacity=0.85,
                    ))
                fig_dm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="stack", height=220, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h", y=1.1, font=dict(size=11)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%d"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="km"),
                )
                st.plotly_chart(fig_dm, use_container_width=True)

                # Heatmap attività del mese
                st.markdown("##### Giorni di allenamento")
                import calendar as _cal
                first_wd, n_days = _cal.monthrange(sel_period.year, sel_period.month)
                active_days_m = set(df_month_r["start_date"].dt.day.tolist())
                cal_html = "<div style='display:flex;gap:3px;flex-wrap:wrap;margin:8px 0'>"
                # Offset lunedì
                for _ in range(first_wd):
                    cal_html += "<div style='width:32px;height:32px'></div>"
                for day in range(1, n_days + 1):
                    active = day in active_days_m
                    bg     = "#4CAF50" if active else "rgba(255,255,255,0.05)"
                    cal_html += (
                        f"<div style='width:32px;height:32px;border-radius:6px;"
                        f"background:{bg};display:flex;align-items:center;"
                        f"justify-content:center;font-size:11px;"
                        f"color:{'#fff' if active else '#555'}'>{day}</div>"
                    )
                cal_html += "</div>"
                c_cal, c_stat = st.columns([2,1])
                with c_cal:
                    st.markdown(cal_html, unsafe_allow_html=True)
                with c_stat:
                    st.metric("Giorni attivi", f"{len(active_days_m)} / {n_days}")
                    st.metric("% mese", f"{len(active_days_m)/n_days*100:.0f}%")
                    if not df_month_py.empty:
                        st.metric("vs stesso mese anno prec.", f"{len(active_days_m) - len(set(df_month_py['start_date'].dt.day.tolist())):+d} giorni")

                st.markdown("##### Top attività del mese")
                activity_list_table(df_month_r, n=10)
            else:
                st.info("Nessuna attività in questo mese.")

        # ────────────────────────────────────
        # TAB 3 — ANNO CORRENTE
        # ────────────────────────────────────
        with tab_year:
            available_years = sorted(df_r["start_date"].dt.year.unique(), reverse=True)
            sel_year = st.selectbox("Anno:", available_years, index=0, key="sel_year")
            df_year_r = df_r[df_r["start_date"].dt.year == sel_year]

            st.markdown(f"#### Anno {sel_year}")

            y1,y2,y3,y4,y5,y6,y7 = st.columns(7)
            km_y  = df_year_r["distance"].sum()/1000
            ore_y = df_year_r["moving_time"].sum()/3600
            tss_y = df_year_r["tss"].sum()
            elev_y= df_year_r["total_elevation_gain"].sum() or 0
            hr_y  = df_year_r["average_heartrate"].dropna()
            cal_y = df_year_r["kilojoules"].sum() or 0

            y1.metric("Sessioni",    len(df_year_r))
            y2.metric("Km totali",   f"{km_y:.0f}")
            y3.metric("Ore totali",  f"{ore_y:.0f}")
            y4.metric("TSS totale",  f"{tss_y:.0f}")
            y5.metric("Dislivello",  f"{elev_y/1000:.1f} km")
            y6.metric("FC media",    f"{hr_y.mean():.0f}" if not hr_y.empty else "N/A")
            y7.metric("Calorie",     f"{cal_y:.0f}" if cal_y else "N/A")

            if not df_year_r.empty:
                st.markdown("##### Distribuzione sport")
                sport_breakdown_chart(df_year_r)

                # Volume mensile per sport (barre stacked)
                st.markdown("##### Volume mensile (km)")
                df_year_r2 = df_year_r.copy()
                df_year_r2["month"] = df_year_r2["start_date"].dt.to_period("M").dt.start_time
                monthly = df_year_r2.groupby(["month","type"])["distance"].sum().reset_index()
                monthly["km"] = monthly["distance"] / 1000
                fig_my = go.Figure()
                for sport in monthly["type"].unique():
                    si  = get_sport_info(sport)
                    sub = monthly[monthly["type"] == sport]
                    fig_my.add_trace(go.Bar(
                        x=sub["month"], y=sub["km"],
                        name=f"{si['icon']} {si['label']}",
                        marker_color=si["color"], opacity=0.85,
                    ))
                fig_my.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="stack", height=280, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h", y=1.08, font=dict(size=11)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat="%b"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="km"),
                )
                st.plotly_chart(fig_my, use_container_width=True)

                # Sessioni per settimana (heatmap stile GitHub)
                st.markdown("##### Heatmap attività annuale")
                year_start_d = datetime(sel_year, 1, 1).date()
                year_end_d   = datetime(sel_year, 12, 31).date()
                active_dates_y = set(df_year_r["start_date"].dt.date.astype(str).tolist())
                tss_by_date    = df_year_r.groupby(df_year_r["start_date"].dt.date.astype(str))["tss"].sum().to_dict()

                # Costruzione griglia ISO settimane
                d = year_start_d
                weeks_data: list = []
                current_wk: list = []
                wd_offset = d.weekday()
                for _ in range(wd_offset):
                    current_wk.append(None)
                while d <= year_end_d:
                    current_wk.append(d)
                    if d.weekday() == 6:
                        weeks_data.append(current_wk)
                        current_wk = []
                    d += timedelta(days=1)
                if current_wk:
                    weeks_data.append(current_wk)

                heat_html = "<div style='display:flex;gap:2px;overflow-x:auto'>"
                for week in weeks_data:
                    heat_html += "<div style='display:flex;flex-direction:column;gap:2px'>"
                    for day_d in week:
                        if day_d is None:
                            heat_html += "<div style='width:12px;height:12px'></div>"
                        else:
                            ds    = str(day_d)
                            tss_d = tss_by_date.get(ds, 0)
                            if tss_d == 0:      bg = "rgba(255,255,255,0.05)"
                            elif tss_d < 50:    bg = "#1b5e20"
                            elif tss_d < 100:   bg = "#388e3c"
                            elif tss_d < 150:   bg = "#66bb6a"
                            else:               bg = "#a5d6a7"
                            heat_html += (
                                f"<div title='{ds} — TSS {tss_d:.0f}' "
                                f"style='width:12px;height:12px;border-radius:2px;"
                                f"background:{bg}'></div>"
                            )
                    heat_html += "</div>"
                heat_html += "</div>"
                heat_html += """
                <div style='display:flex;gap:8px;align-items:center;margin-top:6px;font-size:11px;color:#666'>
                    <span>Meno</span>
                    <div style='width:12px;height:12px;border-radius:2px;background:rgba(255,255,255,0.05)'></div>
                    <div style='width:12px;height:12px;border-radius:2px;background:#1b5e20'></div>
                    <div style='width:12px;height:12px;border-radius:2px;background:#388e3c'></div>
                    <div style='width:12px;height:12px;border-radius:2px;background:#66bb6a'></div>
                    <div style='width:12px;height:12px;border-radius:2px;background:#a5d6a7'></div>
                    <span>Più TSS</span>
                </div>"""
                st.markdown(heat_html, unsafe_allow_html=True)

                # Medie mensili
                st.divider()
                st.markdown("##### Medie mensili")
                avg_sess = len(df_year_r) / 12
                avg_km   = km_y / 12
                avg_ore  = ore_y / 12
                avg_tss  = tss_y / 12
                a1,a2,a3,a4 = st.columns(4)
                a1.metric("Sessioni/mese", f"{avg_sess:.1f}")
                a2.metric("Km/mese",       f"{avg_km:.0f}")
                a3.metric("Ore/mese",      f"{avg_ore:.0f}")
                a4.metric("TSS/mese",      f"{avg_tss:.0f}")
            else:
                st.info(f"Nessuna attività nel {sel_year}.")

        # ────────────────────────────────────
        # TAB 4 — CONFRONTO ANNO SU ANNO
        # ────────────────────────────────────
        with tab_yoy:
            st.markdown("#### ↔️ Confronto Anno su Anno")
            available_years_yoy = sorted(df_r["start_date"].dt.year.unique(), reverse=True)

            if len(available_years_yoy) < 2:
                st.info("Servono almeno 2 anni di dati per il confronto.")
            else:
                col_y1, col_y2 = st.columns(2)
                with col_y1:
                    year_a = st.selectbox("Anno A:", available_years_yoy, index=0, key="yoy_a")
                with col_y2:
                    year_b = st.selectbox("Anno B:", available_years_yoy,
                                           index=min(1, len(available_years_yoy)-1), key="yoy_b")

                df_ya = df_r[df_r["start_date"].dt.year == year_a]
                df_yb = df_r[df_r["start_date"].dt.year == year_b]

                # KPI confronto
                def yoy_metric(label, val_a, val_b, fmt=".0f"):
                    delta = val_a - val_b
                    pct   = (delta / val_b * 100) if val_b else 0
                    color = "#4CAF50" if delta >= 0 else "#F44336"
                    return f"""
                    <div style='background:rgba(255,255,255,0.03);border-radius:10px;padding:12px;text-align:center'>
                        <div style='color:#888;font-size:12px'>{label}</div>
                        <div style='font-size:20px;font-weight:700;color:#fff'>{val_a:{fmt}}</div>
                        <div style='font-size:11px;color:#666'>{year_b}: {val_b:{fmt}}</div>
                        <div style='font-size:13px;font-weight:700;color:{color}'>{delta:+{fmt}} ({pct:+.1f}%)</div>
                    </div>"""

                st.markdown(f"### {year_a} vs {year_b}")
                cols_yoy = st.columns(5)
                metrics_yoy = [
                    ("Sessioni",    len(df_ya),                      len(df_yb),                      "d"),
                    ("Km totali",   df_ya["distance"].sum()/1000,    df_yb["distance"].sum()/1000,    ".0f"),
                    ("Ore totali",  df_ya["moving_time"].sum()/3600, df_yb["moving_time"].sum()/3600, ".0f"),
                    ("TSS totale",  df_ya["tss"].sum(),              df_yb["tss"].sum(),              ".0f"),
                    ("Dislivello",  df_ya["total_elevation_gain"].sum() or 0,
                                    df_yb["total_elevation_gain"].sum() or 0, ".0f"),
                ]
                for i, (label, va, vb, fmt) in enumerate(metrics_yoy):
                    cols_yoy[i].markdown(yoy_metric(label, va, vb, fmt), unsafe_allow_html=True)

                st.divider()

                # Grafico mensile sovrapposto — km
                st.markdown("##### Confronto volume mensile (km)")
                months_range = range(1, 13)
                months_label = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"]

                km_ya_m = [df_ya[df_ya["start_date"].dt.month == m]["distance"].sum()/1000 for m in months_range]
                km_yb_m = [df_yb[df_yb["start_date"].dt.month == m]["distance"].sum()/1000 for m in months_range]

                fig_yoy = go.Figure()
                fig_yoy.add_trace(go.Scatter(
                    x=months_label, y=km_ya_m, name=str(year_a),
                    line=dict(color="#e94560", width=2.5), mode="lines+markers",
                    fill="tozeroy", fillcolor="rgba(233,69,96,0.08)",
                ))
                fig_yoy.add_trace(go.Scatter(
                    x=months_label, y=km_yb_m, name=str(year_b),
                    line=dict(color="#2196F3", width=2, dash="dot"), mode="lines+markers",
                ))
                fig_yoy.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=280, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h", y=1.08),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="km"),
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

                # Confronto TSS mensile
                st.markdown("##### Confronto TSS mensile")
                tss_ya_m = [df_ya[df_ya["start_date"].dt.month == m]["tss"].sum() for m in months_range]
                tss_yb_m = [df_yb[df_yb["start_date"].dt.month == m]["tss"].sum() for m in months_range]

                fig_tss_yoy = go.Figure()
                fig_tss_yoy.add_trace(go.Bar(x=months_label, y=tss_ya_m, name=str(year_a),
                                              marker_color="#e94560", opacity=0.8))
                fig_tss_yoy.add_trace(go.Bar(x=months_label, y=tss_yb_m, name=str(year_b),
                                              marker_color="#2196F3", opacity=0.6))
                fig_tss_yoy.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="group", height=240, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h", y=1.08),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="TSS"),
                )
                st.plotly_chart(fig_tss_yoy, use_container_width=True)

                # Sessioni per sport confronto
                st.markdown("##### Sessioni per sport")
                sports_union = set(df_ya["type"].unique()) | set(df_yb["type"].unique())
                sport_comp = []
                for sport in sports_union:
                    si = get_sport_info(sport)
                    sport_comp.append({
                        "Sport": f"{si['icon']} {si['label']}",
                        f"Sessioni {year_a}": len(df_ya[df_ya["type"]==sport]),
                        f"Km {year_a}":       round(df_ya[df_ya["type"]==sport]["distance"].sum()/1000, 1),
                        f"Sessioni {year_b}": len(df_yb[df_yb["type"]==sport]),
                        f"Km {year_b}":       round(df_yb[df_yb["type"]==sport]["distance"].sum()/1000, 1),
                    })
                sport_comp_df = pd.DataFrame(sport_comp).sort_values(f"Km {year_a}", ascending=False)
                st.dataframe(sport_comp_df, use_container_width=True, hide_index=True)

    # ============================================================
    # CALENDARIO
    # ============================================================
    # ============================================================
    # COACH CHAT
    # ============================================================

    # ============================================================
    # PLANNING ROUTE
    # ============================================================
    elif menu == "🗺️ Planning Route":
        st.markdown("## 🗺️ Planning Route")
        st.caption("Pianifica percorsi per Trail Running, Scialpinismo, Bici da Strada e MTB")

        # ── Costanti ──
        DEFAULT_LAT, DEFAULT_LON = 42.0369, 13.4256   # Avezzano (AQ)
        ORS_PROFILES = {
            "🏔️ Trail Running":  "foot-hiking",
            "🎿 Scialpinismo":   "foot-hiking",
            "🚴 Bici da Strada": "cycling-road",
            "🚵 MTB":            "cycling-mountain",
        }

        # ── Session state init ──
        for _k, _v in [
            ("pr_sport",       "🏔️ Trail Running"),
            ("pr_waypoints",   []),
            ("pr_routes",      []),     # lista di route generate (coordinate + stats)
            ("pr_active_idx",  0),
            ("pr_gpx_loaded",  None),
            ("pr_chat_hist",   []),
            ("pr_show_heat",   True),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        # ── Layout: sidebar sinistra + mappa destra ──
        col_ctrl, col_map = st.columns([1, 2])

        with col_ctrl:
            # ── Selettore sport ──
            _sport = st.selectbox("🏅 Sport", list(ORS_PROFILES.keys()),
                                   index=list(ORS_PROFILES.keys()).index(st.session_state.pr_sport),
                                   key="pr_sport_sel")
            st.session_state.pr_sport = _sport

            # ── Toggle heatmap ──
            _show_heat = st.toggle("🔥 Mostra Heatmap Strava", value=st.session_state.pr_show_heat,
                                    key="pr_heat_tog")
            st.session_state.pr_show_heat = _show_heat

            st.divider()

            # ── Tab: AI Chat / Manuale / GPX ──
            _ptab_ai, _ptab_man, _ptab_gpx = st.tabs(["🤖 Chiedi all'AI", "✏️ Crea Manuale", "📂 Carica GPX"])

            with _ptab_ai:
                st.markdown("**Descrivi il percorso che vuoi:**")
                st.caption("Es: *Voglio fare 35km e 1800m attorno al Monte Sirente* oppure *Un giro MTB da Avezzano che passi per Celano e Ovindoli*")

                # Storico chat
                for _msg in st.session_state.pr_chat_hist[-6:]:
                    _role_icon = "🧑" if _msg["role"] == "user" else "🤖"
                    st.markdown(f"**{_role_icon}** {_msg['content'][:200]}{'...' if len(_msg['content'])>200 else ''}")

                _pr_input = st.text_area("La tua richiesta:", height=90, key="pr_chat_input",
                                          placeholder="Es: voglio fare un trail di 20km con 1200m di dislivello partendo da Rocca di Cambio...")
                if st.button("🗺️ Genera Percorsi", use_container_width=True, key="pr_gen_btn") and _pr_input.strip():
                    with st.spinner("L'AI sta pianificando 3 varianti..."):
                        try:
                            # Step 1: AI estrae parametri
                            _extract_prompt = f"""Sei un esperto di pianificazione percorsi outdoor in Abruzzo e dintorni.
Dall'utente: "{_pr_input}"
Sport selezionato: {_sport}
Posizione default se non specificata: Avezzano (AQ), coordinate 42.0369, 13.4256

Estrai i parametri e rispondi SOLO con JSON (nessun testo prima/dopo):
{{
  "start_lat": float,
  "start_lon": float,
  "target_km": float or null,
  "target_elev_m": float or null,
  "waypoints_text": ["nome1", "nome2"],
  "loop": true/false,
  "area_hint": "descrizione area",
  "difficulty": "facile/medio/difficile",
  "notes": "note percorso"
}}
Se non specificato: start = Avezzano, target_km = 20, loop = true."""

                            _raw_params = ai_fast(_extract_prompt)
                            import json, re as _re
                            _clean = _re.sub(r"```(?:json)?|```", "", _raw_params).strip()
                            try:
                                _params = json.loads(_clean)
                            except Exception:
                                _params = {"start_lat": DEFAULT_LAT, "start_lon": DEFAULT_LON,
                                           "target_km": 20, "target_elev_m": None,
                                           "waypoints_text": [], "loop": True,
                                           "area_hint": _pr_input, "difficulty": "medio", "notes": ""}

                            _slat = float(_params.get("start_lat") or DEFAULT_LAT)
                            _slon = float(_params.get("start_lon") or DEFAULT_LON)
                            _tkm  = float(_params.get("target_km") or 20)
                            _tel  = _params.get("target_elev_m")
                            _loop = _params.get("loop", True)

                            # Step 2: genera 3 varianti via ORS
                            _ors_profile = ORS_PROFILES[_sport]
                            _routes_found = []

                            # Variante 1: andata e ritorno circolare (se loop)
                            # Variante 2: con waypoint intermedi diversi
                            # Variante 3: profilo altimetrico diverso
                            _variants = [
                                {"name": "Variante A — Circolare Principale",
                                 "options": {"round_trip": {"length": _tkm * 1000, "points": 3, "seed": 42}}},
                                {"name": "Variante B — Percorso Alternativo",
                                 "options": {"round_trip": {"length": _tkm * 1000, "points": 5, "seed": 137}}},
                                {"name": "Variante C — Giro Più Corto",
                                 "options": {"round_trip": {"length": _tkm * 800, "points": 3, "seed": 7}}},
                            ]

                            if ORS_API_KEY:
                                import requests as _req
                                for _var in _variants:
                                    try:
                                        _ors_url = "https://api.openrouteservice.org/v2/directions/" + _ors_profile + "/geojson"
                                        _ors_body = {
                                            "coordinates": [[_slon, _slat]],
                                            "options": _var["options"],
                                            "elevation": True,
                                            "instructions": False,
                                        }
                                        _ors_resp = _req.post(_ors_url, json=_ors_body,
                                            headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
                                            timeout=15)
                                        if _ors_resp.status_code == 200:
                                            _gj = _ors_resp.json()
                                            _feat = _gj["features"][0]
                                            _seg  = _feat["properties"]["segments"][0]
                                            _coords = _feat["geometry"]["coordinates"]  # [lon, lat, elev]
                                            _dist_km = _seg["distance"] / 1000
                                            _dur_min = _seg["duration"] / 60
                                            # Calcola dislivello positivo
                                            _ascent = 0
                                            if len(_coords[0]) >= 3:
                                                for _ci in range(1, len(_coords)):
                                                    _dh = _coords[_ci][2] - _coords[_ci-1][2]
                                                    if _dh > 0: _ascent += _dh
                                            _routes_found.append({
                                                "name": _var["name"],
                                                "coords": [[c[1], c[0]] for c in _coords],  # lat,lon
                                                "coords_3d": _coords,
                                                "dist_km": round(_dist_km, 1),
                                                "ascent_m": round(_ascent),
                                                "dur_min": round(_dur_min),
                                                "params": _params,
                                            })
                                        else:
                                            _routes_found.append({
                                                "name": _var["name"] + " (routing N/D)",
                                                "coords": [[_slat, _slon]],
                                                "coords_3d": [],
                                                "dist_km": _tkm, "ascent_m": 0, "dur_min": 0,
                                                "params": _params,
                                                "error": f"ORS {_ors_resp.status_code}",
                                            })
                                    except Exception as _e:
                                        _routes_found.append({
                                            "name": _var["name"] + " (errore)",
                                            "coords": [[_slat, _slon]],
                                            "coords_3d": [],
                                            "dist_km": _tkm, "ascent_m": 0, "dur_min": 0,
                                            "params": _params,
                                            "error": str(_e),
                                        })
                            else:
                                # Senza ORS: crea percorsi simulati come punti attorno alla partenza
                                import math as _math
                                for _vi, _var in enumerate(_variants):
                                    _r_deg = (_tkm * (0.8 + _vi*0.1)) / 111  # gradi approssimativi
                                    _n_pts = 12
                                    _sim_coords = []
                                    for _pi in range(_n_pts + 1):
                                        _ang = 2 * _math.pi * _pi / _n_pts
                                        _sim_coords.append([
                                            _slat + _r_deg * _math.sin(_ang),
                                            _slon + _r_deg * _math.cos(_ang)
                                        ])
                                    _routes_found.append({
                                        "name": _var["name"] + " (simulato — configura ORS_API_KEY)",
                                        "coords": _sim_coords,
                                        "coords_3d": [],
                                        "dist_km": _tkm * (0.8 + _vi*0.1),
                                        "ascent_m": int((_tel or 800) * (0.8 + _vi*0.1)),
                                        "dur_min": int(_tkm * 6),
                                        "params": _params,
                                    })

                            st.session_state.pr_routes    = _routes_found
                            st.session_state.pr_active_idx = 0
                            st.session_state.pr_chat_hist.append({"role": "user", "content": _pr_input})

                            # Step 3: AI commenta le varianti
                            if _routes_found:
                                _r0 = _routes_found[0]
                                _ai_comment_prompt = (
                                    f"Sei un coach outdoor esperto di Abruzzo. L'utente vuole: '{_pr_input}'. "
                                    f"Ho generato 3 varianti di percorso {_sport} partendo da lat={_slat:.4f}, lon={_slon:.4f}. "
                                    f"Variante principale: {_r0['dist_km']:.1f}km, +{_r0['ascent_m']}m D+, ~{_r0['dur_min']:.0f}min. "
                                    f"Commenta brevemente le 3 varianti (max 3 righe ciascuna), suggerisci quale è più adatta "
                                    f"e aggiungi 2 consigli pratici per questo tipo di percorso ({_sport}). "
                                    f"Rispondi in italiano, diretto e pratico."
                                )
                                _ai_comment = ai_fast(_ai_comment_prompt)
                                st.session_state.pr_chat_hist.append({"role": "assistant", "content": _ai_comment})

                            st.rerun()
                        except Exception as _e:
                            st.error(f"Errore generazione: {_e}")

            with _ptab_man:
                st.markdown("**Crea percorso manuale**")
                st.caption("Aggiungi coordinate waypoint. Sulla mappa puoi vedere i punti.")
                _wps = st.session_state.pr_waypoints

                # Input waypoint
                _wp_lat = st.number_input("Latitudine", value=DEFAULT_LAT, format="%.4f", key="pr_wp_lat")
                _wp_lon = st.number_input("Longitudine", value=DEFAULT_LON, format="%.4f", key="pr_wp_lon")
                _wp_name = st.text_input("Nome punto (opzionale)", key="pr_wp_name")
                _c1, _c2 = st.columns(2)
                with _c1:
                    if st.button("➕ Aggiungi", use_container_width=True, key="pr_add_wp"):
                        _wps.append({"lat": _wp_lat, "lon": _wp_lon, "name": _wp_name or f"P{len(_wps)+1}"})
                        st.session_state.pr_waypoints = _wps
                        st.rerun()
                with _c2:
                    if st.button("🗑️ Svuota", use_container_width=True, key="pr_clear_wp"):
                        st.session_state.pr_waypoints = []
                        st.session_state.pr_routes = []
                        st.rerun()

                if _wps:
                    st.markdown(f"**{len(_wps)} waypoint:**")
                    for _i, _wp in enumerate(_wps):
                        _wcols = st.columns([3, 1])
                        _wcols[0].caption(f"{_wp['name']} — {_wp['lat']:.4f}, {_wp['lon']:.4f}")
                        if _wcols[1].button("✕", key=f"pr_del_{_i}"):
                            _wps.pop(_i)
                            st.session_state.pr_waypoints = _wps
                            st.rerun()

                    # Calcola distanza totale
                    import math as _math2
                    def _haversine(la1, lo1, la2, lo2):
                        R = 6371
                        dlat = _math2.radians(la2-la1); dlon = _math2.radians(lo2-lo1)
                        a = _math2.sin(dlat/2)**2 + _math2.cos(_math2.radians(la1))*_math2.cos(_math2.radians(la2))*_math2.sin(dlon/2)**2
                        return R * 2 * _math2.asin(_math2.sqrt(a))

                    _total_km = sum(_haversine(_wps[i]["lat"], _wps[i]["lon"],
                                               _wps[i+1]["lat"], _wps[i+1]["lon"])
                                    for i in range(len(_wps)-1))
                    st.metric("Distanza lineare totale", f"{_total_km:.1f} km")

                    if len(_wps) >= 2 and ORS_API_KEY:
                        if st.button("🗺️ Calcola percorso su sentieri", use_container_width=True, key="pr_calc_man"):
                            with st.spinner("Calcolo percorso..."):
                                try:
                                    import requests as _req2
                                    _ors_profile = ORS_PROFILES[_sport]
                                    _ors_url2 = "https://api.openrouteservice.org/v2/directions/" + _ors_profile + "/geojson"
                                    _ors_body2 = {
                                        "coordinates": [[w["lon"], w["lat"]] for w in _wps],
                                        "elevation": True, "instructions": False,
                                    }
                                    _ors_r2 = _req2.post(_ors_url2, json=_ors_body2,
                                        headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"},
                                        timeout=20)
                                    if _ors_r2.status_code == 200:
                                        _gj2 = _ors_r2.json()
                                        _feat2 = _gj2["features"][0]
                                        _seg2  = _feat2["properties"]["segments"][0]
                                        _coords2 = _feat2["geometry"]["coordinates"]
                                        _dist2 = _seg2["distance"] / 1000
                                        _asc2 = sum(
                                            max(0, _coords2[i][2] - _coords2[i-1][2])
                                            for i in range(1, len(_coords2)) if len(_coords2[i]) >= 3
                                        )
                                        _route_man = {
                                            "name": f"Percorso manuale — {len(_wps)} tappe",
                                            "coords": [[c[1], c[0]] for c in _coords2],
                                            "coords_3d": _coords2,
                                            "dist_km": round(_dist2, 1),
                                            "ascent_m": round(_asc2),
                                            "dur_min": round(_seg2["duration"] / 60),
                                            "params": {},
                                        }
                                        st.session_state.pr_routes = [_route_man]
                                        st.session_state.pr_active_idx = 0
                                        st.success(f"✅ {_dist2:.1f} km · +{_asc2:.0f}m D+")
                                        st.rerun()
                                    else:
                                        st.error(f"ORS error {_ors_r2.status_code}: {_ors_r2.text[:200]}")
                                except Exception as _e2:
                                    st.error(f"Errore routing: {_e2}")

            with _ptab_gpx:
                st.markdown("**Carica un file GPX esistente**")
                _gpx_file = st.file_uploader("GPX file", type=["gpx"], key="pr_gpx_up")
                if _gpx_file:
                    try:
                        import xml.etree.ElementTree as ET
                        _tree = ET.parse(_gpx_file)
                        _root = _tree.getroot()
                        _ns_map = {"gpx": "http://www.topografix.com/GPX/1/1",
                                   "gpx0": "http://www.topografix.com/GPX/1/0"}
                        _ns = "gpx" if "topografix.com/GPX/1/1" in _root.tag else "gpx0"
                        _nsp = "{http://www.topografix.com/GPX/1/1}" if _ns == "gpx" else "{http://www.topografix.com/GPX/1/0}"

                        _trkpts = _root.findall(f".//{_nsp}trkpt")
                        if not _trkpts:
                            _trkpts = _root.findall(".//trkpt")

                        _coords_gpx = []
                        for _pt in _trkpts:
                            _lat2 = float(_pt.get("lat", 0))
                            _lon2 = float(_pt.get("lon", 0))
                            _ele_el = _pt.find(f"{_nsp}ele") or _pt.find("ele")
                            _ele2 = float(_ele_el.text) if _ele_el is not None else 0
                            _coords_gpx.append([_lat2, _lon2, _ele2])

                        if _coords_gpx:
                            import math as _mg
                            _dist_gpx = sum(
                                _mg.sqrt((_coords_gpx[i][0]-_coords_gpx[i-1][0])**2 +
                                         (_coords_gpx[i][1]-_coords_gpx[i-1][1])**2) * 111
                                for i in range(1, len(_coords_gpx))
                            )
                            _asc_gpx = sum(max(0, _coords_gpx[i][2]-_coords_gpx[i-1][2])
                                           for i in range(1, len(_coords_gpx)))
                            _route_gpx = {
                                "name": f"📂 {_gpx_file.name}",
                                "coords": [[c[0], c[1]] for c in _coords_gpx],
                                "coords_3d": [[c[1], c[0], c[2]] for c in _coords_gpx],  # lon,lat,ele per mapbox
                                "dist_km": round(_dist_gpx, 1),
                                "ascent_m": round(_asc_gpx),
                                "dur_min": 0,
                                "params": {},
                            }
                            st.session_state.pr_routes = [_route_gpx]
                            st.session_state.pr_active_idx = 0
                            st.session_state.pr_gpx_loaded = _gpx_file.name
                            st.success(f"✅ {len(_coords_gpx)} punti · {_dist_gpx:.1f}km · +{_asc_gpx:.0f}m")
                            st.rerun()
                    except Exception as _eg:
                        st.error(f"Errore lettura GPX: {_eg}")

            st.divider()

            # ── Route selector + stats ──
            _routes = st.session_state.pr_routes
            if _routes:
                st.markdown("**Percorsi disponibili:**")
                for _ri, _rt in enumerate(_routes):
                    _is_active = _ri == st.session_state.pr_active_idx
                    _btn_style = "primary" if _is_active else "secondary"
                    _stats = f"{_rt['dist_km']:.1f}km · +{_rt['ascent_m']}m"
                    _prefix = "▶ " if _is_active else ""
                    _btn_label = f"{_prefix}{_rt['name'][:28]} — {_stats}"
                    if st.button(_btn_label,
                                  key=f"pr_sel_{_ri}", type=_btn_style, use_container_width=True):
                        st.session_state.pr_active_idx = _ri
                        st.rerun()

                # ── Export GPX ──
                _act_route = _routes[st.session_state.pr_active_idx]
                if _act_route.get("coords"):
                    _gpx_export = '<?xml version="1.0" encoding="UTF-8"?>\n'
                    _gpx_export += '<gpx version="1.1" creator="Elite AI Coach Pro" '
                    _gpx_export += 'xmlns="http://www.topografix.com/GPX/1/1">\n'
                    _gpx_export += f'  <metadata><name>{_act_route["name"]}</name></metadata>\n'
                    _gpx_export += '  <trk><name>' + _act_route["name"] + '</name><trkseg>\n'
                    for _c3 in _act_route["coords"]:
                        _elev_str = ""
                        if _act_route.get("coords_3d"):
                            # Trova elevazione corrispondente
                            _idx_c = _act_route["coords"].index(_c3) if _c3 in _act_route["coords"] else 0
                            if _idx_c < len(_act_route["coords_3d"]):
                                _elev_str = f"<ele>{_act_route['coords_3d'][_idx_c][2]:.1f}</ele>"
                        _gpx_export += f'    <trkpt lat="{_c3[0]:.6f}" lon="{_c3[1]:.6f}">{_elev_str}</trkpt>\n'
                    _gpx_export += '  </trkseg></trk>\n</gpx>'
                    st.download_button(
                        "⬇️ Esporta GPX", data=_gpx_export,
                        file_name=f"route_{_act_route['name'][:20].replace(' ','_')}.gpx",
                        mime="application/gpx+xml", use_container_width=True, key="pr_gpx_dl"
                    )

        with col_map:
            _routes = st.session_state.pr_routes
            _act_idx = st.session_state.pr_active_idx
            _act_route = _routes[_act_idx] if _routes else None

            if not MAPBOX_TOKEN:
                st.warning("⚠️ Configura **MAPBOX_TOKEN** nei Secrets per visualizzare la mappa 3D.")
                st.info("Puoi comunque generare e scaricare percorsi GPX anche senza la mappa.")
            else:
                # ── Costruisci mappa 3D Mapbox con heatmap + percorso ──
                _center_lat = _act_route["coords"][0][0] if _act_route and _act_route["coords"] else DEFAULT_LAT
                _center_lon = _act_route["coords"][0][1] if _act_route and _act_route["coords"] else DEFAULT_LON

                # Prepara GeoJSON linee percorso (tutte le varianti)
                _route_geojson_features = []
                _route_colors = ["#e94560", "#2196F3", "#4CAF50", "#FF9800"]
                if _routes:
                    for _ri, _rt in enumerate(_routes):
                        if _rt.get("coords") and len(_rt["coords"]) > 1:
                            _is_act = _ri == _act_idx
                            _route_geojson_features.append({
                                "type": "Feature",
                                "properties": {
                                    "name": _rt["name"],
                                    "color": _route_colors[_ri % len(_route_colors)],
                                    "width": 5 if _is_act else 2,
                                    "opacity": 1.0 if _is_act else 0.4,
                                },
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": [[c[1], c[0]] for c in _rt["coords"]]
                                }
                            })

                # Waypoint markers
                _wp_geojson_features = []
                for _i, _wp in enumerate(st.session_state.pr_waypoints):
                    _wp_geojson_features.append({
                        "type": "Feature",
                        "properties": {"name": _wp["name"], "idx": _i},
                        "geometry": {"type": "Point", "coordinates": [_wp["lon"], _wp["lat"]]}
                    })

                # Route GeoJSON string
                import json as _json2
                _route_gj_str = _json2.dumps({
                    "type": "FeatureCollection",
                    "features": _route_geojson_features
                })
                _wp_gj_str = _json2.dumps({
                    "type": "FeatureCollection",
                    "features": _wp_geojson_features
                })

                _heat_url = "https://heatmap-external-a.strava.com/tiles/all/hot/{z}/{x}/{y}.png"
                _show_heat_js = "true" if st.session_state.pr_show_heat else "false"

                _map_html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<script src="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css" rel="stylesheet">
<style>html,body,#map{{margin:0;padding:0;height:100%;width:100%;}}
.info-panel{{position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.75);color:#fff;
  padding:10px 14px;border-radius:10px;font-family:sans-serif;font-size:13px;min-width:160px;z-index:10}}
</style>
</head><body>
<div id="map"></div>
<div class="info-panel" id="info">
  {"<br>".join([
    f"<b style=\"color:{_route_colors[i % len(_route_colors)]}\">{rt['name'][:25]}</b><br>{rt['dist_km']:.1f}km · +{rt['ascent_m']}m"
    for i, rt in enumerate(_routes)
  ]) if _routes else "Nessun percorso"}
</div>
<script>
mapboxgl.accessToken = "{MAPBOX_TOKEN}";
var map = new mapboxgl.Map({{
  container: "map",
  style: "mapbox://styles/mapbox/outdoors-v12",
  center: [{_center_lon}, {_center_lat}],
  zoom: 11, pitch: 50, bearing: 0
}});

map.on("load", function() {{
  // Heatmap Strava layer
  if ({_show_heat_js}) {{
    map.addSource("strava-heat", {{
      "type": "raster",
      "tiles": ["{_heat_url}"],
      "tileSize": 256,
      "attribution": "© Strava"
    }});
    map.addLayer({{
      "id": "strava-heatmap",
      "type": "raster",
      "source": "strava-heat",
      "paint": {{"raster-opacity": 0.55}}
    }}, "road-label");
  }}

  // Percorsi
  var routeData = {_route_gj_str};
  map.addSource("routes", {{"type": "geojson", "data": routeData}});
  map.addLayer({{
    "id": "routes-line",
    "type": "line",
    "source": "routes",
    "paint": {{
      "line-color": ["get", "color"],
      "line-width": ["get", "width"],
      "line-opacity": ["get", "opacity"]
    }}
  }});

  // Waypoint markers
  var wpData = {_wp_gj_str};
  map.addSource("waypoints", {{"type": "geojson", "data": wpData}});
  map.addLayer({{
    "id": "wp-circles",
    "type": "circle",
    "source": "waypoints",
    "paint": {{
      "circle-color": "#FF9800",
      "circle-radius": 8,
      "circle-stroke-color": "#fff",
      "circle-stroke-width": 2
    }}
  }});

  // Punto di partenza (primo punto del percorso attivo)
  {'var startPt = { "type": "FeatureCollection", "features": [{ "type": "Feature", "geometry": { "type": "Point", "coordinates": [' + str(_act_route["coords"][0][1]) + ', ' + str(_act_route["coords"][0][0]) + '] }, "properties": {} }] };' if _act_route and _act_route.get("coords") else 'var startPt = { "type": "FeatureCollection", "features": [] };'}
  map.addSource("start-pt", {{"type": "geojson", "data": startPt}});
  map.addLayer({{
    "id": "start-circle",
    "type": "circle",
    "source": "start-pt",
    "paint": {{
      "circle-color": "#4CAF50",
      "circle-radius": 10,
      "circle-stroke-color": "#fff",
      "circle-stroke-width": 3
    }}
  }});

  // Terrain 3D
  map.addSource("mapbox-dem", {{
    "type": "raster-dem",
    "url": "mapbox://mapbox.mapbox-terrain-dem-v1",
    "tileSize": 512
  }});
  map.setTerrain({{"source": "mapbox-dem", "exaggeration": 1.5}});
  map.addLayer({{
    "id": "sky",
    "type": "sky",
    "paint": {{
      "sky-type": "atmosphere",
      "sky-atmosphere-sun": [0.0, 90.0],
      "sky-atmosphere-sun-intensity": 15
    }}
  }});

  map.addControl(new mapboxgl.NavigationControl());
}});
</script>
</body></html>"""

                import streamlit.components.v1 as _comp
                _comp.html(_map_html, height=600, scrolling=False)

            # ── Profilo altimetrico ──
            if _act_route and _act_route.get("coords_3d") and len(_act_route["coords_3d"]) > 2:
                st.markdown("##### 📈 Profilo Altimetrico")
                import math as _mh
                _c3d = _act_route["coords_3d"]
                _dists_km = [0.0]
                for _ci in range(1, len(_c3d)):
                    _dlat = _c3d[_ci][1] - _c3d[_ci-1][1]
                    _dlon = _c3d[_ci][0] - _c3d[_ci-1][0]
                    _dd   = _mh.sqrt(_dlat**2 + _dlon**2) * 111
                    _dists_km.append(_dists_km[-1] + _dd)
                _eles = [c[2] for c in _c3d]
                # Campiona max 200 punti per fluidità
                _step = max(1, len(_dists_km) // 200)
                _xs = _dists_km[::_step]
                _ys = _eles[::_step]
                fig_elev = go.Figure(go.Scatter(
                    x=_xs, y=_ys, mode="lines",
                    fill="tozeroy", fillcolor="rgba(33,150,243,0.12)",
                    line=dict(color="#2196F3", width=2),
                ))
                fig_elev.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=180, margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(title="km", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(title="m slm", gridcolor="rgba(255,255,255,0.08)"),
                )
                st.plotly_chart(fig_elev, use_container_width=True)

            # ── Note ORS se non configurato ──
            if not ORS_API_KEY:
                st.info("💡 Aggiungi **ORS_API_KEY** nei Secrets per ottenere percorsi reali su sentieri OSM (gratuito su openrouteservice.org/dev)")

    elif menu == "💬 Coach Chat":
        st.markdown("## 💬 Parla con il tuo Coach")

        col_btn, col_info, _ = st.columns([1, 3, 4])
        with col_btn:
            if st.button("🗑️ Pulisci chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.pop("coach_snapshot", None)
                st.rerun()
        with col_info:
            _df6m_cnt = len(df[df["start_date"] >= (df["start_date"].max() - pd.Timedelta(days=183))])
            _df_old_cnt = len(df) - _df6m_cnt
            st.caption(f"📊 Contesto: {_df6m_cnt} attività dettagliate (6 mesi) + storico {_df_old_cnt} attività per anno · metriche · RingConn")

        # Snapshot cached
        if "coach_snapshot" not in st.session_state:
            with st.spinner("Costruisco il contesto atleta (6 mesi + storico)..."):
                st.session_state.coach_snapshot = build_athlete_snapshot(
                    df=df, u=u,
                    current_ctl=current_ctl, current_atl=current_atl,
                    current_tsb=current_tsb, status_label=status_label,
                    ctl_daily=ctl_daily, atl_daily=atl_daily, tsb_daily=tsb_daily,
                    vo2max_val=vo2max_val,
                    acwr_v2=acwr_v2, hrv_slope=hrv_slope,
                    tss_budget=tss_budget,
                    readiness=readiness, rc_vitals=rc_vitals, rc_sleep=rc_sleep,
                )

        _snapshot = st.session_state.coach_snapshot

        _system = (
            "Sei un coach sportivo d'élite specializzato in ciclismo e corsa su strada/trail. "
            "Hai accesso completo ai dati di allenamento dell'atleta: 6 mesi dettagliati riga per riga "
            "più un riassunto storico annuale di tutta la carriera. "
            "FOCUS PRIMARIO: ciclismo (Ride, MTB) e corsa (Run, TrailRun) — questi sport hanno obiettivi gara. "
            "Sci alpinismo e sci alpino sono sport ludici: considerali solo per il carico fisico complessivo, "
            "NON proporre piani o periodizzazione per questi sport. "
            "Rispondi sempre in italiano. Sii specifico, pratico, diretto. "
            "Usa i dati reali dell'atleta — non dare consigli generici. "
            "Quando menzioni numeri (TSS, CTL, passo, watt) usa sempre i dati presenti nel contesto.\n\n"
            + _snapshot
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Chiedi al tuo coach..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            _full_prompt = _system + "\n\n"
            for msg in st.session_state.messages:
                _role = "Atleta" if msg["role"] == "user" else "Coach"
                _full_prompt += f"{_role}: {msg['content']}\n"
            _full_prompt += "Coach:"

            with st.chat_message("assistant"):
                with st.spinner(""):
                    try:
                        res = ai_generate(_full_prompt)
                    except Exception as e:
                        res = f"⚠️ Errore AI: {e}"
                st.markdown(res)

            st.session_state.messages.append({"role": "assistant", "content": res})


    # ============================================================
    # RECORD PERSONALI
    # ============================================================

    # ============================================================
    # PROFILO FISICO + RECORD PERSONALI
    # ============================================================
    elif menu == "👤 Profilo Fisico":
        st.markdown("## 👤 Parametri Atleta")

        if athlete:
            col_a, col_b = st.columns([1, 4])
            with col_a:
                if athlete.get("profile_medium"):
                    st.image(athlete["profile_medium"], width=80)
            with col_b:
                st.markdown(f"**{athlete.get('firstname','')} {athlete.get('lastname','')}**")
                st.markdown(f"📍 {athlete.get('city','')}, {athlete.get('country','')}")
                st.markdown(f"🏆 Follower: {athlete.get('follower_count', 'N/A')}")

        st.divider()
        st.info("Questi parametri influenzano il calcolo del TSS e tutti i valori di fitness.")

        # ── Auto-rilevamento parametri ─────────────────────────────────
        _df90 = df[df["start_date"] >= (df["start_date"].max() - timedelta(days=90))]

        # FC Max: massimo HR registrato nelle ultime 90gg
        _auto_fc_max = None
        _auto_fc_max_date = None
        _fcmax_series = _df90["max_heartrate"].dropna()
        if not _fcmax_series.empty:
            _auto_fc_max = int(_fcmax_series.max())
            _row_fcmax = _df90.loc[_df90["max_heartrate"].fillna(0).idxmax()]
            _auto_fc_max_date = _row_fcmax["start_date"].strftime("%d/%m/%Y")

        # FC Riposo: minima HR da RingConn (ultimi 30gg)
        _auto_fc_min = None
        _auto_fc_min_src = None
        rc_v_prof = st.session_state.get("rc_vitals")
        if rc_v_prof is not None and "hr_min" in rc_v_prof.columns:
            _rc_min = rc_v_prof["hr_min"].dropna().tail(30)
            if not _rc_min.empty:
                _auto_fc_min = int(_rc_min.min())
                _auto_fc_min_src = f"RingConn — min 30gg: {_auto_fc_min} bpm"

        # FTP: stima 95% miglior avg_watts su attività ciclismo >45min, ultime 90gg
        _auto_ftp = None
        _auto_ftp_src = None
        _df_ride90 = _df90[_df90["type"].isin(["Ride","VirtualRide"])]
        if not _df_ride90.empty:
            _long_rides = _df_ride90[_df_ride90["moving_time"] > 2700]
            if not _long_rides.empty:
                _w_vals = _long_rides["average_watts"].dropna()
                if not _w_vals.empty:
                    _best_ride = _long_rides.loc[_long_rides["average_watts"].fillna(0).idxmax()]
                    _auto_ftp = int(float(_best_ride["average_watts"]) * 0.95)
                    _auto_ftp_src = f"95% avg watts — {_best_ride['start_date'].strftime('%d/%m/%Y')}"

        # ── Banner suggerimenti ─────────────────────────────────────────
        _suggestions = []
        if _auto_fc_max and _auto_fc_max != u["fc_max"]:
            _suggestions.append(("❤️ FC Max rilevata", f"{_auto_fc_max} bpm",
                                  f"da attività del {_auto_fc_max_date}", "fc_max", _auto_fc_max))
        if _auto_fc_min and _auto_fc_min != u["fc_min"]:
            _suggestions.append(("💚 FC Riposo rilevata", f"{_auto_fc_min} bpm",
                                  _auto_fc_min_src, "fc_min", _auto_fc_min))
        if _auto_ftp and _auto_ftp != u.get("ftp", 200):
            _suggestions.append(("⚡ FTP stimato", f"{_auto_ftp} W",
                                  _auto_ftp_src, "ftp", _auto_ftp))

        if _suggestions:
            st.markdown("#### 🔄 Aggiornamenti Suggeriti")
            st.caption("Basati sui tuoi dati recenti (ultimi 90 giorni). Clicca per applicare.")
            _scols = st.columns(len(_suggestions))
            for _si, (_label, _val, _src, _key, _newval) in enumerate(_suggestions):
                with _scols[_si]:
                    st.markdown(f"""
                    <div style="background:#2196F314;border:1px solid #2196F344;border-radius:12px;
                                padding:12px 16px;text-align:center;margin-bottom:8px">
                        <div style="font-size:13px;font-weight:700;color:#2196F3">{_label}</div>
                        <div style="font-size:22px;font-weight:900;color:#111;margin:4px 0">{_val}</div>
                        <div style="font-size:10px;color:#666">{_src}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"✅ Applica {_val}", key=f"auto_{_key}", use_container_width=True):
                        _ud = dict(st.session_state.user_data)
                        _ud[_key] = _newval
                        st.session_state.user_data = _ud
                        st.cache_data.clear()
                        st.success(f"✅ {_label} aggiornata a {_val}!")
                        st.rerun()

        st.markdown("#### ✏️ Inserimento Manuale")
        with st.form("settings"):
            col1, col2 = st.columns(2)
            with col1:
                peso   = st.number_input("⚖️ Peso (kg)", value=float(u["peso"]),
                                          min_value=30.0, max_value=200.0)
                fc_min = st.number_input(
                    "💚 FC a Riposo", value=int(u["fc_min"]), min_value=30, max_value=100,
                    help=f"Auto-rilevata: {_auto_fc_min} bpm ({_auto_fc_min_src})"
                         if _auto_fc_min else "Carica dati RingConn per il rilevamento automatico"
                )
            with col2:
                fc_max = st.number_input(
                    "❤️ FC Massima", value=int(u["fc_max"]), min_value=100, max_value=250,
                    help=f"Max rilevata nelle ultime attività: {_auto_fc_max} bpm ({_auto_fc_max_date})"
                         if _auto_fc_max else "Non rilevata nelle ultime 90gg"
                )
                ftp    = st.number_input(
                    "⚡ FTP (Watt)", value=int(u.get("ftp", 200)), min_value=50, max_value=600,
                    help=f"Stimato: {_auto_ftp} W ({_auto_ftp_src})"
                         if _auto_ftp else "Nessuna attività ciclismo >45min nelle ultime 90gg"
                )
            if st.form_submit_button("💾 Aggiorna Parametri", use_container_width=True):
                st.session_state.user_data = {"peso": peso, "fc_min": fc_min, "fc_max": fc_max, "ftp": ftp}
                st.cache_data.clear()
                st.success("✅ Salvato! Il fitness verrà ricalcolato.")
                st.rerun()

        st.divider()

        # ── RECORD PERSONALI ────────────────────────────────────────────
        st.markdown("## 🏅 Record Personali")

        sports_available = df["type"].value_counts().index.tolist()
        selected_pr_sport = st.selectbox(
            "Sport:", sports_available,
            format_func=lambda x: f"{get_sport_info(x)['icon']} {get_sport_info(x)['label']}"
        )
        df_s = df[df["type"] == selected_pr_sport].copy()

        if df_s.empty:
            st.info("Nessuna attività per questo sport.")
        else:
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            best_dist = df_s.loc[df_s["distance"].idxmax()]
            col1.metric("📏 Distanza Massima", f"{best_dist['distance']/1000:.2f} km",
                         help=f"{best_dist['name']} — {best_dist['start_date'].strftime('%d/%m/%Y')}")
            best_elev = df_s.loc[df_s["total_elevation_gain"].fillna(0).idxmax()]
            col2.metric("⛰️ Dislivello Max", f"{best_elev['total_elevation_gain']:.0f} m",
                         help=f"{best_elev['name']} — {best_elev['start_date'].strftime('%d/%m/%Y')}")
            best_tss = df_s.loc[df_s["tss"].idxmax()]
            col3.metric("🔥 TSS Massimo", f"{best_tss['tss']:.1f}",
                         help=f"{best_tss['name']} — {best_tss['start_date'].strftime('%d/%m/%Y')}")
            best_time = df_s.loc[df_s["moving_time"].idxmax()]
            hrs = int(best_time["moving_time"] // 3600)
            mins_t = int((best_time["moving_time"] % 3600) // 60)
            col4.metric("⏱️ Sessione più lunga", f"{hrs}h {mins_t:02d}m",
                         help=f"{best_time['name']} — {best_time['start_date'].strftime('%d/%m/%Y')}")

            st.divider()
            if selected_pr_sport in ("Run", "TrailRun", "Hike", "Walk"):
                st.markdown("#### 🏆 Miglior Passo per Distanza")
                pace_cols = st.columns(4)
                for i, (dist_thr, label) in enumerate([(5,"5 km"),(10,"10 km"),(21.097,"Mezza"),(42.195,"Maratona")]):
                    filtered = df_s[df_s["distance"] >= dist_thr * 1000].copy()
                    if not filtered.empty:
                        filtered["pace_sec_km"] = filtered["moving_time"] / (filtered["distance"] / 1000)
                        best = filtered.loc[filtered["pace_sec_km"].idxmin()]
                        pv = best["pace_sec_km"]
                        pace_cols[i].metric(f"🏃 {label}", f"{int(pv//60)}:{int(pv%60):02d} /km",
                                             help=f"{best['name']} — {best['start_date'].strftime('%d/%m/%Y')}")
                    else:
                        pace_cols[i].metric(f"🏃 {label}", "N/A")
            elif selected_pr_sport in ("Ride","VirtualRide","MountainBikeRide"):
                st.markdown("#### 🏆 Velocità Massima Media")
                speed_cols = st.columns(3)
                for i, (dist_thr, label) in enumerate([(20,"20 km"),(50,"50 km"),(100,"100 km")]):
                    filtered = df_s[df_s["distance"] >= dist_thr * 1000].copy()
                    if not filtered.empty:
                        filtered["speed"] = filtered["distance"] / filtered["moving_time"] * 3.6
                        best = filtered.loc[filtered["speed"].idxmax()]
                        speed_cols[i].metric(f"🚴 {label}", f"{best['speed']:.1f} km/h",
                                              help=f"{best['name']} — {best['start_date'].strftime('%d/%m/%Y')}")
                    else:
                        speed_cols[i].metric(f"🚴 {label}", "N/A")

            st.divider()
            st.markdown("#### 📈 Evoluzione Distanza nel Tempo")
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=df_s["start_date"], y=df_s["distance"]/1000, mode="markers+lines",
                marker=dict(color=get_sport_info(selected_pr_sport)["color"], size=7),
                line=dict(color=get_sport_info(selected_pr_sport)["color"], width=1.5, dash="dot"),
                name="Distanza (km)"))
            df_s_cummax = df_s.set_index("start_date")["distance"].cummax()/1000
            fig_pr.add_trace(go.Scatter(
                x=df_s_cummax.index, y=df_s_cummax.values, mode="lines",
                line=dict(color="#FFD700", width=2), name="Record storico",
                fill="tonexty", fillcolor="rgba(255,215,0,0.05)"))
            fig_pr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  height=270, margin=dict(l=0,r=0,t=10,b=0),
                                  xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                  yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_pr, use_container_width=True)

            st.markdown("#### 📋 Top 10 Sessioni per Distanza")
            top10 = df_s.nlargest(10,"distance")[
                ["start_date","name","distance","moving_time","total_elevation_gain","tss"]].copy()
            top10["Km"]       = (top10["distance"]/1000).round(2)
            top10["Durata"]   = top10["moving_time"].apply(lambda x: f"{int(x//3600)}h {int((x%3600)//60):02d}m")
            top10["Dislivello"] = top10["total_elevation_gain"].fillna(0).round(0).astype(int)
            top10["Data"]     = top10["start_date"].dt.strftime("%d/%m/%Y")
            st.dataframe(top10[["Data","name","Km","Durata","Dislivello","tss"]].rename(
                columns={"name":"Nome","tss":"TSS"}), use_container_width=True, hide_index=True)

# ============================================================
# LOGIN
# ============================================================
else:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px">
        <div style="font-size:64px">🏆</div>
        <h1 style="font-size:42px; font-weight:900; margin:16px 0">Elite AI Coach Pro</h1>
        <p style="color:#888; font-size:18px; max-width:500px; margin:0 auto 32px">
            Analisi avanzata delle performance, coaching AI personalizzato<br>
            e monitoraggio del fitness basato sui tuoi dati Strava.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        url = (
            f"https://www.strava.com/oauth/authorize"
            f"?client_id={CLIENT_ID}&response_type=code"
            f"&redirect_uri={REDIRECT_URI}"
            f"&scope=read,activity:read_all&approval_prompt=force"
        )
        st.link_button("🔗 Connetti Strava", url, use_container_width=True)
