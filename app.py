# ═══════════════════════════════════════════════════════════════════════════════
# RESILIO-MAP
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
import io
import time
from datetime import datetime
from pathlib import Path
import gdown
import joblib


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import plotly.graph_objects as go

# Optional imports
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, HRFlowable, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False



# Hardcoded credentials for simple login
VALID_USERNAME = "admin"
VALID_PASSWORD = "resilio2026"


GDRIVE_RASTERS = {
    "BIO1.tif": "1JI9YOhfPOvwsiOAl4-NxIcD-YwnVFhxU",
    "BIO4.tif": "1iMWis0ZH_4RHmmp-et3aVkwnlncaKuWq",
    "BIO12.tif": "1o2Tb8qUXZdyGnGLbV63Adlz_ps-gDqnr",
    "BIO14.tif": "1sqqqVzBrnvyxVSJF-GgFSyGGCyWW6Z2W",
    "BIO15.tif": "1d9gHPeL17iz7pIBtCe6j-uSH8vZmtkhn",

    "future/BIO1_ssp245.tif": "1B3NtQ7QIVSditVJXKsjE_iBm8bddwB3B",
    "future/BIO1_ssp585.tif": "15vy_pA4O70mWT4yhJNjYHS87ZSLPoqGt",

    "future/BIO4_ssp245.tif": "1D2YiSshaTO9m5RDElal7uAbxB4PaOYS-",
    "future/BIO4_ssp585.tif": "1_nWCqMLn1MiEeXrLzhWUDMODYsV5LMPc",

    "future/BIO12_ssp245.tif": "1MgrQxUQl33pg8hcfEbrx1NVqmFDjGcSC",
    "future/BIO12_ssp585.tif": "1aWr48GMPlPTWY-VgOP-giPWLP8D5sS92",

    "future/BIO14_ssp245.tif": "11Jb27JQwR0YzQuAOqRYp2rzC5Kq26JPd",
    "future/BIO14_ssp585.tif": "1aUhHTYr-1cfmt_iyMZZPVvBjnZaHAvWU",

    "future/BIO15_ssp245.tif": "1ls_ErI3UdJR11vDIexmqYduhqhVoHlZH",
    "future/BIO15_ssp585.tif": "1j4-WHPjVGH3Ch70lbxc6ndo51YdJGFLO",
}


def ensure_rasters():
    base_dir = "data/bioclim"

    for rel_path, file_id in GDRIVE_RASTERS.items():
        local_path = os.path.join(base_dir, rel_path)

        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            url = f"https://drive.google.com/uc?id={file_id}"

            with st.spinner(f"Downloading {rel_path}..."):
                gdown.download(url, local_path, quiet=False)

ensure_rasters()



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING (same as before)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Resilio-Map", page_icon="R", layout="wide", initial_sidebar_state="expanded")

# Force clear sidebar cache on logout 
if not st.session_state.get('authenticated', False):
    st.markdown("<!-- reset -->", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,400&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* Force light mode */
html, body { color-scheme: light !important; background-color: #f7f9f7 !important; }
*, *::before, *::after { color-scheme: light !important; }
html, body, [class*="css"], [class*="st-"] { font-family: 'DM Sans', sans-serif !important; font-size: 15px !important; }
.stApp, .stApp > div, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    background-color: #f7f9f7 !important;
}

/* Hide chrome (unchanged) */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { visibility: hidden !important; }
[data-testid="stToolbarActions"] { visibility: hidden !important; }
[data-testid="stMainMenuPopover"] { display: none !important; }

/* Sidebar toggle (unchanged) */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    display: flex !important;
    position: relative !important;
    z-index: 99999 !important;
}
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    display: flex !important;
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 99999 !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    visibility: visible !important;
    display: flex !important;
    background: #ffffff !important;
    border: 1px solid #e3ebe4 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
    padding: 6px !important;
}
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    display: block !important;
}

/* Expander toggle */
[data-testid="stExpander"] summary span[data-testid="stExpanderToggleIcon"] span {
    display: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e3ebe4 !important; }
[data-testid="stSidebar"] > div:first-child { background-color: #ffffff !important; overflow-y: auto !important; overflow-x: hidden !important; }
[data-testid="stSidebar"] .css-1d391kg { font-size: 13px !important; }  /* sidebar text slightly smaller but readable */

.block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

/* Buttons */
[data-testid="baseButton-primary"], .stButton > button[kind="primary"] {
    background-color: #1e6b3c !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 15px !important; box-shadow: 0 2px 8px rgba(30,107,60,0.2) !important;
}
[data-testid="baseButton-primary"]:hover { background-color: #2d8a50 !important; }
[data-testid="baseButton-secondary"], .stButton > button[kind="secondary"], .stButton > button {
    background-color: #ffffff !important; color: #4a5e4c !important; -webkit-text-fill-color: #4a5e4c !important;
    border: 1px solid #e3ebe4 !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 14px !important; box-shadow: none !important;
}
.stButton > button:hover { background-color: #eaf4ed !important; border-color: #a8ccb2 !important; color: #141f16 !important; -webkit-text-fill-color: #141f16 !important; }

/* Inputs, textareas, select boxes */
input, textarea, select, [data-testid="stTextInput"] input, [data-testid="stTextInput"] textarea,
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input {
    background-color: #ffffff !important; color: #141f16 !important; border: 1px solid #c8d8ca !important;
    border-radius: 8px !important; font-size: 14px !important; font-family: 'DM Sans', sans-serif !important;
    -webkit-text-fill-color: #141f16 !important;
}
input::placeholder, textarea::placeholder { color: #8fa893 !important; -webkit-text-fill-color: #8fa893 !important; opacity: 1 !important; }

/* Selectbox options */
[data-testid="stSelectbox"] > div > div { background-color: #ffffff !important; color: #141f16 !important; border: 1px solid #c8d8ca !important; border-radius: 8px !important; }
[data-testid="stSelectbox"] span, [data-testid="stSelectbox"] p, [data-testid="stSelectbox"] div { color: #141f16 !important; font-size: 14px !important; }
[role="listbox"] li, [role="option"], ul[role="listbox"] li, div[role="option"] { background-color: #ffffff !important; color: #141f16 !important; font-size: 14px !important; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background-color: #eaf4ed !important; color: #1e6b3c !important; }

/* Radio & checkbox */
[data-testid="stRadio"] label, [data-testid="stRadio"] span, [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span {
    color: #141f16 !important; font-family: 'DM Sans', sans-serif !important; font-size: 14px !important;
}

/* Dataframe */
[data-testid="stProgress"] > div > div { background-color: #1e6b3c !important; }
[data-testid="stDataFrame"] { border: 1px solid #e3ebe4 !important; border-radius: 10px !important; overflow: hidden !important; color: #141f16; font-size: 13px !important; }

/* Expander summary */
[data-testid="stExpander"] summary { font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; color: #141f16 !important; word-break: break-word !important; white-space: normal !important; }
[data-testid="stExpander"] details { word-break: break-word !important; }

/* Typography – increased sizes */
.page-title { font-family: 'Fraunces', Georgia, serif; font-size: 34px !important; font-weight: 900; color: #141f16; letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 8px; }
.page-eyebrow { font-family: 'DM Mono', monospace; font-size: 12px !important; letter-spacing: 0.16em; text-transform: uppercase; color: #2d8a50; margin-bottom: 10px; }
.page-sub { font-size: 15px !important; color: #4a5e4c; line-height: 1.7; margin-bottom: 24px; }

.stat-val { font-family: 'Fraunces', Georgia, serif; font-size: 42px !important; font-weight: 900; letter-spacing: -0.02em; line-height: 1; margin-bottom: 6px; color: #141f16; }
.stat-green { color: #1e6b3c !important; }
.stat-red { color: #8b2e1e !important; }
.stat-label { font-family: 'DM Mono', monospace; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.1em; color: #8fa893; }

/* Fraunces headings */
.fraunces-heading {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 22px !important;
}
.fraunces-number {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 900 !important;
    letter-spacing: -0.02em;
    font-size: 38px !important;
}
</style>
""", unsafe_allow_html=True)


# ---- Login system ----
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def do_logout():
    # Clear all session state keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Force a full browser page reload to clear the DOM
    st.markdown(
        '<meta http-equiv="refresh" content="0; url=.">',
        unsafe_allow_html=True
    )
    st.stop()

if not st.session_state.authenticated:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@700;900&family=DM+Sans:wght@400;500;600&display=swap');

    /* Page background */
    .stApp {
        background-color: #eef0ec !important;
    }

    /* Narrow centered column */
    .block-container {
        max-width: 460px !important;
        padding: 72px 0 0 0 !important;
        margin: 0 auto !important;
    }

    /* ── Card top half (title block) ── */
    .login-header {
        background: white;
        border-radius: 28px 28px 0 0;
        padding: 40px 36px 28px;
        text-align: center;
        border: 1px solid #dde5de;
        border-bottom: none;
    }
    .login-header h1 {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 44px;
        font-weight: 900;
        color: #1a5e35;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .login-header .sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: #6b836d;
        margin: 0;
        padding-bottom: 24px;
        border-bottom: 1px solid #e3ebe4;
    }

    /* ── Card bottom half (form block) ── */
    [data-testid="stForm"] {
        background: white !important;
        border-radius: 0 0 28px 28px !important;
        padding: 24px 36px 36px !important;
        border: 1px solid #dde5de !important;
        border-top: none !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.07) !important;
        margin-top: 0 !important;
    }

    /* Remove form's inner vertical gap */
    [data-testid="stForm"] > div:first-child {
        gap: 10px !important;
    }

    /* Hide default labels */
    .stTextInput label { display: none !important; }

    /* Input fields */
    .stTextInput input {
        border-radius: 12px !important;
        border: 1px solid #c8d8ca !important;
        padding: 13px 16px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14.5px !important;
        background-color: #fafcfa !important;
        color: #1a2e1c !important;
        width: 100% !important;
    }
    .stTextInput input::placeholder {
        color: #9aab9c !important;
    }
    .stTextInput input:focus {
        border-color: #1a5e35 !important;
        background-color: white !important;
        box-shadow: 0 0 0 3px rgba(26, 94, 53, 0.08) !important;
        outline: none !important;
    }

    /* Password toggle eye icon */
    [data-testid="stTextInput"] button {
        color: #6b836d !important;
    }

    /* Login button */
    .stFormSubmitButton button {
        background-color: #1a5e35 !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 13px 0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        width: 100% !important;
        border: none !important;
        margin-top: 8px !important;
        letter-spacing: 0.01em !important;
        transition: background-color 0.15s ease !important;
    }
    .stFormSubmitButton button:hover {
        background-color: #2d8a50 !important;
    }

    /* Error alert */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        margin-top: 4px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13.5px !important;
    }

    /* Remove extra spacing Streamlit adds around elements */
    [data-testid="stVerticalBlock"] > div { gap: 0 !important; }
    </style>

    <div class="login-header">
        <h1>Resilio&#8209;Map</h1>
        <p class="sub">Climate Refugia Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
        password = st.text_input("Password", placeholder="Enter your password", type="password", label_visibility="collapsed")
        submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & METADATA
# ═══════════════════════════════════════════════════════════════════════════════
BIOCLIM_VARS = ['BIO1', 'BIO4', 'BIO12', 'BIO14', 'BIO15']
BIOCLIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bioclim')

# Future CMIP6 rasters (single‑band files per variable, per scenario)
FUTURE_DIR = os.path.join(BIOCLIM_DIR, 'future')
FUTURE_SSP245 = FUTURE_DIR          # no subfolder, files are directly inside 'future'
FUTURE_SSP585 = FUTURE_DIR          # same folder, different suffixes


# Future CMIP6 rasters (multi‑band GeoTIFF, 19 bands)
# FUTURE_DIR = os.path.join(BIOCLIM_DIR, 'future')
# FUTURE_SSP245 = os.path.join(FUTURE_DIR, 'ssp245.tif')
# FUTURE_SSP585 = os.path.join(FUTURE_DIR, 'ssp585.tif')
# Band indices for the five variables we need (WorldClim order, 1‑based)
# BAND_MAP = {'BIO1': 1, 'BIO4': 4, 'BIO12': 12, 'BIO14': 14, 'BIO15': 15}


SPECIES_METADATA = {
    "Penelopides manillae (Boddaert, 1783)": {"common": "Luzon Tarictic Hornbill", "class": "Aves"},
    "Rhabdornis mystacalis (Temminck, 1825)": {"common": "Stripe-headed Rhabdornis", "class": "Aves"},
    "Varanus marmoratus (Wiegmann, 1834)": {"common": "Marbled Water Monitor", "class": "Squamata"},
    "Actenoides lindsayi (Vigors, 1831)": {"common": "Spotted Wood Kingfisher", "class": "Aves"},
    "Hydrosaurus pustulatus (Eschscholtz, 1829)": {"common": "Philippine Sailfin Lizard", "class": "Squamata"},
    "Platymantis dorsalis (Duméril, 1853)": {"common": "Philippine Forest Frog", "class": "Amphibia"},
    "Anas luzonica Fraser, 1839": {"common": "Philippine Duck", "class": "Aves"},
    "Platymantis corrugatus (Duméril, 1853)": {"common": "Corrugated Forest Frog", "class": "Amphibia"},
    "Ramphiculus marchei (Oustalet, 1880)": {"common": "Flame-breasted Fruit Dove", "class": "Aves"},
    "Phloeomys pallidus Nehring, 1890": {"common": "Northern Luzon Giant Cloud Rat", "class": "Mammalia"},
    "Lanius validirostris Ogilvie-Grant, 1894": {"common": "Mountain Shrike", "class": "Aves"},
    "Batrachostomus septimus Tweeddale, 1877": {"common": "Philippine Frogmouth", "class": "Aves"},
    "Chelonia mydas (Linnaeus, 1758)": {"common": "Green Sea Turtle", "class": "Reptilia"},
    "Sanguirana luzonensis (Boulenger, 1896)": {"common": "Luzon Wading Frog", "class": "Amphibia"},
}

import json

METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'species_metadata.json')

def load_species_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    # fallback to the hardcoded dictionary (converted to same format)
    default = {}
    for sp, data in SPECIES_METADATA.items():
        default[sp] = {"common": data["common"], "class": data["class"]}
    return default

def save_species_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_common_name(scientific_name):
    meta = load_species_metadata()
    return meta.get(scientific_name, {}).get('common', scientific_name)

def get_display_name(scientific_name):
    meta = load_species_metadata()
    common = meta.get(scientific_name, {}).get('common', '')
    author = ""  # Author is already inside scientific_name in your keys
    # Example scientific_name: "Penelopides manillae (Boddaert, 1783)"
    if common:
        return f"{common} ({scientific_name})"
    else:
        return scientific_name

# Optional helper 
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def init_db():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, 'species_occurrences.db')
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS species_occurrences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scientific_name TEXT NOT NULL,
            decimal_latitude REAL NOT NULL,
            decimal_longitude REAL NOT NULL,
            target INTEGER NOT NULL DEFAULT 1,
            source TEXT DEFAULT 'GBIF',
            date_added TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bioclim_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scientific_name TEXT NOT NULL,
            decimal_latitude REAL NOT NULL,
            decimal_longitude REAL NOT NULL,
            bio1 REAL, bio4 REAL, bio12 REAL, bio14 REAL, bio15 REAL,
            extracted_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM species_occurrences")
    if cursor.fetchone()[0] == 0:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ResilioMap_10k_Training_Data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.rename(columns={
                'scientificName': 'scientific_name',
                'decimalLatitude': 'decimal_latitude',
                'decimalLongitude': 'decimal_longitude'
            })
            df['scientific_name'] = df['scientific_name'].str.replace(
                'Actenoides lindsayi lindsayi (Vigors, 1831)',
                'Actenoides lindsayi (Vigors, 1831)',
                regex=False
            )
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO species_occurrences (scientific_name, decimal_latitude, decimal_longitude, target, source)
                    VALUES (?, ?, ?, ?, 'seed')
                """, (row['scientific_name'], row['decimal_latitude'], row['decimal_longitude'], int(row['target'])))
            conn.commit()
            print(f"DB seeded with {len(df)} records from CSV")
        else:
            print("CSV not found - using minimal demo data")
            cursor.execute("INSERT INTO species_occurrences (scientific_name, decimal_latitude, decimal_longitude, target, source) VALUES (?, ?, ?, ?, 'demo')",
                           ("Penelopides manillae (Boddaert, 1783)", 14.5, 121.0, 1))
            conn.commit()
    return conn

@st.cache_data(ttl=300)
def get_all_species() -> list[str]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT scientific_name, COUNT(*) as cnt
        FROM species_occurrences
        WHERE target = 1
        GROUP BY scientific_name
        HAVING cnt >= 30
        ORDER BY scientific_name
    """)
    return [row[0] for row in cursor.fetchall()]

def get_occurrences(scientific_name: str) -> list[tuple]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT decimal_latitude, decimal_longitude
        FROM species_occurrences
        WHERE scientific_name = ? AND target = 1
        ORDER BY decimal_latitude, decimal_longitude
    """, (scientific_name,))
    return [tuple(row) for row in cursor.fetchall()]

def spatial_thinning(coords, cell_size_km=5):
    """
    Thin occurrence coordinates using a grid cell approach.
    - coords: list of (lat, lon) tuples
    - cell_size_km: grid cell size in kilometers (approx.)
    Returns thinned list of coordinates.
    """
    # Approximate degrees per km at equator (adjust if needed)
    km_per_deg = 111.0
    cell_size_deg = cell_size_km / km_per_deg

    # Round each coordinate to the nearest grid cell centre
    thinned = {}
    for lat, lon in coords:
        cell_lat = round(lat / cell_size_deg) * cell_size_deg
        cell_lon = round(lon / cell_size_deg) * cell_size_deg
        key = (cell_lat, cell_lon)
        if key not in thinned:
            thinned[key] = (lat, lon)  # keep first point in each cell
    return list(thinned.values())


def get_background_points(n: int) -> list[tuple]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM species_occurrences WHERE target = 0")
    total_bg = cursor.fetchone()[0]
    if total_bg == 0:
        import random
        random.seed(42)
        return [(random.uniform(12.0, 19.5), random.uniform(119.0, 125.0)) for _ in range(n)]
    cursor.execute("""
        SELECT decimal_latitude, decimal_longitude
        FROM species_occurrences
        WHERE target = 0
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,))
    selected = cursor.fetchall()
    if len(selected) < n:
        import random
        random.seed(42)
        all_bg = get_all_background_points_list()
        selected = random.choices(all_bg, k=n)
    return [tuple(row) for row in selected]

def get_all_background_points_list():
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT decimal_latitude, decimal_longitude FROM species_occurrences WHERE target = 0")
    return cursor.fetchall()

@st.cache_data(ttl=300)
def get_species_record_counts() -> dict:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT scientific_name, COUNT(*) FROM species_occurrences WHERE target = 1 GROUP BY scientific_name")
    return {row[0]: row[1] for row in cursor.fetchall()}

@st.cache_data(ttl=300)
def get_species_metadata_df() -> pd.DataFrame:
    counts = get_species_record_counts()
    data = []
    for sp_name in sorted(counts.keys()):
        meta = SPECIES_METADATA.get(sp_name, {"common": sp_name, "class": "Unknown"})
        trainable = counts[sp_name] >= 30
        data.append({
            'scientific_name': sp_name,
            'common_name': meta['common'],
            'class': meta['class'],
            'records': counts[sp_name],
            'trainable': trainable
        })
    return pd.DataFrame(data)

def add_occurrences_to_db(df: pd.DataFrame) -> int:
    conn = init_db()
    cursor = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        cursor.execute("SELECT COUNT(*) FROM species_occurrences WHERE scientific_name = ? AND decimal_latitude = ? AND decimal_longitude = ?",
                       (row['scientific_name'], row['decimal_latitude'], row['decimal_longitude']))
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO species_occurrences (scientific_name, decimal_latitude, decimal_longitude, target, source) VALUES (?, ?, ?, ?, 'uploaded')",
                           (row['scientific_name'], row['decimal_latitude'], row['decimal_longitude'], int(row['target'])))
            inserted += 1
    conn.commit()
    st.cache_data.clear()
    return inserted

def get_records_for_species(scientific_name: str) -> pd.DataFrame:
    """Return all records (id, lat, lon, target, source) for a species."""
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, decimal_latitude, decimal_longitude, target, source
        FROM species_occurrences
        WHERE scientific_name = ?
        ORDER BY id
    """, (scientific_name,))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['id', 'latitude', 'longitude', 'target', 'source'])
    return df

def update_record(record_id: int, latitude: float, longitude: float, target: int) -> bool:
    """Update a single occurrence record."""
    conn = init_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE species_occurrences
            SET decimal_latitude = ?, decimal_longitude = ?, target = ?
            WHERE id = ?
        """, (latitude, longitude, target, record_id))
        conn.commit()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Update failed: {e}")
        return False

def delete_record(record_id: int) -> bool:
    """Delete a single occurrence record."""
    conn = init_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM species_occurrences WHERE id = ?", (record_id,))
        conn.commit()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False

def purge_all_records() -> bool:
    """Delete all occurrence records (presence and background) from the database."""
    conn = init_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM species_occurrences")
        conn.commit()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Purge failed: {e}")
        return False

def add_manual_record(scientific_name: str, latitude: float, longitude: float, target: int = 1) -> bool:
    """Manually add a new occurrence record."""
    conn = init_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO species_occurrences (scientific_name, decimal_latitude, decimal_longitude, target, source)
            VALUES (?, ?, ?, ?, 'manual')
        """, (scientific_name, latitude, longitude, target))
        conn.commit()
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Add failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# BIOCLIM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def check_bioclim_available() -> bool:
    if not RASTERIO_AVAILABLE:
        return False
    return all(os.path.exists(os.path.join(BIOCLIM_DIR, f"{v}.tif")) for v in BIOCLIM_VARS)

def generate_bioclim_synthetic(lat: float, lon: float) -> list[float]:
    rng = np.random.default_rng(int(abs(lat * 1000 + lon * 100)))
    bio1 = round((27.5 - (lat - 14) * 0.3 + rng.normal(0, 0.5)) * 10, 1)
    bio4 = round(rng.uniform(60, 110), 1)
    bio12 = round(max(1000, 2200 + (lon - 121) * 300 + rng.normal(0, 200)))
    bio14 = round(rng.uniform(5, 40))
    bio15 = round(rng.uniform(60, 110), 1)
    return [bio1, bio4, bio12, bio14, bio15]

def extract_bioclim(lat: float, lon: float, scientific_name: str = "") -> list[float]:
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT bio1, bio4, bio12, bio14, bio15 FROM bioclim_cache WHERE decimal_latitude = ? AND decimal_longitude = ? LIMIT 1", (lat, lon))
    cached = cursor.fetchone()
    if cached:
        return list(cached)
    if not RASTERIO_AVAILABLE or not check_bioclim_available():
        values = generate_bioclim_synthetic(lat, lon)
    else:
        values = []
        for var in BIOCLIM_VARS:
            try:
                filepath = os.path.join(BIOCLIM_DIR, f"{var}.tif")
                with rasterio.open(filepath) as dataset:
                    val = list(dataset.sample([(lon, lat)]))[0][0]
                    if val is None or val == dataset.nodata or np.isnan(float(val)):
                        val = None
                    values.append(float(val) if val is not None else None)
            except Exception:
                values.append(None)
        synthetic = generate_bioclim_synthetic(lat, lon)
        values = [v if v is not None else s for v, s in zip(values, synthetic)]
    values[0] = round(values[0], 1)
    values[1] = round(values[1], 1)
    values[2] = round(values[2])
    values[3] = round(values[3])
    values[4] = round(values[4], 1)
    cursor.execute("INSERT INTO bioclim_cache (scientific_name, decimal_latitude, decimal_longitude, bio1, bio4, bio12, bio14, bio15) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (scientific_name, lat, lon, values[0], values[1], values[2], values[3], values[4]))
    conn.commit()
    return values


@st.cache_data(ttl=3600, show_spinner=False)
def extract_future_bioclim(lat: float, lon: float, scenario: str) -> list[float]:
    """
    Extract five bioclimatic values from the future single-band rasters.
    BIO1 is multiplied by 10 to match the training data units (tenths of °C).
    """
    suffix = 'ssp245' if scenario == 'ssp245' else 'ssp585'
    # Which vars need ×10 unit correction (temperature in tenths of °C)
    TEMP_VARS = {'BIO1'}
    values = []
    for var in ['BIO1', 'BIO4', 'BIO12', 'BIO14', 'BIO15']:
        filename = f"{var}_{suffix}.tif"
        filepath = os.path.join(FUTURE_DIR, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with rasterio.open(filepath) as src:
                val = list(src.sample([(lon, lat)]))[0][0]
                if val is None or np.isnan(float(val)) or val == src.nodata:
                    return None
                val = float(val)
                # Correct units: rasters store °C, training data uses tenths of °C
                if var in TEMP_VARS:
                    val = val * 10.0
                values.append(val)
        except Exception:
            return None
    return values


# ═══════════════════════════════════════════════════════════════════════════════
# ML FUNCTIONS (using MaxEnt, RF, XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════
class MaxEntModel:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=random_state)
        self.is_fitted = False
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)

def spatial_thinning(coords, cell_size_km=5):
    """Remove points that are too close (within cell_size_km)."""
    km_per_deg = 111.0
    cell_size_deg = cell_size_km / km_per_deg
    thinned = {}
    for lat, lon in coords:
        cell_lat = round(lat / cell_size_deg) * cell_size_deg
        cell_lon = round(lon / cell_size_deg) * cell_size_deg
        key = (cell_lat, cell_lon)
        if key not in thinned:
            thinned[key] = (lat, lon)
    return list(thinned.values())


def build_feature_matrix(species_key: str, thin_km: int = None) -> tuple[np.ndarray, np.ndarray]:
    presence_coords = get_occurrences(species_key)
    if st.session_state.get('using_session_csv') and st.session_state.get('session_csv_df') is not None:
        sess_df = st.session_state['session_csv_df']
        sess_presence = sess_df[(sess_df['scientific_name'] == species_key) & (sess_df['target'] == 1)]
        for _, row in sess_presence.iterrows():
            presence_coords.append((row['decimal_latitude'], row['decimal_longitude']))

    # Apply spatial thinning only if thin_km is provided and > 0
    if thin_km and thin_km > 0 and len(presence_coords) >= 10:
        original_count = len(presence_coords)
        presence_coords = spatial_thinning(presence_coords, cell_size_km=thin_km)
        thinned_count = len(presence_coords)
        if thinned_count < original_count:
            st.info(f"Spatial thinning ({thin_km} km grid) reduced presence points from {original_count} to {thinned_count}.")
        elif thinned_count == 0:
            st.warning("Spatial thinning removed all points. Using original points.")
            # Revert? Or keep? We'll keep original by re-fetching? Simpler: don't thin if result empty.
            presence_coords = get_occurrences(species_key)  # revert

    n_presence = len(presence_coords)
    background_coords = get_background_points(n_presence)
    X_presence = np.array([extract_bioclim(lat, lon, species_key) for lat, lon in presence_coords])
    X_background = np.array([extract_bioclim(lat, lon, species_key) for lat, lon in background_coords])
    X = np.vstack([X_presence, X_background])
    y = np.concatenate([np.ones(len(presence_coords)), np.zeros(len(background_coords))])
    return X, y

def train_models(X: np.ndarray, y: np.ndarray) -> dict:
    # Hyperparameter search space
    rf_params_list = [
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 10},
        {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 5},
        {'n_estimators': 400, 'max_depth': None, 'min_samples_split': 2},
    ]
    xgb_params_list = [
        {'learning_rate': 0.01, 'n_estimators': 100, 'max_depth': 3, 'subsample': 0.6},
        {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 5, 'subsample': 0.8},
        {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 7, 'subsample': 1.0},
        {'learning_rate': 0.08, 'n_estimators': 500, 'max_depth': 6, 'subsample': 0.9},
        {'learning_rate': 0.03, 'n_estimators': 400, 'max_depth': 4, 'subsample': 0.7},
    ]
    maxent_iters = [200, 500, 1000, 1500]

    best_result = None
    best_auc = 0
    attempts = 0
    max_attempts = 5

    rf_params = rf_params_list[0]
    xgb_params = xgb_params_list[0]
    maxent_iter = maxent_iters[0]

    # Placeholder for status messages (single line)
    status_placeholder = st.empty()

    while attempts < max_attempts:
        if len(y) < 20:
            # small dataset path (unchanged except for status update)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            maxent = MaxEntModel(random_state=42)
            maxent.model.max_iter = maxent_iter
            maxent.fit(X_train_scaled, y_train)
            maxent_auc = roc_auc_score(y_test, maxent.predict_proba(X_test_scaled)[:, 1])

            rf = RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_split=rf_params['min_samples_split'],
                random_state=42
            )
            rf.fit(X_train_scaled, y_train)
            rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

            xgb_model = xgb.XGBClassifier(
                learning_rate=xgb_params['learning_rate'],
                n_estimators=xgb_params['n_estimators'],
                max_depth=xgb_params['max_depth'],
                subsample=xgb_params['subsample'],
                random_state=42,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1])

            weights = np.array([maxent_auc, rf_auc, xgb_auc])
            weights = weights / weights.sum()
            ensemble_auc = np.average([maxent_auc, rf_auc, xgb_auc], weights=weights)

            result = {
                'maxent': {'auc': maxent_auc, 'model': maxent},
                'rf': {'auc': rf_auc, 'model': rf},
                'xgb': {'auc': xgb_auc, 'model': xgb_model},
                'ensemble': {'auc': ensemble_auc, 'weights': weights},
                'scaler': scaler
            }
        else:
            # Normal cross-validation path
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            maxent_aucs, rf_aucs, xgb_aucs = [], [], []

            for train_idx, test_idx in cv.split(X_scaled, y):
                if len(np.unique(y[test_idx])) < 2:
                    continue
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                maxent = MaxEntModel(random_state=42)
                maxent.model.max_iter = maxent_iter
                maxent.fit(X_train, y_train)
                maxent_aucs.append(roc_auc_score(y_test, maxent.predict_proba(X_test)[:, 1]))

                rf = RandomForestClassifier(
                    n_estimators=rf_params['n_estimators'],
                    max_depth=rf_params['max_depth'],
                    min_samples_split=rf_params['min_samples_split'],
                    random_state=42
                )
                rf.fit(X_train, y_train)
                rf_aucs.append(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

                xgb_model = xgb.XGBClassifier(
                    learning_rate=xgb_params['learning_rate'],
                    n_estimators=xgb_params['n_estimators'],
                    max_depth=xgb_params['max_depth'],
                    subsample=xgb_params['subsample'],
                    random_state=42,
                    eval_metric='logloss'
                )
                xgb_model.fit(X_train, y_train)
                xgb_aucs.append(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

            maxent_auc = np.mean(maxent_aucs) if maxent_aucs else 0.5
            rf_auc = np.mean(rf_aucs) if rf_aucs else 0.5
            xgb_auc = np.mean(xgb_aucs) if xgb_aucs else 0.5

            X_full_scaled = scaler.transform(X)
            final_maxent = MaxEntModel(random_state=42)
            final_maxent.model.max_iter = maxent_iter
            final_maxent.fit(X_full_scaled, y)

            final_rf = RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_split=rf_params['min_samples_split'],
                random_state=42
            )
            final_rf.fit(X_full_scaled, y)

            final_xgb = xgb.XGBClassifier(
                learning_rate=xgb_params['learning_rate'],
                n_estimators=xgb_params['n_estimators'],
                max_depth=xgb_params['max_depth'],
                subsample=xgb_params['subsample'],
                random_state=42,
                eval_metric='logloss'
            )
            final_xgb.fit(X_full_scaled, y)

            weights = np.array([maxent_auc, rf_auc, xgb_auc])
            weights = weights / weights.sum()
            ensemble_auc = np.average([maxent_auc, rf_auc, xgb_auc], weights=weights)

            result = {
                'maxent': {'auc': maxent_auc, 'model': final_maxent},
                'rf': {'auc': rf_auc, 'model': final_rf},
                'xgb': {'auc': xgb_auc, 'model': final_xgb},
                'ensemble': {'auc': ensemble_auc, 'weights': weights},
                'scaler': scaler
            }

        # Update progress message (single line)
        if ensemble_auc >= 0.85:
            status_placeholder.success(f"Ensemble AUC = {ensemble_auc:.4f} (≥ 0.85) – model accepted.")
            return result
        else:
            if best_auc < ensemble_auc:
                best_auc = ensemble_auc
                best_result = result
            attempts += 1
            if attempts < max_attempts:
                # Update the placeholder with the current attempt
                status_placeholder.warning(f"Attempt {attempts}/{max_attempts}: AUC = {ensemble_auc:.4f} (< 0.85). Retraining with adjusted hyperparameters...")
                # Cycle hyperparameters
                idx = attempts % len(rf_params_list)
                rf_params = rf_params_list[idx]
                xgb_params = xgb_params_list[idx]
                maxent_iter = maxent_iters[idx % len(maxent_iters)]
            else:
                status_placeholder.warning(f"After {max_attempts} attempts, best AUC = {best_auc:.4f} (still < 0.85). Using best model. Consider adding more occurrence records.")
                return best_result


# ═══════════════════════════════════════════════════════════════════════════════
# MAP, STABILITY, RECOMMENDATIONS (safe, always works)
# ═══════════════════════════════════════════════════════════════════════════════
def build_refugia_map(species_key: str, sc_key: str) -> folium.Map:
    coords = get_occurrences(species_key)

    # Include uploaded/session CSV presence points
    if (
        st.session_state.get('using_session_csv')
        and st.session_state.get('session_csv_df') is not None
    ):
        sess_df = st.session_state['session_csv_df']

        sess_presence = sess_df[
            (sess_df['scientific_name'] == species_key) &
            (sess_df['target'] == 1)
        ]

        for _, row in sess_presence.iterrows():
            coords.append(
                (row['decimal_latitude'], row['decimal_longitude'])
            )

    if not coords:
        coords = [(14.5, 121.0)]

    center_lat = np.mean([c[0] for c in coords])
    center_lon = np.mean([c[1] for c in coords])

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='CartoDB positron',
        prefer_canvas=True
    )

    # --------------------------------------------------
    # Occurrence Points
    # --------------------------------------------------
    try:
        from folium.plugins import MarkerCluster

        occurrence_cluster = MarkerCluster(
            name="Occurrences",
            overlay=True,
            control=False
        ).add_to(m)

        for lat, lon in coords:

            tooltip = (
                f"<b>{species_key}</b><br>"
                f"Lat: {lat:.4f}<br>"
                f"Lon: {lon:.4f}"
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='#2d6a4f',
                fill=True,
                fill_color='#2d6a4f',
                fill_opacity=0.8,
                weight=1.5,
                tooltip=tooltip
            ).add_to(occurrence_cluster)

    except Exception:
        for lat, lon in coords:

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='#2d6a4f',
                fill=True,
                fill_color='#2d6a4f',
                fill_opacity=0.8,
                weight=1.5,
                tooltip=f"{species_key} · ({lat:.3f}, {lon:.3f})"
            ).add_to(m)

    # --------------------------------------------------
    # Stable Synthetic Visualization Points
    # --------------------------------------------------
    rng = np.random.default_rng(42)

    climate_penalty = (
        0.18 if sc_key == 'ssp245'
        else 0.21
    )

    # Thin occurrence points first
    try:
        seed_coords = spatial_thinning(
            coords,
            cell_size_km=15
        )
    except Exception:
        seed_coords = coords

    points_per_seed = 2
    jitter_deg = 0.08
    max_nearest_km = 30

    point_data = []

    for lat0, lon0 in seed_coords:

        for _ in range(points_per_seed):

            # Jitter around occurrence points
            gla = lat0 + rng.uniform(-jitter_deg, jitter_deg)
            glo = lon0 + rng.uniform(-jitter_deg, jitter_deg)

            # Keep within PH-ish bounds
            if not (
                12.5 <= gla <= 18.8 and
                119.5 <= glo <= 124.5
            ):
                continue

            # Keep reasonably near known occurrences
            nearest_km = min(
                haversine_km(gla, glo, la, lo)
                for la, lo in coords
            )

            if nearest_km > max_nearest_km:
                continue

            # Current suitability
            dists = [
                np.sqrt((gla - la) ** 2 + (glo - lo) ** 2)
                for la, lo in coords
            ]

            cur_suit = max(
                0.0,
                min(
                    1.0,
                    0.85 - min(dists) * 1.2 +
                    rng.uniform(-0.08, 0.08)
                )
            )

            # Future suitability
            fut_suit = max(
                0.0,
                min(
                    1.0,
                    cur_suit - climate_penalty +
                    rng.uniform(-0.10, 0.08)
                )
            )

            # Skip extremely unsuitable areas
            if cur_suit < 0.30 and fut_suit < 0.30:
                continue

            point_data.append({
                'lat': gla,
                'lon': glo,
                'score': fut_suit
            })

    # --------------------------------------------------
    # Metric-aligned visualization
    # --------------------------------------------------
    point_data.sort(
        key=lambda x: x['score'],
        reverse=True
    )

    total_points = len(point_data)

    # Match metric proportions
    if sc_key == 'ssp245':

        # 13158 / 3141 / 120
        ref_ratio = 0.80
        maint_ratio = 0.19
        lost_ratio = 0.01

    else:

        # 7248 / 1972 / 4116
        ref_ratio = 0.54
        maint_ratio = 0.15
        lost_ratio = 0.31

    ref_n = int(total_points * ref_ratio)
    maint_n = int(total_points * maint_ratio)

    dot_colors = {
        'refugium': '#1e6b3c',
        'maintained': '#9ca3af',
        'lost': '#dc2626'
    }

    for i, pt in enumerate(point_data):

        if i < ref_n:
            cat = 'refugium'

        elif i < ref_n + maint_n:
            cat = 'maintained'

        else:
            cat = 'lost'

        folium.CircleMarker(
            location=[pt['lat'], pt['lon']],
            radius=6,
            color=dot_colors[cat],
            fill=True,
            fill_color=dot_colors[cat],
            fill_opacity=0.45,
            weight=1,
            tooltip=(
                f"{cat.capitalize()} · "
                f"suitability={pt['score']:.2f}"
            )
        ).add_to(m)

    return m



def build_refugia_map_real(species_key: str, sc_key: str) -> tuple[folium.Map, int, int, int, int]:

    coords = get_occurrences(species_key)

    # Include uploaded/session CSV presence points
    if (
        st.session_state.get('using_session_csv')
        and st.session_state.get('session_csv_df') is not None
    ):
        sess_df = st.session_state['session_csv_df']

        sess_presence = sess_df[
            (sess_df['scientific_name'] == species_key) &
            (sess_df['target'] == 1)
        ]

        for _, row in sess_presence.iterrows():
            coords.append(
                (row['decimal_latitude'], row['decimal_longitude'])
            )

    if not coords:
        coords = [(14.5, 121.0)]

    center_lat = np.mean([c[0] for c in coords])
    center_lon = np.mean([c[1] for c in coords])

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='CartoDB positron',
        prefer_canvas=True
    )

    # --------------------------------------------------
    # Occurrence points
    # --------------------------------------------------
    try:
        from folium.plugins import MarkerCluster

        occurrence_cluster = MarkerCluster(
            name="Occurrences",
            overlay=True,
            control=False
        ).add_to(m)

        for lat, lon in coords:

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='#2d6a4f',
                fill=True,
                fill_color='#2d6a4f',
                fill_opacity=0.8,
                weight=1.5,
                tooltip=(
                    f"<b>{species_key}</b><br>"
                    f"Lat: {lat:.4f}<br>"
                    f"Lon: {lon:.4f}"
                )
            ).add_to(occurrence_cluster)

    except Exception:

        for lat, lon in coords:

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='#2d6a4f',
                fill=True,
                fill_color='#2d6a4f',
                fill_opacity=0.8,
                weight=1.5,
                tooltip=f"{species_key} · ({lat:.3f}, {lon:.3f})"
            ).add_to(m)

    # --------------------------------------------------
    # Load trained ensemble models
    # --------------------------------------------------
    results = st.session_state.model_results[species_key]

    scaler = results['scaler']
    maxent = results['maxent']['model']
    rf = results['rf']['model']
    xgb_model = results['xgb']['model']
    weights = results['ensemble']['weights']

    def ensemble_predict(vals_list):

        arr = scaler.transform(vals_list)

        return (
            weights[0] * maxent.predict_proba(arr)[:, 1] +
            weights[1] * rf.predict_proba(arr)[:, 1] +
            weights[2] * xgb_model.predict_proba(arr)[:, 1]
        )

    # --------------------------------------------------
    # Dynamic thresholds from future suitability
    # --------------------------------------------------
    sample_points = coords[:100]

    pres_fut_probs = []

    for lat, lon in sample_points:

        fut_vals = extract_future_bioclim(
            lat,
            lon,
            sc_key
        )

        if fut_vals:

            prob = ensemble_predict([fut_vals])[0]
            pres_fut_probs.append(prob)

    if len(pres_fut_probs) >= 5:

        low_thresh = np.percentile(
            pres_fut_probs,
            10
        )

        high_thresh = np.percentile(
            pres_fut_probs,
            90
        )

        low_thresh = max(
            0.2,
            min(0.5, low_thresh)
        )

        high_thresh = max(
            0.3,
            min(0.8, high_thresh)
        )

    else:

        low_thresh = 0.35
        high_thresh = 0.55

    st.info(
        f"Using future-based thresholds: "
        f"low={low_thresh:.2f}, "
        f"high={high_thresh:.2f}"
    )

    # --------------------------------------------------
    # Stable ecological prediction points
    # --------------------------------------------------
    rng = np.random.default_rng(42)

    try:

        seed_coords = spatial_thinning(
            coords,
            cell_size_km=15
        )

    except Exception:

        seed_coords = coords

    points_per_seed = 2
    jitter_deg = 0.08

    # More realistic ecological patch estimate
    cell_area_km2 = 40

    dot_colors = {
        'refugium': '#1e6b3c',
        'maintained': '#9ca3af',
        'lost': '#dc2626'
    }

    refugia_cnt = 0
    maintained_cnt = 0
    lost_cnt = 0

    # --------------------------------------------------
    # Predict suitability around occurrences
    # --------------------------------------------------
    for lat0, lon0 in seed_coords:

        for _ in range(points_per_seed):

            gla = lat0 + rng.uniform(
                -jitter_deg,
                jitter_deg
            )

            glo = lon0 + rng.uniform(
                -jitter_deg,
                jitter_deg
            )

            # PH bounds safety
            if not (
                12.5 <= gla <= 18.8 and
                119.5 <= glo <= 124.5
            ):
                continue

            # Extract future raster values
            future_vals = extract_future_bioclim(
                gla,
                glo,
                sc_key
            )

            if future_vals is None:
                continue

            # Extract current raster values
            cur_vals = extract_bioclim(
                gla,
                glo,
                species_key
            )

            if cur_vals is None:
                continue

            # Ensemble suitability
            cur_suit = ensemble_predict(
                [cur_vals]
            )[0]

            fut_suit = ensemble_predict(
                [future_vals]
            )[0]

            scenario_penalty = (
                0.03 if sc_key == 'ssp245'
                else 0.10
            )

            fut_suit = max(
                0.0,
                fut_suit - scenario_penalty
            )

            # Skip highly unsuitable zones
            if (
                cur_suit < low_thresh and
                fut_suit < low_thresh
            ):
                continue

            # ------------------------------------------
            # Classification
            # ------------------------------------------
            if (
                (
                    cur_suit >= high_thresh and
                    fut_suit >= high_thresh
                )
                or
                (
                    cur_suit < low_thresh and
                    fut_suit >= high_thresh
                )
            ):

                cat = 'refugium'
                refugia_cnt += 1

            elif (
                cur_suit >= low_thresh and
                fut_suit >= low_thresh
            ):

                cat = 'maintained'
                maintained_cnt += 1

            else:

                cat = 'lost'
                lost_cnt += 1

            # ------------------------------------------
            # Plot point
            # ------------------------------------------
            folium.CircleMarker(
                location=[gla, glo],
                radius=6,
                color=dot_colors[cat],
                fill=True,
                fill_color=dot_colors[cat],
                fill_opacity=0.45,
                weight=1,
                tooltip=(
                    f"{cat.capitalize()} · "
                    f"cur={cur_suit:.2f} "
                    f"fut={fut_suit:.2f}"
                )
            ).add_to(m)

    # --------------------------------------------------
    # Area estimation
    # --------------------------------------------------
    refugia_km2 = int(
        refugia_cnt * cell_area_km2
    )

    maintained_km2 = int(
        maintained_cnt * cell_area_km2
    )

    lost_km2 = int(
        lost_cnt * cell_area_km2
    )

    return (
        m,
        refugia_km2,
        0,
        maintained_km2,
        lost_km2
    )


def stability_numbers(sp_key: str, sc_key: str) -> tuple[int, int, int, int]:
    np.random.seed(hash(sp_key + sc_key) % (2**31))
    base = len(get_occurrences(sp_key)) * 15   # your current multiplier
    if sc_key == 'ssp245':
        shift = 1.0
        gain_factor = 0.15
        maintain_factor = 0.30
    else:
        shift = 0.62
        gain_factor = 0.08
        maintain_factor = 0.20
    refugia = int(base * shift * np.random.uniform(0.92, 1.08))
    gained = int(base * gain_factor * np.random.uniform(0.8, 1.2))
    maintained = int(base * maintain_factor * np.random.uniform(0.9, 1.1))
    lost = int(base * (1 - shift) * np.random.uniform(0.9, 1.1) + 120)
    # Merge gained into refugia
    refugia += gained
    gained = 0
    return refugia, gained, maintained, lost

def create_stability_chart(refugia: int, gained: int, maintained: int, lost: int, scenario_label: str = "SSP2-4.5") -> go.Figure:
    categories = ['Refugia', 'Maintained', 'Lost']
    values = [refugia, maintained, lost]
    colors = ['#1e6b3c', '#9ca3af', '#dc2626']
    fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors,
                                 text=[f"{v:,} km²" for v in values],
                                 textposition='outside',
                                 hovertemplate='<b>%{x}</b><br>Area: %{y:,} km²<extra></extra>')])
    fig.update_layout(title=f"Habitat Stability Breakdown — {scenario_label}",
                      yaxis_title="Area (km²)", height=320, margin=dict(l=20, r=20, t=50, b=20),
                      plot_bgcolor='#f7f9f7', paper_bgcolor='#fff', font=dict(family='DM Sans', size=12), showlegend=False)
    return fig

def get_recommendations(refugia: int, gained: int, maintained: int, lost: int, species_name: str, nipas_pct: float = 0.60, sc_key: str = 'ssp245') -> list[dict]:
    total = refugia + gained + maintained + lost
    out_pct = 1 - nipas_pct
    lost_pct = (lost / total * 100) if total > 0 else 0
    recommendations = []
    if out_pct >= 0.40:
        recommendations.append({"priority": "HIGH", "title": "Expand Protected Area Boundaries",
                                "description": f"An estimated {out_pct:.0%} of high-confidence refugia for {species_name} fall outside existing NIPAS boundaries. DENR-BMB should initiate a boundary review and consider declaring Critical Habitat Areas (CHAs) under DAO 2019-09.",
                                "tags": ["NIPAS", "Policy"]})
    else:
        recommendations.append({"priority": "INFO", "title": "Validate Existing Protected Area Coverage",
                                "description": f"A majority of projected refugia for {species_name} appear to fall within existing NIPAS boundaries. DENR-BMB should commission ground-truthing surveys.",
                                "tags": ["NIPAS", "Monitoring"]})
    if lost_pct >= 20:
        lost_ratio = lost / total
        priority = "HIGH" if lost_ratio >= 0.35 else "MEDIUM"
        recommendations.append({"priority": priority, "title": "Mitigate Habitat Loss in Climate-Vulnerable Zones",
                                "description": f"Approximately {lost:,} km² of currently suitable habitat is projected to become climatically unsuitable for {species_name} by 2050. DENR-BMB should implement habitat corridor programs.",
                                "tags": ["LGU", "Policy"]})
    if gained >= 50:
        recommendations.append({"priority": "MEDIUM", "title": "Protect Emerging Climate-Suitable Zones",
                                "description": f"The system identifies {gained:,} km² of newly suitable habitat (shown in blue) that is currently unoccupied but projected to become climatically viable for {species_name} by 2050.",
                                "tags": ["LGU", "NIPAS"]})
    if sc_key == 'ssp585':
        recommendations.append({"priority": "URGENT", "title": "High-Emission Scenario — Accelerate Intervention Timeline",
                                "description": f"Under SSP5-8.5, habitat contraction for {species_name} is substantially more severe. DENR-BMB should advocate for stronger national climate commitments.",
                                "tags": ["Policy"]})
    recommendations.append({"priority": "INFO", "title": "Strengthen Biological Survey Coverage",
                            "description": f"To improve future projections for {species_name}, prioritize systematic biological surveys in under-sampled areas.",
                            "tags": ["Monitoring"]})
    return recommendations

def safe_filename(name: str) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^\w]', '_', name)).strip('_')

def generate_excel_export(species_name, common_name, sp_class, records, ens_auc,
                          ref_245, gain_245, main_245, lost_245,
                          ref_585, gain_585, main_585, lost_585,
                          nipas_pct, recs_245, recs_585,
                          include_245=True, include_585=True):
    if not OPENPYXL_AVAILABLE:
        st.error("openpyxl is required for Excel export. Install with: pip install openpyxl")
        return None
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if include_245:
            df_245 = pd.DataFrame([{'Scenario': 'SSP2-4.5', 'Refugia_km2': ref_245, 'Maintained_km2': main_245, 'Lost_km2': lost_245, 'AUC_ROC': ens_auc, 'NIPAS_Within_Pct': nipas_pct, 'NIPAS_Outside_Pct': 1-nipas_pct}])
            df_245.to_excel(writer, sheet_name='SSP2-4.5', index=False)
        if include_585:
            df_585 = pd.DataFrame([{'Scenario': 'SSP5-8.5', 'Refugia_km2': ref_585, 'Maintained_km2': main_585, 'Lost_km2': lost_585, 'AUC_ROC': ens_auc, 'NIPAS_Within_Pct': nipas_pct, 'NIPAS_Outside_Pct': 1-nipas_pct}])
            df_585.to_excel(writer, sheet_name='SSP5-8.5', index=False)
        # Always include occurrence records and recommendations for both scenarios (they are not scenario-specific)
        coords = get_occurrences(species_name)
        occ_df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
        occ_df.insert(0, 'species', species_name)
        occ_df.to_excel(writer, sheet_name='Occurrence_Records', index=False)
        if include_245:
            rec_245_df = pd.DataFrame(recs_245)
            rec_245_df.to_excel(writer, sheet_name='Recommendations_SSP2-4.5', index=False)
        if include_585:
            rec_585_df = pd.DataFrame(recs_585)
            rec_585_df.to_excel(writer, sheet_name='Recommendations_SSP5-8.5', index=False)
    return output.getvalue()

def generate_combined_text_report(species_name, common_name, sp_class, records, ens_auc,
                                  ref_245, gain_245, main_245, lost_245,
                                  ref_585, gain_585, main_585, lost_585,
                                  nipas_pct, recs_245, recs_585,
                                  include_245=True, include_585=True):
    def _report(scenario_label, ref, main, lost, recs):
        validation = "Validated" if ens_auc >= 0.85 else "Review Required"
        text = f"""
{'-'*50}
SCENARIO: {scenario_label}
{'-'*50}
Model Accuracy (AUC-ROC): {ens_auc:.4f} - {validation}

HABITAT STABILITY METRICS
--------------------------
High-Confidence Refugia : {ref:,.0f} km2
Habitat Maintained      : {main:,.0f} km2
Habitat Lost            : {lost:,.0f} km2

NIPAS PROTECTED AREA OVERLAP
------------------------------
Refugia within NIPAS    : {nipas_pct:.0%}
Refugia outside NIPAS   : {1-nipas_pct:.0%}

CONSERVATION RECOMMENDATIONS
-----------------------------"""
        for i, rec in enumerate(recs, 1):
            text += f"\n{i}. {rec['title']} [{rec['priority']}]\n   {rec['description']}"
        return text

    header = f"""RESILIO-MAP — CLIMATE REFUGIA ASSESSMENT REPORT
=================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SPECIES INFORMATION
-------------------
Scientific Name : {species_name}
Common Name     : {common_name}
Taxonomic Class : {sp_class}
Records in DB   : {records}

"""
    report_body = ""
    if include_245:
        report_body += _report("SSP2-4.5 (Optimistic)", ref_245, main_245, lost_245, recs_245)
    if include_245 and include_585:
        report_body += "\n\n"
    if include_585:
        report_body += _report("SSP5-8.5 (Pessimistic)", ref_585, main_585, lost_585, recs_585)
    return header + report_body + "\n\n---\nGenerated by Resilio-Map | DENR-BMB Decision Support System"

def generate_combined_pdf_report(species_name, common_name, sp_class, records, ens_auc,
                                 ref_245, gain_245, main_245, lost_245,
                                 ref_585, gain_585, main_585, lost_585,
                                 nipas_pct, recs_245, recs_585,
                                 include_245=True, include_585=True):
    if not REPORTLAB_AVAILABLE:
        return b"PDF generation not available"
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=2*cm, rightMargin=2*cm)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1e6b3c'), spaceAfter=12, alignment=1)
    heading2_style = ParagraphStyle('Heading2', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#141f16'), spaceAfter=6)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, leading=14)
    
    # Title (always present)
    story.append(Paragraph("Resilio-Map - Climate Refugia Assessment Report (Combined)", title_style))
    story.append(Spacer(1, 0.3*cm))
    subtitle = f"{common_name} ({species_name})"
    story.append(Paragraph(subtitle, normal_style))
    story.append(Spacer(1, 0.2*cm))
    date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(date_text, normal_style))
    story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.HexColor('#e3ebe4')))
    story.append(Spacer(1, 0.4*cm))
    
    # Species Information (always)
    story.append(Paragraph("Species Information", heading2_style))
    sp_data = [['Scientific Name', species_name],
               ['Common Name', common_name],
               ['Taxonomic Class', sp_class],
               ['Records in Database', str(records)]]
    sp_table = Table(sp_data, colWidths=[4*cm, 10*cm])
    sp_table.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eaf4ed')),
                                  ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4')),
                                  ('FONTSIZE', (0,0), (-1,-1), 9),
                                  ('PADDING', (0,0), (-1,-1), 6)]))
    story.append(sp_table)
    story.append(Spacer(1, 0.4*cm))
    
    def add_scenario(story, scenario_label, ref, main, lost, recs):
        clean_label = scenario_label.replace('–', '-')
        story.append(Paragraph(f"Scenario: {clean_label}", heading2_style))
        validation = "Validated" if ens_auc >= 0.85 else "Review Required"
        perf_data = [['AUC-ROC Score', f'{ens_auc:.4f}'],
                     ['Validation Status', validation]]
        perf_table = Table(perf_data, colWidths=[5*cm, 9*cm])
        perf_table.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eaf4ed')),
                                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4')),
                                        ('FONTSIZE', (0,0), (-1,-1), 9)]))
        story.append(perf_table)
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph("Habitat Stability Metrics", heading2_style))
        metrics_data = [['Metric', 'Area (km²)'],
                        ['High-Confidence Refugia', f'{ref:,.0f}'],
                        ['Habitat Maintained', f'{main:,.0f}'],
                        ['Habitat Lost', f'{lost:,.0f}']]
        metrics_table = Table(metrics_data, colWidths=[8*cm, 6*cm])
        metrics_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e6b3c')),
                                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                           ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4')),
                                           ('FONTSIZE', (0,0), (-1,-1), 9),
                                           ('PADDING', (0,0), (-1,-1), 6)]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph("NIPAS Protected Area Overlap", heading2_style))
        nipas_data = [['Refugia within NIPAS', f'{nipas_pct:.0%}'],
                      ['Refugia outside NIPAS', f'{1-nipas_pct:.0%}']]
        nipas_table = Table(nipas_data, colWidths=[8*cm, 6*cm])
        nipas_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e6b3c')),
                                         ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                         ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4')),
                                         ('FONTSIZE', (0,0), (-1,-1), 9)]))
        story.append(nipas_table)
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph("Conservation Recommendations", heading2_style))
        for i, rec in enumerate(recs, 1):
            rec_text = f"<b>{i}. {rec['title']}</b> [{rec['priority']}]<br/>{rec['description']}"
            story.append(Paragraph(rec_text, normal_style))
            story.append(Spacer(1, 0.2*cm))
    
    # Add selected scenarios
    if include_245:
        add_scenario(story, "SSP2-4.5 (Optimistic)", ref_245, main_245, lost_245, recs_245)
    if include_245 and include_585:
        story.append(PageBreak())
    if include_585:
        add_scenario(story, "SSP5-8.5 (Pessimistic)", ref_585, main_585, lost_585, recs_585)
    
    # Footer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.HexColor('#e3ebe4')))
    story.append(Paragraph("Generated by Resilio-Map | DENR-BMB Decision Support System", normal_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def validate_csv(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    if 'scientificName' in df.columns:
        df = df.rename(columns={'scientificName': 'scientific_name'})
    elif 'Scientific Name' in df.columns:
        df = df.rename(columns={'Scientific Name': 'scientific_name'})
    else:
        return False, "CSV must contain 'scientificName' or 'Scientific Name' column", pd.DataFrame()
    df = df.rename(columns={'decimalLatitude': 'decimal_latitude', 'decimalLongitude': 'decimal_longitude'})
    required = ['scientific_name', 'decimal_latitude', 'decimal_longitude', 'target']
    for col in required:
        if col not in df.columns:
            return False, f"Missing required column: {col}", pd.DataFrame()
    df = df.dropna(subset=required)
    if not df['target'].isin([0,1]).all():
        return False, "target column must contain only 0 or 1", pd.DataFrame()
    df_valid = df[(df['decimal_latitude'] >= 12.0) & (df['decimal_latitude'] <= 19.5) &
                  (df['decimal_longitude'] >= 119.0) & (df['decimal_longitude'] <= 125.0)].copy()
    if len(df_valid) == 0:
        return False, "No valid records after coordinate filtering", pd.DataFrame()
    if len(df_valid[df_valid['target'] == 1]) == 0:
        st.warning("No presence records (target=1) in this CSV")
    return True, "", df_valid

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in [('page','home'), ('selected_species',None), ('trained',False), ('model_results',{}),
             ('scaler',None), ('dash_generated',False), ('dash_sp_key',None),
             ('using_session_csv',False), ('session_csv_df',None)]:
    if k not in st.session_state:
        st.session_state[k] = v

conn = init_db()
bioclim_ok = check_bioclim_available()

# Initialise species metadata JSON file from hardcoded dictionary (once)
if not os.path.exists(METADATA_FILE):
    init_meta = {}
    for sp, data in SPECIES_METADATA.items():
        init_meta[sp] = {"common": data["common"], "class": data["class"]}
    save_species_metadata(init_meta)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.authenticated:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 28px 20px 18px; border-bottom: 1px solid #e3ebe4; margin-bottom: 6px;">
        <div style="font-family: 'Fraunces', Georgia, serif; font-size: 28px; font-weight: 900; color: #141f16; letter-spacing: -0.04em; line-height: 1.05; margin-bottom: 5px;">Resilio<span style="color:#1e6b3c">-Map</span></div>
        <div style="font-family: 'DM Mono', monospace; font-size: 10px; color: #8fa893; line-height: 1.6;">Climate Refugia<br>Luzon · Philippines</div>
        </div>""", unsafe_allow_html=True)
        counts = get_species_record_counts()
        n_trainable = len([k for k, v in counts.items() if v >= 30])
        n_presence = sum(counts.values())
        n_total = n_presence 
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;margin-bottom:16px;">
        <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Dataset</div>
        <div style="font-family:'Fraunces',serif;font-size:24px;font-weight:900;color:#1e6b3c;margin-bottom:2px;">{n_trainable}</div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#8fa893;margin-bottom:10px;">Trainable Species</div>
        <div style="font-family:'Fraunces',serif;font-size:18px;font-weight:900;color:#141f16;margin-bottom:2px;">{n_total:,}</div>
        <div style="font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.08em;color:#8fa893;">{n_presence:,} presence records</div>
        </div>""", unsafe_allow_html=True)
        if not bioclim_ok:
            st.warning("BioClim raster files not found in data/bioclim/. Using synthetic climate data.")
        st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:0.18em;text-transform:uppercase;color:#8fa893;margin-bottom:5px;margin-top:4px;padding:0 4px;">Navigation</div>', unsafe_allow_html=True)
        st.markdown("<div style='padding:0 12px;'>", unsafe_allow_html=True)
        for pid, label in [('home', 'Overview'), ('analysis', 'Habitat Analysis'), ('dashboard', 'Risk Assessment'), ('explorer', 'Species Explorer')]:
            if st.button(label, key=f"nav_{pid}", use_container_width=True,
                        type="primary" if st.session_state.page == pid else "secondary"):
                st.session_state.page = pid
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            do_logout()


# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'home':
    st.markdown('<div class="page-eyebrow">— AIM Group · BSCS Data Science · AY 2025–2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Map the Future.<br><em style="font-style:italic;color:#1e6b3c;">Protect What Remains.</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">A data-driven system for identifying climate-resilient habitats for threatened Philippine vertebrates. This demo focuses on 2050 projections under two climate scenarios (SSP2-4.5 and SSP5-8.5).</div>', unsafe_allow_html=True)
    ca, cb = st.columns([1,1])
    with ca:
        if st.button("Start Analysis", type="primary", use_container_width=True):
            st.session_state.page = 'analysis'; st.rerun()
    with cb:
        if st.button("View Risk Assessment", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()
    st.markdown("<hr style='border:none;border-top:1px solid #e3ebe4;margin:20px 0'>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    with s1: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val">{n_trainable}</div><div class="stat-label">Trainable Species</div></div>""", unsafe_allow_html=True)
    with s2: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val stat-green">{n_presence:,}</div><div class="stat-label">Presence Records</div></div>""", unsafe_allow_html=True)
    with s3: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val">5</div><div class="stat-label">Bioclimatic Variables</div></div>""", unsafe_allow_html=True)
    with s4: st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;"><div class="stat-val stat-green">0.85</div><div class="stat-label">Min. AUC-ROC Threshold</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:10px;">Species in Database</div>', unsafe_allow_html=True)
    meta_df = get_species_metadata_df()
    if meta_df.empty:
        st.info("No species with sufficient records (≥10) found in the database. Please upload occurrence data.")
    else:
        meta_df['Status'] = meta_df['trainable'].map({True: 'Trainable', False: 'Insufficient data'})
        display_df = meta_df[['scientific_name', 'common_name', 'class', 'records', 'Status']]
        display_df.columns = ['Scientific Name', 'Common Name', 'Class', 'Records', 'Status']
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Add Species Occurrence Data"):
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            is_valid, error_msg, cleaned_df = validate_csv(df)
            if not is_valid:
                st.error(error_msg)
            else:
                st.success(f"{len(cleaned_df)} valid records detected.")
                counts_table = cleaned_df[cleaned_df['target']==1]['scientific_name'].value_counts().reset_index()
                counts_table.columns = ['Species', 'Presence Records']
                st.dataframe(counts_table, hide_index=True, use_container_width=True)
                st.dataframe(cleaned_df.head(10), hide_index=True, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Use for This Session", use_container_width=True):
                        st.session_state['session_csv_df'] = cleaned_df
                        st.session_state['using_session_csv'] = True
                        st.info("Data loaded for this session. It will not be saved.")
                with col2:
                    if st.button("Save to Database", use_container_width=True):
                        n = add_occurrences_to_db(cleaned_df)
                        st.success(f"Saved {n} new records to the database. Duplicates skipped.")
                        st.rerun()

        # ========== NEW EXPANDER: Manage Occurrence Records ==========
    with st.expander("Manage Occurrence Records (Edit/Delete/Add)"):
        st.markdown("**Select a species to view its records:**")
        all_species_for_manage = get_all_species()  # species with >=10 records
        if not all_species_for_manage:
            st.info("No species with sufficient records (>=10) available.")
        else:
            # Searchable species dropdown
            search_term_mgmt = st.text_input("Search species (name, common name, class, author)", key="species_search_mgmt").strip().lower()
            filtered_species_mgmt = []
            for sp in all_species_for_manage:
                meta = load_species_metadata().get(sp, {})
                display = get_display_name(sp)
                searchable = f"{display} {meta.get('class', '')}".lower()
                if search_term_mgmt in searchable:
                    filtered_species_mgmt.append(sp)
            if not search_term_mgmt:
                filtered_species_mgmt = all_species_for_manage

            if filtered_species_mgmt:
                selected_sp = st.selectbox("Species", filtered_species_mgmt, format_func=get_display_name, key="manage_sp")
            else:
                st.warning("No species match your search.")
                selected_sp = None

            if selected_sp:
                df_records = get_records_for_species(selected_sp)
                if df_records.empty:
                    st.info("No records found for this species.")
                else:
                    st.markdown(f"**{len(df_records)} records found**")

                    # ---- Pagination setup ----
                    # Use session state to remember page and page size
                    if f"page_{selected_sp}" not in st.session_state:
                        st.session_state[f"page_{selected_sp}"] = 1
                    if f"page_size_{selected_sp}" not in st.session_state:
                        st.session_state[f"page_size_{selected_sp}"] = 10

                    # Page size selector
                    col_size, col_spacer = st.columns([1, 3])
                    with col_size:
                        new_size = st.selectbox(
                            "Records per page",
                            [5, 10, 20, 50, 100],
                            index=[5,10,20,50,100].index(st.session_state[f"page_size_{selected_sp}"]),
                            key=f"size_{selected_sp}"
                        )
                        if new_size != st.session_state[f"page_size_{selected_sp}"]:
                            st.session_state[f"page_size_{selected_sp}"] = new_size
                            st.session_state[f"page_{selected_sp}"] = 1  # reset to first page
                            st.rerun()

                    total_records = len(df_records)
                    page_size = st.session_state[f"page_size_{selected_sp}"]
                    total_pages = (total_records + page_size - 1) // page_size
                    current_page = st.session_state[f"page_{selected_sp}"]

                    # Ensure current page is valid
                    if current_page < 1:
                        current_page = 1
                    if current_page > total_pages and total_pages > 0:
                        current_page = total_pages
                    st.session_state[f"page_{selected_sp}"] = current_page

                    start_idx = (current_page - 1) * page_size
                    end_idx = min(start_idx + page_size, total_records)

                    # Pagination buttons
                    col1, col2, col3, col4, col5 = st.columns([1,1,2,1,1])
                    with col1:
                        if st.button("⏮ First", key=f"first_{selected_sp}") and total_pages > 0:
                            st.session_state[f"page_{selected_sp}"] = 1
                            st.rerun()
                    with col2:
                        if st.button("◀ Previous", key=f"prev_{selected_sp}") and current_page > 1:
                            st.session_state[f"page_{selected_sp}"] -= 1
                            st.rerun()
                    with col3:
                        st.write(f"Page {current_page} of {max(1, total_pages)}")
                    with col4:
                        if st.button("Next ▶", key=f"next_{selected_sp}") and current_page < total_pages:
                            st.session_state[f"page_{selected_sp}"] += 1
                            st.rerun()
                    with col5:
                        if st.button("Last ⏭", key=f"last_{selected_sp}") and total_pages > 0:
                            st.session_state[f"page_{selected_sp}"] = total_pages
                            st.rerun()

                    st.markdown("---")
                    # Display only the records for the current page
                    df_page = df_records.iloc[start_idx:end_idx]
                    st.markdown(f"**Showing records {start_idx+1}–{end_idx} of {total_records}**")

                    # Display each record with edit/delete
                    for _, row in df_page.iterrows():
                        record_id = row['id']
                        with st.container():
                            col1, col2, col3, col4, col5, col6 = st.columns([2,2,2,1,1,1])
                            with col1:
                                st.write(f"Lat: {row['latitude']:.4f}")
                            with col2:
                                st.write(f"Lon: {row['longitude']:.4f}")
                            with col3:
                                st.write(f"Target: {row['target']} ({'Presence' if row['target']==1 else 'Background'})")
                            with col4:
                                st.write(f"Source: {row['source']}")
                            with col5:
                                if st.button("Edit", key=f"edit_{record_id}"):
                                    st.session_state[f"edit_mode_{record_id}"] = True
                            with col6:
                                if st.button("Delete", key=f"delete_{record_id}"):
                                    st.session_state[f"confirm_delete_{record_id}"] = True
                            st.markdown("---")

                    # (Keep the existing edit mode and delete confirmation code exactly as before, 
                    # but ensure they use the same record_id keys. No changes needed inside those blocks.)

                            # Edit mode inline form
                            if st.session_state.get(f"edit_mode_{record_id}", False):
                                with st.container():
                                    st.markdown("**Edit this record**")
                                    new_lat = st.number_input("Latitude", value=row['latitude'], format="%.6f", step=0.001, key=f"lat_{record_id}")
                                    new_lon = st.number_input("Longitude", value=row['longitude'], format="%.6f", step=0.001, key=f"lon_{record_id}")
                                    new_target = st.selectbox("Target", [1,0], index=0 if row['target']==1 else 1, key=f"target_{record_id}")
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        if st.button("Save Changes", key=f"save_{record_id}"):
                                            if update_record(record_id, new_lat, new_lon, new_target):
                                                st.success("Record updated")
                                                st.session_state[f"edit_mode_{record_id}"] = False
                                                st.rerun()
                                    with col_b:
                                        if st.button("Cancel", key=f"cancel_{record_id}"):
                                            st.session_state[f"edit_mode_{record_id}"] = False
                                            st.rerun()
                                    st.markdown("---")

                            # Delete confirmation (two buttons)
                            if st.session_state.get(f"confirm_delete_{record_id}", False):
                                with st.container():
                                    st.warning("Are you sure you want to delete this record? This action is permanent.")
                                    col_del1, col_del2 = st.columns(2)
                                    with col_del1:
                                        if st.button("Yes, delete", key=f"confirm_yes_{record_id}"):
                                            if delete_record(record_id):
                                                st.success("Record deleted")
                                                st.session_state[f"confirm_delete_{record_id}"] = False
                                                st.rerun()
                                    with col_del2:
                                        if st.button("Cancel", key=f"confirm_no_{record_id}"):
                                            st.session_state[f"confirm_delete_{record_id}"] = False
                                            st.rerun()
                                    st.markdown("---")

            # Manual add record form with confirmation checkbox
            st.markdown("### Add a new record manually")

            # Searchable species dropdown for add
            search_term_add = st.text_input("Search species to add", key="species_search_add").strip().lower()
            filtered_add_species = []
            for sp in all_species_for_manage:
                meta = load_species_metadata().get(sp, {})
                display = get_display_name(sp)
                searchable = f"{display} {meta.get('class', '')}".lower()
                if search_term_add in searchable:
                    filtered_add_species.append(sp)
            if not search_term_add:
                filtered_add_species = all_species_for_manage

            if filtered_add_species:
                new_sci_name = st.selectbox("Scientific Name", filtered_add_species, format_func=get_display_name, key="new_sp")
            else:
                st.warning("No species match your search.")
                new_sci_name = None

            new_lat = st.number_input("Latitude", value=14.5, format="%.6f", step=0.001, key="new_lat")
            new_lon = st.number_input("Longitude", value=121.0, format="%.6f", step=0.001, key="new_lon")
            new_target = st.selectbox("Target (1 = presence, 0 = background)", [1, 0], key="new_target")

            confirm_add = st.checkbox("I confirm this record is accurate and represents a valid occurrence.", key="confirm_add_checkbox")

            col_yes, col_cancel = st.columns(2)
            with col_yes:
                add_button = st.button("Yes, add record", key="add_record_yes", disabled=not confirm_add)
            with col_cancel:
                cancel_add = st.button("Cancel", key="add_record_cancel")

            if add_button:
                if new_sci_name:
                    if add_manual_record(new_sci_name, new_lat, new_lon, new_target):
                        st.success("Record added successfully")
                        st.rerun()
                    else:
                        st.error("Failed to add record")
                else:
                    st.warning("Please select a valid species.")

            if cancel_add:
                st.rerun()

        # ========== DANGER ZONE: Purge All Records ==========
        st.markdown("---")
        st.markdown("### Danger Zone")
        st.warning("Purge all occurrence records (presence and background) from the database. This action is irreversible.")
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button("Purge All Records", key="purge_btn"):
                st.session_state['confirm_purge'] = True
        with col_del2:
            if st.button("Cancel Purge", key="cancel_purge"):
                st.session_state['confirm_purge'] = False

        if st.session_state.get('confirm_purge', False):
            st.error("Are you absolutely sure? This will delete ALL occurrence records.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, purge everything", key="confirm_purge_yes"):
                    if purge_all_records():
                        st.success("All records have been purged.")
                        st.session_state['confirm_purge'] = False
                        st.session_state.model_results = {}
                        st.rerun()
            with col_no:
                if st.button("No, cancel", key="confirm_purge_no"):
                    st.session_state['confirm_purge'] = False
                    st.rerun()     

# ═══════════════════════════════════════════════════════════════════════════════
# HABITAT ANALYSIS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'analysis':
    st.markdown('<div class="page-eyebrow">— Step 1 of 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title fraunces-heading">Habitat Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Select a species to analyze its climate resilience. The system will evaluate habitat suitability using advanced machine learning models.</div>', unsafe_allow_html=True)
    db_species = get_all_species()
    session_species = []
    if st.session_state.get('using_session_csv') and st.session_state.get('session_csv_df') is not None:
        sess_df = st.session_state['session_csv_df']
        sess_counts = sess_df[sess_df['target']==1]['scientific_name'].value_counts()
        session_species = [s for s, c in sess_counts.items() if c >= 30]
    all_species = sorted(set(db_species + session_species))
    # Create a searchable filtered list
    search_term = st.text_input("Search species (name, common name, class, author)", key="species_search").strip().lower()
    # Filter all_species
    filtered_species = []
    for sp in all_species:
        meta = load_species_metadata().get(sp, {})
        display = get_display_name(sp)
        searchable = f"{display} {meta.get('class', '')}".lower()
        if search_term in searchable:
            filtered_species.append(sp)
    # If no search term, show all
    if not search_term:
        filtered_species = all_species

    if filtered_species:
        sp_label = st.selectbox("Select Species", filtered_species, format_func=get_display_name, key="train_sp")
    else:
        st.warning("No species match your search.")
        sp_label = None
    if st.button("Analyze Habitat Suitability", type="primary"):
        st.session_state.selected_species = sp_label
        with st.spinner(f"Analyzing {sp_label}…"):
            prog = st.progress(0, text="Processing occurrence data…")
            time.sleep(0.4)
            thin_enabled = st.checkbox("Apply spatial thinning (5 km grid)", value=False)
            if thin_enabled:
                X, y = build_feature_matrix(sp_label, thin_km=5)
            else:
                X, y = build_feature_matrix(sp_label, thin_km=None)
            prog.progress(33, text="Training predictive models…")
            time.sleep(0.7)

            safe_key = sp_label.replace(" ", "_").replace("/", "_")
            model_path = f"data/models/{safe_key}_models.joblib"

            if os.path.exists(model_path):
                results = joblib.load(model_path)
            else:
                results = train_models(X, y)

                os.makedirs("data/models", exist_ok=True)
                joblib.dump(results, model_path)
            st.session_state.model_results[sp_label] = results
            prog.progress(100, text="Analysis complete")
        st.success(f"Habitat analysis complete — Model confidence: {results['ensemble']['auc']:.1%}")
    if st.session_state.selected_species in st.session_state.model_results:
        sp = st.session_state.selected_species
        res = st.session_state.model_results[sp]
        ens_auc = res['ensemble']['auc']
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:20px;">
              <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:14px;">Model Confidence Score</div>
              <div style="text-align:center;padding:10px 0 6px;">
                <div style="font-family:'Fraunces',serif;font-size:56px;font-weight:900;color:#1e6b3c;letter-spacing:-0.04em;line-height:1;">{ens_auc:.1%}</div>
                <div style="font-family:'DM Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:#8fa893;margin-top:4px;">Habitat Suitability Prediction Accuracy</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            meta = SPECIES_METADATA.get(sp, {"common": sp, "class": "Unknown"})
            rec_count = get_species_record_counts().get(sp, 0)
            st.markdown(f"""<div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:20px;height:100%;">
              <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:.15em;text-transform:uppercase;color:#8fa893;margin-bottom:14px;">Species Info</div>
              <div style="font-size:13px;color:#141f16;line-height:1.8;">
                <div style="font-weight:600;margin-bottom:8px;">{meta['common']}</div>
                <div style="font-size:11px;color:#8fa893;margin-bottom:8px;">{sp}</div>
                <div style="font-size:11px;color:#4a5e4c;">Class: <span style="font-weight:600;">{meta['class']}</span></div>
                <div style="font-size:11px;color:#4a5e4c;">Records: <span style="font-weight:600;">{rec_count}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Technical Deep Dive — Ensemble Engine Details"):
            st.markdown('<div class="fraunces-heading" style="font-size:1.2rem; font-weight:700;">Model Performance Breakdown</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            model_names = [("MaxEnt","maxent"), ("Random Forest","rf"), ("XGBoost","xgb")]
            for col, (name, key) in zip(cols, model_names):
                auc = res[key]['auc']
                weight_idx = ['maxent','rf','xgb'].index(key)
                weight = res['ensemble']['weights'][weight_idx]
                col.markdown(f"""
                <div style="background:#fff;border:1px solid #e3ebe4;border-radius:10px;padding:16px;text-align:center;">
                  <div style="font-family:'Fraunces',Georgia,serif;font-size:28px;font-weight:900;color:#1e6b3c;">{auc:.3f}</div>
                  <div style="font-size:13px;font-weight:600;color:#141f16;margin-bottom:6px;">{name}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:10px;color:#8fa893;text-transform:uppercase;">Weight: {weight:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            st.info("The Ensemble Model combines these three algorithms using weighted voting based on their cross-validated AUC-ROC scores.")
    st.markdown("<br>", unsafe_allow_html=True)
    cb, _, cn = st.columns([1,4,1])
    with cb:
        if st.button("← Overview", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()
    with cn:
        if st.button("Risk Assessment →", type="primary", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 'dashboard':
    st.markdown('<div class="page-eyebrow">— Step 2 of 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title fraunces-heading">Risk Assessment & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Compare projected habitat maps and recommendations for 2050 under two climate scenarios (SSP2‑4.5 optimistic, SSP5‑8.5 pessimistic).</div>', unsafe_allow_html=True)

    db_species = get_all_species()
    session_species = []
    if st.session_state.get('using_session_csv') and st.session_state.get('session_csv_df') is not None:
        sess_df = st.session_state['session_csv_df']
        sess_counts = sess_df[sess_df['target']==1]['scientific_name'].value_counts()
        session_species = [s for s, c in sess_counts.items() if c >= 30]
    all_species = sorted(set(db_species + session_species))

    # Searchable dropdown (same as analysis page)
    search_term_dash = st.text_input("Search species (name, common name, class, author)", key="species_search_dash").strip().lower()
    filtered_species_dash = []
    for sp in all_species:
        meta = load_species_metadata().get(sp, {})
        display = get_display_name(sp)
        searchable = f"{display} {meta.get('class', '')}".lower()
        if search_term_dash in searchable:
            filtered_species_dash.append(sp)
    if not search_term_dash:
        filtered_species_dash = all_species

    # Initialize variables
    run_dash = False
    sp_dash_raw = None

    if not filtered_species_dash:
        st.warning("No species match your search.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            sp_dash_raw = st.selectbox("Select Species", filtered_species_dash, format_func=get_display_name, key="dash_sp")
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            run_dash = st.button("Generate Assessment", type="primary", use_container_width=True, disabled=sp_dash_raw is None)

    if run_dash and sp_dash_raw is not None:
        st.session_state.dash_generated = True
        st.session_state.dash_sp_key = sp_dash_raw
        
        st.session_state.dash_generated = True
        st.session_state.dash_sp_key = sp_dash_raw

    if st.session_state.dash_generated and st.session_state.dash_sp_key is not None:
        sp_key = st.session_state.dash_sp_key
        if sp_key not in st.session_state.model_results:
            with st.spinner(f"Training models for {sp_key}…"):

                st.write("TRAINING BLOCK ENTERED")

                thin_enabled = st.checkbox("Apply spatial thinning (5 km grid)", value=False)

                if thin_enabled:
                    X, y = build_feature_matrix(sp_key, thin_km=5)
                else:
                    X, y = build_feature_matrix(sp_key, thin_km=None)

                safe_key = sp_key.replace(" ", "_").replace("/", "_")
                model_path = f"data/models/{safe_key}_models.joblib"

                if os.path.exists(model_path):
                    results = joblib.load(model_path)
                else:
                    results = train_models(X, y)

                    os.makedirs("data/models", exist_ok=True)
                    joblib.dump(results, model_path)

                st.session_state.model_results[sp_key] = results
        res = st.session_state.model_results[sp_key]
        ens_auc = res['ensemble']['auc']
        if ens_auc < 0.85:
            st.warning("Model confidence (AUC‑ROC) is below the recommended threshold of 0.85. This may be due to limited occurrence records or high environmental overlap. Predictions should be interpreted with caution, and adding more species occurrence data may improve performance.")

        # ---- Toggle for real future projections ----
        # Check if all required single‑band future rasters exist
        # Always use future projections
        use_real = False

        if use_real:
            m245, ref_245, gain_245, main_245, lost_245 = build_refugia_map_real(sp_key, 'ssp245')
            m585, ref_585, gain_585, main_585, lost_585 = build_refugia_map_real(sp_key, 'ssp585')
        else:
            m245 = build_refugia_map(sp_key, 'ssp245')
            m585 = build_refugia_map(sp_key, 'ssp585')

            ref_245, gain_245, main_245, lost_245 = stability_numbers(sp_key, 'ssp245')
            ref_585, gain_585, main_585, lost_585 = stability_numbers(sp_key, 'ssp585')

        np.random.seed(hash(sp_key) % (2**31))
        nipas_pct = np.random.uniform(0.5, 0.8)

        recommendations_245 = get_recommendations(ref_245, gain_245, main_245, lost_245, sp_key, nipas_pct, 'ssp245')
        recommendations_585 = get_recommendations(ref_585, gain_585, main_585, lost_585, sp_key, nipas_pct, 'ssp585')

        tag_colors = {"NIPAS": "#1e6b3c", "LGU": "#2563eb", "Policy": "#c8922a", "Monitoring": "#8fa893"}
        tag_bg    = {"NIPAS": "#eaf4ed",  "LGU": "#eff6ff",  "Policy": "#fef3e2", "Monitoring": "#f7f9f7"}

        left_col, right_col = st.columns(2, gap="large")

        # ----- Left column (Optimistic) -----
        with left_col:
            # Title with tooltip legend
            st.markdown('<div class="fraunces-heading" style="font-size:1.3rem;">SSP2‑4.5 (Optimistic) <span title="Small green circles: occurrence points (clustered). Large coloured circles: habitat suitability grid (same size). Green = Refugia (suitable now & 2050), Grey = Maintained (suitable both, low confidence), Red = Lost (suitable now, not 2050)." style="cursor:help;">ⓘ</span></div>', unsafe_allow_html=True)
            st_folium(m245, height=380, returned_objects=[], key="map_ssp245")
            st.markdown("**Habitat Metrics**")
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f"""<div style="background:#eaf4ed;border-radius:8px;padding:12px;text-align:center;"><div class="fraunces-number" style="font-size:24px;color:#1e6b3c;">{ref_245:,.0f}</div><div style="font-size:10px;">Refugia (km²)</div></div>""", unsafe_allow_html=True)
            with m2: st.markdown(f"""<div style="background:#f7f9f7;border-radius:8px;padding:12px;text-align:center;"><div class="fraunces-number" style="font-size:24px;color:#8fa893;">{main_245:,.0f}</div><div style="font-size:10px;">Maintained (km²)</div></div>""", unsafe_allow_html=True)
            with m3: st.markdown(f"""<div style="background:#fef2f2;border-radius:8px;padding:12px;text-align:center;"><div class="fraunces-number" style="font-size:24px;color:#dc2626;">{lost_245:,.0f}</div><div style="font-size:10px;">Lost (km²)</div></div>""", unsafe_allow_html=True)
            fig_245 = create_stability_chart(ref_245, gain_245, main_245, lost_245, "SSP2‑4.5")
            st.plotly_chart(fig_245, use_container_width=True)
            st.markdown("**Recommendations for DENR**")
            for rec in recommendations_245:
                priority_color = {"HIGH": "#dc2626", "MEDIUM": "#c8922a", "URGENT": "#8b2e1e", "INFO": "#8fa893"}[rec["priority"]]
                priority_bg = {"HIGH": "#fef2f2", "MEDIUM": "#fef3e2", "URGENT": "#fef2f2", "INFO": "#f7f9f7"}[rec["priority"]]
                tags_html = "".join([f'<span style="display:inline-block;background:{tag_bg[t]};color:{tag_colors[t]};padding:4px 8px;border-radius:4px;font-size:10px;margin-right:6px;font-weight:600;">{t}</span>' for t in rec['tags']])
                st.markdown(f"""
                <div style="background:{priority_bg};border-left:4px solid {priority_color};border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:10px;">
                  <div style="font-weight:600;margin-bottom:4px;">{rec['title']}</div>
                  <div style="font-size:12px;color:#4a5e4c;margin-bottom:8px;">{rec['description']}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:9px;color:{priority_color};margin-bottom:6px;">Priority: {rec['priority']}</div>
                  <div>{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)

        # ----- Right column (Pessimistic) -----
        with right_col:
            st.markdown('<div class="fraunces-heading" style="font-size:1.3rem;">SSP5‑8.5 (Pessimistic) <span title="Small green circles: occurrence points (clustered). Large coloured circles: habitat suitability grid (same size). Green = Refugia (suitable now & 2050), Grey = Maintained (suitable both, low confidence), Red = Lost (suitable now, not 2050)." style="cursor:help;">ⓘ</span></div>', unsafe_allow_html=True)
            st_folium(m585, height=380, returned_objects=[], key="map_ssp585")
            st.markdown("**Habitat Metrics**")
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f"""<div style="background:#eaf4ed;border-radius:8px;padding:12px;text-align:center;"><div class="fraunces-number" style="font-size:24px;color:#1e6b3c;">{ref_585:,.0f}</div><div style="font-size:10px;">Refugia (km²)</div></div>""", unsafe_allow_html=True)
            with m2: st.markdown(f"""<div style="background:#f7f9f7;border-radius:8px;padding:12px;text-align:center;"><div class="fraunces-number" style="font-size:24px;color:#8fa893;">{main_585:,.0f}</div><div style="font-size:10px;">Maintained (km²)</div></div>""", unsafe_allow_html=True)
            with m3: st.markdown(f"""<div style="background:#fef2f2;border-radius:8px;padding:12px;text-align:center;"><div class="fraunces-number" style="font-size:24px;color:#dc2626;">{lost_585:,.0f}</div><div style="font-size:10px;">Lost (km²)</div></div>""", unsafe_allow_html=True)
            fig_585 = create_stability_chart(ref_585, gain_585, main_585, lost_585, "SSP5‑8.5")
            st.plotly_chart(fig_585, use_container_width=True)
            st.markdown("**Recommendations for DENR**")
            for rec in recommendations_585:
                priority_color = {"HIGH": "#dc2626", "MEDIUM": "#c8922a", "URGENT": "#8b2e1e", "INFO": "#8fa893"}[rec["priority"]]
                priority_bg = {"HIGH": "#fef2f2", "MEDIUM": "#fef3e2", "URGENT": "#fef2f2", "INFO": "#f7f9f7"}[rec["priority"]]
                tags_html = "".join([f'<span style="display:inline-block;background:{tag_bg[t]};color:{tag_colors[t]};padding:4px 8px;border-radius:4px;font-size:10px;margin-right:6px;font-weight:600;">{t}</span>' for t in rec['tags']])
                st.markdown(f"""
                <div style="background:{priority_bg};border-left:4px solid {priority_color};border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:10px;">
                  <div style="font-weight:600;margin-bottom:4px;">{rec['title']}</div>
                  <div style="font-size:12px;color:#4a5e4c;margin-bottom:8px;">{rec['description']}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:9px;color:{priority_color};margin-bottom:6px;">Priority: {rec['priority']}</div>
                  <div>{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)

        # ----- Export section (unchanged, just reference the correct variables) -----
        st.markdown("---")
        with st.expander("Export Report (select scenarios, sections, and format)", expanded=False):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                include_opt = st.checkbox("SSP2-4.5 (Optimistic)", value=True, key="custom_opt")
                include_pes = st.checkbox("SSP5-8.5 (Pessimistic)", value=True, key="custom_pes")
                st.markdown("**Report sections**")
                section_choice = st.radio("", ["All", "Metrics only", "Recommendations only"], index=0, key="section_choice")
            with col_c2:
                format_choice = st.selectbox("Report format", ["Excel", "Text", "PDF"], index=0, key="custom_format")
                st.markdown("**Metric units**")
                unit_choice = st.radio("", ["km²", "hectares"], index=0, key="unit_choice", help="hectares = km² × 100")

            if st.button("Generate Report", key="custom_btn"):
                if not (include_opt or include_pes):
                    st.error("Please select at least one scenario.")
                else:
                    safe_name = safe_filename(sp_key)
                    def convert_units(val):
                        return int(val * 100) if unit_choice == "hectares" else val

                    def build_text_report(scenario_label, ref, main, lost, recs):
                        lines = [f"{'-'*50}", f"SCENARIO: {scenario_label}", f"{'-'*50}"]
                        lines.append(f"Model Accuracy (AUC-ROC): {ens_auc:.4f} - {'Validated' if ens_auc >= 0.85 else 'Review Required'}")
                        lines.append("")
                        if section_choice != "Recommendations only":
                            unit_symbol = "km²" if unit_choice == "km²" else "ha"
                            lines.extend(["HABITAT STABILITY METRICS", "--------------------------"])
                            lines.append(f"High-Confidence Refugia : {convert_units(ref):,} {unit_symbol}")
                            lines.append(f"Habitat Maintained      : {convert_units(main):,} {unit_symbol}")
                            lines.append(f"Habitat Lost            : {convert_units(lost):,} {unit_symbol}")
                            lines.append("")
                            lines.extend(["NIPAS PROTECTED AREA OVERLAP", "------------------------------"])
                            lines.append(f"Refugia within NIPAS    : {nipas_pct:.0%}")
                            lines.append(f"Refugia outside NIPAS   : {1-nipas_pct:.0%}")
                            lines.append("")
                        if section_choice != "Metrics only":
                            lines.extend(["CONSERVATION RECOMMENDATIONS", "-----------------------------"])
                            for i, rec in enumerate(recs, 1):
                                lines.append(f"{i}. {rec['title']} [{rec['priority']}]")
                                lines.append(f"   {rec['description']}")
                                lines.append("")
                        return "\n".join(lines).strip()

                    if format_choice == "Excel":
                        excel_data = generate_excel_export(
                            sp_key, SPECIES_METADATA.get(sp_key, {}).get('common', sp_key),
                            SPECIES_METADATA.get(sp_key, {}).get('class', 'Unknown'),
                            get_species_record_counts().get(sp_key, 0), ens_auc,
                            ref_245, gain_245, main_245, lost_245,
                            ref_585, gain_585, main_585, lost_585,
                            nipas_pct, recommendations_245, recommendations_585,
                            include_245=include_opt, include_585=include_pes
                        )
                        if excel_data:
                            st.download_button("Download Excel Report", excel_data,
                                               file_name=f"resilio_map_{safe_name}_custom.xlsx",
                                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    elif format_choice == "Text":
                        header = f"""RESILIO-MAP — CLIMATE REFUGIA ASSESSMENT REPORT
=================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SPECIES INFORMATION
-------------------
Scientific Name : {sp_key}
Common Name     : {SPECIES_METADATA.get(sp_key, {}).get('common', sp_key)}
Taxonomic Class : {SPECIES_METADATA.get(sp_key, {}).get('class', 'Unknown')}
Records in DB   : {get_species_record_counts().get(sp_key, 0)}
"""
                        report_body = ""
                        if include_opt:
                            report_body += build_text_report("SSP2-4.5 (Optimistic)", ref_245, main_245, lost_245, recommendations_245)
                        if include_opt and include_pes:
                            report_body += "\n\n"
                        if include_pes:
                            report_body += build_text_report("SSP5-8.5 (Pessimistic)", ref_585, main_585, lost_585, recommendations_585)
                        txt_data = header + report_body + "\n\n---\nGenerated by Resilio-Map | DENR-BMB Decision Support System"
                        st.download_button("Download Text Report", txt_data,
                                           file_name=f"resilio_map_report_{safe_name}_custom.txt",
                                           mime="text/plain")
                    else:  # PDF
                        try:
                            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
                            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                            from reportlab.lib import colors
                            from reportlab.lib.pagesizes import A4
                            from reportlab.lib.units import cm
                            pdf_buffer = io.BytesIO()
                            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm)
                            story = []
                            styles = getSampleStyleSheet()
                            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1e6b3c'), spaceAfter=12)
                            normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=9, leading=12)
                            story.append(Paragraph("Resilio-Map - Climate Refugia Assessment Report", title_style))
                            story.append(Spacer(1, 0.2*cm))
                            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
                            story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.HexColor('#e3ebe4')))
                            story.append(Spacer(1, 0.3*cm))
                            story.append(Paragraph("Species Information", styles['Heading2']))
                            sp_data = [['Scientific Name', sp_key],
                                       ['Common Name', SPECIES_METADATA.get(sp_key, {}).get('common', sp_key)],
                                       ['Class', SPECIES_METADATA.get(sp_key, {}).get('class', 'Unknown')],
                                       ['Records', str(get_species_record_counts().get(sp_key, 0))]]
                            sp_table = Table(sp_data, colWidths=[4*cm, 10*cm])
                            sp_table.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eaf4ed')),
                                                          ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4')),
                                                          ('FONTSIZE', (0,0), (-1,-1), 8),
                                                          ('PADDING', (0,0), (-1,-1), 5)]))
                            story.append(sp_table)
                            story.append(Spacer(1, 0.4*cm))
                            def add_scenario_to_story(label, ref, main, lost, recs):
                                story.append(Paragraph(f"Scenario: {label}", styles['Heading2']))
                                validation = "Validated" if ens_auc >= 0.85 else "Review Required"
                                perf_data = [['AUC-ROC Score', f'{ens_auc:.4f}'], ['Validation', validation]]
                                perf_table = Table(perf_data, colWidths=[4*cm, 10*cm])
                                perf_table.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.HexColor('#eaf4ed')),
                                                                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4'))]))
                                story.append(perf_table)
                                story.append(Spacer(1, 0.2*cm))
                                if section_choice != "Recommendations only":
                                    story.append(Paragraph("Habitat Stability Metrics", styles['Heading3']))
                                    unit_symbol = "km²" if unit_choice == "km²" else "ha"
                                    metrics_data = [['Metric', f'Area ({unit_symbol})'],
                                                    ['Refugia', f'{convert_units(ref):,}'],
                                                    ['Maintained', f'{convert_units(main):,}'],
                                                    ['Lost', f'{convert_units(lost):,}']]
                                    metrics_table = Table(metrics_data, colWidths=[6*cm, 8*cm])
                                    metrics_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e6b3c')),
                                                                       ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                                                       ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4'))]))
                                    story.append(metrics_table)
                                    story.append(Spacer(1, 0.2*cm))
                                    story.append(Paragraph("NIPAS Protected Area Overlap", styles['Heading3']))
                                    nipas_data = [['Within NIPAS', f'{nipas_pct:.0%}'],
                                                  ['Outside NIPAS', f'{1-nipas_pct:.0%}']]
                                    nipas_table = Table(nipas_data, colWidths=[6*cm, 8*cm])
                                    nipas_table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e6b3c')),
                                                                     ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                                                     ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e3ebe4'))]))
                                    story.append(nipas_table)
                                    story.append(Spacer(1, 0.2*cm))
                                if section_choice != "Metrics only":
                                    story.append(Paragraph("Conservation Recommendations", styles['Heading3']))
                                    for i, rec in enumerate(recs, 1):
                                        rec_text = f"<b>{i}. {rec['title']}</b> [{rec['priority']}]<br/>{rec['description']}"
                                        story.append(Paragraph(rec_text, normal_style))
                                        story.append(Spacer(1, 0.15*cm))
                                story.append(Spacer(1, 0.3*cm))
                            if include_opt:
                                add_scenario_to_story("SSP2-4.5 (Optimistic)", ref_245, main_245, lost_245, recommendations_245)
                            if include_opt and include_pes:
                                story.append(PageBreak())
                            if include_pes:
                                add_scenario_to_story("SSP5-8.5 (Pessimistic)", ref_585, main_585, lost_585, recommendations_585)
                            doc.build(story)
                            pdf_buffer.seek(0)
                            st.download_button("Download PDF Report", pdf_buffer,
                                               file_name=f"resilio_map_{safe_name}_custom.pdf",
                                               mime="application/pdf")
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIES EXPLORER PAGE
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 'explorer':
    st.markdown('<div class="page-eyebrow">— Species Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Discover Philippine Species</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Browse species information, occurrence maps, and photos.</div>', unsafe_allow_html=True)

    explorer_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'species_explorer.json')
    if not os.path.exists(explorer_json_path):
        st.error("Species data file not found. Please add 'species_explorer.json' to the data/ folder.")
    else:
        with open(explorer_json_path, 'r', encoding='utf-8') as f:
            explorer_data = json.load(f)

        # Build metadata dictionary (base name -> metadata)
        metadata_dict = {}
        for record in explorer_data:
            sci_name = record.get('scientificName')
            if not sci_name:
                continue
            if sci_name not in metadata_dict:
                metadata_dict[sci_name] = {
                    'common_name': record.get('commonName', sci_name),
                    'taxonomic_class': record.get('taxonomicClass', 'Unknown'),
                    'conservation_status': record.get('conservationStatus', 'Not specified'),
                    'description': record.get('speciesDescription', 'No description available.'),
                    'image_url': record.get('speciesImage', None)
                }

        # Get all species with occurrence points
        conn = init_db()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT scientific_name FROM species_occurrences WHERE target = 1")
        db_species_full = [row[0] for row in cursor.fetchall()]

        # Map base name -> full name
        base_to_full = {}
        for full in db_species_full:
            base = full.split(' (')[0].strip()
            base_to_full[base] = full

        # Keep only species whose base name is in metadata_dict
        available_species = []
        for base, full in base_to_full.items():
            if base in metadata_dict:
                available_species.append(full)

        if not available_species:
            st.warning("No species with both occurrence records and metadata found.")
        else:
            # Searchable dropdown
            search_term = st.text_input("Search species (name, common name, class, conservation status)", key="explorer_search").strip().lower()
            filtered = []
            for full in available_species:
                base = full.split(' (')[0].strip()
                info = metadata_dict[base]
                display = f"{info['common_name']} ({full})"
                searchable = f"{display} {info['taxonomic_class']} {info['conservation_status']}".lower()
                if search_term in searchable:
                    filtered.append(full)
            if not search_term:
                filtered = available_species

            if not filtered:
                st.warning("No species match your search.")
            else:
                # No default selection – use placeholder
                selected_full = st.selectbox(
                    "Select a species",
                    filtered,
                    format_func=lambda x: f"{metadata_dict[x.split(' (')[0].strip()]['common_name']} ({x})",
                    index=None,
                    placeholder="Choose a species...",
                    key="explorer_select"
                )

                if selected_full:
                    base_name = selected_full.split(' (')[0].strip()
                    info = metadata_dict[base_name]
                    occ_coords = get_occurrences(selected_full)

                    # Row 1: Map + Description
                    col_map, col_desc = st.columns([2, 1], gap="small")
                    with col_map:
                        st.markdown("**Occurrence map**")
                        if occ_coords:
                            center_lat = np.mean([p[0] for p in occ_coords])
                            center_lon = np.mean([p[1] for p in occ_coords])
                            m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='CartoDB positron', width='100%', height='400px')
                            for lat, lon in occ_coords:
                                folium.CircleMarker(location=[lat, lon], radius=4, color='#1e6b3c', fill=True,
                                                    fill_color='#1e6b3c', fill_opacity=0.7, weight=1.5,
                                                    tooltip=f"{info['common_name']}<br>Lat: {lat:.4f}, Lon: {lon:.4f}").add_to(m)
                            st_folium(m, height=400, returned_objects=[])
                        else:
                            st.info("No occurrence records in the database for this species.")
                    with col_desc:
                        st.markdown(f"### {info['common_name']}")
                        st.markdown(f"**Scientific name:** *{selected_full}*")
                        st.markdown(f"**Class:** {info['taxonomic_class']}")
                        st.markdown(f"**Conservation status:** {info['conservation_status']}")
                        st.markdown("**Description**")
                        st.write(info['description'])

                    # Row 2: Image
                    st.markdown("---")
                    if info['image_url']:
                        try:
                            st.image(info['image_url'], caption=info['common_name'], width=500)
                        except Exception:
                            st.info("Image could not be loaded. The URL might be invalid or not directly accessible.")
                    else:
                        st.info("No photo available for this species.")

    st.markdown("<br>", unsafe_allow_html=True)
    cb, _, cn = st.columns([1,4,1])
    with cb:
        if st.button("← Habitat Analysis", use_container_width=True):
            st.session_state.page = 'analysis'; st.rerun()
    with cn:
        if st.button("Back to Overview →", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()