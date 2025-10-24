"""
How to boost your Shortgame – Webapp
powered by Niko Schicketanz

A modern (dual‑mode) app to discover trending YouTube Shorts by **topic** (not channel),
filter by views and strict short length, show thumbnails + links, compute velocity metrics,
suggest best posting windows, pull transcripts on demand, and auto‑translate to German.

✅ Fix for `ModuleNotFoundError: No module named 'streamlit'`
Runs in two modes:
1) **Streamlit UI** (preferred) – if `streamlit` is installed.
2) **CLI fallback** – discovery/export in terminal if `streamlit` is missing.

Latest upgrades
- Ultra metrics: **Views/Tag**, **Views/Stunde**, **Like/View-%**, **Velocity-Score**.
- Best Posting Window (Europe/Berlin), weighted by Velocity.
- Card grid with **thumbnails** + key KPIs.
- NEW: **Hook & Caption Generator (DE)** — always ≥3 Varianten, aus Titel/Thema/Transcript.

---
Quickstart (UI)
1) Create env & install deps
   pip install -U streamlit google-api-python-client google-auth google-auth-oauthlib \
      python-dotenv youtube-transcript-api deepl openai tiktoken pandas python-dateutil
2) Put credentials in `.env`
   YT_API_KEY=YOUR_YOUTUBE_DATA_API_KEY
   DEEPL_API_KEY=...
   OPENAI_API_KEY=...
3) (Optional) Google OAuth JSON → `client_secret.json` (Desktop app)
4) Run UI
   streamlit run app.py

Quickstart (CLI)
   python app.py --topics "AI,Business" --regions US DE GB \
      --min-views 20000 --min-vpd 500 --min-like 1.0 --max-sec 40 --window-days 14 --limit 15 --sort velocity

Notes
- Uses YouTube Data API v3. "Shorts" approximated by duration; we enforce **≤60s** and you choose the stricter **max (default 40s)**.
- Transcripts via `youtube-transcript-api`; translation via DeepL/OpenAI if keys exist.
- Preview thumbnails; Views/Hour for fresh uploads; Best Posting Window (Europe/Berlin).
- **Google OAuth (ohne API‑Key):** In Streamlit Cloud unter *Settings → Secrets* folgende Werte setzen:
  - `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (**Typ "Web application"**)  
  - `OAUTH_REDIRECT_URI` = **deine Streamlit‑App‑URL** (z. B. `https://deinname-shortgame-webapp.streamlit.app`)  
  Danach in der App auf **„Mit Google anmelden (ohne API‑Key)“** klicken.

"""

from __future__ import annotations

import os
import re
import sys
import csv
import json
import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dateutil import parser as dtparser
from typing import List, Dict, Optional, Tuple

import pandas as pd
# dotenv is optional (Streamlit Cloud sometimes lacks it until deps finish). Fallback no-op.
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

# --- Optional UI dependency -------------------------------------------------
try:
    import streamlit as _st  # type: ignore
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    _st = None
    STREAMLIT_AVAILABLE = False

# Unified cache decorator: no‑op if streamlit is missing

def cache_data(*dargs, **dkwargs):
    def _wrap(fn):
        if STREAMLIT_AVAILABLE:
            return _st.cache_data(*dargs, **dkwargs)(fn)  # type: ignore
        return fn
    return _wrap

# --- YouTube & Auth ---------------------------------------------------------
# Try official google client; if missing we will fallback to a lightweight HTTP client
try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
except Exception:
    build = None  # type: ignore
    InstalledAppFlow = None  # type: ignore
    Credentials = None  # type: ignore

# Minimal HTTP fallback (no google-api-python-client required)
import urllib.parse as _urlparse
import urllib.request as _urlrequest

def _http_get_json(url: str, params: dict, headers: dict | None = None) -> dict:
    qs = _urlparse.urlencode({k: v for k, v in params.items() if v is not None})
    full = url + ("?" + qs if qs else "")
    req = _urlrequest.Request(full, headers=headers or {})
    with _urlrequest.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)

class _YTReq:
    def __init__(self, url: str, params: dict, headers: dict):
        self.url = url
        self.params = params
        self.headers = headers
    def execute(self) -> dict:
        return _http_get_json(self.url, self.params, self.headers)

class SimpleYouTubeClient:
    """Tiny wrapper mimicking googleapiclient's .search().list(...).execute() API."""
    def __init__(self, api_key: str | None = None, oauth_token: str | None = None):
        self.api_key = api_key
        self.oauth_token = oauth_token
    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.oauth_token:
            h["Authorization"] = f"Bearer {self.oauth_token}"
        return h
    def search(self):
        parent = self
        class _Search:
            def list(self, **kwargs):
                params = kwargs.copy()
                if parent.api_key:
                    params["key"] = parent.api_key
                return _YTReq("https://www.googleapis.com/youtube/v3/search", params, parent._headers())
        return _Search()
    def videos(self):
        parent = self
        class _Videos:
            def list(self, **kwargs):
                params = kwargs.copy()
                if parent.api_key:
                    params["key"] = parent.api_key
                return _YTReq("https://www.googleapis.com/youtube/v3/videos", params, parent._headers())
        return _Videos()

# Transcripts
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound,
    )
except Exception:
    YouTubeTranscriptApi = None  # type: ignore
    TranscriptsDisabled = Exception  # type: ignore
    NoTranscriptFound = Exception  # type: ignore

# Optional translation
try:
    import deepl
except Exception:
    deepl = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# ----------------------------
# Config
# ----------------------------
APP_TITLE = "How to boost your Shortgame"
APP_SUBTITLE = "powered by Niko Schicketanz"
DEFAULT_MIN_VIEWS = 20000
DEFAULT_MAX_SECONDS = 40  # UI default; hard cap is 60
DEFAULT_WINDOW_DAYS = 14
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

# ----------------------------
# Helpers & Metrics
# ----------------------------

@cache_data(show_spinner=False)
def yt_client_with_key(api_key: str):
    """Return a YouTube client. Falls back to HTTP shim if google client not installed."""
    if build is not None:
        try:
            return build("youtube", "v3", developerKey=api_key, static_discovery=False)
        except Exception:
            pass
    # Fallback
    return SimpleYouTubeClient(api_key=api_key)


def yt_client_with_oauth(creds: Credentials):
    """Return a YouTube client using OAuth credentials; fallback to HTTP shim."""
    if build is not None:
        try:
            return build("youtube", "v3", credentials=creds, static_discovery=False)
        except Exception:
            pass
    token = getattr(creds, "token", None)
    return SimpleYouTubeClient(api_key=None, oauth_token=token)


def ensure_credentials_st() -> Optional["Credentials"]:
    """Google OAuth Sign‑in for Streamlit Cloud (no API key needed).

    Priority:
    1) If creds already in session, return them.
    2) If secrets ENV (GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET/OAUTH_REDIRECT_URI) exist → do **web OAuth**
       with redirect back to this Streamlit app (handles `code` in query params).
    3) Fallback (local dev): if `client_secret.json` is present → Desktop OAuth.
    """
    if not STREAMLIT_AVAILABLE:
        return None

    # 3) Fallback Desktop OAuth (lokale Entwicklung)
    if os.path.exists("client_secret.json") and InstalledAppFlow is not None:
        if st.button("Mit Google anmelden (lokal)", type="secondary"):
            creds = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES).run_local_server(port=0, prompt="consent")
            st.session_state["oauth_creds"] = creds
            st.success("Erfolgreich angemeldet.")
            return creds

    return None
    st = _st
    if "oauth_creds" in st.session_state and st.session_state["oauth_creds"]:
        return st.session_state["oauth_creds"]
    if not os.path.exists("client_secret.json") or InstalledAppFlow is None:
        return None
    if st.button("Mit Google anmelden", type="secondary"):
        flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent")
        st.session_state["oauth_creds"] = creds
        st.success("Erfolgreich angemeldet.")
        return creds
    return None


def parse_iso8601_duration(iso_duration: str) -> int:
    """Return duration in seconds from ISO8601 durations like PT#H#M#S.

    >>> parse_iso8601_duration("PT40S")
    40
    >>> parse_iso8601_duration("PT1M20S")
    80
    >>> parse_iso8601_duration("PT2H")
    7200
    >>> parse_iso8601_duration("PT")  # empty duration → 0
    0
    >>> parse_iso8601_duration("n/a")
    0
    """
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration or "")
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + s


def _safe_like_view_ratio(likes: Optional[int], views: Optional[int]) -> float:
    """Return like/view percentage (0..100). None or 0 views → 0.

    >>> _safe_like_view_ratio(100, 10000)
    1.0
    >>> _safe_like_view_ratio(None, 50000)
    0.0
    >>> _safe_like_view_ratio(100, 0)
    0.0
    """
    if not likes or not views:
        return 0.0
    if views <= 0:
        return 0.0
    return round((likes / views) * 100.0, 3)


def _age_days(published_at: str, now: Optional[datetime] = None) -> float:
    """Return age in days (float). Handles missing/invalid timestamps.

    >>> round(_age_days("2025-01-02T00:00:00Z", now=datetime(2025,1,03,tzinfo=timezone.utc)), 3)
    1.0
    >>> _age_days("invalid") >= 0
    True
    """
    if not now:
        now = datetime.now(timezone.utc)
    try:
        dt = dtparser.parse(published_at)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        return max(delta.total_seconds() / 86400.0, 0.0001)
    except Exception:
        return 0.0001


def attach_metrics(df: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """Attach Age (days), Views/Day, Views/Hour (for fresh videos) and Velocity-Score.

    Velocity-Score = Views/Day * (1 + 2 * Like/View_fraction)
    (Like/View_fraction = Like/View-% divided by 100)

    >>> import pandas as _pd
    >>> _df = _pd.DataFrame({
    ...   'viewCount':[50000, 20000],
    ...   'likeCount':[1000, 100],
    ...   'publishedAt':["2025-01-01T00:00:00Z","2025-01-02T00:00:00Z"]
    ... })
    >>> out = attach_metrics(_df, now=datetime(2025,1,03,tzinfo=timezone.utc))
    >>> set(['ageDays','viewsPerDay','viewsPerHour','likeViewPct','velocityScore']).issubset(out.columns)
    True
    """
    if df.empty:
        return df
    now = now or datetime.now(timezone.utc)
    ages = df["publishedAt"].apply(lambda x: _age_days(str(x), now))
    vpd = df["viewCount"].astype(float) / ages
    hours = (ages * 24.0).clip(lower=0.01)
    vph = (df["viewCount"].astype(float) / hours).round(2)
    lvp = df.apply(lambda r: _safe_like_view_ratio(r.get("likeCount"), r.get("viewCount")), axis=1)
    vel = vpd * (1.0 + 2.0 * (lvp / 100.0))
    out = df.copy()
    out["ageDays"] = ages
    out["viewsPerDay"] = vpd.round(2)
    out["viewsPerHour"] = vph
    out["likeViewPct"] = lvp
    out["velocityScore"] = vel.round(2)
    return out


def filter_candidates(df: pd.DataFrame, min_views: int, max_seconds: int,
                      min_vpd: float = 0.0, min_like_pct: float = 0.0) -> pd.DataFrame:
    """Filter candidates by views and length, enforce ≤60s, plus velocity & like ratio.

    >>> import pandas as _pd
    >>> _df = _pd.DataFrame({
    ...     'videoId': ['a','b','c','d'],
    ...     'viewCount': [50000, 10000, 25000, 30000],
    ...     'durationSec': [35, 45, 30, 60],
    ...     'publishedAt': ["2025-01-01T00:00:00Z"]*4,
    ...     'likeCount': [1000, 10, 50, 900]
    ... })
    >>> out = attach_metrics(_df, now=datetime(2025,1,02,tzinfo=timezone.utc))
    >>> list(filter_candidates(out, 20000, 40)['videoId'])
    ['a', 'c']
    >>> list(filter_candidates(out, 20000, 40, min_vpd=20000)['videoId'])  # very strict
    ['a']
    >>> list(filter_candidates(out, 20000, 60, min_like_pct=2.0)['videoId'])
    ['a', 'd']
    """
    cap = min(int(max_seconds), 60)
    base = df[(df["durationSec"] <= cap) & (df["viewCount"] >= int(min_views))]
    if min_vpd > 0 and "viewsPerDay" in base.columns:
        base = base[base["viewsPerDay"] >= float(min_vpd)]
    if min_like_pct > 0 and "likeViewPct" in base.columns:
        base = base[base["likeViewPct"] >= float(min_like_pct)]
    return base.copy()


def make_watch_url(video_id: str) -> str:
    """Build a standard YouTube watch URL.

    >>> make_watch_url('abc123')
    'https://www.youtube.com/watch?v=abc123'
    """
    return f"https://www.youtube.com/watch?v={video_id}"


# ----------------------------
# Language helpers
# ----------------------------

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

def _infer_lang_from_row(row: pd.Series) -> str:
    la = (row.get("defaultAudioLanguage") or row.get("defaultLanguage") or "").lower()
    if la.startswith("de"):
        return "de"
    if la.startswith("en"):
        return "en"
    # heuristic: if title has many CJK chars → 'other'
    title = str(row.get("title", ""))
    if _CJK_RE.search(title):
        return "other"
    # unknown default to 'en' if ASCII heavy, else 'other'
    try:
        if all(ord(ch) < 128 for ch in title):
            return "en"
    except Exception:
        pass
    return "other"

def filter_languages(df: pd.DataFrame, allowed: List[str]) -> pd.DataFrame:
    if df.empty or not allowed:
        return df
    langs = df.apply(_infer_lang_from_row, axis=1)
    return df[langs.isin([x.lower() for x in allowed])].copy()

# ----------------------------
# YouTube Search Logic
# ----------------------------

def search_shorts(yt, query: str, regions: List[str], published_after: datetime, max_results: int = 50, langs: Optional[List[str]] = None) -> List[str]:
    """Search for videos matching query; return a de‑duplicated list of video IDs.
    NOTE: We refine by duration later.
    """
    video_ids: List[str] = []
    langs = langs or []
    seen: set[str] = set()
    for rc in regions:
        # If user picked exactly one language, hint the API to boost that language
        relevance_lang = (langs[0] if len(langs) == 1 else None) or None
        req = yt.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=min(max_results, 50),
            order="viewCount",
            publishedAfter=published_after.isoformat("T") + "Z",
            regionCode=rc,
            relevanceLanguage=relevance_lang,
            videoDuration="short",  # API short = <4 min
        )
        res = req.execute()
        for item in res.get("items", []):
            vid = item["id"].get("videoId")
            if vid and vid not in seen:
                seen.add(vid)
                video_ids.append(vid)
    return video_ids


def fetch_video_stats(yt, video_ids: List[str]) -> pd.DataFrame:
    rows = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        if not chunk:
            continue
        res = yt.videos().list(part="snippet,contentDetails,statistics", id=",".join(chunk)).execute()
        for it in res.get("items", []):
            dur = parse_iso8601_duration(it.get("contentDetails", {}).get("duration", ""))
            stats = it.get("statistics", {})
            sn = it.get("snippet", {})
            thumbs = (sn.get("thumbnails", {}) or {})
            thumb_url = (
                thumbs.get("high", {}).get("url")
                or thumbs.get("medium", {}).get("url")
                or thumbs.get("default", {}).get("url")
                or ""
            )
            rows.append(
                {
                    "videoId": it.get("id"),
                    "title": sn.get("title", ""),
                    "channelId": sn.get("channelId", ""),
                    "channelTitle": sn.get("channelTitle", ""),
                    "publishedAt": sn.get("publishedAt", ""),
                    "durationSec": dur,
                    "viewCount": int(stats.get("viewCount", 0) or 0),
                    "likeCount": int(stats.get("likeCount", 0) or 0) if stats.get("likeCount") else None,
                    "commentCount": int(stats.get("commentCount", 0) or 0) if stats.get("commentCount") else None,
                    "thumbnailUrl": thumb_url,
                    "defaultAudioLanguage": sn.get("defaultAudioLanguage", ""),
                    "defaultLanguage": sn.get("defaultLanguage", ""),
                }
            )
    df = pd.DataFrame(rows)
    return attach_metrics(df)


# ----------------------------
# Transcripts & Translation
# ----------------------------

def get_transcript(video_id: str, preferred_langs=("en", "de")) -> Tuple[str, str]:
    """Return (language, transcript_text). Returns ("", "") if unavailable.

    Implementation tries requested languages first, then auto captions.
    """
    if not YouTubeTranscriptApi:
        return "", ""
    try:
        trs = YouTubeTranscriptApi.list_transcripts(video_id)
        for lang in preferred_langs:
            try:
                t = trs.find_transcript([lang]).fetch()
                text = " ".join(x.get("text", "") for x in t)
                return lang, text
            except Exception:
                pass
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception:
        pass
    try:
        t = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(x.get("text", "") for x in t)
        return "auto", text
    except Exception:
        return "", ""


def translate_to_de(text: str) -> str:
    if not text:
        return ""
    if os.getenv("DEEPL_API_KEY") and deepl:
        try:
            tr = deepl.Translator(os.getenv("DEEPL_API_KEY"))
            return tr.translate_text(text, target_lang="DE").text
        except Exception:
            pass
    if os.getenv("OPENAI_API_KEY") and OpenAI:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = f"Übersetze ins Deutsche. Erhalte Eigennamen/URLs.\n\n{text[:120000]}"
            resp = client.responses.create(model="gpt-4o-mini", input=prompt)
            return resp.output_text.strip()
        except Exception:
            pass
    return text  # fallback


# ----------------------------
# Hook & Caption Generator (DE)
# ----------------------------

def generate_hooks_captions_de(title: str = "", transcript: str = "", topic: str = "", n: int = 3) -> Tuple[List[str], List[str]]:
    """Generate ≥n hooks and captions in German. Uses OpenAI if available, else rule-based.

    >>> hs, cs = generate_hooks_captions_de("Titel", "Kurz Transcript", "Thema", n=3)
    >>> len(hs) >= 3 and len(cs) >= 3
    True
    """
    n = max(int(n), 3)
    ctx = (topic or title or "Dein Thema").strip()[:200]
    base_caps = [
        "Jetzt umsetzen → #Shorts #Tipps #Growth",
        "Speichern & später testen. #Business #KI",
        "Mehr davon? Folge für täglich 1 Nugget."
    ]

    # If OpenAI available, try a structured JSON response
    if OpenAI and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                "Erzeuge kreative deutsche HOOKS und CAPTIONS für ein YouTube Short. "
                "Liefere JSON mit Schlüsseln 'hooks' und 'captions'. "
                "Kontext: " + ctx + ". "
                f"Mindestens {n} Varianten, knackig, maximal 90 Zeichen je Hook, 140 je Caption. "
                "Vermeide Clickbait ohne Inhalt; gib klare Nutzenversprechen."
            )
            resp = client.responses.create(model="gpt-4o-mini", input=prompt)
            txt = resp.output_text.strip()
            # naive JSON sniff
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(txt[start:end+1])
                hooks = [h.strip() for h in data.get("hooks", []) if h.strip()]
                caps = [c.strip() for c in data.get("captions", []) if c.strip()]
                # pad if fewer than n
                while len(hooks) < n:
                    hooks.append(f"{ctx}: So gehst du in 3 Schritten vor →")
                while len(caps) < n:
                    caps.append(f"{ctx} in 3 Schritten. Speichern & ausprobieren. #Shorts")
                return hooks[:n], caps[:n]
        except Exception:
            pass

    # Fallback: rule-based templates using topic/title and transcript keywords
    key = ctx
    hooks = [
        f"{key}: 3 Hacks in 30s",
        f"{key} – der eine Move, den alle vergessen",
        f"{key} schneller umsetzen: so geht’s",
        f"{key} in 3 Schritten (kurz & knackig)",
    ][:max(n, 3)]
    caps = [
        f"{key} kompakt erklärt. Teste heute 1 Schritt. {base_caps[0]}",
        f"So startest du mit {key} – einfach & messbar. {base_caps[1]}",
        f"{key}: Framework + Mini-Checkliste im Video. {base_caps[2]}",
        f"Speicher dir {key} ab und bau’s diese Woche ein.",
    ][:max(n, 3)]
    return hooks[:n], caps[:n]


# ----------------------------
# Hashtag Generator (DE) + Copy helpers
# ----------------------------

def generate_hashtags_de(topic: str = "", title: str = "", n: int = 12) -> List[str]:
    """Simple relevant hashtag generator for DE niches; returns unique tags.

    >>> tags = generate_hashtags_de("KI Business", "How to level up your Business with KI", n=10)
    >>> any(t.lower() == '#shorts' for t in tags) and len(tags) >= 10
    True
    """
    base = (topic + " " + title).lower()
    tokens = list(dict.fromkeys(re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", base)))
    tags: List[str] = []

    def add(t: str):
        if not t:
            return
        if t[0] != '#':
            t = '#' + t
        if t.lower() not in [x.lower() for x in tags]:
            tags.append(t)

    # core keywords → hashtags
    for t in tokens:
        if len(t) <= 20 and not t.isdigit():
            add(t)

    # mappings
    if any(k in base for k in ["ki", "ai", "gpt", "modell"]):
        for t in ["KI", "AI", "künstlicheintelligenz", "automation", "noCode", "prompting"]:
            add(t)
    if any(k in base for k in ["business", "unternehmen", "sales", "growth", "marketing"]):
        for t in ["Business", "Unternehmen", "Marketing", "Growth", "Vertrieb", "Strategie"]:
            add(t)
    if any(k in base for k in ["youtube", "shorts"]):
        for t in ["YouTubeShorts", "Shorts", "Content"]:
            add(t)

    # always include some general reach tags
    for t in ["Shorts", "YouTubeShorts", "lernen", "Tipps", "Deutsch", "viral"]:
        add(t)

    # limit & pretty-case
    tags = [('#' + t.lstrip('#')).replace(' ', '') for t in tags]
    return tags[: max(3, n)]


def _render_copy_button(st, label: str, text: str, key: str):
    """Render a small HTML copy button using the Clipboard API.

    This uses unsafe_allow_html=True; Streamlit may restrict in some environments.
    """
    js_text = json.dumps(text)
    html = f"""
    <button onclick="navigator.clipboard.writeText({js_text}); this.innerText='Kopiert!'; setTimeout(()=>this.innerText='{label}',1200);" style="margin-left:6px;padding:4px 8px;">{label}</button>
    """
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------
# STREAMLIT UI (only when available)
# ----------------------------

def _in_streamlit_runtime() -> bool:
    """Detect if the script is running inside Streamlit's runtime."""
    if not STREAMLIT_AVAILABLE:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False

# ----------------------------

def _best_posting_windows(df: pd.DataFrame, tz_name: str = "Europe/Berlin", top_k: int = 3) -> List[Tuple[int, float]]:
    """Return top-k local hours (0-23) weighted by velocityScore.

    >>> import pandas as _pd
    >>> _df = _pd.DataFrame({
    ...   'publishedAt':["2025-01-01T10:00:00Z","2025-01-02T18:00:00Z","2025-01-03T18:30:00Z"],
    ...   'velocityScore':[1000,2000,1500]
    ... })
    >>> tops = _best_posting_windows(_df, tz_name="UTC", top_k=2)
    >>> [h for h,_ in tops]
    [18, 10]
    """
    if df.empty or "publishedAt" not in df.columns:
        return []
    try:
        ts = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce").dt.tz_convert(ZoneInfo(tz_name))
        hours = ts.dt.hour
        weights = df.get("velocityScore", pd.Series([1]*len(df)))
        agg = pd.DataFrame({"hour": hours, "w": weights}).groupby("hour")["w"].sum().sort_values(ascending=False)
        return [(int(h), float(s)) for h, s in agg.head(top_k).items()]
    except Exception:
        return []


def _hour_range_label(h: int) -> str:
    """Format hour as a 2h posting window, e.g., 18 → "18:00–20:00".

    >>> _hour_range_label(23)
    '23:00–01:00'
    """
    h2 = (h + 2) % 24
    return f"{h:02d}:00–{h2:02d}:00"


def _render_card(st, row: pd.Series):
    url = make_watch_url(row["videoId"])
# ----------------------------
# Entry points for Streamlit & CLI
# ----------------------------
if _in_streamlit_runtime():
    run_streamlit_ui()
elif __name__ == "__main__":
    run_cli()
