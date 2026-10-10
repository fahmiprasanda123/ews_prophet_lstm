"""
theme.py: Token warna, CSS kecil, dan helper Plotly untuk Agri-AI EWS.

Warna dasar, font, dan pergantian terang/gelap diatur oleh tema bawaan Streamlit
(.streamlit/config.toml, lihat DESIGN.md), sehingga tabel, input, dan grafik
ikut berganti tanpa CSS override. Modul ini hanya menambah komponen yang tidak
dimiliki Streamlit: chip status, kartu EWS, panel, dan footer. Semua warnanya
dipilih agar lolos WCAG AA di kedua tema, jadi tidak perlu tahu tema aktif.
"""
import html

import streamlit as st
import plotly.graph_objects as go


# ======================================================================
# Token
# ======================================================================
# Chip padat: latar + teks dengan kontras >= 5.4:1, berlaku di kedua tema.
STATUS = {
    "danger":  {"bg": "#B42318", "fg": "#FFFFFF"},
    "alert":   {"bg": "#B54708", "fg": "#FFFFFF"},
    "watch":   {"bg": "#F5C451", "fg": "#1C2421"},
    "normal":  {"bg": "#1F6E35", "fg": "#FFFFFF"},
    "neutral": {"bg": "#475467", "fg": "#FFFFFF"},
}

# Garis grafik: kontras >= 3:1 terhadap latar terang (#F7F6F2) dan gelap (#121613).
CHART = {
    "actual": "#3E8E55",
    "forecast": "#A86A10",
    "target": "#C2412D",
    "compare": "#4E7BA8",
    "reference": "#7A847E",
}

FONT_FAMILY = "IBM Plex Sans, sans-serif"

EWS_LEVEL_STATUS = {"Danger": "danger", "Alert": "alert", "Watch": "watch", "Normal": "normal"}
# Sama dengan kosakata pesan engine/ews_engine_v2.py
EWS_LEVEL_LABEL = {"Danger": "Bahaya", "Alert": "Waspada", "Watch": "Perhatian", "Normal": "Normal"}


def get_theme() -> str:
    """Tema aktif menurut Streamlit ('light' atau 'dark'); 'light' bila belum diketahui."""
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


# ======================================================================
# Komponen HTML
# ======================================================================
def status_chip(text: str, status: str) -> str:
    """HTML chip status padat. Warna selalu disertai teks, tidak pernah warna saja."""
    c = STATUS.get(status, STATUS["neutral"])
    return (f'<span class="status-chip" style="background:{c["bg"]};color:{c["fg"]};">'
            f'{html.escape(text)}</span>')


def ews_card(level: str, score: float, message: str) -> str:
    """Kartu status EWS: satu warna solid per level, teks lolos AA."""
    c = STATUS[EWS_LEVEL_STATUS.get(level, "neutral")]
    label = EWS_LEVEL_LABEL.get(level, "Belum ada")
    # Tanpa level yang dikenal, skor 0 akan terbaca sebagai "aman"; jangan tampilkan
    score_html = (f'<div class="ews-card__score">{score:.0f}<span>/100</span></div>'
                  if level in EWS_LEVEL_LABEL else "")
    # Satu baris tanpa indentasi: baris kosong + indentasi akan dibaca markdown sebagai blok kode
    return (f'<div class="ews-card" style="background:{c["bg"]};color:{c["fg"]};" role="status">'
            f'<div class="ews-card__label">Status peringatan dini</div>'
            f'<div class="ews-card__level">{html.escape(label)}</div>'
            f'{score_html}'
            f'<div class="ews-card__message">{html.escape(message)}</div></div>')


def render_sidebar_brand(subtitle: str = "Peringatan dini harga pangan"):
    """Nama produk sebagai wordmark teks (pengganti logo sampai ada logo resmi)."""
    st.sidebar.markdown(
        f'<div class="brand"><div class="brand__name">Agri-AI EWS</div>'
        f'<div class="brand__sub">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def render_theme_toggle():
    """Petunjuk ganti tema. Pergantian tema memakai menu bawaan Streamlit agar
    semua komponen (termasuk tabel dan input) ikut berganti."""
    st.sidebar.caption("Tema terang/gelap: menu ⋮ di kanan atas > Settings > Theme.")


def render_footer(*items: str):
    """Footer satu baris yang membungkus di layar sempit."""
    cells = "".join(f"<span>{html.escape(i)}</span>" for i in items)
    st.markdown(f'<div class="theme-footer">{cells}</div>', unsafe_allow_html=True)


# ======================================================================
# Plotly
# ======================================================================
def apply_theme_to_plotly(fig, **layout_kwargs) -> go.Figure:
    """Rapikan figur Plotly tanpa mengunci warna teks.

    Warna teks, grid, dan latar diserahkan ke tema Streamlit
    (st.plotly_chart memakai theme="streamlit"), sehingga figur tetap terbaca
    saat pengguna mengganti tema.
    """
    legend_config = dict(bgcolor="rgba(0,0,0,0)")
    if isinstance(layout_kwargs.get("legend"), dict):
        legend_config.update(layout_kwargs.pop("legend"))

    title_config = dict(font=dict(size=14))
    user_title = layout_kwargs.pop("title", None)
    if isinstance(user_title, dict):
        title_config.update(user_title)
    elif isinstance(user_title, str):
        title_config["text"] = user_title

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=12),
        title=title_config,
        legend=legend_config,
        **layout_kwargs
    )
    return fig


# ======================================================================
# CSS
# ======================================================================
def inject_theme_css():
    """CSS untuk komponen kustom. Tidak menimpa widget bawaan Streamlit,
    supaya ring fokus, warna tema, dan tata letak mobile tetap bawaan."""
    st.markdown("""
    <style>
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        /* Nilai dan delta metrik membungkus di kolom sempit, tidak terpotong "IDR 2..." */
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        /* Caption bawaan memakai opacity 0.6: hanya 4.0:1 di sidebar terang. 0.8 = 7.3:1 */
        [data-testid="stCaptionContainer"] { opacity: 0.8 !important; }
        /* Fokus keyboard: ring bawaan transparan 50% dan link navigasi hanya diberi latar tipis.
           #3E8E55 = 3.7:1 di latar terang, 4.5:1 di latar gelap. */
        a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, summary:focus-visible {
            outline: 2px solid #3E8E55 !important;
            outline-offset: 2px;
        }
        /* Tabel markdown lebar menggulir di tempat, tidak terpotong di layar ponsel */
        [data-testid="stMarkdownContainer"] table { display: block; max-width: 100%; overflow-x: auto; }

        .brand { margin: 0 0 4px 0; }
        .brand__name { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em; }
        .brand__sub { font-size: 0.85rem; opacity: 0.85; }

        .status-chip {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            line-height: 1.5;
            white-space: nowrap;
        }

        .ews-card {
            border-radius: 6px;
            padding: 16px;
            overflow-wrap: normal;
            word-break: normal;
            hyphens: auto;
        }
        .ews-card__label { font-size: 0.9rem; font-weight: 500; }
        .ews-card__level { font-size: clamp(1.25rem, 2.4vw, 1.75rem); font-weight: 700; line-height: 1.2; margin-top: 2px; }
        .ews-card__score { font-size: clamp(1.75rem, 3vw, 2.25rem); font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.1; margin-top: 8px; }
        .ews-card__score span { font-size: 1rem; font-weight: 500; }
        .ews-card__message { font-size: 0.9rem; line-height: 1.4; margin-top: 8px; }

        .panel {
            border: 1px solid rgba(127,127,127,0.35);
            border-radius: 6px;
            padding: 16px 18px;
            margin: 0 0 12px 0;
        }
        .panel h4 { margin: 0 0 6px 0; padding: 0; }
        .panel p { margin: 0; line-height: 1.6; }

        .direction {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 8px 12px;
        }
        .direction__summary { flex-basis: 100%; margin-top: 6px; line-height: 1.5; }

        .factor { border-top: 1px solid rgba(127,127,127,0.35); padding: 12px 0; }
        .factor__head { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; font-weight: 600; }
        .factor__body { margin-top: 6px; line-height: 1.6; }

        .theme-table { width: 100%; border-collapse: collapse; }
        .theme-table td { padding: 8px 12px 8px 0; border-bottom: 1px solid rgba(127,127,127,0.35); vertical-align: top; }
        .theme-table td:first-child { width: 160px; font-weight: 600; }
        @media (max-width: 640px) {
            .theme-table td { display: block; width: auto !important; border-bottom: none; padding: 2px 0; }
            .theme-table tr { display: block; padding: 8px 0; border-bottom: 1px solid rgba(127,127,127,0.35); }
        }

        .theme-footer {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 4px 24px;
            font-size: 0.85rem;
            padding: 8px 0;
        }
    </style>
    """, unsafe_allow_html=True)
