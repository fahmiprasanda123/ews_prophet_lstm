"""
theme.py — Centralized Theme Manager for Agri-AI EWS.

Provides light/dark mode toggle, comprehensive CSS injection covering
all Streamlit native widgets, Plotly figure styling helpers, and inline color helpers.
"""
import streamlit as st
import plotly.graph_objects as go


# ======================================================================
# Color Palettes
# ======================================================================
DARK_PALETTE = {
    "bg_primary": "#0E1117",
    "bg_secondary": "#161B22",
    "bg_app": "linear-gradient(135deg, #0E1117 0%, #161B22 100%)",
    "bg_card": "rgba(255, 255, 255, 0.04)",
    "bg_card_hover": "rgba(255, 255, 255, 0.07)",
    "sidebar_bg": "#0B0E14",
    "sidebar_border": "rgba(255, 255, 255, 0.08)",
    "sidebar_nav_active_bg": "rgba(79, 172, 254, 0.15)",
    "sidebar_nav_active_text": "#4facfe",
    "border": "rgba(255, 255, 255, 0.1)",
    "border_accent": "rgba(79, 172, 254, 0.3)",
    "border_card": "rgba(255, 255, 255, 0.15)",
    "text_primary": "#F0F6FC",
    "text_secondary": "#8B949E",
    "text_muted": "#C9D1D9",
    "text_heading": "#FFFFFF",
    "text_table": "#E6EDF3",
    "accent": "#4facfe",
    "accent_secondary": "#00f2fe",
    "accent_bg": "linear-gradient(135deg, rgba(79,172,254,0.12) 0%, rgba(0,242,254,0.06) 100%)",
    "accent_border": "rgba(79,172,254,0.3)",
    "shadow": "rgba(0, 0, 0, 0.4)",
    "shadow_heavy": "rgba(0, 0, 0, 0.6)",
    "input_bg": "#161B22",
    "input_border": "rgba(255, 255, 255, 0.15)",
    "popover_bg": "#161B22",
    "menu_item_hover": "rgba(255, 255, 255, 0.08)",
    "expander_bg": "rgba(255, 255, 255, 0.03)",
    "btn_bg": "linear-gradient(90deg, #4facfe 0%, #00f2fe 100%)",
    "btn_text": "#FFFFFF",
    "btn_border": "transparent",
    "btn_shadow": "rgba(0, 242, 254, 0.25)",
    "toggle_btn_bg": "linear-gradient(135deg, #1f293d 0%, #111827 100%)",
    "toggle_btn_border": "rgba(79, 172, 254, 0.35)",
    "toggle_btn_text": "#4facfe",
    "plotly_paper": "rgba(0,0,0,0)",
    "plotly_plot": "rgba(0,0,0,0)",
    "plotly_grid": "rgba(255, 255, 255, 0.08)",
    "plotly_actual_line": "#FFFFFF",
    "plotly_hline": "#E0E0E0",
    "chat_msg_bg": "rgba(255, 255, 255, 0.03)",
    "chat_msg_border": "rgba(255, 255, 255, 0.08)",
}

LIGHT_PALETTE = {
    "bg_primary": "#F8FAFC",
    "bg_secondary": "#FFFFFF",
    "bg_app": "linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%)",
    "bg_card": "#FFFFFF",
    "bg_card_hover": "#F8FAFC",
    "sidebar_bg": "#FFFFFF",
    "sidebar_border": "#E2E8F0",
    "sidebar_nav_active_bg": "rgba(37, 99, 235, 0.08)",
    "sidebar_nav_active_text": "#2563EB",
    "border": "#E2E8F0",
    "border_accent": "rgba(37, 99, 235, 0.25)",
    "border_card": "#E2E8F0",
    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#475569",
    "text_heading": "#0F172A",
    "text_table": "#1E293B",
    "accent": "#2563EB",
    "accent_secondary": "#0284C7",
    "accent_bg": "linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(2,132,199,0.03) 100%)",
    "accent_border": "rgba(37,99,235,0.2)",
    "shadow": "rgba(0, 0, 0, 0.05)",
    "shadow_heavy": "rgba(0, 0, 0, 0.1)",
    "input_bg": "#FFFFFF",
    "input_border": "#CBD5E1",
    "popover_bg": "#FFFFFF",
    "menu_item_hover": "#F1F5F9",
    "expander_bg": "#FFFFFF",
    "btn_bg": "linear-gradient(90deg, #2563EB 0%, #0284C7 100%)",
    "btn_text": "#FFFFFF",
    "btn_border": "transparent",
    "btn_shadow": "rgba(37, 99, 235, 0.25)",
    "toggle_btn_bg": "linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)",
    "toggle_btn_border": "rgba(37, 99, 235, 0.3)",
    "toggle_btn_text": "#1D4ED8",
    "plotly_paper": "rgba(0,0,0,0)",
    "plotly_plot": "rgba(0,0,0,0)",
    "plotly_grid": "rgba(0,0,0,0.07)",
    "plotly_actual_line": "#0F172A",
    "plotly_hline": "#334155",
    "chat_msg_bg": "#FFFFFF",
    "chat_msg_border": "#E2E8F0",
}


# ======================================================================
# Theme State Management
# ======================================================================
def get_theme() -> str:
    """Return the current theme name: 'dark' or 'light'."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"
    return st.session_state.theme_mode


def _get_palette() -> dict:
    """Return the active palette dict."""
    return DARK_PALETTE if get_theme() == "dark" else LIGHT_PALETTE


def theme_color(key: str) -> str:
    """Return a single color value for use in inline HTML/CSS."""
    return _get_palette()[key]


# ======================================================================
# Sidebar Toggle
# ======================================================================
def render_theme_toggle():
    """Render the light/dark mode toggle in the sidebar."""
    current = get_theme()
    if current == "dark":
        label = "☀️ Switch to Light Mode"
    else:
        label = "🌙 Switch to Dark Mode"

    if st.sidebar.button(label, key="theme_toggle_btn", use_container_width=True):
        st.session_state.theme_mode = "light" if current == "dark" else "dark"
        st.rerun()


# ======================================================================
# Plotly Complete Theming Helper
# ======================================================================
def get_plotly_template() -> str:
    """Return appropriate Plotly template name."""
    return "plotly_dark" if get_theme() == "dark" else "plotly_white"


def get_plotly_layout() -> dict:
    """Return a dict of common Plotly layout overrides for the current theme."""
    p = _get_palette()
    return dict(
        template=get_plotly_template(),
        paper_bgcolor=p["plotly_paper"],
        plot_bgcolor=p["plotly_plot"],
        font=dict(color=p["text_primary"], family="sans-serif", size=12),
        title=dict(font=dict(color=p["text_heading"], size=14)),
        legend=dict(font=dict(color=p["text_primary"], size=11)),
    )


def get_plotly_yaxis() -> dict:
    """Return yaxis style dict."""
    p = _get_palette()
    return dict(
        gridcolor=p["plotly_grid"],
        zerolinecolor=p["plotly_grid"],
        tickfont=dict(color=p["text_secondary"], size=11),
        title_font=dict(color=p["text_primary"], size=12),
    )


def get_plotly_xaxis() -> dict:
    """Return xaxis style dict."""
    p = _get_palette()
    return dict(
        gridcolor=p["plotly_grid"],
        zerolinecolor=p["plotly_grid"],
        tickfont=dict(color=p["text_secondary"], size=11),
        title_font=dict(color=p["text_primary"], size=12),
    )


def apply_theme_to_plotly(fig, **layout_kwargs) -> go.Figure:
    """Apply complete, crisp theme formatting to any Plotly figure.
    
    Ensures all titles, axis labels, tick marks, legend texts, colorbars,
    and gridlines have crystal-clear contrast in both Light and Dark modes.
    """
    p = _get_palette()
    template = get_plotly_template()

    # Dynamic legend handling (avoids duplicate keyword argument error)
    legend_config = dict(
        font=dict(color=p["text_primary"], size=11),
        bgcolor="rgba(0,0,0,0)",
    )
    if "legend" in layout_kwargs:
        user_legend = layout_kwargs.pop("legend")
        if isinstance(user_legend, dict):
            if "font" not in user_legend:
                user_legend["font"] = legend_config["font"]
            legend_config.update(user_legend)

    # Dynamic title handling
    title_config = dict(font=dict(color=p["text_heading"], size=14))
    if "title" in layout_kwargs:
        user_title = layout_kwargs.pop("title")
        if isinstance(user_title, dict):
            if "font" not in user_title:
                user_title["font"] = title_config["font"]
            title_config.update(user_title)
        elif isinstance(user_title, str):
            title_config["text"] = user_title

    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["text_primary"], family="sans-serif", size=12),
        title=title_config,
        legend=legend_config,
        **layout_kwargs
    )

    fig.update_xaxes(
        tickfont=dict(color=p["text_secondary"], size=11),
        title_font=dict(color=p["text_primary"], size=12),
        gridcolor=p["plotly_grid"],
        zerolinecolor=p["plotly_grid"],
    )

    fig.update_yaxes(
        tickfont=dict(color=p["text_secondary"], size=11),
        title_font=dict(color=p["text_primary"], size=12),
        gridcolor=p["plotly_grid"],
        zerolinecolor=p["plotly_grid"],
    )

    # If figure has colorbar (e.g. from px.bar or px.imshow)
    if hasattr(fig.layout, "coloraxis") and fig.layout.coloraxis is not None:
        try:
            fig.update_coloraxes(
                colorbar=dict(
                    tickfont=dict(color=p["text_secondary"], size=10),
                    title=dict(font=dict(color=p["text_primary"], size=11))
                )
            )
        except Exception:
            pass

    return fig


# ======================================================================
# CSS Injection (Full Coverage for Streamlit Native Elements)
# ======================================================================
def inject_theme_css():
    """Inject global CSS variables and comprehensive styles covering all Streamlit components."""
    p = _get_palette()
    theme = get_theme()

    css = f"""
    <style>
        /* === THEME: {theme.upper()} === */
        :root {{
            --bg-primary: {p['bg_primary']};
            --bg-secondary: {p['bg_secondary']};
            --bg-app: {p['bg_app']};
            --bg-card: {p['bg_card']};
            --bg-card-hover: {p['bg_card_hover']};
            --border-color: {p['border']};
            --border-accent: {p['border_accent']};
            --border-card: {p['border_card']};
            --text-primary: {p['text_primary']};
            --text-secondary: {p['text_secondary']};
            --text-muted: {p['text_muted']};
            --text-heading: {p['text_heading']};
            --text-table: {p['text_table']};
            --accent: {p['accent']};
            --accent-secondary: {p['accent_secondary']};
            --accent-bg: {p['accent_bg']};
            --accent-border: {p['accent_border']};
            --shadow: {p['shadow']};
            --shadow-heavy: {p['shadow_heavy']};
            --sidebar-bg: {p['sidebar_bg']};
            --sidebar-border: {p['sidebar_border']};
            --input-bg: {p['input_bg']};
            --input-border: {p['input_border']};
            --popover-bg: {p['popover_bg']};
            --expander-bg: {p['expander_bg']};
            --btn-bg: {p['btn_bg']};
            --btn-text: {p['btn_text']};
            --btn-border: {p['btn_border']};
            --btn-shadow: {p['btn_shadow']};
        }}

        /* --- 1. Root & App Container Background --- */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        .main,
        section.main {{
            background: {p['bg_app']} !important;
            color: {p['text_primary']} !important;
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
            visibility: hidden !important;
        }}

        /* --- 2. Typography --- */
        h1, h2, h3, h4, h5, h6 {{
            color: {p['text_heading']} !important;
            font-weight: 700 !important;
        }}

        p, span, label, li, small {{
            color: {p['text_primary']};
        }}

        .stMarkdown p, .stMarkdown span {{
            color: {p['text_primary']} !important;
        }}

        .stCaption, caption, [data-testid="stCaptionContainer"] p {{
            color: {p['text_secondary']} !important;
        }}

        hr, [data-testid="stDivider"] {{
            border-color: {p['border']} !important;
            opacity: 0.8 !important;
        }}

        /* --- 3. Sidebar Styling --- */
        [data-testid="stSidebar"],
        [data-testid="stSidebarUserContent"] {{
            background-color: {p['sidebar_bg']} !important;
            border-right: 1px solid {p['sidebar_border']} !important;
            color: {p['text_primary']} !important;
        }}

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {{
            color: {p['text_primary']} !important;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {p['text_heading']} !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: {p['sidebar_border']} !important;
        }}

        /* Sidebar Navigation Links */
        [data-testid="stSidebarNav"] {{
            background-color: transparent !important;
        }}

        [data-testid="stSidebarNav"] li a {{
            color: {p['text_primary']} !important;
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }}

        [data-testid="stSidebarNav"] li a:hover {{
            background-color: {p['bg_card_hover']} !important;
        }}

        [data-testid="stSidebarNav"] li a[aria-current="page"] {{
            background-color: {p['sidebar_nav_active_bg']} !important;
            color: {p['sidebar_nav_active_text']} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebarNav"] li a span {{
            color: inherit !important;
        }}

        /* --- 4. Form Controls & Inputs (Selectbox, DateInput, Slider, Text) --- */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] span {{
            color: {p['text_primary']} !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }}

        /* Select & Input Containers */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div {{
            background-color: {p['input_bg']} !important;
            border: 1px solid {p['input_border']} !important;
            color: {p['text_primary']} !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px {p['shadow']} !important;
        }}

        /* Text & Icons inside Selects & Inputs */
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] svg {{
            color: {p['text_primary']} !important;
            fill: {p['text_primary']} !important;
        }}

        input, textarea {{
            color: {p['text_primary']} !important;
            background-color: transparent !important;
        }}

        /* Dropdown popover menu */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[data-baseweb="menu"] {{
            background-color: {p['popover_bg']} !important;
            border: 1px solid {p['border']} !important;
            box-shadow: 0 10px 25px {p['shadow_heavy']} !important;
            border-radius: 8px !important;
        }}

        li[data-baseweb="menu-item"] {{
            background-color: {p['popover_bg']} !important;
            color: {p['text_primary']} !important;
        }}

        li[data-baseweb="menu-item"]:hover,
        li[data-baseweb="menu-item"][aria-selected="true"] {{
            background-color: {p['menu_item_hover']} !important;
            color: {p['accent']} !important;
        }}

        /* Date picker calendar */
        div[data-baseweb="calendar"] {{
            background-color: {p['popover_bg']} !important;
            color: {p['text_primary']} !important;
        }}
        div[data-baseweb="calendar"] * {{
            color: {p['text_primary']} !important;
        }}

        /* Slider */
        [data-testid="stSlider"] * {{
            color: {p['text_primary']} !important;
        }}
        div[data-testid="stSliderTickBarMin"],
        div[data-testid="stSliderTickBarMax"] {{
            color: {p['text_secondary']} !important;
        }}

        /* Number Input buttons */
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {{
            background-color: {p['input_bg']} !important;
            color: {p['text_primary']} !important;
            border-color: {p['input_border']} !important;
        }}

        /* --- 5. Expanders --- */
        [data-testid="stExpander"] {{
            background-color: {p['expander_bg']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 10px !important;
            margin-bottom: 8px !important;
        }}

        [data-testid="stExpander"] summary {{
            color: {p['text_primary']} !important;
            font-weight: 600 !important;
        }}

        [data-testid="stExpander"] summary:hover {{
            color: {p['accent']} !important;
        }}

        [data-testid="stExpander"] summary svg {{
            fill: {p['text_secondary']} !important;
        }}

        [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
            background-color: transparent !important;
            color: {p['text_primary']} !important;
            border-top: 1px solid {p['border']} !important;
        }}

        /* --- 6. Metric Cards --- */
        .stMetric {{
            background: {p['bg_card']} !important;
            backdrop-filter: blur(10px);
            padding: 16px !important;
            border-radius: 12px !important;
            border: 1px solid {p['border']} !important;
            box-shadow: 0 2px 10px {p['shadow']} !important;
        }}

        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] span {{
            color: {p['text_secondary']} !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }}

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {{
            color: {p['text_heading']} !important;
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            overflow-wrap: break-word;
            white-space: normal !important;
        }}

        /* --- 7. Buttons & Action Elements --- */
        .stButton > button {{
            background: {p['btn_bg']} !important;
            color: {p['btn_text']} !important;
            border: 1px solid {p['btn_border']} !important;
            padding: 8px 20px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px {p['btn_shadow']} !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px {p['btn_shadow']} !important;
        }}

        /* Theme Toggle Button Specific Styling */
        button[key="theme_toggle_btn"] {{
            background: {p['toggle_btn_bg']} !important;
            border: 1px solid {p['toggle_btn_border']} !important;
            color: {p['toggle_btn_text']} !important;
            font-weight: 700 !important;
        }}

        /* --- 8. Tabs --- */
        [data-testid="stTabs"] button {{
            color: {p['text_secondary']} !important;
            font-weight: 600 !important;
            border-bottom: 2px solid transparent !important;
            padding: 8px 16px !important;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {p['accent']} !important;
            border-bottom: 2px solid {p['accent']} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stTabs"] button:hover {{
            color: {p['accent']} !important;
        }}

        /* --- 9. Spinners, Alerts, Notifications --- */
        .stSpinner > div {{
            color: {p['text_primary']} !important;
            border-top-color: {p['accent']} !important;
        }}

        [data-testid="stAlert"] {{
            border-radius: 10px !important;
            border: 1px solid {p['border']} !important;
        }}

        /* --- 10. DataFrames & Tables --- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {p['border']} !important;
            border-radius: 8px !important;
        }}

        /* --- 11. Custom Reusable Classes --- */
        .theme-card {{
            background: {p['bg_card']};
            border: 1px solid {p['border']};
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
            color: {p['text_primary']};
            box-shadow: 0 2px 8px {p['shadow']};
            transition: all 0.2s ease;
        }}

        .theme-card:hover {{
            box-shadow: 0 6px 16px {p['shadow_heavy']};
            border-color: {p['border_accent']};
        }}

        .theme-card h4 {{
            color: {p['text_heading']};
            margin-top: 0;
        }}

        .theme-card p {{
            color: {p['text_secondary']};
            margin-bottom: 0;
        }}

        .theme-accent-box {{
            background: {p['accent_bg']};
            border: 1px solid {p['accent_border']};
            border-radius: 16px;
            padding: 26px 30px;
            margin: 20px 0;
        }}

        .theme-accent-box h4 {{
            color: {p['accent']};
            margin-top: 0;
        }}

        .theme-accent-box p {{
            color: {p['text_primary']};
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 0;
        }}

        .theme-table {{
            width: 100%;
            color: {p['text_table']};
            border-collapse: collapse;
        }}

        .theme-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid {p['border']};
        }}

        .theme-table a {{
            color: {p['accent']};
            text-decoration: none;
            font-weight: 600;
        }}

        .theme-table a:hover {{
            text-decoration: underline;
        }}

        .commodity-item {{
            background: {p['bg_card']};
            border: 1px solid {p['border']};
            border-radius: 8px;
            padding: 12px 16px;
            margin: 6px 0;
            color: {p['text_primary']};
            box-shadow: 0 1px 4px {p['shadow']};
        }}

        .commodity-item span {{
            color: {p['text_secondary']};
        }}

        .credits-card {{
            background: {p['bg_card']};
            border: 1px solid {p['border']};
            border-radius: 12px;
            padding: 20px;
            color: {p['text_primary']};
            box-shadow: 0 2px 8px {p['shadow']};
        }}

        .theme-footer {{
            display: flex;
            justify-content: space-between;
            opacity: 0.6;
            font-size: 0.8rem;
            color: {p['text_secondary']};
            padding: 10px 0;
        }}

        /* Chat header & messages */
        .chat-header {{
            background: {p['accent_bg']};
            border: 1px solid {p['accent_border']};
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            text-align: center;
        }}

        .chat-header h2 {{
            color: {p['accent']} !important;
            margin: 0 0 8px 0;
        }}

        .chat-header p {{
            color: {p['text_secondary']} !important;
            margin: 0;
            font-size: 0.95rem;
        }}

        [data-testid="stChatMessage"] {{
            background: {p['chat_msg_bg']} !important;
            border: 1px solid {p['chat_msg_border']} !important;
            border-radius: 12px !important;
            margin-bottom: 8px;
            box-shadow: 0 1px 4px {p['shadow']} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
