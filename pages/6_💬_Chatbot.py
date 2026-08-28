"""
Page 6: Interactive Chatbot — Ask questions about food prices in natural language.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.chatbot_engine import ChatbotEngine

st.set_page_config(page_title="Chatbot | Agri-AI EWS", page_icon="💬", layout="wide")

# --- Theme ---
from theme import inject_theme_css, render_theme_toggle, theme_color, get_plotly_template, get_plotly_layout, apply_theme_to_plotly
inject_theme_css()

# --- Additional Chatbot CSS ---
st.markdown(f"""
<style>
    /* Chat container */
    .chat-header {{
        background: var(--accent-bg);
        border: 1px solid var(--accent-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
    }}
    .chat-header h2 {{
        color: var(--accent);
        margin: 0 0 8px 0;
    }}
    .chat-header p {{
        color: var(--text-secondary);
        margin: 0;
        font-size: 0.95rem;
    }}

    /* Quick action buttons */
    .quick-actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 16px 0;
        justify-content: center;
    }}

    /* Streamlit chat message styling */
    [data-testid="stChatMessage"] {{
        background: {theme_color('chat_msg_bg')} !important;
        border: 1px solid {theme_color('chat_msg_border')};
        border-radius: 12px !important;
        margin-bottom: 8px;
    }}

    /* Status badge */
    .status-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(0, 200, 83, 0.15);
        color: #00c853;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }}
</style>
""", unsafe_allow_html=True)

# --- Initialize ---
@st.cache_resource
def get_chatbot():
    return ChatbotEngine()

bot = get_chatbot()

# Session state for chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "pending_quick_action" not in st.session_state:
    st.session_state.pending_quick_action = None

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2534/2534044.png", width=60)
st.sidebar.title("💬 Agri-AI Chatbot")
st.sidebar.caption("Asisten virtual harga pangan")
render_theme_toggle()
st.sidebar.markdown("---")

st.sidebar.markdown("### 💡 Contoh Pertanyaan")
example_questions = [
    "Berapa harga beras di Jakarta?",
    "Prediksi harga cabai di Jakarta",
    "Forecast harga beras 30 hari",
    "Tren harga cabai merah",
    "Statistik harga bawang merah",
    "Bandingkan harga beras Jakarta dan Jawa Barat",
]
for q in example_questions:
    st.sidebar.markdown(f"- *{q}*")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
    st.session_state.chat_messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="status-badge">● Online</span> &nbsp; Chatbot aktif',
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown("""
<div class="chat-header">
    <h2>💬 Agri-AI Chatbot</h2>
    <p>Tanyakan apa saja tentang harga pangan Indonesia — harga terbaru, <strong>prediksi AI</strong>, tren, statistik, dan perbandingan antar provinsi.</p>
</div>
""", unsafe_allow_html=True)

# --- Quick Action Buttons ---
if not st.session_state.chat_messages:
    st.markdown("#### 🚀 Mulai dengan pertanyaan cepat:")
    quick_cols = st.columns(3)
    quick_actions = [
        ("💰 Harga Terbaru", "Berapa harga terbaru semua komoditas?"),
        ("🔮 Prediksi Beras", "Prediksi harga beras di DKI Jakarta"),
        ("🔮 Prediksi Cabai", "Prediksi harga cabai merah di Jawa Timur"),
        ("📈 Tren Cabai", "Bagaimana tren harga cabai merah?"),
        ("📊 Statistik Beras", "Statistik harga beras"),
        ("🔄 Bandingkan", "Bandingkan harga beras Jakarta dan Jawa Barat"),
    ]
    for i, (label, query) in enumerate(quick_actions):
        col = quick_cols[i % 3]
        if col.button(label, key=f"quick_{i}", use_container_width=True):
            st.session_state.pending_quick_action = query
            st.rerun()

    st.markdown("---")

# --- Helper Functions (defined before use) ---
def _render_chart(chart_data, chart_type, chart_title):
    """Render a Plotly chart from chatbot response data."""
    if chart_data is None or chart_data.empty:
        return

    if chart_type == "compare_line":
        fig = px.line(
            chart_data,
            x="date",
            y="price",
            color="province",
            title=chart_title,
            template=get_plotly_template(),
        )
    elif chart_type == "forecast":
        fig = go.Figure()

        # Determine split between historical and forecast data
        has_bounds = "lower" in chart_data.columns and "upper" in chart_data.columns
        if has_bounds:
            # Historical = where lower == price (no uncertainty band)
            hist_mask = chart_data["lower"] == chart_data["price"]
            hist_data = chart_data[hist_mask]
            pred_data = chart_data[~hist_mask]
        else:
            hist_data = chart_data
            pred_data = pd.DataFrame()

        # Historical line
        if not hist_data.empty:
            fig.add_trace(go.Scatter(
                x=hist_data["date"], y=hist_data["price"],
                mode="lines+markers", name="Historis",
                line=dict(color="#4facfe", width=2),
                marker=dict(size=3),
            ))

        # Forecast line + confidence band
        if not pred_data.empty:
            # Upper bound (invisible line for fill)
            fig.add_trace(go.Scatter(
                x=pred_data["date"], y=pred_data["upper"],
                mode="lines", name="Batas Atas",
                line=dict(width=0),
                showlegend=False,
            ))
            # Lower bound with fill to upper
            fig.add_trace(go.Scatter(
                x=pred_data["date"], y=pred_data["lower"],
                mode="lines", name="Confidence Interval",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 165, 0, 0.15)",
            ))
            # Prediction line
            fig.add_trace(go.Scatter(
                x=pred_data["date"], y=pred_data["price"],
                mode="lines+markers", name="Prediksi",
                line=dict(color="#FFA500", width=2, dash="dot"),
                marker=dict(size=3),
            ))

    apply_theme_to_plotly(
        fig,
        title=chart_title,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="",
        yaxis_title="Harga (Rp)",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{id(chart_data)}")


def _process_and_store(user_input: str):
    """Process user input through the chatbot engine and store results."""
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Get bot response
    response = bot.process(user_input)

    # Store bot message
    bot_msg = {"role": "assistant", "content": response["text"]}
    if response.get("chart_data") is not None:
        bot_msg["chart_data"] = response["chart_data"]
        bot_msg["chart_type"] = response["chart_type"]
        bot_msg["chart_title"] = response.get("chart_title", "")
    st.session_state.chat_messages.append(bot_msg)


# --- Render Chat History ---
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

        # Render chart if present
        if message.get("chart_data") is not None and message.get("chart_type"):
            _render_chart(message["chart_data"], message["chart_type"], message.get("chart_title", ""))

# Handle pending quick action
if st.session_state.pending_quick_action:
    query = st.session_state.pending_quick_action
    st.session_state.pending_quick_action = None
    _process_and_store(query)
    st.rerun()

# Chat input
if user_input := st.chat_input("Tanyakan tentang harga pangan... 🌾"):
    _process_and_store(user_input)
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="theme-footer">
    <div>AGRI-AI CHATBOT v1.0</div>
    <div>RULE-BASED + DATA-DRIVEN</div>
    <div>© 2026 Fahmi Prasanda</div>
</div>
""", unsafe_allow_html=True)
