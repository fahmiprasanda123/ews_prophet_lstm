"""
Page 6: Interactive Chatbot: Ask questions about food prices in natural language.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.chatbot_engine import ChatbotEngine

st.set_page_config(page_title="Chatbot | Agri-AI EWS", page_icon="🌾", layout="wide")

# --- Theme ---
from theme import inject_theme_css, render_theme_toggle, render_sidebar_brand, render_footer, apply_theme_to_plotly, CHART
inject_theme_css()

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
render_sidebar_brand("Chatbot harga pangan")
render_theme_toggle()
st.sidebar.divider()

st.sidebar.markdown("**Contoh pertanyaan**")
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

st.sidebar.divider()
if st.sidebar.button("Hapus riwayat chat", use_container_width=True,
                     disabled=not st.session_state.chat_messages):
    st.session_state.chat_messages = []
    st.rerun()

# --- Header ---
st.title("Chatbot harga pangan")
st.markdown("Tanyakan harga terbaru, prediksi, tren, statistik, atau perbandingan antarprovinsi. "
            "Jawaban dihitung dari data PIHPS dengan aturan, bukan model bahasa.")

# --- Quick Action Buttons ---
if not st.session_state.chat_messages:
    st.markdown("**Mulai dengan salah satu pertanyaan ini**")
    quick_cols = st.columns(3)
    quick_actions = [
        ("Harga terbaru", "Berapa harga terbaru semua komoditas?"),
        ("Prediksi beras", "Prediksi harga beras di DKI Jakarta"),
        ("Prediksi cabai", "Prediksi harga cabai merah di Jawa Timur"),
        ("Tren cabai", "Bagaimana tren harga cabai merah?"),
        ("Statistik beras", "Statistik harga beras"),
        ("Bandingkan provinsi", "Bandingkan harga beras Jakarta dan Jawa Barat"),
    ]
    for i, (label, query) in enumerate(quick_actions):
        col = quick_cols[i % 3]
        if col.button(label, key=f"quick_{i}", use_container_width=True):
            st.session_state.pending_quick_action = query
            st.rerun()

    st.divider()

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
                line=dict(color=CHART["actual"], width=2),
                marker=dict(size=3),
            ))

        # Forecast line + confidence band
        if not pred_data.empty:
            # Upper bound (invisible line for fill)
            fig.add_trace(go.Scatter(
                x=pred_data["date"], y=pred_data["upper"],
                mode="lines", name="Batas atas",
                line=dict(width=0),
                showlegend=False,
            ))
            # Lower bound with fill to upper
            fig.add_trace(go.Scatter(
                x=pred_data["date"], y=pred_data["lower"],
                mode="lines", name="Interval prediksi",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(168,106,16,0.15)",
            ))
            # Prediction line
            fig.add_trace(go.Scatter(
                x=pred_data["date"], y=pred_data["price"],
                mode="lines+markers", name="Prediksi",
                line=dict(color=CHART["forecast"], width=2, dash="dot"),
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
    try:
        response = bot.process(user_input)
    except Exception as e:
        response = {"text": "Maaf, pertanyaan ini gagal diproses. Coba tulis ulang dengan menyebut "
                            "komoditas dan provinsi, misalnya \"harga beras di Jawa Barat\". "
                            f"(Detail teknis: {type(e).__name__})"}

    # Store bot message
    bot_msg = {"role": "assistant", "content": response["text"]}
    if response.get("chart_data") is not None:
        bot_msg["chart_data"] = response["chart_data"]
        bot_msg["chart_type"] = response["chart_type"]
        bot_msg["chart_title"] = response.get("chart_title", "")
    st.session_state.chat_messages.append(bot_msg)


# --- Render Chat History ---
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
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
if user_input := st.chat_input("Tulis pertanyaan, misalnya: harga cabai rawit di Bali"):
    _process_and_store(user_input)
    st.rerun()

# --- Footer ---
st.divider()
render_footer("Chatbot berbasis aturan dan data PIHPS", "© 2026 Fahmi Prasanda")
