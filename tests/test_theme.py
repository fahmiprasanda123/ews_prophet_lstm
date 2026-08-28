"""
Unit tests for theme.py module.
"""
import unittest
import sys
import os
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theme


class TestTheme(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        if not hasattr(st, "session_state"):
            st.session_state = {}
        else:
            st.session_state.clear()

    def test_default_theme_is_dark(self):
        import streamlit as st
        self.assertEqual(theme.get_theme(), "dark")
        self.assertEqual(st.session_state.get("theme_mode"), "dark")

    def test_switch_theme_to_light(self):
        import streamlit as st
        st.session_state["theme_mode"] = "light"
        self.assertEqual(theme.get_theme(), "light")

    def test_plotly_template(self):
        import streamlit as st
        st.session_state["theme_mode"] = "dark"
        self.assertEqual(theme.get_plotly_template(), "plotly_dark")
        
        st.session_state["theme_mode"] = "light"
        self.assertEqual(theme.get_plotly_template(), "plotly_white")

    def test_theme_color_keys_match(self):
        self.assertEqual(set(theme.DARK_PALETTE.keys()), set(theme.LIGHT_PALETTE.keys()))

    def test_theme_color_function(self):
        import streamlit as st
        st.session_state["theme_mode"] = "dark"
        self.assertEqual(theme.theme_color("bg_primary"), "#0E1117")

        st.session_state["theme_mode"] = "light"
        self.assertEqual(theme.theme_color("bg_primary"), "#F8FAFC")

    def test_get_plotly_layout(self):
        import streamlit as st
        st.session_state["theme_mode"] = "light"
        layout = theme.get_plotly_layout()
        self.assertEqual(layout["template"], "plotly_white")
        self.assertIn("paper_bgcolor", layout)
        self.assertIn("plot_bgcolor", layout)

    def test_apply_theme_to_plotly_with_custom_legend_and_title(self):
        import streamlit as st
        st.session_state["theme_mode"] = "light"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
        
        # Test calling with legend and title kwargs to ensure no duplicate keyword error
        res = theme.apply_theme_to_plotly(
            fig,
            title="My Test Title",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450
        )
        self.assertEqual(res.layout.title.text, "My Test Title")
        self.assertEqual(res.layout.legend.orientation, "h")
        self.assertEqual(res.layout.height, 450)
        self.assertEqual(res.layout.font.color, "#0F172A")


if __name__ == "__main__":
    unittest.main()
