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


def _luminance(hex_color):
    def channel(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


LIGHT_BG, DARK_BG = "#F7F6F2", "#121613"


class TestTheme(unittest.TestCase):
    def test_get_theme_defaults_to_light_outside_streamlit(self):
        self.assertEqual(theme.get_theme(), "light")

    def test_status_chips_meet_wcag_aa(self):
        for name, c in theme.STATUS.items():
            self.assertGreaterEqual(_contrast(c["fg"], c["bg"]), 4.5, name)

    def test_chart_colors_meet_non_text_contrast_in_both_themes(self):
        for name, color in theme.CHART.items():
            self.assertGreaterEqual(_contrast(color, LIGHT_BG), 3.0, f"{name} on light")
            self.assertGreaterEqual(_contrast(color, DARK_BG), 3.0, f"{name} on dark")

    def test_status_chip_escapes_text(self):
        html = theme.status_chip("<b>Bahaya</b>", "danger")
        self.assertIn("&lt;b&gt;Bahaya&lt;/b&gt;", html)
        self.assertIn(theme.STATUS["danger"]["bg"], html)

    def test_ews_card_unknown_level_uses_neutral(self):
        html = theme.ews_card("Unknown", 0, "pesan")
        self.assertIn(theme.STATUS["neutral"]["bg"], html)
        self.assertIn("Belum ada", html)
        self.assertNotIn("/100", html)

    def test_ews_card_translates_level(self):
        self.assertIn("Waspada", theme.ews_card("Alert", 55, ""))

    def test_ews_card_shows_message(self):
        html = theme.ews_card("Alert", 55, "Harga diprediksi naik 21.0%.")
        self.assertIn("Harga diprediksi naik 21.0%.", html)
        self.assertIn("55", html)

    def test_apply_theme_to_plotly_with_custom_legend_and_title(self):
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
        # Warna teks diserahkan ke tema Streamlit agar ikut berganti terang/gelap
        self.assertIsNone(res.layout.font.color)


if __name__ == "__main__":
    unittest.main()
