"""Regresi untuk skrip prakiraan ke depan."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))


@pytest.mark.parametrize('isi', [None, {}, {'weights': None}])
def test_prakiraan_weights_null(tmp_path, isi):
    import prakiraan_ke_depan
    path = tmp_path / 'bobot.json'
    path.write_text(json.dumps(isi))
    assert prakiraan_ke_depan.load_weights(str(path)) == (None, None)


def test_prakiraan_weights_ok(tmp_path):
    import prakiraan_ke_depan
    path = tmp_path / 'bobot.json'
    path.write_text(json.dumps({'weights': {'Prophet': 1.0}, 'best_single': 'Prophet'}))
    assert prakiraan_ke_depan.load_weights(str(path)) == ({'Prophet': 1.0}, 'Prophet')
