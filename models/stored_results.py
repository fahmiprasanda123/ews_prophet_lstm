"""Hasil evaluasi tersimpan (thesis_outputs/) agar Dashboard tidak menghitung ulang protokol evaluasi."""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stored_evaluation(province, commodity, folder=None):
    """Metrik dan bobot ensemble dari skrip evaluasi bila kombinasi dan mode cocok; selain itu None."""
    folder = folder or os.path.join(ROOT, 'thesis_outputs')
    try:
        with open(os.path.join(folder, 'konfigurasi.json')) as f:
            meta = json.load(f)
        if (meta.get('provinsi'), meta.get('komoditas')) != (province, commodity) or meta.get('mode_cepat'):
            return None
        metrics = pd.read_csv(os.path.join(folder, '4_2_metrik_model.csv'))
        with open(os.path.join(folder, '4_2_bobot_ensemble.json')) as f:
            ensemble = json.load(f)
        return {'metrics': metrics, 'ensemble': ensemble, 'stored': True,
                'covariates_used': bool(meta.get('kovariat_iklim'))}
    except Exception:
        return None
