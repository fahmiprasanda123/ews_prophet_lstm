#!/usr/bin/env python3
"""
Analisis tinjauan literatur berbasis jaringan untuk Bab II.

Korpus = studi penelitian periode 2021–2026 yang tercantum di daftar pustaka naskah.
Istilah diekstraksi dari judul memakai tesaurus (pendekatan analisis istilah seperti
VOSviewer), lalu dibentuk jaringan ko-okurensi, klaster (modularitas), overlay rata-rata
tahun terbit, tren tema per tahun, dan peta jalan riset 2021–2031.

Pemakaian:
    python scripts/analisis_slr.py --naskah naskah.docx --output slr_outputs
"""
import argparse
import itertools
import os
import re
import zipfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from lxml import etree  # noqa: E402

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
q = lambda t: f'{{{W}}}{t}'  # noqa: E731
EXCLUDE = ('Badan Pusat Statistik', 'Bank Indonesia', 'FAO', 'ISO/IEC', 'NOAA', 'Oxford Economics', 'Peraturan',
           'Undang-Undang', 'UNISDR', 'Zippenfenig', 'Page, M. J.', 'Hyndman, R. J., & Athanasopoulos', 'Sugiyono',
           'Fajri', 'Ryu Tantanu')
TERMS = [
    ('lstm', r'\blstm\b|long short[- ]?term memory'),
    ('bilstm', r'bilstm|bidirectional'),
    ('gru', r'\bgru\b'),
    ('cnn', r'\bcnn\b|convolutional'),
    ('transformer', r'transformer'),
    ('attention', r'attention'),
    ('prophet', r'prophet'),
    ('arima', r'arima'),
    ('hybrid model', r'hybrid'),
    ('ensemble/combination', r'ensemble|combination|combining|stacking'),
    ('deep learning', r'deep learning|deep hybrid|neural network|\brnn\b|\bgnn\b|recurrent'),
    ('machine learning', r'machine learning|xgboost|boost|\btcn\b|tslm'),
    ('agricultural commodity', r'agricultur|crop|commodit|wheat|vegetable|groceries|bahan pokok|pangan|food'),
    ('price', r'price|harga'),
    ('energy', r'energy|electricity|power market|consumption'),
    ('finance', r'financ|stock|bitcoin|cryptocurrency|futures'),
    ('climate/weather', r'climate|weather|el ni|southern oscillation|iklim|cuaca'),
    ('early warning system', r'early[- ]warning|peringatan dini'),
    ('risk', r'\brisk|risiko'),
    ('uncertainty', r'uncertaint|probabilistic|quantile|interval'),
    ('multi-horizon', r'multi-horizon|long-term|multi-step|short-term'),
    ('time series', r'time[- ]series|deret waktu'),
    ('seasonality/trend', r'seasonal|trend|decomposition'),
    ('comparative study', r'comparative|comparison|comparing|benchmark|\bvs\b|versus'),
    ('decision support/BI', r'business intelligence|decision support|knowledge graph|monitoring'),
]
THEMES = {
    'Statistik klasik (ARIMA, Prophet)': ['arima', 'prophet'],
    'Jaringan rekuren (LSTM, BiLSTM, GRU)': ['lstm', 'bilstm', 'gru'],
    'Perhatian/Transformer': ['transformer', 'attention'],
    'Hibrida/ensemble': ['hybrid model', 'ensemble/combination'],
    'Kovariat iklim/cuaca': ['climate/weather'],
    'Peringatan dini/risiko': ['early warning system', 'risk'],
}


def parse_bibliography(docx):
    root = etree.fromstring(zipfile.ZipFile(docx).read('word/document.xml'))
    paras = list(root.iter(q('p')))
    txt = lambda p: ''.join(t.text or '' for t in p.iter(q('t')))  # noqa: E731
    start = max(i for i, p in enumerate(paras) if txt(p).strip() == 'DAFTAR PUSTAKA')
    rows = []
    for p in paras[start + 1:]:
        full = txt(p).strip()
        if not full:
            continue
        if full in ('LAMPIRAN', 'RIWAYAT HIDUP') or full.startswith('Lampiran 1'):
            break
        runs = []
        for r in p.iter(q('r')):
            t = ''.join(x.text or '' for x in r.iter(q('t')))
            if t:
                rpr = r.find(q('rPr'))
                ital = rpr is not None and rpr.find(q('i')) is not None
                runs.append((t, ital))
        m = re.search(r'\((\d{4})[a-z]?(?:, [^)]*)?\)\.\s*', full)
        if not m:
            continue
        year = int(m.group(1))
        pos, title, source, title_ital, in_source = 0, '', '', None, False
        for t, ital in runs:  # segmentasi judul/sumber berdasarkan run miring
            seg_start, pos = pos, pos + len(t)
            if pos <= m.end():
                continue
            piece = t[max(0, m.end() - seg_start):]
            if not title:
                title, title_ital = piece, ital
                continue
            if title_ital is False and not in_source:
                if ital:
                    source, in_source = piece, True
                else:
                    title += piece
                continue
            if in_source:  # gabungkan run miring berurutan (nama jurnal bisa terpecah)
                if ital or not piece.strip() or (piece.strip() and not re.search(r'[,\d]', piece)):
                    source += piece
                    continue
                break
            if title_ital and not ital:
                break
        title = re.split(r'\s*\[(?:Preprint|Perangkat lunak)\]', title)[0].strip().rstrip('.').strip()
        source = source.strip().strip(',').strip()
        doi = re.search(r'https?://doi\.org/(\S+)', full)
        arx = re.search(r'arxiv\.org/abs/(\S+)|arXiv:(\S+)', full)
        ident = doi.group(1).rstrip('.') if doi else (f"arXiv:{(arx.group(1) or arx.group(2)).rstrip('.')}" if arx else '')
        authors = full[:m.start()].strip().rstrip('.').strip()
        rows.append({'Penulis': authors, 'Tahun': year, 'Judul': title, 'Sumber': source, 'DOI': ident, 'Entri': full})
    return pd.DataFrame(rows)


def short_author(a):
    first = a.split(',')[0].strip()
    return first + (' dkk.' if a.count(',') >= 3 or ' & ' in a and a.count(',') >= 2 else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--naskah', required=True)
    ap.add_argument('--output', default='slr_outputs')
    ap.add_argument('--tahun-awal', type=int, default=2021)
    ap.add_argument('--tahun-akhir', type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    bib = parse_bibliography(a.naskah)
    corpus = bib[(bib['Tahun'] >= a.tahun_awal) & (bib['Tahun'] <= a.tahun_akhir)
                 & ~bib['Penulis'].str.startswith(EXCLUDE)].reset_index(drop=True)
    corpus['Istilah'] = [sorted({lab for lab, pat in TERMS if re.search(pat, f"{t} {s}".lower())})
                         for t, s in zip(corpus['Judul'], corpus['Sumber'])]
    corpus['Kutipan'] = [f"{short_author(p)} ({y})" for p, y in zip(corpus['Penulis'], corpus['Tahun'])]

    freq = pd.Series([t for ts in corpus['Istilah'] for t in ts]).value_counts()
    keep = set(freq[freq >= 2].index)
    G = nx.Graph()
    for t in keep:
        years = corpus.loc[corpus['Istilah'].apply(lambda ts: t in ts), 'Tahun']
        G.add_node(t, freq=int(freq[t]), year=float(years.mean()))
    for ts in corpus['Istilah']:
        for u, v in itertools.combinations(sorted(set(ts) & keep), 2):
            G.add_edge(u, v, weight=G[u][v]['weight'] + 1 if G.has_edge(u, v) else 1)
    comms = sorted(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'),
                   key=lambda c: (-sum(G.nodes[x]['freq'] for x in c), sorted(c)))
    cluster = {n: i + 1 for i, c in enumerate(comms) for n in c}
    pos = nx.spring_layout(G, k=1.6, iterations=400, seed=7, weight='weight')

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8})
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']

    def draw(colors, title, path, cbar=None):
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        widths = [0.4 + 0.9 * G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color='#bbbbbb', alpha=0.7)
        sizes = [90 + 75 * G.nodes[n]['freq'] for n in G.nodes()]
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, linewidths=0.5,
                                       edgecolors='white', cmap=cbar[0] if cbar else None,
                                       vmin=cbar[1] if cbar else None, vmax=cbar[2] if cbar else None)
        for n, (x, y) in pos.items():
            ax.text(x, y + 0.055, n, ha='center', va='bottom', fontsize=7.2,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.4))
        if not cbar:
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=7,
                                  label=f'Klaster {i + 1}') for i in range(len(comms))]
            ax.legend(handles=handles, loc='lower left', fontsize=7, frameon=False)
        if cbar:
            cb = fig.colorbar(nodes, ax=ax, fraction=0.035, pad=0.01)
            cb.set_label('Rata-rata tahun terbit')
        ax.set_title(title, fontsize=8.5, loc='left')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    draw([palette[(cluster[n] - 1) % len(palette)] for n in G.nodes()],
         f'Jaringan ko-okurensi istilah (n = {len(corpus)} studi, {G.number_of_nodes()} istilah, '
         f'{len(comms)} klaster)', os.path.join(a.output, 'gambar_jaringan_klaster.png'))
    draw([G.nodes[n]['year'] for n in G.nodes()], 'Overlay rata-rata tahun terbit setiap istilah',
         os.path.join(a.output, 'gambar_jaringan_overlay_tahun.png'),
         cbar=(plt.cm.viridis, a.tahun_awal, a.tahun_akhir))

    # Tren tema per tahun
    years = list(range(a.tahun_awal, a.tahun_akhir + 1))
    trend = pd.DataFrame(0, index=years, columns=list(THEMES))
    for ts, y in zip(corpus['Istilah'], corpus['Tahun']):
        for th, terms in THEMES.items():
            if set(terms) & set(ts):
                trend.loc[y, th] += 1
    trend['Jumlah studi'] = corpus.groupby('Tahun').size().reindex(years).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = ['#8d6e63', '#ff7f0e', '#d62728', '#2ca02c', '#1f77b4', '#9467bd']
    width = 0.13
    for i, th in enumerate(THEMES):
        ax.bar([y + (i - 2.5) * width for y in years], trend[th], width=width, label=th, color=colors[i])
    ax.plot(years, trend['Jumlah studi'], color='black', marker='o', lw=1.2, label='Jumlah studi')
    ax.set_xticks(years)
    ax.set_ylabel('Jumlah studi')
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc='upper left')
    ax.grid(axis='y', color='#eeeeee')
    fig.tight_layout()
    fig.savefig(os.path.join(a.output, 'gambar_tren_tema.png'), dpi=220)
    plt.close(fig)

    # Peta jalan 2021–2031: tiga periode lalu (berbasis data) + tiga periode proyeksi
    periods = [(2021, 2022), (2023, 2024), (2025, 2026)]
    past = []
    for p0, p1 in periods:
        sub = corpus[(corpus['Tahun'] >= p0) & (corpus['Tahun'] <= p1)]
        counts = {th: sum(bool(set(t) & set(terms)) for t in sub['Istilah']) for th, terms in THEMES.items()}
        short = {'Statistik klasik (ARIMA, Prophet)': 'Statistik', 'Jaringan rekuren (LSTM, BiLSTM, GRU)': 'RNN/LSTM',
                 'Perhatian/Transformer': 'Transformer', 'Hibrida/ensemble': 'Hibrida', 'Kovariat iklim/cuaca': 'Iklim',
                 'Peringatan dini/risiko': 'EWS/risiko'}
        top = [f"{short[th]} ({c})" for th, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c > 0][:5]
        past.append((f'{p0}–{p1}', len(sub), top))
    future = [
        ('2027–2028', 'Kalibrasi interval ketidakpastian; data produksi, stok, dan cuaca sentra produksi; '
                      'validasi seluruh 334 deret'),
        ('2029–2030', 'Model spasio-temporal antarprovinsi (graf) dan transfer learning lintas komoditas'),
        ('2031', 'EWS terintegrasi kebijakan: rekomendasi intervensi otomatis dan evaluasi dampaknya'),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.annotate('', xy=(5.95, 0.82), xytext=(0.05, 0.82), arrowprops=dict(arrowstyle='->', lw=1.4))
    ax.text(1.5, 0.93, '5 tahun sebelumnya (berbasis korpus)', ha='center', fontsize=8, weight='bold')
    ax.text(4.5, 0.93, '5 tahun sesudahnya (proyeksi peta jalan)', ha='center', fontsize=8, weight='bold')
    ax.axvline(3, ymin=0.05, ymax=0.9, color='#999999', ls='--', lw=0.8)
    for i, (label, n, top) in enumerate(past):
        x = 0.5 + i
        ax.add_patch(plt.Rectangle((x - 0.47, 0.18), 0.94, 0.56, color='#e8f0fa', ec='#4c72b0'))
        ax.text(x, 0.69, label, ha='center', fontsize=8, weight='bold')
        ax.text(x, 0.64, f'{n} studi', ha='center', fontsize=7)
        ax.text(x - 0.40, 0.58, '\n'.join(f'• {t}' for t in top), ha='left', va='top', fontsize=6.4, linespacing=1.5)
    ax.text(2.5, 0.12, 'Penelitian ini (2026): Smart Ensemble Prophet–BiLSTM–TFT\n+ kovariat iklim + EWS 34 provinsi',
            ha='center', fontsize=6.8, color='#c44e52', weight='bold')
    for i, (label, text) in enumerate(future):
        x = 3.5 + i
        ax.add_patch(plt.Rectangle((x - 0.47, 0.18), 0.94, 0.56, color='#fdf2e9', ec='#dd8452'))
        ax.text(x, 0.69, label, ha='center', fontsize=8, weight='bold')
        import textwrap
        ax.text(x, 0.62, textwrap.fill(text, 17), ha='center', va='top', fontsize=6.3, linespacing=1.4)
    fig.tight_layout()
    fig.savefig(os.path.join(a.output, 'gambar_peta_jalan.png'), dpi=220)
    plt.close(fig)

    # Keluaran tabel
    corpus['Klaster'] = corpus['Istilah'].apply(
        lambda ts: ', '.join(str(c) for c in sorted({cluster[t] for t in ts if t in cluster})) or '-')
    corpus.drop(columns=['Entri']).assign(Istilah=corpus['Istilah'].apply(', '.join)).to_csv(
        os.path.join(a.output, 'korpus_slr.csv'), index=False)
    crows = []
    for i, c in enumerate(comms, 1):
        terms = sorted(c, key=lambda n: -G.nodes[n]['freq'])
        studies = corpus[corpus['Istilah'].apply(lambda ts: len(set(ts) & set(c)) >= 2)]
        crows.append({'Klaster': i, 'Istilah utama': ', '.join(terms),
                      'Rata-rata tahun': round(sum(G.nodes[n]['year'] for n in c) / len(c), 1),
                      'Jumlah studi': len(studies),
                      'Contoh studi': '; '.join(studies.sort_values('Tahun', ascending=False)['Kutipan'].head(4))})
    pd.DataFrame(crows).to_csv(os.path.join(a.output, 'klaster_slr.csv'), index=False)
    trend.to_csv(os.path.join(a.output, 'tren_tema.csv'))
    pd.DataFrame(past, columns=['Periode', 'Jumlah studi', 'Tema dominan']).to_csv(
        os.path.join(a.output, 'peta_jalan_periode.csv'), index=False)
    print(f'Korpus: {len(corpus)} studi | tanpa DOI: {int((corpus["DOI"] == "").sum())}')
    print(f'Jaringan: {G.number_of_nodes()} istilah, {G.number_of_edges()} tautan, {len(comms)} klaster')


if __name__ == '__main__':
    main()
