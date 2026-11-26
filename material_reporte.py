"""Material para el reporte y las diapositivas, generado desde los resultados del paso 2:
- resultados/paso2/fig_ejemplo_senal.png: una señal real antes/después del preprocesamiento y su espectro.
- resultados/paso2/tablas/*.tex: filas de tablas LaTeX leídas de los CSV (ningún número se copia a mano).

Uso: .venv/bin/python -I material_reporte.py   (después de paso2_features.py)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import periodogram

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paso2_features as p2   # mismo preprocesamiento y misma paleta que el paso 2

TABLAS = p2.SALIDA / 'tablas'
CORTO = {'TEPT (Rahmani)': 'TEPT', 'Alzheimer (ds004504)': 'Alzheimer', 'Depresión (Mumtaz)': 'Depresión'}
GRIEGO = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ'}
ETIQ = {**{f'abs_{b}': rf'Pot.\ absoluta $\{b}$' for b in p2.BANDAS},
        **{f'rel_{b}': rf'Pot.\ relativa $\{b}$' for b in p2.BANDAS},
        'ent_espectral': 'Entropía espectral', 'ent_permutacion': 'Entropía de permutación'}


def coma(v, d=2):
    return f'{v:.{d}f}'.replace('.', '{,}')


def figura_ejemplo():
    """Control 'típico' de ds004504: el de alfa relativa mediana en O1 (ventanas de 4 s, sin solapamiento)."""
    d = np.load(p2.SALIDA / 'features' / 'alzheimer_v4s_s0.npz')
    col = list(d['nombres']).index('rel_alpha_O1')
    medias = {s: d['X'][d['sujeto'] == s, col].mean() for s in np.unique(d['sujeto'][d['grupo'] == 0])}
    sujeto = sorted(medias, key=medias.get)[len(medias) // 2]
    ruta = p2.DATOS / 'ds004504' / sujeto / 'eeg' / f'{sujeto}_task-eyesclosed_eeg.set'
    crudo, fs = p2.cargar_mne(mne.io.read_raw_eeglab(ruta, preload=True, verbose='error'))
    x, malo = p2.preprocesar(crudo, fs)
    o1 = p2.CANALES.index('O1')

    fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)
    rejilla = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    a1, a2, b = fig.add_subplot(rejilla[0, 0]), fig.add_subplot(rejilla[1, 0]), fig.add_subplot(rejilla[:, 1])
    t0, t1 = 60, 64
    tc, tp = np.arange(crudo.shape[1]) / fs, p2.INICIO + np.arange(x.shape[1]) / p2.FS
    c, q = (tc >= t0) & (tc < t1), (tp >= t0) & (tp < t1)
    a1.plot(tc[c], crudo[o1, c] - crudo[o1, c].mean(), color=p2.TENUE, lw=1)
    a1.set_title('A. Canal O1 crudo: 500 Hz, referencia original (4 s)', loc='left', fontsize=10)
    a2.plot(tp[q], x[o1, q], color=p2.SERIES[0], lw=1.6)
    a2.set_title('Mismo tramo preprocesado: 256 Hz, 1–40 Hz, referencia promedio', loc='left', fontsize=10)
    a1.set(ylabel='µV')
    a2.set(ylabel='µV', xlabel='Tiempo (s)')
    a1.sharex(a2)
    a1.tick_params(labelbottom=False)

    ini, X, _ = p2.ventanear(x, malo, 4, 0)
    f, P = periodogram(X[o1], fs=p2.FS, window='hann', axis=-1)   # (ventanas, frecuencias)
    m, media = (f >= 1) & (f < 40), P.mean(0)
    for i, (banda, (lo, hi)) in enumerate(p2.BANDAS.items()):
        if i % 2 == 0:
            b.axvspan(lo, hi, color='#f0efec', lw=0)
        rel = media[(f >= lo) & (f < hi)].sum() / media[m].sum()
        b.text((lo + hi) / 2, 1.01, f'{GRIEGO[banda]}\n{100 * rel:.0f} %', transform=b.get_xaxis_transform(),
               ha='center', va='bottom', fontsize=8.5, color=p2.TINTA2, linespacing=1.1)
    b.semilogy(f[m], P[len(P) // 2, m], color=p2.TENUE, lw=1, label='Una ventana de 4 s')
    b.semilogy(f[m], media[m], color=p2.SERIES[0], lw=2, label=f'Promedio de las {len(ini)} ventanas')
    b.set(xlabel='Frecuencia (Hz)', ylabel='Densidad de potencia (µV²/Hz)', xlim=(1, 40))
    b.set_title('B. Espectro del mismo canal y potencia relativa por banda', loc='left', fontsize=10, pad=30)
    b.legend(frameon=False, fontsize=8, loc='lower left')
    for ax in (a1, a2, b):
        ax.grid(color=p2.REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)
    fig.savefig(p2.SALIDA / 'fig_ejemplo_senal.png', dpi=150)
    return sujeto


def tablas():
    TABLAS.mkdir(exist_ok=True)
    icc = pd.read_csv(p2.SALIDA / 'estabilidad_icc.csv')
    res = pd.read_csv(p2.SALIDA / 'resumen.csv').set_index(['dataset', 'ventana_s', 'solape_pct'])
    suj = pd.read_csv(p2.SALIDA / 'sujetos.csv')
    tiempos = pd.read_csv(p2.SALIDA / 'tiempos.csv')

    m = icc[icc.solape_pct == 0].groupby(['tipo', 'dataset', 'ventana_s']).icc.median()
    filas = [ETIQ[t] + ' & ' + ' & '.join(coma(m[(t, ds, v)]) for ds in p2.DATASETS for v in p2.VENTANAS) + r' \\'
             for t in p2.TIPOS]
    filas.insert(5, r'\midrule')
    filas.insert(11, r'\midrule')
    (TABLAS / 'icc_tipos.tex').write_text('\n'.join(filas) + '\n')

    filas = []
    for ds in p2.DATASETS:
        for i, (v, s) in enumerate(p2.CONFIGS):
            r = res.loc[(ds, v, s)]
            filas.append(f'{CORTO[ds] if i == 0 else ""} & {v} s & {s} \\% & {r.ventanas_por_sujeto:.0f} & '
                         f'{coma(r.pct_descartadas, 1)} & {r.t_ms_por_sujeto:.0f} & {coma(r.icc_mediana, 3)} & '
                         f'{coma(r.pct_icc_ge_075, 1)} \\\\')
        filas.append(r'\midrule')
    (TABLAS / 'resumen.tex').write_text('\n'.join(filas[:-1]) + '\n')

    med = tiempos.groupby(['ventana_s', 'solape_pct'])[['n_ventanas', 't_extraccion_s']].median()
    filas = [f'{v} s & {s} \\% & {med.loc[(v, s)].n_ventanas:.0f} & {1000 * med.loc[(v, s)].t_extraccion_s:.0f} & '
             + ' & '.join(coma(res.loc[(ds, v, s)].pct_descartadas, 1) for ds in p2.DATASETS) + r' \\'
             for v, s in p2.CONFIGS]
    (TABLAS / 'costo.tex').write_text('\n'.join(filas) + '\n')

    # Diapositivas: ICC mediano por familia, mínimo-máximo en las 16 configuraciones
    def rango(c, d=2):
        lo, hi = coma(c.min(), d), coma(c.max(), d)
        return lo if lo == hi else f'{lo}--{hi}'
    familias = {'icc_abs': 'Potencia absoluta', 'icc_rel': 'Potencia relativa', 'icc_ent_espectral': 'Entropía espectral',
                'icc_ent_permutacion': 'Entropía de permutación', 'icc_mediana': r'\textbf{Las 228}'}
    filas = [nombre + ' & ' + ' & '.join(rango(res.loc[ds, col]) for ds in p2.DATASETS) + r' \\'
             for col, nombre in familias.items()]
    filas.insert(4, r'\midrule')
    (TABLAS / 'estabilidad_familias.tex').write_text('\n'.join(filas) + '\n')

    # Diapositivas: tiempo por sujeto según solapamiento y ventanas descartadas (rango entre conjuntos, 0 %)
    filas = [f'{v} s & ' + ' & '.join(f'{1000 * med.loc[(v, s)].t_extraccion_s:.0f}' for s in p2.SOLAPES) + ' & '
             + rango(pd.Series([res.loc[(ds, v, 0)].pct_descartadas for ds in p2.DATASETS]), 1) + r' \\'
             for v in p2.VENTANAS]
    (TABLAS / 'costo_diapositivas.tex').write_text('\n'.join(filas) + '\n')

    filas = []
    for ds in p2.DATASETS:
        for g, nombre in ((1, 'Pacientes'), (0, 'Controles')):
            x = suj[(suj.dataset == ds) & (suj.grupo == g)]
            filas.append(f'{CORTO[ds] if g == 1 else ""} & {nombre} & {len(x)} & {int(x.incluido.sum())} & '
                         f'{coma(x.pct_segundos_malos.median(), 1)} \\\\')
    (TABLAS / 'muestra.tex').write_text('\n'.join(filas) + '\n')


if __name__ == '__main__':
    print('sujeto del ejemplo:', figura_ejemplo())
    tablas()
    for f in sorted(TABLAS.glob('*.tex')):
        print(f'--- {f.name}\n{f.read_text()}')
