"""Paso 2: el mismo preprocesamiento para los 3 datasets, 16 configuraciones de ventana
(1/2/4/8 s × 0/25/50/75 % de solapamiento), potencia por bandas + entropía, estabilidad
entre la 1.ª y la 2.ª mitad de cada registro (ICC) y tiempo de cómputo.

Uso:    .venv/bin/python -I paso2_features.py   (Windows: .venv\Scripts\python -I paso2_features.py)
Salida: resultados/paso2/  (features por ventana en .npz para el paso 3, CSV y figuras)
"""
import hashlib
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, periodogram, resample_poly, sosfiltfilt

BASE = Path(__file__).resolve().parent
DATOS, SALIDA = BASE / 'datos', BASE / 'resultados' / 'paso2'

FS = 256                    # Hz común: con 256 todos los pasos de ventana son enteros (250 × 0,25 = 62,5)
INICIO, DURACION = 1, 240   # s: se descarta el 1.er segundo (borde del filtro) y todos aportan 240 s
UMBRAL_PTP = 150            # µV: un segundo es "malo" si algún canal supera este pico a pico
MIN_VENTANAS_MITAD = 5      # ventanas limpias mínimas por mitad, en todas las configuraciones, para el ICC
CANALES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
           'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
BANDAS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 40)}
TIPOS = [f'abs_{b}' for b in BANDAS] + [f'rel_{b}' for b in BANDAS] + ['ent_espectral', 'ent_permutacion']
NOMBRES = [f'{t}_{c}' for t in TIPOS for c in CANALES]          # 12 tipos × 19 canales = 228
VENTANAS, SOLAPES = (1, 2, 4, 8), (0, 25, 50, 75)
CONFIGS = [(v, s) for v in VENTANAS for s in SOLAPES]           # (ventana en s, solapamiento en %): 16
SOS = butter(4, [1, 40], btype='bandpass', fs=FS, output='sos')


# ---------- lectura: cada cargador entrega (sujeto, grupo 1=paciente/0=control, x (19, n) en µV, fs) ----------
def cargar_rahmani():
    for f in sorted((DATOS / 'rahmani').glob('Origin_*.txt')):
        df = pd.read_csv(f, sep=r'\s+').rename(columns={'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'})
        sujeto = f.stem.split('_')[1]                        # P1..P6 = TEPT, H1..H6 = controles
        yield sujeto, int(sujeto[0] == 'P'), df[CANALES].to_numpy().T, 250   # 250 Hz según el artículo


def cargar_mne(raw):
    return raw.get_data()[[raw.ch_names.index(c) for c in CANALES]] * 1e6, raw.info['sfreq']


def cargar_ds004504():
    part = pd.read_csv(DATOS / 'ds004504' / 'participants.tsv', sep='\t')
    for s, g in part.loc[part.Group.isin(['A', 'C']), ['participant_id', 'Group']].itertuples(index=False):
        ruta = DATOS / 'ds004504' / s / 'eeg' / f'{s}_task-eyesclosed_eeg.set'
        yield s, int(g == 'A'), *cargar_mne(mne.io.read_raw_eeglab(ruta, preload=True, verbose='error'))


def cargar_mumtaz():
    for f in sorted((DATOS / 'mumtaz').glob('*_EC.edf')):
        raw = mne.io.read_raw_edf(f, preload=True, verbose='error')
        raw.rename_channels(lambda c: c.removeprefix('EEG ').removesuffix('-LE'))
        yield f.stem.removesuffix('_EC'), int(f.name.startswith('MDD')), *cargar_mne(raw)


DATASETS = {'TEPT (Rahmani)': cargar_rahmani,
            'Alzheimer (ds004504)': cargar_ds004504,
            'Depresión (Mumtaz)': cargar_mumtaz}
CLAVE = {'TEPT (Rahmani)': 'tept', 'Alzheimer (ds004504)': 'alzheimer', 'Depresión (Mumtaz)': 'depresion'}


# ---------- preprocesamiento común ----------
def preprocesar(x, fs):
    """250/256/500 Hz -> 256 Hz, pasa-banda 1-40 Hz (Butterworth 4, fase cero), referencia promedio,
    recorte [INICIO, INICIO+DURACION) s y máscara de segundos con artefactos. (None, None) si es corto."""
    x = resample_poly(x, FS, int(fs), axis=1) if fs != FS else x
    x = sosfiltfilt(SOS, x, axis=1)
    x = (x - x.mean(0))[:, INICIO * FS:(INICIO + DURACION) * FS]
    if x.shape[1] < DURACION * FS:
        return None, None
    ptp = np.ptp(x.reshape(len(x), DURACION, FS), axis=2)       # (canales, segundos)
    return x, (ptp > UMBRAL_PTP).any(0)


def ventanear(x, malo_seg, v, s):
    """Ventanas de v s con s % de solapamiento; descarta las que tocan un segundo malo."""
    n = v * FS
    inicios = np.arange(0, x.shape[1] - n + 1, n * (100 - s) // 100)
    malo = np.concatenate([[0], np.cumsum(np.repeat(malo_seg, FS))])
    total = len(inicios)
    inicios = inicios[malo[inicios + n] == malo[inicios]]
    return inicios, sliding_window_view(x, n, axis=1)[:, inicios], total


# ---------- features ----------
def ent_permutacion(X):
    """Entropía de permutación normalizada, orden 3 y retardo 1 (Bandt y Pompe, 2002)."""
    a, b, c = X[..., :-2], X[..., 1:-1], X[..., 2:]
    codigo = ((a < b).astype(np.uint8) << 2) | ((b < c).astype(np.uint8) << 1) | (a < c)
    p = np.stack([(codigo == k).sum(-1) for k in range(8)], -1) / codigo.shape[-1]
    return -(p * np.log(p + 1e-30)).sum(-1) / np.log(6)


def extraer(X):
    """X: (canales, ventanas, muestras) -> (12 tipos, canales, ventanas)."""
    f, P = periodogram(X, fs=FS, window='hann', axis=-1)          # quita la media de cada ventana
    df, rango = f[1] - f[0], (f >= 1) & (f < 40)
    total = P[..., rango].sum(-1) * df
    pot = [P[..., (f >= lo) & (f < hi)].sum(-1) * df for lo, hi in BANDAS.values()]
    p = P[..., rango] / P[..., rango].sum(-1, keepdims=True)
    ent_esp = -(p * np.log(p + 1e-30)).sum(-1) / np.log(rango.sum())
    return np.stack([np.log10(b) for b in pot] + [b / total for b in pot] + [ent_esp, ent_permutacion(X)])


def icc_a1(Y):
    """ICC(A,1) de McGraw y Wong (1996): acuerdo absoluto, una medida. Y: (sujetos, k mediciones, ...)."""
    n, k = Y.shape[:2]
    gm = Y.mean((0, 1))
    msr = k * ((Y.mean(1) - gm) ** 2).sum(0) / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum(0) / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum((0, 1)) / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def autotest():
    sf = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    assert abs(icc_a1(sf) - 0.29) < 0.01                         # ICC(2,1) de Shrout y Fleiss (1979)
    t = np.arange(4 * FS) / FS
    F = extraer(np.stack([np.sin(2 * np.pi * 10 * t), np.random.default_rng(0).standard_normal(t.size), t])[:, None])[..., 0]
    seno, ruido, rampa = 0, 1, 2
    assert F[TIPOS.index('rel_alpha'), seno] > 0.95              # un seno de 10 Hz cae entero en alfa
    assert F[TIPOS.index('ent_espectral'), seno] < 0.3 and F[TIPOS.index('ent_espectral'), ruido] > 0.85
    assert F[TIPOS.index('ent_permutacion'), rampa] < 1e-9 and F[TIPOS.index('ent_permutacion'), ruido] > 0.98
    malo = np.zeros(DURACION, bool)
    malo[10] = True                                              # artefacto en el segundo 10
    ini, _, total = ventanear(np.zeros((1, DURACION * FS)), malo, 4, 75)
    assert total == 237 and not ((ini < 11 * FS) & (ini + 4 * FS > 10 * FS)).any()
    # Avances enteros a 256 Hz y, para cada ventana, las ventanas de cada solapamiento son un
    # subconjunto de las de 75 % (todos los avances son múltiplos de n/4)
    for v, s in CONFIGS:
        assert (v * FS * (100 - s)) % 100 == 0
        a, _, _ = ventanear(np.zeros((1, DURACION * FS)), malo, v, s)
        b, _, _ = ventanear(np.zeros((1, DURACION * FS)), malo, v, 75)
        assert np.isin(a, b).all()


# ---------- figuras (paleta de referencia: rampa azul secuencial y 3 primeros colores categóricos) ----------
AZUL = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
        '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#f2a600']
TINTA, TINTA2, TENUE, REJILLA, EJE, FONDO = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
ETIQUETAS = ['Pot. absoluta δ', 'Pot. absoluta θ', 'Pot. absoluta α', 'Pot. absoluta β', 'Pot. absoluta γ',
             'Pot. relativa δ', 'Pot. relativa θ', 'Pot. relativa α', 'Pot. relativa β', 'Pot. relativa γ',
             'Entropía espectral', 'Entropía de permutación']
plt.rcParams.update({'font.family': 'sans-serif', 'text.color': TINTA, 'axes.labelcolor': TINTA2,
                     'xtick.color': TENUE, 'ytick.color': TENUE, 'axes.edgecolor': EJE,
                     'figure.facecolor': FONDO, 'axes.facecolor': FONDO, 'savefig.facecolor': FONDO})


def figura_icc(icc, n_icc):
    nv, ns = len(VENTANAS), len(SOLAPES)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True, constrained_layout=True)
    for ax, ds in zip(axes, DATASETS):
        m = icc[icc.dataset == ds].groupby(['tipo', 'ventana_s', 'solape_pct']).icc.median().unstack(['ventana_s', 'solape_pct'])
        im = ax.imshow(m.loc[TIPOS, CONFIGS].to_numpy(), cmap=LinearSegmentedColormap.from_list('azul', AZUL),
                       vmin=0, vmax=1, aspect='auto')
        for i in range(1, nv):
            ax.axvline(ns * i - 0.5, color=FONDO, lw=4)
        for borde in (4.5, 9.5):
            ax.axhline(borde, color=FONDO, lw=4)
        ax.set_xticks(range(len(CONFIGS)), [f'{s}' for _, s in CONFIGS], fontsize=7)
        for i, v in enumerate(VENTANAS):
            ax.text(ns * i + (ns - 1) / 2, -0.85, f'Ventana {v} s', ha='center', va='bottom', fontsize=9, color=TINTA2)
        ax.set_title(f'{ds}  (n = {n_icc[ds]})', fontsize=11, pad=22, loc='left')
        ax.set_xlabel('Solapamiento (%)')
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
    axes[0].set_yticks(range(12), ETIQUETAS, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01).set_label('ICC(A,1) entre mitades (mediana de los 19 canales)')
    fig.savefig(SALIDA / 'fig_estabilidad_icc.png', dpi=150)


def figura_costo(tiempos):
    med = tiempos.groupby(['ventana_s', 'solape_pct'])[['n_ventanas', 't_extraccion_s']].median()
    med['t_ms'] = med.t_extraccion_s * 1000
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
    ancho = 0.8 / len(SOLAPES)
    for ax, col, titulo, fmt in [(axes[0], 'n_ventanas', 'Ventanas limpias por sujeto (mediana)', '{:.0f}'),
                                 (axes[1], 't_ms', 'Tiempo de extracción por sujeto en ms (mediana)', '{:.0f}')]:
        for j, s in enumerate(SOLAPES):
            y = [med.loc[(v, s), col] for v in VENTANAS]
            barras = ax.bar(np.arange(len(VENTANAS)) + (j - (len(SOLAPES) - 1) / 2) * ancho, y, width=ancho - 0.02,
                            color=SERIES[j], label=f'{s} %')
            ax.bar_label(barras, [fmt.format(val) for val in y], padding=2, fontsize=7, color=TINTA2)
        ax.set_xticks(range(len(VENTANAS)), [f'{v} s' for v in VENTANAS])
        ax.set_xlabel('Longitud de ventana')
        ax.set_title(titulo, fontsize=11, loc='left')
        ax.grid(axis='y', color=REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for lado in ('top', 'right', 'left'):
            ax.spines[lado].set_visible(False)
    axes[0].legend(title='Solapamiento', frameon=False, fontsize=8, title_fontsize=8)
    fig.savefig(SALIDA / 'fig_ventanas_tiempo.png', dpi=150)


# ---------- flujo principal ----------
def main():
    autotest()
    (SALIDA / 'features').mkdir(parents=True, exist_ok=True)
    sujetos, tiempos, iccs, n_icc = [], [], [], {}
    for ds, cargar in DATASETS.items():
        acum, vistos = {cfg: [] for cfg in CONFIGS}, {}
        for sujeto, grupo, x, fs in cargar():
            duracion, huella = x.shape[1] / fs, hashlib.md5(x.tobytes()).hexdigest()
            t0 = time.perf_counter()
            x, malo = (None, None) if huella in vistos else preprocesar(x, fs)
            motivo = (f'duplicado de {vistos[huella]}' if huella in vistos else
                      f'dura {duracion:.0f} s' if x is None else '')
            vistos.setdefault(huella, sujeto)
            sujetos.append(dict(dataset=ds, sujeto=sujeto, grupo=grupo, duracion_s=duracion,
                                t_preproc_s=time.perf_counter() - t0, incluido=not motivo, excluido_por=motivo,
                                pct_segundos_malos=np.nan if motivo else 100 * malo.mean()))
            if motivo:
                print(f'{ds:22s} {sujeto:8s} EXCLUIDO: {motivo}', flush=True)
                continue
            for v, s in CONFIGS:
                t0 = time.perf_counter()
                inicios, X, total = ventanear(x, malo, v, s)
                F = extraer(X).transpose(2, 0, 1).reshape(len(inicios), len(NOMBRES))
                tiempos.append(dict(dataset=ds, sujeto=sujeto, ventana_s=v, solape_pct=s, n_total=total,
                                    n_ventanas=len(inicios), t_extraccion_s=time.perf_counter() - t0))
                mitad = np.select([inicios + v * FS <= DURACION * FS // 2, inicios >= DURACION * FS // 2], [0, 1], -1)
                acum[(v, s)].append((F.astype(np.float32), np.full(len(inicios), sujeto),
                                     np.full(len(inicios), grupo), inicios / FS, mitad))
            print(f'{ds:22s} {sujeto:8s} grupo={grupo} segundos malos={100 * malo.mean():5.1f} %', flush=True)

        datos = {}
        for (v, s), partes in acum.items():
            X, suj, grp, ini, mit = (np.concatenate(c) for c in zip(*partes))
            np.savez(SALIDA / 'features' / f'{CLAVE[ds]}_v{v}s_s{s}.npz', X=X, sujeto=suj, grupo=grp,
                     inicio_s=ini, mitad=mit, nombres=np.array(NOMBRES))
            datos[(v, s)] = (X, suj, mit)
        # Mismo conjunto de sujetos en las 16 configuraciones: así el ICC solo cambia por la configuración
        incluidos = [f['sujeto'] for f in sujetos if f['dataset'] == ds and f['incluido']]
        ok = [u for u in incluidos if all(((suj == u) & (mit == h)).sum() >= MIN_VENTANAS_MITAD
                                          for _, suj, mit in datos.values() for h in (0, 1))]
        n_icc[ds] = len(ok)
        for (v, s), (X, suj, mit) in datos.items():
            Y = np.stack([[X[(suj == u) & (mit == h)].mean(0) for h in (0, 1)] for u in ok])
            iccs += [dict(dataset=ds, ventana_s=v, solape_pct=s, feature=nf, tipo=nf.rsplit('_', 1)[0], icc=i)
                     for nf, i in zip(NOMBRES, icc_a1(Y.astype(np.float64)))]

    sujetos, tiempos, iccs = pd.DataFrame(sujetos), pd.DataFrame(tiempos), pd.DataFrame(iccs)
    sujetos.to_csv(SALIDA / 'sujetos.csv', index=False)
    tiempos.to_csv(SALIDA / 'tiempos.csv', index=False)
    iccs.to_csv(SALIDA / 'estabilidad_icc.csv', index=False)

    g = ['dataset', 'ventana_s', 'solape_pct']
    iccs['familia'] = np.where(iccs.tipo.str.startswith('ent'), iccs.tipo, iccs.tipo.str[:3])   # abs / rel / ent_*
    tot = tiempos.groupby(g)[['n_ventanas', 'n_total']].sum()
    resumen = pd.DataFrame({
        'ventanas_por_sujeto': tiempos.groupby(g).n_ventanas.median(),
        'pct_descartadas': 100 * (1 - tot.n_ventanas / tot.n_total),
        't_ms_por_sujeto': 1000 * tiempos.groupby(g).t_extraccion_s.median(),
        'icc_mediana': iccs.groupby(g).icc.median(),
        'pct_icc_ge_075': 100 * iccs.assign(b=iccs.icc >= 0.75).groupby(g).b.mean(),
    }).join(iccs.pivot_table(index=g, columns='familia', values='icc', aggfunc='median').add_prefix('icc_'))
    resumen.round(3).to_csv(SALIDA / 'resumen.csv')

    figura_icc(iccs, n_icc)
    figura_costo(tiempos)
    pd.set_option('display.width', 200)
    print(sujetos.groupby(['dataset', 'grupo']).agg(n=('sujeto', 'size'), incluidos=('incluido', 'sum'),
                                                    pct_seg_malos_mediana=('pct_segundos_malos', 'median'),
                                                    t_preproc_s=('t_preproc_s', 'median')).round(2))
    print('sujetos en el ICC:', n_icc)
    print(resumen.round(3).to_string())


if __name__ == '__main__':
    main()
