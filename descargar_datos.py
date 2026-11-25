"""Descarga los 3 datasets elegidos en el paso 1 a datos/<dataset>/.

- Rahmani 2018 (TEPT):        figshare 5393338, 12 .txt
- Mumtaz 2017 (depresión):    figshare 4244171, solo registros de ojos cerrados (EC)
- ds004504 (Alzheimer):       OpenNeuro, solo grupos A (Alzheimer) y C (controles)

Es reanudable: salta los archivos que ya están completos.
Uso: python -I descargar_datos.py
"""
import csv
import hashlib
import io
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parent / 'datos'
S3 = 'https://s3.amazonaws.com/openneuro.org/ds004504/'


def leer(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def bajar(url, destino, tam=None, md5=None):
    if destino.exists() and (tam is None or destino.stat().st_size == tam):
        return 'ya estaba'
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + '.part')
    for intento in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
                esperado = tam or int(r.headers.get('Content-Length', -1))
                shutil.copyfileobj(r, f, 1 << 20)
            if esperado >= 0 and tmp.stat().st_size != esperado:
                raise OSError(f'tamaño {tmp.stat().st_size} != {esperado}')
            if md5 and hashlib.md5(tmp.read_bytes()).hexdigest() != md5:
                raise OSError('md5 no coincide')
            tmp.rename(destino)
            return 'ok'
        except OSError as e:
            error = e
    raise OSError(f'{destino.name}: {error}')


def figshare(articulo, filtro, carpeta, renombrar=lambda n: n):
    archivos = json.loads(leer(f'https://api.figshare.com/v2/articles/{articulo}/files?page_size=1000'))
    return [(f['download_url'], RAIZ / carpeta / renombrar(f['name']), f['size'], f.get('computed_md5'))
            for f in archivos if filtro(f['name'])]


tareas = figshare(5393338, lambda n: n.endswith('.txt'), 'rahmani')
tareas += figshare(4244171, lambda n: n.endswith(' EC.edf'), 'mumtaz', lambda n: '_'.join(n.split()))

participantes = leer(S3 + 'participants.tsv').decode()
(RAIZ / 'ds004504').mkdir(parents=True, exist_ok=True)
(RAIZ / 'ds004504' / 'participants.tsv').write_text(participantes)
for fila in csv.DictReader(io.StringIO(participantes), delimiter='\t'):
    if fila['Group'] in ('A', 'C'):
        s = fila['participant_id']
        ruta = f'{s}/eeg/{s}_task-eyesclosed_eeg.set'
        tareas.append((S3 + ruta, RAIZ / 'ds004504' / ruta, None, None))

print(len(tareas), 'archivos')
with ThreadPoolExecutor(4) as ex:
    futuros = [ex.submit(bajar, *t) for t in tareas]
    fallos = 0
    for t, fut in zip(tareas, futuros):
        try:
            print(fut.result(), t[1].relative_to(RAIZ), flush=True)
        except OSError as e:
            fallos += 1
            print('FALLO', e, flush=True)
print('fallos:', fallos)
