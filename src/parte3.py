"""
parte3.py -- Parte 3: Tendencia, estacionalidad y autocorrelacion.

Lo que hacen 3.1 y 3.2 es una descomposicion clasica multiplicativa:
serie = tendencia x estacionalidad x resto.

Salidas:
  figuras/p3_tendencia.png
  figuras/p3_indices.png
  figuras/p3_acf.png
  tablas/p3_indices_estacionales.csv
  tablas/p3_acf.csv
  tablas/p3_ljungbox.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import estilo
from estilo import plt
from utils_ts import acf_manual, banda_ruido_blanco, ljung_box

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(RAIZ, "figuras"), os.path.join(RAIZ, "tablas")
DER = os.path.join(RAIZ, "datos_derivados")

FIN_ENTRENAMIENTO = pd.Timestamp("2011-09-30")
DIAS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]


def cargar_serie():
    s = pd.read_csv(os.path.join(DER, "serie_diaria.csv"), parse_dates=["fecha"]) \
          .set_index("fecha")["venta_neta"]
    s.index.freq = None
    return s


def media_movil_centrada(serie, ventana=7):
    """Media movil centrada de `ventana` dias sobre el calendario completo.

    Los dias cerrados se rellenan con cero, como pide el inciso 3.1: para
    medir el nivel de la semana, un sabado cerrado aporta cero libras, no un
    dato faltante.
    """
    lleno = serie.fillna(0.0)
    return lleno.rolling(ventana, center=True, min_periods=ventana).mean()


def main():
    for c in (FIG, TAB):
        os.makedirs(c, exist_ok=True)
    print("PARTE 3 -- tendencia, estacionalidad y autocorrelacion")

    serie = cargar_serie()
    ma7 = media_movil_centrada(serie, 7)
    ma7.to_frame("ma7").to_csv(os.path.join(DER, "ma7.csv"))

    # ---------------------------------------------------------------- 3.1 --
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(serie.index, serie.values, color="#AAB4C0", lw=0.8, label="venta neta diaria")
    ax.plot(ma7.index, ma7.values, color=estilo.NARANJA, lw=2,
            label="media movil centrada de 7 dias")
    ax.axvline(FIN_ENTRENAMIENTO, color=estilo.VERDE, ls="--", lw=1.1,
               label="fin del entrenamiento")
    ax.set_title("Tendencia: media movil centrada de 7 dias sobre el calendario completo")
    ax.set_ylabel("GBP"); ax.legend(loc="upper left")
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p3_tendencia.png"))

    tr_ma = ma7[ma7.index <= FIN_ENTRENAMIENTO].dropna()
    print(f"  MA7: nivel al inicio {tr_ma.iloc[:14].mean():,.0f} GBP, "
          f"al final del entrenamiento {tr_ma.iloc[-14:].mean():,.0f} GBP, "
          f"maximo {tr_ma.max():,.0f} GBP el {tr_ma.idxmax().date()}")
    print(f"  MA7 sobre toda la serie: maximo {ma7.max():,.0f} GBP el {ma7.idxmax().date()}")

    # ---------------------------------------------------------------- 3.2 --
    # Solo entrenamiento, solo dias abiertos: cociente y_t / MA_t.
    ent = serie[serie.index <= FIN_ENTRENAMIENTO]
    coc = (ent / ma7.reindex(ent.index)).dropna()
    coc = coc[ent.reindex(coc.index).notna()]          # solo dias abiertos
    tabla = pd.DataFrame({"cociente": coc.values,
                          "dow": [DIAS_ES[d] for d in coc.index.dayofweek]})
    idx = tabla.groupby("dow")["cociente"].agg(["mean", "std", "count"])
    orden = [d for d in DIAS_ES if d in idx.index]
    idx = idx.loc[orden].rename(columns={"mean": "indice_estacional",
                                         "std": "desv_est", "count": "n_dias"})
    idx["indice_normalizado"] = idx["indice_estacional"] / idx["indice_estacional"].mean()
    idx.index.name = "dia_semana"
    idx.to_csv(os.path.join(TAB, "p3_indices_estacionales.csv"))
    print("\n3.2 -- indices estacionales (promedio de y_t / MA_t):")
    print(idx.round(4).to_string())
    print(f"  promedio de los seis indices: {idx['indice_estacional'].mean():.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colores = [estilo.AZUL if v < 1 else estilo.NARANJA for v in idx["indice_estacional"]]
    ax.bar(idx.index, idx["indice_estacional"], color=colores)
    ax.axhline(1.0, color=estilo.ROJO, ls="--", lw=1.1, label="indice = 1")
    for i, v in enumerate(idx["indice_estacional"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title("Indice estacional por dia de la semana (entrenamiento)")
    ax.set_ylabel("$y_t$ / MA$_t$"); ax.legend()
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p3_indices.png"))

    # ---------------------------------------------------------------- 3.3 --
    # Dias abiertos del entrenamiento, una fila por dia, SIN calendario.
    abiertos = ent.dropna()
    y = abiertos.values
    n = y.size
    kmax = 21
    r = acf_manual(y, kmax)
    banda = banda_ruido_blanco(n)
    fuera = np.where(np.abs(r) > banda)[0] + 1
    pico = int(np.argmax(np.abs(r))) + 1

    t_acf = pd.DataFrame({"k": np.arange(1, kmax + 1), "r_k": r,
                          "fuera_de_banda": np.abs(r) > banda})
    t_acf.to_csv(os.path.join(TAB, "p3_acf.csv"), index=False)

    print(f"\n3.3 -- ACF sobre {n} dias abiertos del entrenamiento "
          f"(banda +-{banda:.4f})")
    print(f"  rezagos fuera de banda ({len(fuera)} de {kmax}): {list(fuera)}")
    print(f"  r_1 = {r[0]:.4f} -> la prueba de rezago 1 "
          f"{'RECHAZA' if abs(r[0]) > banda else 'no rechaza'} ruido blanco")
    print(f"  pico en k = {pico} con r_{pico} = {r[pico-1]:.4f}")

    # Cuantos dias calendario representa el rezago del pico:
    # con la tienda cerrada los sabados, 6 dias abiertos = 7 dias calendario.
    dias_cal = []
    fechas = abiertos.index
    for i in range(pico, len(fechas)):
        dias_cal.append((fechas[i] - fechas[i - pico]).days)
    print(f"  el rezago {pico} corresponde a {np.median(dias_cal):.0f} dias calendario "
          f"(mediana; min {min(dias_cal)}, max {max(dias_cal)})")

    filas_lb = []
    for m in (6, 12):
        Q, gl, p = ljung_box(r, n, m)
        filas_lb.append({"serie": "venta neta, dias abiertos del entrenamiento",
                         "m": m, "Q": Q, "gl": gl, "valor_p": p,
                         "rechaza_5pct": p < 0.05})
        print(f"  Ljung-Box Q({m}) = {Q:.2f}, gl = {gl}, p = {p:.3g}")
    pd.DataFrame(filas_lb).to_csv(os.path.join(TAB, "p3_ljungbox.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.vlines(np.arange(1, kmax + 1), 0, r, color=estilo.AZUL, lw=2.5)
    ax.plot(np.arange(1, kmax + 1), r, "o", color=estilo.AZUL, ms=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(banda, color=estilo.ROJO, ls="--", lw=1)
    ax.axhline(-banda, color=estilo.ROJO, ls="--", lw=1,
               label=r"$\pm 1.96/\sqrt{n}$")
    ax.plot([pico], [r[pico - 1]], "o", color=estilo.NARANJA, ms=8, zorder=5,
            label=f"pico en k = {pico}")
    ax.set_xticks(range(1, kmax + 1))
    ax.set_xlabel("rezago k (en dias abiertos)"); ax.set_ylabel("$r_k$")
    ax.set_title("ACF de la venta neta diaria (dias abiertos del entrenamiento)")
    ax.legend()
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p3_acf.png"))

    abiertos.to_frame("venta_neta").to_csv(os.path.join(DER, "entrenamiento_abiertos.csv"))
    print("  Parte 3 lista.")


if __name__ == "__main__":
    main()
