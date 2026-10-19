"""
parte0.py -- Parte 0: Procesos simulados (incisos 0.2 a 0.5).

El inciso 0.1 es una derivacion a mano y vive en informe/tarea4.tex.

Salidas:
  figuras/p0_realizaciones.png
  figuras/p0_medias_moviles.png
  tablas/p0_tabla_ar1.csv
  tablas/p0_tabla_media_no_constante.csv
  tablas/p0_tabla_ljungbox.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import signal

import estilo
from estilo import plt
from utils_ts import acf_manual, banda_ruido_blanco, ljung_box, simular_ar1

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(RAIZ, "figuras")
TAB = os.path.join(RAIZ, "tablas")

N = 300          # largo de cada realizacion
BURN = 200       # pasos descartados para que el valor inicial no importe
REPS = 5000      # replicas por escenario
PHIS = [0.0, 0.5, 0.9, 1.0]
SEMILLAS = {0.0: 2026, 0.5: 2027, 0.9: 2028, 1.0: 2029}
SEMILLA_TEND = 2030
SEMILLA_SEM = 2031


# ----------------------------------------------------------------------------
# Simulacion vectorizada (0.2)
# ----------------------------------------------------------------------------
def simular_ar1_lote(reps, n, phi, c=0.0, sigma=1.0, burn=BURN, rng=None):
    """Simula `reps` realizaciones de un AR(1) de largo n, en bloque.

    Es la misma recursion que utils_ts.simular_ar1 pero resuelta con
    scipy.signal.lfilter, que aplica el filtro 1/(1 - phi B) de un solo golpe.
    El modulo test_sarima.py y la funcion escalar sirven de contraste.
    """
    rng = np.random.default_rng() if rng is None else rng
    eps = rng.normal(0.0, sigma, size=(reps, n + burn))
    eps[:, 0] += 0.0
    y = signal.lfilter([1.0], [1.0, -phi], eps + c, axis=1)
    return y[:, burn:]


def phi_ols(y):
    """Estimado OLS de phi: regresion de y_t contra y_{t-1} con intercepto."""
    y0, y1 = y[:-1], y[1:]
    X = np.column_stack([np.ones_like(y0), y0])
    beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
    return beta[1]


def resumen_escenario(Y, nombre, rezagos=(1, 5)):
    """Calcula la tabla de 0.2 para una matriz de replicas (reps x n)."""
    reps, n = Y.shape
    banda = banda_ruido_blanco(n)
    kmax = max(rezagos)
    R = np.empty((reps, kmax))
    phis = np.empty(reps)
    for i in range(reps):
        R[i] = acf_manual(Y[i], kmax)
        phis[i] = phi_ols(Y[i])
    fila = {"escenario": nombre, "n": n, "replicas": reps, "banda_1.96/sqrt(n)": banda}
    for k in rezagos:
        fila[f"prom_r{k}"] = R[:, k - 1].mean()
        fila[f"ee_r{k}"] = R[:, k - 1].std(ddof=1) / np.sqrt(reps)
    rech = np.abs(R[:, 0]) > banda
    fila["tasa_rechazo_r1"] = rech.mean()
    fila["ee_tasa_rechazo"] = np.sqrt(rech.mean() * (1 - rech.mean()) / reps)
    fila["prom_phi_ols"] = phis.mean()
    fila["ee_phi_ols"] = phis.std(ddof=1) / np.sqrt(reps)
    return fila, R


def tasa_ljungbox(Y, m=10, alpha=0.05):
    """Fraccion de replicas en que Ljung-Box con m rezagos rechaza al nivel alpha."""
    reps, n = Y.shape
    rech = np.empty(reps, dtype=bool)
    for i in range(reps):
        r = acf_manual(Y[i], m)
        _, _, p = ljung_box(r, n, m)
        rech[i] = p < alpha
    return rech.mean(), np.sqrt(rech.mean() * (1 - rech.mean()) / reps)


# ----------------------------------------------------------------------------
# Procesos sin memoria con media no constante (0.4)
# ----------------------------------------------------------------------------
def media_tendencia(n=N):
    """m_t = 2t/300: sube de 0 a 2 a lo largo de los 300 pasos."""
    t = np.arange(1, n + 1, dtype=float)
    return 2.0 * t / 300.0


def media_semanal(n=N):
    """m_t = -1.5 si t mod 7 == 6, y 0.25 en los demas dias."""
    t = np.arange(1, n + 1)
    return np.where(t % 7 == 6, -1.5, 0.25)


def simular_sin_memoria(reps, m, rng):
    """Y_t = m_t + eps_t con eps iid N(0,1): no hay memoria, solo media movil."""
    return m[None, :] + rng.normal(0.0, 1.0, size=(reps, m.size))


# ----------------------------------------------------------------------------
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    print("PARTE 0 -- procesos simulados")

    # -- 0.2 una realizacion de cada phi ------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True)
    for ax, phi in zip(axes.ravel(), PHIS):
        y = simular_ar1(N, phi, rng=np.random.default_rng(SEMILLAS[phi]))
        ax.plot(y, color=estilo.AZUL)
        ax.axhline(0, color=estilo.GRIS, lw=0.8, ls="--")
        etiqueta = "caminata aleatoria" if phi == 1.0 else ("ruido blanco" if phi == 0 else "")
        ax.set_title(rf"$\phi = {phi}$" + (f"  ({etiqueta})" if etiqueta else ""))
        ax.set_xlabel("t")
    fig.suptitle("Una realizacion de un AR(1) para cada $\\phi$ (n = 300)", y=1.0)
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p0_realizaciones.png"))

    # -- 0.2 tabla de 5000 replicas -----------------------------------------
    filas = []
    guardadas = {}
    for phi in PHIS:
        rng = np.random.default_rng(SEMILLAS[phi])
        Y = simular_ar1_lote(REPS, N, phi, rng=rng)
        guardadas[phi] = Y
        fila, _ = resumen_escenario(Y, f"AR(1) phi={phi}", rezagos=(1, 5))
        fila["phi"] = phi
        fila["phi^5_teorico"] = phi ** 5
        fila["semilla"] = SEMILLAS[phi]
        filas.append(fila)
        print(f"  phi={phi}: prom r1={fila['prom_r1']:.4f}  prom r5={fila['prom_r5']:.4f}"
              f"  (phi^5={phi**5:.4f})  rechazo={fila['tasa_rechazo_r1']:.4f}"
              f"  phi_ols={fila['prom_phi_ols']:.4f}")
    t02 = pd.DataFrame(filas)[
        ["escenario", "phi", "n", "replicas", "semilla", "prom_r1", "ee_r1",
         "prom_r5", "ee_r5", "phi^5_teorico", "tasa_rechazo_r1", "ee_tasa_rechazo",
         "prom_phi_ols", "ee_phi_ols", "banda_1.96/sqrt(n)"]]
    t02.to_csv(os.path.join(TAB, "p0_tabla_ar1.csv"), index=False)

    # -- 0.4 medias que cambian con el tiempo -------------------------------
    mt, ms = media_tendencia(), media_semanal()
    escen = {
        "tendencia lineal m_t=2t/300": (mt, SEMILLA_TEND),
        "patron semanal m_t=-1.5 si t mod 7 = 6": (ms, SEMILLA_SEM),
    }
    filas4 = []
    guardadas4 = {}
    for nombre, (m, sem) in escen.items():
        rng = np.random.default_rng(sem)
        Y = simular_sin_memoria(REPS, m, rng)
        guardadas4[nombre] = Y
        fila, _ = resumen_escenario(Y, nombre, rezagos=(1, 7))
        fila["semilla"] = sem
        filas4.append(fila)
        print(f"  {nombre}: prom r1={fila['prom_r1']:.4f}  prom r7={fila['prom_r7']:.4f}"
              f"  rechazo r1={fila['tasa_rechazo_r1']:.4f}")
    # para comparar, el ruido blanco puro (phi = 0)
    fila0, _ = resumen_escenario(guardadas[0.0], "ruido blanco (phi=0)", rezagos=(1, 7))
    fila0["semilla"] = SEMILLAS[0.0]
    filas4.insert(0, fila0)
    t04 = pd.DataFrame(filas4)[
        ["escenario", "n", "replicas", "semilla", "prom_r1", "ee_r1", "prom_r7",
         "ee_r7", "tasa_rechazo_r1", "ee_tasa_rechazo", "banda_1.96/sqrt(n)"]]
    t04.to_csv(os.path.join(TAB, "p0_tabla_media_no_constante.csv"), index=False)

    # figura de apoyo: las dos medias y una realizacion de cada una
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (nombre, (m, sem)) in zip(axes, escen.items()):
        y = guardadas4[nombre][0]
        ax.plot(y, color=estilo.GRIS, lw=0.9, label="una realizacion")
        ax.plot(m, color=estilo.NARANJA, lw=2, label="$m_t$")
        ax.set_title(nombre.split(" m_t")[0], fontsize=9)
        ax.set_xlabel("t")
        ax.legend()
    fig.suptitle("Procesos sin memoria cuya media cambia con el tiempo", y=1.02)
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p0_medias_moviles.png"))

    # -- 0.5 Ljung-Box con m = 10 -------------------------------------------
    filas5 = []
    casos = [("ruido blanco (phi=0)", guardadas[0.0])] + \
            [(k, v) for k, v in guardadas4.items()]
    for nombre, Y in casos:
        tasa, ee = tasa_ljungbox(Y, m=10, alpha=0.05)
        # para contraste, la tasa de la prueba de un solo rezago
        banda = banda_ruido_blanco(Y.shape[1])
        r1 = np.array([acf_manual(Y[i], 1)[0] for i in range(Y.shape[0])])
        tasa1 = np.mean(np.abs(r1) > banda)
        filas5.append({"escenario": nombre, "tasa_rechazo_LB_m10": tasa,
                       "ee_LB": ee, "tasa_rechazo_r1": tasa1,
                       "ee_r1": np.sqrt(tasa1 * (1 - tasa1) / Y.shape[0])})
        print(f"  Ljung-Box m=10 | {nombre}: rechazo={tasa:.4f}  (rezago 1: {tasa1:.4f})")
    pd.DataFrame(filas5).to_csv(os.path.join(TAB, "p0_tabla_ljungbox.csv"), index=False)
    print("  Parte 0 lista.")


if __name__ == "__main__":
    main()
