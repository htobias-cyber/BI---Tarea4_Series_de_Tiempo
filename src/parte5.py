"""
parte5.py -- Parte 5: Pronostico y fuga.

Siete modelos (B1, B2, M1, M2, S1, S2 y el M2+MA7 de la trampa) evaluados de
tres formas: validacion cruzada aleatoria, origen movil dentro del
entrenamiento y origen movil en la prueba.

Reglas del origen movil, tal como las fija el enunciado:
  - el origen es un domingo, y su venta ya forma parte de lo observado;
  - los objetivos son los dias ABIERTOS del lunes al domingo siguientes;
  - en (b) no se evalua ningun dia despues del 30 de septiembre;
  - en (c) el ultimo dia evaluado es el 8 de diciembre;
  - para S1 y S2 el horizonte de cada objetivo es su posicion entre los dias
    abiertos posteriores al origen;
  - el MAE y el RMSE se calculan juntando todos los dias pronosticados.

Salidas:
  tablas/p5_resultados.csv
  tablas/p5_cv_detalle.csv
  tablas/p5_pronosticos_prueba.csv
  tablas/p5_trampa.csv
  figuras/p5_prueba.png
  figuras/p5_trampa.png
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import estilo
from estilo import plt
from utils_ts import SARIMA, mae, ols, rmse

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(RAIZ, "figuras"), os.path.join(RAIZ, "tablas")
DER = os.path.join(RAIZ, "datos_derivados")

FIN_ENT = pd.Timestamp("2011-09-30")      # fin del entrenamiento
INI_PRUEBA = pd.Timestamp("2011-10-03")   # primer dia evaluado de la prueba
FIN_PRUEBA = pd.Timestamp("2011-12-08")   # ultimo dia evaluado
ORIGEN0_PRUEBA = pd.Timestamp("2011-10-02")  # primer origen: sirve de historia
S = 6
MODELOS = ["B1", "B2", "M1", "M2", "S1", "S2"]


# ---------------------------------------------------------------------------
# Rezagos "mismo dia de la semana abierto mas reciente"
# ---------------------------------------------------------------------------
def rezagos_mismo_dow(fechas_abiertas):
    """Para cada dia abierto devuelve los dos dias abiertos anteriores del
    mismo dia de la semana.

    El enunciado pide que, si y_{t-7} cae en un dia cerrado, se use el mismo
    dia de la semana abierto mas reciente. Aplicado tambien a y_{t-14}, eso
    significa el primero y el segundo dia abierto anterior con el mismo dia
    de la semana. Ejemplo del enunciado: para el lunes 9 de mayo, el lunes
    anterior (2 de mayo) y el de hace dos semanas (25 de abril) fueron
    feriados, asi que y_{t-7} es el 18 de abril.
    """
    fechas = pd.DatetimeIndex(fechas_abiertas)
    por_dow = {}
    l1, l2 = [], []
    for f in fechas:
        prev = por_dow.setdefault(f.dayofweek, [])
        l1.append(prev[-1] if len(prev) >= 1 else pd.NaT)
        l2.append(prev[-2] if len(prev) >= 2 else pd.NaT)
        prev.append(f)
    return pd.Series(l1, index=fechas), pd.Series(l2, index=fechas)


def construir_panel(serie):
    """Panel de dias abiertos con t, dia de la semana y los dos rezagos."""
    ab = serie.dropna()
    l1, l2 = rezagos_mismo_dow(ab.index)
    p = pd.DataFrame({"y": ab.values}, index=ab.index)
    p["t"] = np.arange(len(p))                       # posicion entre dias abiertos
    p["dow"] = p.index.dayofweek
    p["f_lag1"] = l1.values
    p["f_lag2"] = l2.values
    p["y_lag7"] = p["f_lag1"].map(ab).values         # mismo dow mas reciente
    p["y_lag14"] = p["f_lag2"].map(ab).values        # segundo mas reciente
    return p


def _dummies(dows, niveles):
    X = np.zeros((len(dows), len(niveles)))
    for j, d in enumerate(niveles):
        X[:, j] = (np.asarray(dows) == d).astype(float)
    return X


# ---------------------------------------------------------------------------
# Los modelos. Cada uno recibe el panel de historia y el panel de objetivos.
# ---------------------------------------------------------------------------
def ajustar_predecir(nombre, hist, obj, extra_col=None):
    """Ajusta `nombre` sobre `hist` y predice las filas de `obj`."""
    niveles = sorted(hist["dow"].unique())

    if nombre == "B1":
        # Media historica de los dias abiertos.
        return np.full(len(obj), hist["y"].mean())

    if nombre == "B2":
        # Naive estacional: el mismo dia de la semana abierto mas reciente.
        # Es el pronostico de una caminata aleatoria estacional.
        return obj["y_lag7"].values.astype(float)

    if nombre == "M1":
        # OLS con tendencia lineal en t e indicadoras de dia de la semana.
        X = np.column_stack([hist["t"].values, _dummies(hist["dow"], niveles)])
        beta, *_ = np.linalg.lstsq(X, hist["y"].values, rcond=None)
        Xo = np.column_stack([obj["t"].values, _dummies(obj["dow"], niveles)])
        return Xo @ beta

    if nombre in ("M2", "M2t"):
        # OLS con indicadoras de dia de la semana y los dos rezagos.
        cols = ["y_lag7", "y_lag14"] + ([extra_col] if nombre == "M2t" else [])
        h = hist.dropna(subset=cols)          # filas sin rezago no se usan
        X = np.column_stack([_dummies(h["dow"], niveles), h[cols].values])
        beta, *_ = np.linalg.lstsq(X, h["y"].values, rcond=None)
        Xo = np.column_stack([_dummies(obj["dow"], niveles), obj[cols].values])
        return Xo @ beta

    raise ValueError(nombre)


def predecir_sarima(clave, y_hist, h_max, maxiter=500):
    """Ajusta S1 o S2 sobre log(y_hist) y pronostica h_max pasos.

    Devuelve (pronostico en GBP, convergio). Se aplica exp() al pronostico
    sin correccion de sesgo, como indica el enunciado.
    """
    ly = np.log(y_hist)
    if clave == "S1":
        mod = SARIMA(ly, order=(1, 0, 0), seasonal_order=(1, 0, 0, S), trend="c")
    else:
        mod = SARIMA(ly, order=(0, 1, 1), seasonal_order=(0, 1, 1, S), trend=None)
    res = mod.fit(maxiter=maxiter)
    return np.exp(res.forecast(h_max)), res.mle_retvals["converged"]


# ---------------------------------------------------------------------------
# Origen movil
# ---------------------------------------------------------------------------
def origenes_domingo(panel, desde, hasta):
    """Domingos abiertos dentro del rango, que sirven de origen."""
    f = panel.index
    return [d for d in f if d.dayofweek == 6 and desde <= d <= hasta]


def origen_movil(panel, origenes, fin_evaluacion, modelos, extra_col=None):
    """Corre el origen movil semanal y devuelve una fila por dia pronosticado."""
    filas = []
    conv = {"S1": [], "S2": []}
    for o in origenes:
        hist = panel.loc[:o]                                  # el origen ya se observo
        objetivos = panel.loc[(panel.index > o) &
                              (panel.index <= o + pd.Timedelta(days=7)) &
                              (panel.index <= fin_evaluacion)]
        if objetivos.empty:
            continue
        pred = {}
        for m in modelos:
            if m in ("S1", "S2"):
                # el horizonte es la posicion entre los dias abiertos posteriores
                h = len(objetivos)
                f, ok = predecir_sarima(m, hist["y"].values, h)
                conv[m].append(ok)
                pred[m] = f
            else:
                pred[m] = ajustar_predecir(m, hist, objetivos, extra_col)
        for i, (fecha, fila) in enumerate(objetivos.iterrows()):
            r = {"origen": o, "fecha": fecha, "h": i + 1,
                 "dow": fecha.dayofweek, "y": fila["y"]}
            for m in modelos:
                r[m] = pred[m][i]
            filas.append(r)
    for m, v in conv.items():
        if v:
            print(f"     {m}: convergio en {sum(v)}/{len(v)} origenes")
    return pd.DataFrame(filas)


def metricas(df, modelos):
    """MAE y RMSE juntando todos los dias pronosticados (no por semana)."""
    out = {}
    for m in modelos:
        ok = df[m].notna() & df["y"].notna()
        out[m] = {"MAE": mae(df.loc[ok, "y"], df.loc[ok, m]),
                  "RMSE": rmse(df.loc[ok, "y"], df.loc[ok, m]),
                  "n_dias": int(ok.sum())}
    return out


# ---------------------------------------------------------------------------
# (a) Validacion cruzada aleatoria de 5 folds
# ---------------------------------------------------------------------------
def cv_aleatoria(panel_ent, modelos, semillas=range(10), k=5, extra_col=None):
    """5 folds aleatorios sobre el entrenamiento, repetidos con 10 semillas.

    Solo tiene sentido para B1, M1 y M2 (y M2t): son modelos de fila, que no
    necesitan que las filas esten en orden. B2, S1 y S2 estan definidos por el
    orden del tiempo y un fold aleatorio lo destruye.
    """
    res = {m: {"MAE": [], "RMSE": []} for m in modelos}
    detalle = []
    n = len(panel_ent)
    for sem in semillas:
        rng = np.random.default_rng(2026 + sem)
        orden = rng.permutation(n)
        folds = np.array_split(orden, k)
        ys, preds = {m: [] for m in modelos}, {m: [] for m in modelos}
        for f in folds:
            prueba = panel_ent.iloc[np.sort(f)]
            entrena = panel_ent.iloc[np.sort(np.setdiff1d(orden, f))]
            for m in modelos:
                p = ajustar_predecir(m, entrena, prueba, extra_col)
                ok = ~np.isnan(np.asarray(p, dtype=float))
                preds[m].append(np.asarray(p)[ok])
                ys[m].append(prueba["y"].values[ok])
        for m in modelos:
            yv, pv = np.concatenate(ys[m]), np.concatenate(preds[m])
            res[m]["MAE"].append(mae(yv, pv))
            res[m]["RMSE"].append(rmse(yv, pv))
            detalle.append({"semilla": 2026 + sem, "modelo": m,
                            "MAE": mae(yv, pv), "RMSE": rmse(yv, pv)})
    resumen = {m: {"MAE": np.mean(v["MAE"]), "MAE_sd": np.std(v["MAE"], ddof=1),
                   "RMSE": np.mean(v["RMSE"]), "RMSE_sd": np.std(v["RMSE"], ddof=1)}
               for m, v in res.items()}
    return resumen, pd.DataFrame(detalle)


# ---------------------------------------------------------------------------
def main():
    print("PARTE 5 -- pronostico y fuga")
    serie = pd.read_csv(os.path.join(DER, "serie_diaria.csv"),
                        parse_dates=["fecha"]).set_index("fecha")["venta_neta"]
    panel = construir_panel(serie)
    panel_ent = panel.loc[:FIN_ENT]
    print(f"  dias abiertos: {len(panel)} en total, {len(panel_ent)} de entrenamiento")

    filas_res = []

    # ------------------------------------------------------------- 5.2 (a) --
    print("\n5.2(a) -- validacion cruzada aleatoria de 5 folds, 10 semillas")
    mods_cv = ["B1", "M1", "M2"]
    cv, cv_det = cv_aleatoria(panel_ent, mods_cv)
    cv_det.to_csv(os.path.join(TAB, "p5_cv_detalle.csv"), index=False)
    for m in mods_cv:
        print(f"  {m}: MAE {cv[m]['MAE']:>9,.0f} (sd {cv[m]['MAE_sd']:>6,.0f})   "
              f"RMSE {cv[m]['RMSE']:>9,.0f} (sd {cv[m]['RMSE_sd']:>6,.0f})")
        filas_res.append({"modelo": m, "estimacion": "(a) CV aleatoria 5 folds",
                          "MAE": cv[m]["MAE"], "MAE_sd": cv[m]["MAE_sd"],
                          "RMSE": cv[m]["RMSE"], "RMSE_sd": cv[m]["RMSE_sd"],
                          "n_dias": len(panel_ent)})
    for m in ["B2", "S1", "S2"]:
        filas_res.append({"modelo": m, "estimacion": "(a) CV aleatoria 5 folds",
                          "MAE": np.nan, "RMSE": np.nan, "n_dias": 0,
                          "nota": "no aplica: el modelo depende del orden del tiempo"})

    # ------------------------------------------------------------- 5.2 (b) --
    print("\n5.2(b) -- origen movil semanal dentro del entrenamiento (mayo a septiembre)")
    orig_b = origenes_domingo(panel, pd.Timestamp("2011-05-01"), FIN_ENT)
    print(f"     {len(orig_b)} origenes, del {orig_b[0].date()} al {orig_b[-1].date()}")
    om_b = origen_movil(panel, orig_b, FIN_ENT, MODELOS)
    mb = metricas(om_b, MODELOS)
    for m in MODELOS:
        print(f"  {m}: MAE {mb[m]['MAE']:>9,.0f}   RMSE {mb[m]['RMSE']:>9,.0f}   "
              f"({mb[m]['n_dias']} dias)")
        filas_res.append({"modelo": m, "estimacion": "(b) origen movil en entrenamiento",
                          **mb[m]})

    # ------------------------------------------------------------- 5.2 (c) --
    print("\n5.2(c) -- origen movil semanal en el periodo de prueba")
    orig_c = origenes_domingo(panel, ORIGEN0_PRUEBA, FIN_PRUEBA)
    print(f"     {len(orig_c)} origenes, del {orig_c[0].date()} al {orig_c[-1].date()} "
          f"(el {ORIGEN0_PRUEBA.date()} sirve de historia y no se evalua)")
    om_c = origen_movil(panel, orig_c, FIN_PRUEBA, MODELOS)
    om_c.to_csv(os.path.join(TAB, "p5_pronosticos_prueba.csv"), index=False)
    mc = metricas(om_c, MODELOS)
    for m in MODELOS:
        print(f"  {m}: MAE {mc[m]['MAE']:>9,.0f}   RMSE {mc[m]['RMSE']:>9,.0f}   "
              f"({mc[m]['n_dias']} dias)")
        filas_res.append({"modelo": m, "estimacion": "(c) origen movil en prueba", **mc[m]})

    tabla = pd.DataFrame(filas_res)
    tabla.to_csv(os.path.join(TAB, "p5_resultados.csv"), index=False)

    print("\n5.3 -- tabla unica (MAE en GBP)")
    piv = tabla.pivot_table(index="modelo", columns="estimacion", values="MAE")
    piv = piv.reindex(MODELOS)
    print(piv.round(0).to_string())
    ganador = min(MODELOS, key=lambda m: mc[m]["MAE"])
    print(f"  gana en prueba (MAE): {ganador}")
    ganador_r = min(MODELOS, key=lambda m: mc[m]["RMSE"])
    print(f"  gana en prueba (RMSE): {ganador_r}")

    # ---------------------------------------------------------------- 5.3 --
    fig, ax = plt.subplots(figsize=(10, 3.8))
    real = om_c.drop_duplicates("fecha").set_index("fecha")["y"]
    ax.plot(real.index, real.values, "o-", color="black", ms=3.5, lw=1.4,
            label="venta real", zorder=6)
    estilos = {"B1": ":", "B2": "--", "M1": "-.", "M2": "-", "S1": "--", "S2": "-"}
    for m in MODELOS:
        p = om_c.drop_duplicates("fecha").set_index("fecha")[m]
        ax.plot(p.index, p.values, estilos[m], lw=1.1, alpha=0.85,
                label=f"{m} (MAE {mc[m]['MAE']:,.0f})")
    ax.set_title("Pronosticos contra la serie real, octubre a diciembre de 2011")
    ax.set_ylabel("GBP")
    ax.legend(ncol=7, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), columnspacing=1.0)
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p5_prueba.png"))

    # ---------------------------------------------------------------- 5.4 --
    print("\n5.4 -- la trampa: agregar la media movil centrada de 7 dias a M2")
    ma7 = pd.read_csv(os.path.join(DER, "ma7.csv"), parse_dates=["fecha"]) \
            .set_index("fecha")["ma7"]
    panel_t = panel.copy()
    panel_t["ma7"] = ma7.reindex(panel_t.index).values
    panel_t_ent = panel_t.loc[:FIN_ENT]

    mods_t = ["M2", "M2t"]
    cvt, _ = cv_aleatoria(panel_t_ent, mods_t, extra_col="ma7")
    print("  (a) CV aleatoria:")
    for m in mods_t:
        print(f"      {m:4s}: MAE {cvt[m]['MAE']:>9,.0f} (sd {cvt[m]['MAE_sd']:>6,.0f})   "
              f"RMSE {cvt[m]['RMSE']:>9,.0f}")
    om_bt = origen_movil(panel_t, orig_b, FIN_ENT, mods_t, extra_col="ma7")
    mbt = metricas(om_bt, mods_t)
    print("  (b) origen movil en entrenamiento:")
    for m in mods_t:
        print(f"      {m:4s}: MAE {mbt[m]['MAE']:>9,.0f}   RMSE {mbt[m]['RMSE']:>9,.0f}")

    filas_t = []
    for m in mods_t:
        filas_t.append({"modelo": m, "estimacion": "(a) CV aleatoria",
                        "MAE": cvt[m]["MAE"], "MAE_sd": cvt[m]["MAE_sd"],
                        "RMSE": cvt[m]["RMSE"], "RMSE_sd": cvt[m]["RMSE_sd"]})
        filas_t.append({"modelo": m, "estimacion": "(b) origen movil en entrenamiento",
                        "MAE": mbt[m]["MAE"], "RMSE": mbt[m]["RMSE"]})
    filas_t.append({"modelo": "M2t", "estimacion": "(c) origen movil en prueba",
                    "MAE": np.nan, "RMSE": np.nan,
                    "nota": "NO SE PUEDE CALCULAR SIN HACER TRAMPA: la MA centrada "
                            "del dia t necesita y_{t+1}, y_{t+2} e y_{t+3}"})
    pd.DataFrame(filas_t).to_csv(os.path.join(TAB, "p5_trampa.csv"), index=False)

    mejora_a = 100 * (1 - cvt["M2t"]["MAE"] / cvt["M2"]["MAE"])
    mejora_b = 100 * (1 - mbt["M2t"]["MAE"] / mbt["M2"]["MAE"])
    print(f"  la MA7 'mejora' el MAE {mejora_a:.1f}% en (a) y {mejora_b:.1f}% en (b),"
          " y ninguna de las dos lo detecta.")

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(2)
    ax.bar(x - 0.18, [cvt["M2"]["MAE"], mbt["M2"]["MAE"]], 0.36,
           color=estilo.AZUL, label="M2")
    ax.bar(x + 0.18, [cvt["M2t"]["MAE"], mbt["M2t"]["MAE"]], 0.36,
           color=estilo.ROJO, label="M2 + MA7 centrada (ilegal)")
    ax.set_xticks(x); ax.set_xticklabels(["(a) CV aleatoria", "(b) origen movil"])
    ax.set_ylabel("MAE (GBP)")
    ax.set_title("La trampa: un predictor que mira al futuro mejora ambas estimaciones")
    ax.legend()
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p5_trampa.png"))

    print("  Parte 5 lista.")


if __name__ == "__main__":
    main()
