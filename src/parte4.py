"""
parte4.py -- Parte 4: Del AR(1) al SARIMA.

La receta: quitar la estacionalidad, mirar el residuo y modelar la memoria
que quede. Todo sobre los dias abiertos del entrenamiento, una fila por dia.

IMPORTANTE: statsmodels no estaba disponible en el entorno donde se corrio
esta tarea (sin acceso a PyPI), asi que el SARIMAX se implemento a mano en
src/utils_ts.py y se verifico contra procesos simulados en src/test_sarima.py.
La interfaz imita la de statsmodels: mod.fit(maxiter=...), res.resid,
res.mle_retvals['converged'], res.forecast(h).

Salidas:
  figuras/p4_acf_residuos_ols.png
  figuras/p4_acf_residuos_ar1.png
  tablas/p4_ols_dow.csv
  tablas/p4_ar1.csv
  tablas/p4_sarima.csv
  tablas/p4_diagnostico.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import estilo
from estilo import plt
from utils_ts import SARIMA, acf_manual, banda_ruido_blanco, ljung_box, ols

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(RAIZ, "figuras"), os.path.join(RAIZ, "tablas")
DER = os.path.join(RAIZ, "datos_derivados")
DIAS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
S = 6     # seis dias abiertos por semana: la tienda cierra los sabados
KMAX = 21


def matriz_dow(fechas):
    """Indicadoras de dia de la semana, una por dia abierto, sin intercepto."""
    dows = sorted({d for d in fechas.dayofweek})
    X = np.zeros((len(fechas), len(dows)))
    for j, d in enumerate(dows):
        X[:, j] = (fechas.dayofweek == d).astype(float)
    return X, [DIAS_ES[d] for d in dows]


def grafica_acf(r, n, titulo, ruta, extra=None):
    banda = banda_ruido_blanco(n)
    k = np.arange(1, len(r) + 1)
    fig, ax = plt.subplots(figsize=(8, 3.1))
    ax.vlines(k, 0, r, color=estilo.AZUL, lw=2.5)
    ax.plot(k, r, "o", color=estilo.AZUL, ms=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(banda, color=estilo.ROJO, ls="--", lw=1)
    ax.axhline(-banda, color=estilo.ROJO, ls="--", lw=1, label=r"$\pm 1.96/\sqrt{n}$")
    if extra is not None:
        ax.plot(k, extra, "s--", color=estilo.NARANJA, ms=3, lw=1,
                label=r"$\hat\phi^{\,k}$ de un AR(1)")
    ax.set_xticks(k)
    ax.set_xlabel("rezago k (en dias abiertos)"); ax.set_ylabel("$r_k$")
    ax.set_title(titulo); ax.legend()
    fig.tight_layout()
    estilo.guardar(fig, ruta)
    return banda


def main():
    print("PARTE 4 -- del AR(1) al SARIMA")
    ent = pd.read_csv(os.path.join(DER, "entrenamiento_abiertos.csv"),
                      parse_dates=["fecha"]).set_index("fecha")["venta_neta"]
    y = ent.values
    n = y.size
    print(f"  {n} dias abiertos de entrenamiento, "
          f"{ent.index.min().date()} a {ent.index.max().date()}")

    # ---------------------------------------------------------------- 4.1 --
    # OLS de la venta contra indicadoras de dia de la semana, sin intercepto.
    X, nombres = matriz_dow(ent.index)
    beta, e, fit = ols(X, y)
    t_ols = pd.DataFrame({"dia_semana": nombres, "coeficiente_GBP": beta,
                          "n_dias": X.sum(axis=0).astype(int)})
    t_ols.to_csv(os.path.join(TAB, "p4_ols_dow.csv"), index=False)
    print("\n4.1 -- OLS contra indicadoras de dia de la semana (sin intercepto):")
    print(t_ols.round(2).to_string(index=False))
    print(f"  R^2 = {1 - e.var()/y.var():.4f}  |  desv. est. de los residuos = {e.std():,.0f} GBP")

    r_e = acf_manual(e, KMAX)
    banda = grafica_acf(r_e, n,
                        "4.1  ACF de los residuos del OLS contra dia de la semana",
                        os.path.join(FIG, "p4_acf_residuos_ols.png"))
    fuera_e = list(np.where(np.abs(r_e) > banda)[0] + 1)
    print(f"  rezagos fuera de banda: {fuera_e}")
    print(f"  r_1 = {r_e[0]:.4f}, r_6 = {r_e[5]:.4f}")

    # ---------------------------------------------------------------- 4.2 --
    # AR(1) sobre los residuos: OLS de e_t contra e_{t-1}.
    Xa = np.column_stack([np.ones(n - 1), e[:-1]])
    (c_hat, phi_hat), u, _ = ols(Xa, e[1:])
    print(f"\n4.2 -- AR(1) sobre los residuos: c = {c_hat:,.1f}, phi = {phi_hat:.4f}")
    r_u = acf_manual(u, KMAX)
    banda_u = grafica_acf(r_u, u.size,
                          "4.2  ACF de los residuos del AR(1) ajustado a $e_t$",
                          os.path.join(FIG, "p4_acf_residuos_ar1.png"))
    fuera_u = list(np.where(np.abs(r_u) > banda_u)[0] + 1)
    print(f"  rezagos fuera de banda: {fuera_u}")

    filas_ar1 = [{"parametro": "c", "valor": c_hat},
                 {"parametro": "phi", "valor": phi_hat},
                 {"parametro": "desv_est_residuos_u", "valor": float(u.std())}]
    for m in (6, 12):
        Q, gl, p = ljung_box(r_u, u.size, m, dof_ajuste=1)   # un gl menos por phi
        filas_ar1.append({"parametro": f"LjungBox Q({m})", "valor": Q})
        filas_ar1.append({"parametro": f"LjungBox Q({m}) gl", "valor": gl})
        filas_ar1.append({"parametro": f"LjungBox Q({m}) valor_p", "valor": p})
        print(f"  Ljung-Box Q({m}) = {Q:.2f}, gl = {gl}, p = {p:.3g} "
              f"-> {'RECHAZA' if p < 0.05 else 'no rechaza'} ruido blanco")
    pd.DataFrame(filas_ar1).to_csv(os.path.join(TAB, "p4_ar1.csv"), index=False)

    # ---------------------------------------------------------------- 4.3 --
    # SARIMA sobre el LOGARITMO de la venta diaria, con s = 6.
    assert (y > 0).all(), "hay dias con venta neta <= 0: el log no esta definido"
    ly = np.log(y)

    print("\n4.3 -- SARIMA sobre log(venta), s = 6")
    resultados = {}
    filas = []
    for etiqueta, kwargs in [
        ("S1  SARIMA(1,0,0)(1,0,0)_6 con constante",
         dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, S), trend="c")),
        ("S2  SARIMA(0,1,1)(0,1,1)_6 sin constante",
         dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, S), trend=None)),
    ]:
        clave = etiqueta.split()[0]
        # Primero con el maxiter por omision; si no converge, se sube a 500 y
        # se comparan los parametros antes y despues, como pide el enunciado.
        r50 = SARIMA(ly, **kwargs).fit(maxiter=50)
        r500 = SARIMA(ly, **kwargs).fit(maxiter=500)
        print(f"  {etiqueta}")
        print(f"     maxiter=50 : convergio={r50.converged}  loglik={r50.llf:.3f}  "
              f"params={np.round(r50.params, 4)}")
        print(f"     maxiter=500: convergio={r500.converged}  loglik={r500.llf:.3f}  "
              f"params={np.round(r500.params, 4)}")
        res = r500
        resultados[clave] = res
        fila = {"modelo": etiqueta, "convergio_50": r50.converged,
                "convergio_500": res.converged, "loglik": res.llf,
                "sigma2": res.sigma2, "nobs": res.nobs, "aic": res.aic,
                "iteraciones": res.mle_retvals["iterations"]}
        for nom, val in zip(["phi", "Phi", "theta", "Theta"],
                            [res.phi, res.Phi, res.theta, res.Theta]):
            if len(val):
                fila[nom] = float(val[0])
        if res.modelo.trend == "c":
            fila["mu"] = res.mu
            # c de la forma phi(B)(y_t - mu) = eps_t  =>  c = mu * phi(1)
            fila["c = mu*phi(1)"] = res.mu * (1 - res.phi[0]) * (1 - res.Phi[0])
        filas.append(fila)
    pd.DataFrame(filas).to_csv(os.path.join(TAB, "p4_sarima.csv"), index=False)

    # ---------------------------------------------------------------- 4.4 --
    print("\n4.4 -- diagnostico: Ljung-Box sobre los residuos")
    diag = []
    for clave, res in resultados.items():
        resid = res.resid
        if clave == "S2":
            # (1-B)(1-B^6) se come 7 observaciones; esos primeros residuos son
            # artefactos del arranque del filtro, no informacion del modelo.
            resid = resid[7:]
            nota = "sin los primeros 7 residuos (artefactos del arranque)"
        else:
            nota = "todos los residuos"
        r = acf_manual(resid, 12)
        for m in (6, 12):
            Q, gl, p = ljung_box(r, resid.size, m, dof_ajuste=2)  # m - 2 por los 2 parametros
            diag.append({"modelo": clave, "m": m, "n_residuos": resid.size,
                         "Q": Q, "gl": gl, "valor_p": p,
                         "ruido_blanco_5pct": p >= 0.05, "nota": nota})
            print(f"  {clave}  Q({m}) = {Q:7.2f}  gl = {gl}  p = {p:.3g}  "
                  f"-> {'ruido blanco' if p >= 0.05 else 'NO es ruido blanco'}  ({nota})")
    pd.DataFrame(diag).to_csv(os.path.join(TAB, "p4_diagnostico.csv"), index=False)

    # ---------------------------------------------------------------- 4.5 --
    print("\n4.5 -- hacia donde van los pronosticos cuando el horizonte crece")
    for clave, res in resultados.items():
        f = res.forecast(120)
        print(f"  {clave}: h=1 -> {np.exp(f[0]):>10,.0f} GBP | "
              f"h=30 -> {np.exp(f[29]):>10,.0f} | h=120 -> {np.exp(f[119]):>10,.0f}")
        if clave == "S1":
            print(f"        media del modelo exp(mu) = {np.exp(res.mu):,.0f} GBP "
                  f"(el pronostico converge a ella)")
    print("  Parte 4 lista.")
    return resultados


if __name__ == "__main__":
    main()
