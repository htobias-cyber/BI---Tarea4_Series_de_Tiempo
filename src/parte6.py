"""
parte6.py -- Parte 6: Detras de la serie hay clientes (analisis RFM).

R = dias desde la ultima compra, medida al 1 de octubre de 2011
F = numero de facturas de venta distintas
M = venta neta total del cliente

Todo se calcula usando SOLO las transacciones anteriores al 1 de octubre de
2011. La ventana del 1 de octubre al 8 de diciembre es la que se pronostica,
asi que mirarla al construir R, F y M seria exactamente la fuga de la Parte 5.

Salidas:
  tablas/p6_rfm_resumen.csv
  tablas/p6_fraccion_por_puntaje.csv
  tablas/p6_auc.csv
  tablas/p6_campeones.csv
  tablas/p6_nuevos.csv
  datos_derivados/rfm.csv
  figuras/p6_fraccion_por_puntaje.png
  figuras/p6_roc.png
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import estilo
from estilo import plt
from limpieza import cargar, limpiar

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(RAIZ, "figuras"), os.path.join(RAIZ, "tablas")
DER = os.path.join(RAIZ, "datos_derivados")
XLSX = os.path.join(RAIZ, "data", "Online Retail.xlsx")

CORTE = pd.Timestamp("2011-10-01")      # fecha a la que se mide la recencia
FIN_VENTANA = pd.Timestamp("2011-12-08")
SEMILLA = 2026


def main():
    print("PARTE 6 -- analisis RFM")
    limpio = limpiar(cargar(XLSX))

    # Para RFM si se excluyen las filas sin CustomerID: aqui la pregunta es
    # *quien* compro, y una venta sin cliente no se le puede asignar a nadie.
    con_id = limpio[limpio["CustomerID"].notna()].copy()
    print(f"  filas limpias: {len(limpio):,} | con CustomerID: {len(con_id):,} "
          f"({len(con_id)/len(limpio):.1%})")

    antes = con_id[con_id["InvoiceDate"] < CORTE]
    ventana = con_id[(con_id["InvoiceDate"] >= CORTE) &
                     (con_id["InvoiceDate"] <= FIN_VENTANA + pd.Timedelta(days=1))]

    # ---------------------------------------------------------------- 6.1 --
    ventas = antes[~antes["cancelacion"]]
    rfm = pd.DataFrame({
        # R: dias desde la ultima compra, medidos al 1 de octubre
        "R": (CORTE - antes.groupby("CustomerID")["InvoiceDate"].max()).dt.days,
        # F: numero de facturas de VENTA distintas (las cancelaciones no cuentan)
        "F": ventas.groupby("CustomerID")["InvoiceNo"].nunique(),
        # M: venta neta total, las cancelaciones restan
        "M": antes.groupby("CustomerID")["monto"].sum(),
    })
    rfm["F"] = rfm["F"].fillna(0).astype(int)
    rfm = rfm.sort_index()
    n_cli = len(rfm)
    n_m0 = int((rfm["M"] <= 0).sum())
    print(f"\n6.1 -- clientes con al menos una transaccion antes del {CORTE.date()}: {n_cli}")
    print(f"  con M <= 0: {n_m0} ({n_m0/n_cli:.2%}) -- devolvieron mas de lo que compraron")
    print(f"  con F = 0 (solo cancelaciones): {int((rfm['F'] == 0).sum())}")
    print(rfm.describe().round(2).to_string())

    # ---------------------------------------------------------------- 6.2 --
    # Desempate al azar, exactamente como lo pide el enunciado.
    rfm = rfm.sort_index()                                   # CustomerID ascendente
    rng = np.random.default_rng(SEMILLA)
    rfm = rfm.iloc[rng.permutation(len(rfm))]                # barajar

    # En recencia, 5 es el mas reciente: menos dias = mejor, por eso van
    # las etiquetas al reves.
    rfm["R_score"] = pd.qcut(rfm["R"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["F"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["M"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_suma"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)
    rfm = rfm.sort_index()

    emp = rfm["F"].value_counts().sort_index()
    print(f"\n6.2 -- F toma solo {rfm['F'].nunique()} valores distintos entre {n_cli} clientes")
    print(f"  los cuatro valores mas comunes de F: "
          f"{dict(emp.sort_values(ascending=False).head(4))}")
    print("  cortes efectivos de F por quintil (min y max de F en cada puntaje):")
    print(rfm.groupby("F_score")["F"].agg(["min", "max", "count"]).to_string())

    # ---------------------------------------------------------------- 6.3 --
    compradores = set(ventana[~ventana["cancelacion"]]["CustomerID"].unique())
    rfm["compra"] = rfm.index.isin(compradores).astype(int)
    print(f"\n6.3 -- compraron entre el {CORTE.date()} y el {FIN_VENTANA.date()}: "
          f"{rfm['compra'].sum()} de {n_cli} ({rfm['compra'].mean():.2%})")

    filas = []
    for dim in ("R", "F", "M"):
        g = rfm.groupby(f"{dim}_score")["compra"].agg(["mean", "size"])
        for s, fila in g.iterrows():
            filas.append({"dimension": dim, "puntaje": int(s),
                          "fraccion_compro": fila["mean"], "n_clientes": int(fila["size"])})
    t_frac = pd.DataFrame(filas)
    t_frac.to_csv(os.path.join(TAB, "p6_fraccion_por_puntaje.csv"), index=False)
    piv = t_frac.pivot(index="puntaje", columns="dimension", values="fraccion_compro")
    print(piv.round(4).to_string())
    rango = (piv.max() - piv.min()).sort_values(ascending=False)
    print(f"  separacion (max - min de la fraccion): "
          f"{', '.join(f'{k} {v:.3f}' for k, v in rango.items())}")

    # AUC con los valores crudos; R con el signo invertido porque menos dias
    # es mejor.
    aucs = {
        "R (signo invertido)": roc_auc_score(rfm["compra"], -rfm["R"]),
        "F (crudo)": roc_auc_score(rfm["compra"], rfm["F"]),
        "M (crudo)": roc_auc_score(rfm["compra"], rfm["M"]),
        "suma de los tres puntajes": roc_auc_score(rfm["compra"], rfm["RFM_suma"]),
    }
    pd.DataFrame([{"clasificador": k, "AUC": v} for k, v in aucs.items()]) \
      .to_csv(os.path.join(TAB, "p6_auc.csv"), index=False)
    print("  AUC como clasificadores de compra_i:")
    for k, v in aucs.items():
        print(f"    {k:28s} {v:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    an = np.arange(1, 6)
    for i, dim in enumerate(("R", "F", "M")):
        ax.plot(an, piv[dim].values, "o-", label=dim)
    ax.axhline(rfm["compra"].mean(), color=estilo.GRIS, ls=":",
               label=f"base {rfm['compra'].mean():.1%}")
    ax.set_xticks(an); ax.set_xlabel("puntaje (5 = mejor)")
    ax.set_ylabel("fraccion que compro")
    ax.set_title("Fraccion que compro, por puntaje"); ax.legend()
    ax = axes[1]
    for k, pts in [("R (signo invertido)", -rfm["R"]), ("F (crudo)", rfm["F"]),
                   ("M (crudo)", rfm["M"]), ("suma de los tres puntajes", rfm["RFM_suma"])]:
        fpr, tpr, _ = roc_curve(rfm["compra"], pts)
        ax.plot(fpr, tpr, label=f"{k}  AUC {aucs[k]:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=estilo.GRIS, lw=0.9)
    ax.set_xlabel("falsos positivos"); ax.set_ylabel("verdaderos positivos")
    ax.set_title("Curvas ROC"); ax.legend(fontsize=7)
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p6_roc.png"))

    fig2, ax2 = plt.subplots(figsize=(6.4, 3.2))
    w = 0.26
    for i, dim in enumerate(("R", "F", "M")):
        ax2.bar(an + (i - 1) * w, piv[dim].values, w, label=dim)
    ax2.axhline(rfm["compra"].mean(), color=estilo.GRIS, ls=":")
    ax2.set_xticks(an); ax2.set_xlabel("puntaje (5 = mejor)")
    ax2.set_ylabel("fraccion que compro")
    ax2.set_title("6.3  Fraccion de clientes que compro, por puntaje")
    ax2.legend()
    fig2.tight_layout()
    estilo.guardar(fig2, os.path.join(FIG, "p6_fraccion_por_puntaje.png"))

    # ---------------------------------------------------------------- 6.4 --
    campeones = rfm[(rfm["R_score"] >= 4) & (rfm["F_score"] >= 4) & (rfm["M_score"] >= 4)]
    venta_ventana_total = ventana["monto"].sum()     # TODOS los clientes con CustomerID
    venta_campeones = ventana[ventana["CustomerID"].isin(campeones.index)]["monto"].sum()
    print(f"\n6.4 -- campeones (puntaje 4 o 5 en las tres dimensiones): {len(campeones)} "
          f"= {len(campeones)/n_cli:.2%} de los clientes")
    print(f"  generaron {venta_campeones:,.0f} GBP de {venta_ventana_total:,.0f} GBP "
          f"= {venta_campeones/venta_ventana_total:.2%} de la venta de la ventana")
    pd.DataFrame([{"n_campeones": len(campeones),
                   "fraccion_de_clientes": len(campeones) / n_cli,
                   "venta_campeones_GBP": venta_campeones,
                   "venta_total_ventana_GBP": venta_ventana_total,
                   "fraccion_de_la_venta": venta_campeones / venta_ventana_total}]) \
      .to_csv(os.path.join(TAB, "p6_campeones.csv"), index=False)

    # ---------------------------------------------------------------- 6.5 --
    # Cliente nuevo: ninguna factura de VENTA antes del 1 de octubre.
    con_venta_antes = set(ventas["CustomerID"].unique())
    v = ventana.copy()
    v["nuevo"] = ~v["CustomerID"].isin(con_venta_antes)
    venta_nuevos = v.loc[v["nuevo"], "monto"].sum()
    n_nuevos = v.loc[v["nuevo"], "CustomerID"].nunique()
    print(f"\n6.5 -- clientes nuevos en la ventana: {n_nuevos}")
    print(f"  generaron {venta_nuevos:,.0f} GBP = "
          f"{venta_nuevos/venta_ventana_total:.2%} de la venta de la ventana")
    print("  ningun modelo cliente por cliente construido al 1 de octubre puede "
          "verlos: no existian como filas.")
    pd.DataFrame([{"clientes_nuevos": n_nuevos, "venta_nuevos_GBP": venta_nuevos,
                   "venta_total_ventana_GBP": venta_ventana_total,
                   "fraccion_de_la_venta": venta_nuevos / venta_ventana_total}]) \
      .to_csv(os.path.join(TAB, "p6_nuevos.csv"), index=False)

    resumen = pd.DataFrame([
        {"concepto": "clientes con transaccion antes del 1-oct", "valor": n_cli},
        {"concepto": "clientes con M <= 0", "valor": n_m0},
        {"concepto": "valores distintos de F", "valor": int(rfm["F"].nunique())},
        {"concepto": "compraron en la ventana", "valor": int(rfm["compra"].sum())},
        {"concepto": "tasa base de compra", "valor": round(float(rfm["compra"].mean()), 4)},
        {"concepto": "campeones", "valor": len(campeones)},
        {"concepto": "clientes nuevos en la ventana", "valor": n_nuevos},
        {"concepto": "venta de la ventana (GBP, clientes con ID)",
         "valor": round(float(venta_ventana_total), 2)},
    ] + [{"concepto": f"AUC {k}", "valor": round(v, 4)} for k, v in aucs.items()])
    resumen.to_csv(os.path.join(TAB, "p6_rfm_resumen.csv"), index=False)
    rfm.to_csv(os.path.join(DER, "rfm.csv"))
    print("  Parte 6 lista.")


if __name__ == "__main__":
    main()
