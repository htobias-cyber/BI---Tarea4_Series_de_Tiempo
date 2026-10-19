"""
parte1_2.py -- Parte 1 (Limpieza) y Parte 2 (El calendario).

Salidas:
  tablas/p1_conteos.csv
  tablas/p1_casos.csv
  tablas/p1_bitacora.csv
  tablas/p1_bitacora_orden_alterno.csv
  tablas/p1_extremos.csv
  tablas/p2_dias_sin_transacciones.csv
  tablas/p2_resumen_calendario.csv
  figuras/p2_serie.png
  datos_derivados/serie_diaria.csv   (la usan las partes 3, 4 y 5)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import estilo
from estilo import plt
from limpieza import (CODIGOS_SE_QUEDAN, CODIGOS_SE_VAN, _es_producto_o_venta,
                      cargar, limpiar, serie_venta_neta_diaria)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(RAIZ, "figuras"), os.path.join(RAIZ, "tablas")
DER = os.path.join(RAIZ, "datos_derivados")
XLSX = os.path.join(RAIZ, "data", "Online Retail.xlsx")

# Feriados bancarios de Inglaterra y Gales en el periodo de los datos.
# Fuente: calendario oficial de bank holidays del Reino Unido.
FERIADOS = {
    "2010-12-24": ("Nochebuena (la tienda cierra)", "fijo"),
    "2010-12-25": ("Navidad", "fijo"),
    "2010-12-26": ("Boxing Day", "fijo"),
    "2010-12-27": ("Navidad (dia sustituto)", "fijo"),
    "2010-12-28": ("Boxing Day (dia sustituto)", "fijo"),
    "2011-01-03": ("Ano Nuevo (dia sustituto)", "fijo"),
    "2011-04-22": ("Viernes Santo", "MOVIL"),
    "2011-04-25": ("Lunes de Pascua", "MOVIL"),
    "2011-04-29": ("Boda real (feriado extraordinario de 2011)", "irrepetible"),
    "2011-05-02": ("Early May Bank Holiday", "fijo (primer lunes de mayo)"),
    "2011-05-30": ("Spring Bank Holiday", "fijo (ultimo lunes de mayo)"),
    "2011-08-29": ("Summer Bank Holiday", "fijo (ultimo lunes de agosto)"),
}
DIAS_ES = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]


def main():
    for c in (FIG, TAB, DER):
        os.makedirs(c, exist_ok=True)
    print("PARTE 1 -- limpieza")

    crudo = cargar(XLSX)
    crudo["monto"] = crudo["Quantity"] * crudo["UnitPrice"]

    # ---------------------------------------------------------------- 1.1 --
    conteos = pd.DataFrame([
        {"concepto": "filas", "valor": len(crudo)},
        {"concepto": "facturas distintas", "valor": crudo.InvoiceNo.nunique()},
        {"concepto": "productos distintos (StockCode)", "valor": crudo.StockCode.nunique()},
        {"concepto": "clientes distintos (CustomerID)", "valor": crudo.CustomerID.nunique()},
        {"concepto": "paises", "valor": crudo.Country.nunique()},
        {"concepto": "filas sin CustomerID", "valor": int(crudo.CustomerID.isna().sum())},
        {"concepto": "fraccion de filas sin CustomerID",
         "valor": round(float(crudo.CustomerID.isna().mean()), 4)},
        {"concepto": "primera fecha", "valor": str(crudo.InvoiceDate.min())},
        {"concepto": "ultima fecha", "valor": str(crudo.InvoiceDate.max())},
        {"concepto": "venta bruta sin limpiar (GBP)", "valor": round(float(crudo.monto.sum()), 2)},
    ])
    conteos.to_csv(os.path.join(TAB, "p1_conteos.csv"), index=False)
    print(conteos.to_string(index=False))

    # ---------------------------------------------------------------- 1.2 --
    # Un renglon por caso, con el tamano del problema y la decision tomada.
    casos = []

    m = crudo.duplicated(keep="first")
    casos.append(("(a) filas duplicadas exactas", int(m.sum()), float(crudo.loc[m, "monto"].sum()),
                  "SE VAN",
                  "Misma factura, producto, cantidad y minuto: es un error de captura "
                  "o de exportacion. Si se dejan, el monto se cuenta dos veces."))

    pre = crudo.InvoiceNo.str[0]
    mA = (~pre.str.isdigit()) & (pre != "C")
    mC = pre == "C"
    casos.append(("(b) facturas con prefijo 'A'", int(mA.sum()), float(crudo.loc[mA, "monto"].sum()),
                  "SE VAN",
                  "Las tres dicen 'Adjust bad debt': es un ajuste contable por deuda "
                  "incobrable, no una venta ni la cancelacion de una venta."))
    casos.append(("(b) facturas con prefijo 'C'", int(mC.sum()), float(crudo.loc[mC, "monto"].sum()),
                  "SE QUEDAN",
                  "Son las cancelaciones. El enunciado pide que resten el dia en que "
                  "ocurren, asi que son parte de la venta neta."))

    mNP = ~_es_producto_o_venta(crudo["StockCode"])
    mSQ = crudo.StockCode.isin(CODIGOS_SE_QUEDAN)
    casos.append(("(c) StockCode que no es venta (AMAZONFEE, BANK CHARGES, CRUK, B, S)",
                  int(mNP.sum()), float(crudo.loc[mNP, "monto"].sum()), "SE VAN",
                  "Comisiones de terceros, cargos bancarios, ajustes de deuda y muestras: "
                  "son gastos o movimientos de inventario, no ventas al cliente."))
    casos.append(("(c) StockCode especial que SI es venta (POST, DOT, C2, D, M, PADS, gift_*, DCGS*)",
                  int(mSQ.sum()), float(crudo.loc[mSQ, "monto"].sum()), "SE QUEDAN",
                  "Envios y descuentos son dinero que el cliente paga o deja de pagar; "
                  "DCGS*, PADS y los vales si son productos; 'M' cancela ventas reales."))

    mP0 = crudo.UnitPrice == 0
    mPn = crudo.UnitPrice < 0
    casos.append(("(d) precio unitario igual a cero", int(mP0.sum()), 0.0, "SE VAN",
                  f"No aportan nada a la venta neta y {int(crudo.loc[mP0].Description.isna().sum())} "
                  "de ellas ni siquiera tienen descripcion; el resto dice 'check', "
                  "'damages', '?': son movimientos de bodega."))
    casos.append(("(d) precio unitario negativo", int(mPn.sum()), float(crudo.loc[mPn, "monto"].sum()),
                  "SE VAN",
                  "Son las mismas facturas 'A' de deuda incobrable, asi que la regla (b) "
                  "ya se las habia llevado."))

    mNeg = (crudo.Quantity < 0) & (~crudo.InvoiceNo.str.startswith("C"))
    top_desc = crudo.loc[mNeg, "Description"].value_counts(dropna=False).head(5)
    casos.append(("(e) cantidad negativa en factura que no es cancelacion", int(mNeg.sum()),
                  float(crudo.loc[mNeg, "monto"].sum()), "SE VAN",
                  "La columna Description las delata: 'damages', 'check', '?', 'lost', "
                  "'thrown away'. Son ajustes de inventario, no devoluciones de clientes. "
                  f"Todas tienen precio cero ({int((crudo.loc[mNeg,'UnitPrice']==0).sum())} de "
                  f"{int(mNeg.sum())}), asi que su monto es exactamente 0."))

    mSC = crudo.CustomerID.isna()
    casos.append(("(f) filas sin CustomerID", int(mSC.sum()), float(crudo.loc[mSC, "monto"].sum()),
                  "SE QUEDAN (para la serie)",
                  f"Son el {mSC.mean():.1%} de las filas y "
                  f"{float(crudo.loc[mSC,'monto'].sum()):,.0f} GBP de venta. Para la serie "
                  "agregada siguen siendo dinero que entro a la caja. Para la Parte 6 hay "
                  "que saber quien compro, y ahi si se excluyen."))

    t_casos = pd.DataFrame(casos, columns=["caso", "filas", "monto_GBP", "decision", "justificacion"])
    t_casos.to_csv(os.path.join(TAB, "p1_casos.csv"), index=False)
    print("\n" + t_casos[["caso", "filas", "monto_GBP", "decision"]].to_string(index=False))

    # ---------------------------------------------------------------- 1.4 --
    # Las seis lineas mas grandes en valor absoluto, ANTES de limpiar.
    idx = crudo.monto.abs().sort_values(ascending=False).index[:6]
    ext = crudo.loc[idx].copy()
    filas_ext = []
    for i, fila in ext.iterrows():
        inv = fila.InvoiceNo
        dia = fila.InvoiceDate.normalize()
        base = inv[1:] if inv.startswith(("C", "A")) else inv
        # que mas paso ese dia con ese cliente / esa factura
        mismo_dia = crudo[(crudo.InvoiceDate.dt.normalize() == dia) &
                          (crudo.InvoiceNo.str.lstrip("CA") == base)]
        pareja = mismo_dia[mismo_dia.InvoiceNo != inv]
        filas_ext.append({
            "InvoiceNo": inv, "StockCode": fila.StockCode,
            "Description": fila.Description, "Quantity": fila.Quantity,
            "UnitPrice": fila.UnitPrice, "monto_GBP": round(fila.monto, 2),
            "fecha_hora": str(fila.InvoiceDate),
            "contraparte_mismo_dia": "; ".join(
                f"{r.InvoiceNo} {r.monto:+,.2f} a las {r.InvoiceDate.time()}"
                for _, r in pareja.iterrows()) or "ninguna con el mismo numero base",
            "sobrevive_a_limpiar": bool(_es_producto_o_venta(pd.Series([fila.StockCode])).iloc[0]
                                        and fila.UnitPrice > 0 and not inv.startswith("A")),
        })
    t_ext = pd.DataFrame(filas_ext)
    t_ext.to_csv(os.path.join(TAB, "p1_extremos.csv"), index=False)
    print("\n1.4 -- seis lineas de mayor |Quantity x UnitPrice| antes de limpiar:")
    print(t_ext[["InvoiceNo", "Description", "Quantity", "monto_GBP", "fecha_hora",
                 "sobrevive_a_limpiar"]].to_string(index=False))

    # ---------------------------------------------------------------- 1.3 --
    limpio, bitacora = limpiar(crudo, devolver_bitacora=True)
    bitacora.to_csv(os.path.join(TAB, "p1_bitacora.csv"), index=False)
    print("\n1.3 -- bitacora de limpieza (orden principal):")
    print(bitacora[["orden", "regla", "filas_eliminadas", "monto_eliminado_GBP"]].to_string(index=False))

    # El orden cambia los conteos: aqui la demostracion. Si la regla de
    # cantidad negativa se aplica ANTES que la de precio cero, se lleva 1336
    # filas que en el orden principal ya se habia llevado la regla de precio.
    alt = crudo.copy()
    alt_log = []
    m = alt.duplicated(keep="first"); alt_log.append(("R1 duplicados exactos", int(m.sum()), float(alt.loc[m,"monto"].sum()))); alt = alt[~m]
    pre = alt.InvoiceNo.str[0]; m = (~pre.str.isdigit()) & (pre != "C")
    alt_log.append(("R2 prefijo distinto de digito o C", int(m.sum()), float(alt.loc[m,"monto"].sum()))); alt = alt[~m]
    m = ~_es_producto_o_venta(alt.StockCode); alt_log.append(("R3 StockCode que no es venta", int(m.sum()), float(alt.loc[m,"monto"].sum()))); alt = alt[~m]
    m = (alt.Quantity < 0) & (~alt.InvoiceNo.str.startswith("C"))
    alt_log.append(("R5 cantidad negativa sin cancelacion (ADELANTADA)", int(m.sum()), float(alt.loc[m,"monto"].sum()))); alt = alt[~m]
    m = alt.UnitPrice <= 0; alt_log.append(("R4 precio <= 0 (ATRASADA)", int(m.sum()), float(alt.loc[m,"monto"].sum()))); alt = alt[~m]
    t_alt = pd.DataFrame(alt_log, columns=["regla", "filas_eliminadas", "monto_eliminado_GBP"])
    t_alt["orden"] = range(1, len(t_alt) + 1)
    t_alt.to_csv(os.path.join(TAB, "p1_bitacora_orden_alterno.csv"), index=False)
    print("\n1.3 -- el mismo conjunto de reglas en otro orden:")
    print(t_alt[["orden", "regla", "filas_eliminadas"]].to_string(index=False))
    print(f"   filas finales: orden principal {len(limpio):,} | orden alterno {len(alt):,} "
          f"(iguales: {len(limpio) == len(alt)})")

    # ================================================================ 2 ====
    print("\nPARTE 2 -- el calendario")
    s = serie_venta_neta_diaria(limpio)

    # ---------------------------------------------------------------- 2.1 --
    ini, fin = s.index.min(), s.index.max()
    cal = pd.date_range(ini, fin, freq="D")
    print(f"  dias calendario entre {ini.date()} y {fin.date()} inclusive: {len(cal)}")
    print(f"  dias con al menos una transaccion: {len(s)}")
    print(f"  dias sin transacciones: {len(cal) - len(s)}")

    # ---------------------------------------------------------------- 2.2 --
    faltan = cal.difference(s.index)
    filas = []
    for f in faltan:
        clave = f.strftime("%Y-%m-%d")
        nombre, tipo = FERIADOS.get(clave, ("", ""))
        dow = DIAS_ES[f.dayofweek]
        if f.dayofweek == 5:
            motivo, mov = "sabado: la tienda no abre", "—"
        elif nombre:
            motivo, mov = f"feriado bancario: {nombre}", tipo
        else:
            motivo, mov = "sin explicacion en el calendario", "—"
        filas.append({"fecha": clave, "dia_semana": dow, "motivo": motivo, "tipo": mov})
    t_falt = pd.DataFrame(filas)
    t_falt.to_csv(os.path.join(TAB, "p2_dias_sin_transacciones.csv"), index=False)
    print(f"\n  dias sin transacciones, clasificados ({len(t_falt)}):")
    print(t_falt.to_string(index=False))

    sab = int((t_falt.dia_semana == "sabado").sum())
    fer = int(t_falt.motivo.str.startswith("feriado").sum())
    otros = len(t_falt) - sab - fer
    print(f"\n  sabados: {sab} | feriados: {fer} | sin explicar: {otros}")
    print("  dia de la semana por conteo de dias abiertos:")
    abiertos_dow = pd.Series(s.index.dayofweek).map(lambda i: DIAS_ES[i]).value_counts()
    print(abiertos_dow.to_string())

    # ---------------------------------------------------------------- 2.4 --
    ultimo = limpio[limpio.InvoiceDate.dt.normalize() == fin]
    print(f"\n2.4 -- ultimo dia {fin.date()}: {len(ultimo)} lineas, "
          f"ultima factura a las {ultimo.InvoiceDate.max().time()}, "
          f"venta {s.iloc[-1]:,.2f} GBP")
    hora_cierre = limpio.groupby(limpio.InvoiceDate.dt.normalize())["InvoiceDate"].max().dt.hour
    print(f"   hora de la ultima factura en el resto de los dias: "
          f"mediana {hora_cierre[:-1].median():.0f}h, minimo {hora_cierre[:-1].min():.0f}h")

    # ------------------------------------------------- serie definitiva ----
    # Calendario completo, dias cerrados como faltantes, sin el ultimo dia.
    serie = s.reindex(cal)
    serie = serie.iloc[:-1]                      # se descarta 2011-12-09
    serie.index.name = "fecha"
    serie.name = "venta_neta"
    serie.to_frame().to_csv(os.path.join(DER, "serie_diaria.csv"))

    resumen = pd.DataFrame([
        {"concepto": "primer dia", "valor": str(ini.date())},
        {"concepto": "ultimo dia de los datos", "valor": str(fin.date())},
        {"concepto": "dias calendario (inclusive)", "valor": len(cal)},
        {"concepto": "dias con al menos una transaccion", "valor": len(s)},
        {"concepto": "dias sin transacciones", "valor": len(cal) - len(s)},
        {"concepto": "   de ellos, sabados", "valor": sab},
        {"concepto": "   de ellos, feriados bancarios", "valor": fer},
        {"concepto": "   sin explicacion", "valor": otros},
        {"concepto": "serie final: dias en el calendario (sin el ultimo dia)", "valor": len(serie)},
        {"concepto": "serie final: dias abiertos", "valor": int(serie.notna().sum())},
        {"concepto": "venta neta total de la serie (GBP)", "valor": round(float(serie.sum()), 2)},
    ])
    resumen.to_csv(os.path.join(TAB, "p2_resumen_calendario.csv"), index=False)

    # ------------------------------------------------------------ figura ---
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(serie.index, serie.values, color=estilo.AZUL, lw=0.9)
    cerr = serie.index[serie.isna()]
    ax.plot(cerr, np.zeros(len(cerr)), "|", color=estilo.ROJO, ms=5,
            label=f"dias cerrados ({len(cerr)})")
    ax.axvline(pd.Timestamp("2011-09-30"), color=estilo.VERDE, ls="--", lw=1.1)
    ax.text(pd.Timestamp("2011-10-02"), ax.get_ylim()[1] * 0.92, " prueba →",
            color=estilo.VERDE, fontsize=8)
    ax.set_title("Venta neta diaria (GBP), calendario completo")
    ax.set_ylabel("GBP"); ax.legend(loc="upper left")
    fig.tight_layout()
    estilo.guardar(fig, os.path.join(FIG, "p2_serie.png"))
    print("  Partes 1 y 2 listas.")


if __name__ == "__main__":
    main()
