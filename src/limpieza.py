"""
limpieza.py
===========
La funcion `limpiar(df)` de la Parte 1.5, en su propio archivo, tal como
pide el enunciado. Se reusa en la Parte 6 y quedara disponible para el
Proyecto 1.

Criterio general
----------------
La variable a pronosticar es la **venta neta diaria**: la suma de
Quantity x UnitPrice de todas las lineas de un dia, con las cancelaciones
restando el dia en que ocurren. Entonces la pregunta que decide cada regla
es una sola: *esta linea es dinero que un cliente pago (o dejo de pagar)
por algo que la tienda le vendio?* Si la respuesta es si, la linea se queda,
aunque el monto sea negativo. Si es un ajuste contable, un cargo de un
tercero o un movimiento de inventario, se va.

Las seis reglas (el orden importa y esta documentado en la bitacora)
---------------------------------------------------------------------
R1  Filas duplicadas exactas -> SE VAN.
R2  Facturas con prefijo que no es digito ni 'C' -> SE VAN.
R3  StockCode que no corresponde a una venta al cliente -> SE VAN.
R4  Precio unitario menor o igual a cero -> SE VAN.
R5  Cantidad negativa en factura que no es cancelacion -> SE VAN.
R6  Filas sin CustomerID -> SE QUEDAN (pero se marcan).

La justificacion de cada una esta en el docstring de la regla y, mas
extensa, en informe/tarea4.tex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Clasificacion de los StockCode que no son codigos de producto de 5 digitos.
# El enunciado avisa: "No todos son iguales: algunos si son productos."
# ---------------------------------------------------------------------------

# Se quedan: son dinero que el cliente efectivamente paga.
CODIGOS_SE_QUEDAN = {
    "POST",   # POSTAGE: el envio se le cobra al cliente, es ingreso de la tienda
    "DOT",    # DOTCOM POSTAGE: lo mismo para el canal web
    "C2",     # CARRIAGE: otro cargo de transporte facturado al cliente
    "D",      # Discount: un descuento es venta neta que se deja de cobrar
    "M", "m", # Manual: lineas capturadas a mano que cancelan o ajustan ventas
              #   reales (ver la nota sobre la factura C556445 mas abajo)
    "PADS",   # PADS TO MATCH ALL CUSHIONS: es un producto
}

# Se van: no son una venta a un cliente.
CODIGOS_SE_VAN = {
    "BANK CHARGES",  # comision que cobra el banco, es un gasto
    "AMAZONFEE",     # comision que cobra Amazon, es un gasto
    "CRUK",          # CRUK Commission: comision de una organizacion benefica
    "B",             # Adjust bad debt: ajuste contable por deuda incobrable
    "S",             # SAMPLES: muestras que salen de inventario sin venderse
}

# Nota sobre 'M' (Manual). Es tentador borrarlo por ambiguo, pero seria un
# error: la factura C556445 es una linea 'Manual' de -38,970 GBP que cancela
# la cesta de picnic de la factura 556444 del mismo dia. Si se borra 'M', la
# venta de 38,970 GBP se queda sin su cancelacion y aparece un pico falso en
# la serie. Esto se discute en el inciso 1.4.


def _es_producto_o_venta(stock_code: pd.Series) -> pd.Series:
    """True si el StockCode representa una venta al cliente."""
    sc = stock_code.astype(str)
    cinco_digitos = sc.str.match(r"^\d{5}")          # codigos de producto normales
    dcgs = sc.str.upper().str.startswith("DCGS")      # productos del canal dotcom
    vales = sc.str.lower().str.startswith("gift_")    # vales de regalo: son venta
    en_lista = sc.isin(CODIGOS_SE_QUEDAN)
    return cinco_digitos | dcgs | vales | en_lista


def cargar(ruta="data/Online Retail.xlsx") -> pd.DataFrame:
    """Lee el .xlsx original.

    InvoiceNo, StockCode y CustomerID se leen como texto a proposito:
      - InvoiceNo trae prefijos ('C', 'A'), asi que no es un numero;
      - StockCode mezcla digitos con letras ('POST', '22423H');
      - CustomerID es una etiqueta, no una cantidad, y si se lee como numero
        pandas lo vuelve float por los faltantes y termina siendo '17850.0'.
    """
    df = pd.read_excel(ruta, dtype={"InvoiceNo": str, "StockCode": str,
                                    "CustomerID": str})
    df["InvoiceNo"] = df["InvoiceNo"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["CustomerID"] = df["CustomerID"].str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def limpiar(df: pd.DataFrame, devolver_bitacora: bool = False):
    """Aplica las seis reglas de limpieza y devuelve el DataFrame limpio.

    Parametros
    ----------
    df : DataFrame crudo, tal como sale de `cargar`.
    devolver_bitacora : si es True devuelve (df_limpio, bitacora) donde la
        bitacora es un DataFrame con una fila por regla, el numero de filas
        que elimina y el monto en GBP que esas filas representaban.

    Devuelve
    --------
    DataFrame limpio con dos columnas nuevas:
        monto       = Quantity * UnitPrice
        cancelacion = True si la factura empieza con 'C'
    """
    d = df.copy()
    d["monto"] = d["Quantity"] * d["UnitPrice"]
    bitacora = []
    n_ini, m_ini = len(d), d["monto"].sum()

    def registrar(regla, mascara_quitar, detalle):
        """Anota cuantas filas y cuanto dinero quita una regla, y las quita."""
        q = int(mascara_quitar.sum())
        monto = float(d.loc[mascara_quitar, "monto"].sum())
        bitacora.append({"orden": len(bitacora) + 1, "regla": regla,
                         "filas_eliminadas": q, "monto_eliminado_GBP": monto,
                         "detalle": detalle})

    # -- R1: filas duplicadas exactas ---------------------------------------
    # Una linea de factura repetida caracter por caracter es casi seguro un
    # error de captura o de exportacion: la misma factura, el mismo producto,
    # la misma cantidad y el mismo minuto. Si se dejan, el monto se cuenta dos
    # veces. Se conserva la primera aparicion.
    m = d.duplicated(keep="first")
    registrar("R1 duplicados exactos", m, "se conserva la primera aparicion")
    d = d[~m]

    # -- R2: prefijos de factura que no son digito ni 'C' -------------------
    # Hay dos prefijos: 'C' (cancelacion) y 'A'. Las tres facturas 'A' dicen
    # "Adjust bad debt": es un ajuste contable por deuda incobrable, no una
    # venta ni la cancelacion de una venta. Las 'C' SE QUEDAN porque el
    # enunciado pide que las cancelaciones resten el dia en que ocurren.
    primera = d["InvoiceNo"].str[0]
    m = (~primera.str.isdigit()) & (primera != "C")
    registrar("R2 facturas con prefijo distinto de digito o C", m,
              "prefijo 'A' = ajuste por deuda incobrable; las 'C' se conservan")
    d = d[~m]

    # -- R3: StockCode que no es una venta al cliente -----------------------
    # Se van comisiones de terceros (AMAZONFEE, CRUK), cargos bancarios,
    # ajustes de deuda ('B') y muestras ('S'). Se quedan los cargos de envio
    # (POST, DOT, C2) porque el cliente los paga, los descuentos ('D') porque
    # reducen la venta neta, los vales de regalo y los codigos DCGS/PADS,
    # que si son productos. 'M' (Manual) se queda: cancela ventas reales.
    m = ~_es_producto_o_venta(d["StockCode"])
    codigos = ", ".join(sorted(d.loc[m, "StockCode"].unique())[:10])
    registrar("R3 StockCode que no es venta al cliente", m, f"codigos: {codigos}")
    d = d[~m]

    # -- R4: precio unitario menor o igual a cero ---------------------------
    # Un precio de cero no aporta nada a la venta neta y casi siempre viene
    # con la descripcion vacia o con textos de bodega ("check", "damages",
    # "?"): son movimientos de inventario, no ventas. Los precios negativos
    # que quedaban eran los ajustes de deuda, que R2 ya se llevo.
    m = d["UnitPrice"] <= 0
    registrar("R4 precio unitario <= 0", m,
              "lineas sin ingreso: ajustes de bodega y capturas vacias")
    d = d[~m]

    # -- R5: cantidad negativa en factura que no es cancelacion -------------
    # Devoluciones sin su factura 'C'. Despues de R4 casi no queda ninguna,
    # porque todas tenian precio cero; el conteo depende del orden y por eso
    # la bitacora lleva la columna 'orden'.
    m = (d["Quantity"] < 0) & (~d["InvoiceNo"].str.startswith("C"))
    registrar("R5 cantidad negativa sin factura de cancelacion", m,
              "ajustes de inventario que sobreviven a R4")
    d = d[~m]

    # -- R6: filas sin CustomerID -> SE QUEDAN ------------------------------
    # Aqui la respuesta depende de para que se usa el dato. Para la serie de
    # ventas, una transaccion sin cliente identificado sigue siendo dinero que
    # entro a la caja: son casi una cuarta parte de las filas y alrededor de
    # 1.45 millones de GBP. Borrarlas mutilaria la serie. Para la Parte 6 (RFM)
    # la pregunta es otra, porque hay que saber *quien* compro, y ahi si se
    # filtran. Por eso aqui solo se marcan.
    sin_cliente = d["CustomerID"].isna()
    bitacora.append({"orden": len(bitacora) + 1,
                     "regla": "R6 filas sin CustomerID (SE QUEDAN)",
                     "filas_eliminadas": 0,
                     "monto_eliminado_GBP": 0.0,
                     "detalle": (f"{int(sin_cliente.sum())} filas "
                                 f"({sin_cliente.mean():.2%}) por "
                                 f"{float(d.loc[sin_cliente, 'monto'].sum()):,.2f} GBP "
                                 "se conservan para la serie; se excluyen en la Parte 6")})

    d["cancelacion"] = d["InvoiceNo"].str.startswith("C")
    d = d.reset_index(drop=True)

    bit = pd.DataFrame(bitacora)
    bit.loc[len(bit)] = {"orden": len(bit) + 1, "regla": "TOTAL",
                         "filas_eliminadas": n_ini - len(d),
                         "monto_eliminado_GBP": m_ini - d["monto"].sum(),
                         "detalle": f"de {n_ini:,} filas quedan {len(d):,}"}

    if devolver_bitacora:
        return d, bit
    return d


def serie_venta_neta_diaria(d: pd.DataFrame) -> pd.Series:
    """Venta neta por dia: suma de `monto` de todas las lineas de cada dia.

    Las cancelaciones restan el dia en que ocurren, no el dia de la venta
    original, tal como pide el enunciado.
    """
    s = d.groupby(d["InvoiceDate"].dt.normalize())["monto"].sum()
    s.index.name = "fecha"
    s.name = "venta_neta"
    return s
