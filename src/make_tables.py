"""
make_tables.py -- convierte las tablas de tablas/*.csv en tablas de LaTeX
listas para \\input{} desde informe/tarea4.tex.

    python src/make_tables.py

Escribe en tablas/tex/*.tex. No recalcula nada: solo da formato a lo que
produjeron los modulos de cada parte.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(RAIZ, "tablas")
OUT = os.path.join(TAB, "tex")


# ---------------------------------------------------------------------------
def esc(x):
    """Escapa los caracteres que LaTeX interpreta."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    s = str(x)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    # acentos y caracteres sueltos del castellano escritos sin tilde en los CSV
    s = s.replace("±", r"$\pm$").replace("—", "---").replace("ñ", r"\~n")
    return s


def num(x, dec=0, miles=True):
    if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    s = f"{v:,.{dec}f}" if miles else f"{v:.{dec}f}"
    return s.replace(",", r"\,")


def tabla(nombre, encabezados, filas, alineacion, caption, label=None,
          nota=None, ancho=None, pequena=True, largo=False):
    """Escribe una tabla de LaTeX en tablas/tex/<nombre>.tex."""
    label = label or nombre
    L = []
    entorno = "longtable" if largo else "table"
    if largo:
        L.append(r"\begingroup\footnotesize" if pequena else r"\begingroup")
        L.append(r"\begin{longtable}{" + alineacion + "}")
        L.append(r"\caption{" + caption + r"}\label{tab:" + label + r"}\\")
        L.append(r"\toprule")
        L.append(" & ".join(r"\textbf{" + h + "}" for h in encabezados) + r" \\")
        L.append(r"\midrule\endfirsthead")
        L.append(r"\toprule")
        L.append(" & ".join(r"\textbf{" + h + "}" for h in encabezados) + r" \\")
        L.append(r"\midrule\endhead")
        L.append(r"\bottomrule\endfoot")
    else:
        L.append(r"\begin{table}[htbp]")
        L.append(r"\centering")
        if pequena:
            L.append(r"\footnotesize")
        L.append(r"\caption{" + caption + r"}")
        L.append(r"\label{tab:" + label + r"}")
        if ancho:
            L.append(r"\begin{tabularx}{\textwidth}{" + alineacion + "}")
        else:
            L.append(r"\begin{tabular}{" + alineacion + "}")
        L.append(r"\toprule")
        L.append(" & ".join(r"\textbf{" + h + "}" for h in encabezados) + r" \\")
        L.append(r"\midrule")
    for f in filas:
        L.append(" & ".join(f) + r" \\")
    if largo:
        L.append(r"\end{longtable}")
        L.append(r"\endgroup")
    else:
        L.append(r"\bottomrule")
        L.append(r"\end{tabularx}" if ancho else r"\end{tabular}")
        if nota:
            L.append(r"\\[0.3em]{\scriptsize " + nota + r"}")
        L.append(r"\end{table}")
    ruta = os.path.join(OUT, nombre + ".tex")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  -> {nombre}.tex")


def leer(n):
    return pd.read_csv(os.path.join(TAB, n + ".csv"))


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    print("Generando tablas de LaTeX")

    # ------------------------------------------------------------ Parte 0 --
    d = leer("p0_tabla_ar1")
    tabla("p0_tabla_ar1",
          [r"$\phi$", r"prom. $r_1$", r"prom. $r_5$", r"$\phi^5$ te\'orico",
           r"tasa rechazo $|r_1|$", r"prom. $\hat\phi_{\text{OLS}}$", "semilla"],
          [[num(r["phi"], 2, False),
            f"{num(r['prom_r1'], 4, False)} \\tiny$\\pm${num(r['ee_r1'], 4, False)}",
            f"{num(r['prom_r5'], 4, False)} \\tiny$\\pm${num(r['ee_r5'], 4, False)}",
            num(r["phi^5_teorico"], 4, False),
            f"{num(r['tasa_rechazo_r1'], 4, False)} \\tiny$\\pm${num(r['ee_tasa_rechazo'], 4, False)}",
            f"{num(r['prom_phi_ols'], 4, False)} \\tiny$\\pm${num(r['ee_phi_ols'], 4, False)}",
            str(int(r["semilla"]))] for _, r in d.iterrows()],
          "ccccccc",
          r"\textbf{0.2} AR(1) simulado: $n=300$, 5\,000 r\'eplicas por escenario. "
          r"Cada celda trae el promedio sobre las r\'eplicas y, en peque\~no, su error "
          r"est\'andar. La banda de la prueba de rezago~1 es $\pm 1.96/\sqrt{300} = 0.1132$.",
          nota=r"\textbf{Qu\'e leer:} con $\phi=0$ la tasa de rechazo es exactamente el "
               r"5\,\% nominal. Con $\phi=1$, $r_5=0.88$ en vez del $\phi^5=1$ te\'orico, "
               r"porque la f\'ormula supone estacionariedad. Y $\hat\phi_{\text{OLS}}$ est\'a "
               r"sesgado hacia abajo, cada vez m\'as conforme $\phi\to1$.")

    d = leer("p0_tabla_media_no_constante")
    tabla("p0_tabla_media_no_constante",
          ["escenario", r"prom. $r_1$", r"prom. $r_7$", r"tasa rechazo $|r_1|$", "semilla"],
          [[esc(r["escenario"].split(" m\\_t")[0].replace("m_t", "")[:38]),
            f"{num(r['prom_r1'], 4, False)} \\tiny$\\pm${num(r['ee_r1'], 4, False)}",
            f"{num(r['prom_r7'], 4, False)} \\tiny$\\pm${num(r['ee_r7'], 4, False)}",
            f"{num(r['tasa_rechazo_r1'], 4, False)} \\tiny$\\pm${num(r['ee_tasa_rechazo'], 4, False)}",
            str(int(r["semilla"]))] for _, r in d.iterrows()],
          "lcccc",
          r"\textbf{0.4} Procesos \emph{sin memoria} cuya media cambia con el tiempo. "
          r"5\,000 r\'eplicas de $n=300$ por escenario.",
          nota=r"\textbf{Qu\'e leer:} la tendencia reprueba la prueba de rezago~1 el "
               r"98.5\,\% de las veces sin tener ninguna memoria. El patr\'on semanal la "
               r"\emph{pasa} (11.5\,\%) aunque su $r_7=0.26$ delata una estructura enorme: "
               r"la prueba mira donde no hay nada.")

    d = leer("p0_tabla_ljungbox")
    tabla("p0_tabla_ljungbox",
          ["escenario", r"Ljung--Box $m=10$", "prueba de rezago 1"],
          [[esc(r["escenario"].split(" m_t")[0][:40]),
            f"{num(r['tasa_rechazo_LB_m10'], 4, False)} \\tiny$\\pm${num(r['ee_LB'], 4, False)}",
            f"{num(r['tasa_rechazo_r1'], 4, False)} \\tiny$\\pm${num(r['ee_r1'], 4, False)}"]
           for _, r in d.iterrows()],
          "lcc",
          r"\textbf{0.5} Tasa de rechazo al 5\,\% de Ljung--Box con $m=10$, "
          r"frente a la prueba de un solo rezago.",
          nota=r"\textbf{Qu\'e leer:} el caso semanal es el que cambia. Ljung--Box lo "
               r"rechaza el 95.5\,\% de las veces contra el 11.5\,\% de la prueba de "
               r"rezago~1, porque $Q(10)$ incluye el rezago~7 en la suma.")

    # ------------------------------------------------------------ Parte 1 --
    d = leer("p1_conteos")
    tabla("p1_conteos", ["concepto", "valor"],
          [[esc(r["concepto"]), esc(r["valor"])] for _, r in d.iterrows()],
          "lr", r"\textbf{1.1} El archivo antes de limpiar.",
          nota=r"\textbf{Qu\'e leer:} una cuarta parte de las filas no tiene cliente "
               r"identificado; qu\'e hacer con ellas es la decisi\'on m\'as delicada "
               r"de la limpieza.")

    d = leer("p1_casos")
    tabla("p1_casos", ["caso", "filas", r"monto (\pounds)", "decisi\\'on"],
          [[esc(r["caso"])[:72], num(r["filas"]), num(r["monto_GBP"]),
            r"\textbf{" + esc(r["decision"]) + "}"] for _, r in d.iterrows()],
          "p{8.3cm}rrl",
          r"\textbf{1.2} Los seis casos del enunciado, con su tama\~no y la decisi\'on "
          r"tomada. La justificaci\'on de cada uno est\'a en el texto.",
          nota=r"\textbf{Qu\'e leer:} las dos decisiones de \emph{conservar} son tan "
               r"importantes como las de eliminar. Las cancelaciones deben restar el "
               r"d\'ia en que ocurren, y las filas sin cliente son 1.45 millones de "
               r"libras de venta real.")

    for nom, cap, nota in [
        ("p1_bitacora",
         r"\textbf{1.3} Bit\'acora de limpieza. El orden en que se aplican las reglas "
         r"es parte de la bit\'acora.",
         r"\textbf{Qu\'e leer:} R5 elimina cero filas porque R4 ya se llev\'o las "
         r"1\,336 filas de cantidad negativa, que ten\'ian todas precio cero. "
         r"La tabla siguiente muestra qu\'e pasa si se invierte el orden."),
        ("p1_bitacora_orden_alterno",
         r"\textbf{1.3} Las mismas reglas con R4 y R5 intercambiadas.",
         r"\textbf{Qu\'e leer:} la misma regla elimina 1\,336 filas aqu\'i y 0 en la "
         r"tabla anterior. El conjunto final es id\'entico (533\,979 filas), pero la "
         r"bit\'acora cuenta otra historia: sin el orden documentado no es reproducible.")]:
        d = leer(nom)
        tabla(nom, ["orden", "regla", "filas eliminadas", r"monto (\pounds)"],
              [[str(int(r["orden"])), esc(r["regla"])[:56],
                num(r["filas_eliminadas"]), num(r["monto_eliminado_GBP"], 2)]
               for _, r in d.iterrows()],
              "clrr", cap, nota=nota)

    d = leer("p1_extremos")
    tabla("p1_extremos",
          ["factura", "descripci\\'on", "cantidad", r"monto (\pounds)", "fecha y hora",
           "sobrevive"],
          [[esc(r["InvoiceNo"]), esc(str(r["Description"]))[:30], num(r["Quantity"]),
            num(r["monto_GBP"], 2), esc(str(r["fecha_hora"])[:16]),
            r"s\'i" if r["sobrevive_a_limpiar"] else "no"] for _, r in d.iterrows()],
          "llrrlc",
          r"\textbf{1.4} Las seis l\'ineas de mayor $|$Quantity $\times$ UnitPrice$|$ "
          r"\emph{antes} de limpiar.",
          nota=r"\textbf{Qu\'e leer:} son tres pares venta/cancelaci\'on, no seis "
               r"eventos. Las seis sobreviven a la limpieza porque ninguna es un error; "
               r"lo que las resuelve es netear por d\'ia, ya que en los tres casos la "
               r"cancelaci\'on ocurre el mismo d\'ia (12, 16 y 3 minutos despu\'es).")

    # ------------------------------------------------------------ Parte 2 --
    d = leer("p2_resumen_calendario")
    tabla("p2_resumen_calendario", ["concepto", "valor"],
          [[esc(r["concepto"]), esc(r["valor"])] for _, r in d.iterrows()],
          "lr", r"\textbf{2.1--2.4} El calendario de la serie.",
          nota=r"\textbf{Qu\'e leer:} de 374 d\'ias calendario solo 305 tuvieron "
               r"ventas. Los 53 s\'abados son el hallazgo que fija $s=6$ en la Parte~4.")

    d = leer("p2_dias_sin_transacciones")
    d = d[d["dia_semana"] != "sabado"]
    tabla("p2_feriados", ["fecha", "d\\'ia", "motivo", "tipo"],
          [[esc(r["fecha"]), esc(r["dia_semana"]), esc(r["motivo"])[:46],
            r"\textbf{" + esc(r["tipo"]) + "}" if "MOVIL" in str(r["tipo"]) or
            "irrepetible" in str(r["tipo"]) else esc(r["tipo"])]
           for _, r in d.iterrows()],
          "llll",
          r"\textbf{2.2} Los d\'ias sin transacciones que \emph{no} son s\'abado. "
          r"Los 53 s\'abados se omiten de la tabla por brevedad.",
          nota=r"\textbf{Qu\'e leer:} tres de los feriados son m\'oviles o "
               r"irrepetibles. La Pascua se mueve hasta un mes entre a\~nos y la boda "
               r"real de 2011 no se repite nunca, as\'i que ning\'un modelo entrenado "
               r"con un a\~no de datos puede anticiparlos.")

    # ------------------------------------------------------------ Parte 3 --
    d = leer("p3_indices_estacionales")
    tabla("p3_indices",
          ["d\\'ia", r"\'indice $y_t/\mathrm{MA}_t$", "desv. est.", r"$n$ d\'ias",
           "normalizado"],
          [[esc(r["dia_semana"]), num(r["indice_estacional"], 4, False),
            num(r["desv_est"], 4, False), str(int(r["n_dias"])),
            num(r["indice_normalizado"], 4, False)] for _, r in d.iterrows()],
          "lcccc",
          r"\textbf{3.2} \'Indice estacional por d\'ia de la semana, calculado solo "
          r"sobre el entrenamiento.",
          nota=r"\textbf{Qu\'e leer:} jueves y martes venden m\'as del triple que el "
               r"domingo. El promedio de los seis \'indices es 1.2005 y no 1 porque el "
               r"denominador promedia siete d\'ias calendario de los cuales uno es un "
               r"s\'abado cerrado que entra como cero, lo que lo subestima por un "
               r"factor de $6/7$.")

    d = leer("p3_ljungbox")
    tabla("p3_ljungbox", [r"$m$", r"$Q(m)$", "gl", r"valor-$p$", r"rechaza al 5\,\%"],
          [[str(int(r["m"])), num(r["Q"], 2, False), str(int(r["gl"])),
            f"{r['valor_p']:.2e}", r"\textbf{s\'i}" if r["rechaza_5pct"] else "no"]
           for _, r in d.iterrows()],
          "ccccc",
          r"\textbf{3.3} Ljung--Box sobre la venta neta de los 245 d\'ias abiertos "
          r"del entrenamiento.",
          nota=r"\textbf{Qu\'e leer:} los dos criterios de $m$ que discuten FPP2 y "
               r"Tsay coinciden sin ambig\"uedad: la serie no es ruido blanco.")

    # ------------------------------------------------------------ Parte 4 --
    d = leer("p4_ols_dow")
    tabla("p4_ols_dow", ["d\\'ia", r"coeficiente (\pounds)", r"$n$ d\'ias"],
          [[esc(r["dia_semana"]), num(r["coeficiente_GBP"], 2), str(int(r["n_dias"]))]
           for _, r in d.iterrows()],
          "lrc",
          r"\textbf{4.1} OLS de la venta contra indicadoras de d\'ia de la semana, "
          r"sin intercepto. $R^2 = 0.2606$.",
          nota=r"\textbf{Qu\'e leer:} los coeficientes son la venta promedio de cada "
               r"d\'ia y reproducen el orden de los \'indices de 3.2. El modelo le "
               r"asigna el mismo valor a todos los jueves del a\~no, lo que deja la "
               r"tendencia entera dentro de los residuos.")

    d = leer("p4_ar1")
    tabla("p4_ar1", ["cantidad", "valor"],
          [[esc(r["parametro"]), num(r["valor"], 4, False)
            if abs(float(r["valor"])) < 100 else num(r["valor"], 2)]
           for _, r in d.iterrows()],
          "lr",
          r"\textbf{4.2} AR(1) ajustado por OLS a los residuos $e_t$ del inciso 4.1, "
          r"y Ljung--Box sobre sus residuos $u_t$ (un grado de libertad menos por "
          r"el $\phi$ estimado).",
          nota=r"\textbf{Qu\'e leer:} $\hat\phi=0.278$ limpia el rezago~1, pero ambas "
               r"pruebas siguen rechazando: quedan fuera de banda los rezagos 2, 4, 6 "
               r"y 9. Un AR(1) implica $\rho_6=0.28^6\approx0.0005$, o sea que predice "
               r"que no hay nada en el rezago 6, y s\'i lo hay.")

    d = leer("p4_sarima")
    enc = ["modelo", r"$\hat\phi$", r"$\hat\Phi$", r"$\hat\theta$", r"$\hat\Theta$",
           r"$\hat\mu$", r"$\hat\sigma^2$", "log-verosim.", "conv."]
    filas = []
    for _, r in d.iterrows():
        filas.append([esc(str(r["modelo"])[:34]),
                      num(r.get("phi"), 4, False), num(r.get("Phi"), 4, False),
                      num(r.get("theta"), 4, False), num(r.get("Theta"), 4, False),
                      num(r.get("mu"), 4, False), num(r["sigma2"], 5, False),
                      num(r["loglik"], 3, False),
                      r"s\'i" if r["convergio_500"] else "no"])
    tabla("p4_sarima", enc, filas, "lcccccccc",
          r"\textbf{4.3} Los dos modelos SARIMA ajustados sobre $\log$(venta diaria) "
          r"con $s=6$, por m\'axima verosimilitud exacta (filtro de Kalman).",
          nota=r"\textbf{Qu\'e leer:} en S1 el t\'ermino estacional ($\hat\Phi=0.53$) "
               r"pesa mucho m\'as que el regular ($\hat\phi=0.12$), confirmando que el "
               r"rezago 6 es el que importa. Ambos modelos convergieron ya con el "
               r"\texttt{maxiter} por omisi\'on; con \texttt{maxiter}=500 los "
               r"par\'ametros son id\'enticos a cuatro decimales.")

    d = leer("p4_diagnostico")
    tabla("p4_diagnostico",
          ["modelo", r"$m$", r"$n$ residuos", r"$Q(m)$", "gl", r"valor-$p$",
           "ruido blanco"],
          [[esc(r["modelo"]), str(int(r["m"])), str(int(r["n_residuos"])),
            num(r["Q"], 2, False), str(int(r["gl"])), num(r["valor_p"], 4, False),
            r"\textbf{s\'i}" if r["ruido_blanco_5pct"] else "no"]
           for _, r in d.iterrows()],
          "cccccccc".replace("c", "c", 1)[:7],
          r"\textbf{4.4} Ljung--Box sobre los residuos. En S2 se descartan los "
          r"primeros 7 residuos, que son artefactos del arranque de las diferencias "
          r"$(1-B)(1-B^6)$. Los grados de libertad bajan a $m-2$ en ambos modelos.",
          nota=r"\textbf{Qu\'e leer:} S2 es el \'unico que deja residuos de ruido "
               r"blanco. S1 es estacionario y supone una media fija que no existe, as\'i "
               r"que la tendencia de agosto en adelante se le queda en los residuos. "
               r"Este resultado anticipa correctamente qui\'en gana en la Parte~5.")

    # ------------------------------------------------------------ Parte 5 --
    d = leer("p5_resultados")
    orden = ["B1", "B2", "M1", "M2", "S1", "S2"]
    cols = ["(a) CV aleatoria 5 folds", "(b) origen movil en entrenamiento",
            "(c) origen movil en prueba"]
    filas = []
    for m in orden:
        fila = [r"\textbf{" + m + "}"]
        for c in cols:
            sub = d[(d["modelo"] == m) & (d["estimacion"] == c)]
            if sub.empty or pd.isna(sub.iloc[0]["MAE"]):
                fila += ["---", "---"]
            else:
                s = sub.iloc[0]
                sd = s.get("MAE_sd")
                mae_s = num(s["MAE"])
                if pd.notna(sd):
                    mae_s += r" \tiny$\pm$" + num(sd)
                fila += [mae_s, num(s["RMSE"])]
        filas.append(fila)
    tabla("p5_resultados",
          ["modelo", "MAE (a)", "RMSE (a)", "MAE (b)", "RMSE (b)",
           r"\textbf{MAE (c)}", r"\textbf{RMSE (c)}"],
          filas, "lrrrrrr",
          r"\textbf{5.2--5.3} MAE y RMSE en libras de los seis modelos, estimados de "
          r"tres formas: (a) validaci\'on cruzada aleatoria de 5 \emph{folds} sobre el "
          r"entrenamiento con 10 semillas (media $\pm$ desviaci\'on est\'andar entre "
          r"semillas), (b) origen m\'ovil semanal dentro del entrenamiento con 22 "
          r"or\'igenes de mayo a septiembre, (c) origen m\'ovil semanal en la prueba "
          r"con 10 or\'igenes, 58 d\'ias evaluados.",
          nota=r"\textbf{Qu\'e leer:} (c) es la columna honesta y S2 gana por los dos "
               r"criterios. (a) no aplica a B2, S1 y S2 porque los tres dependen del "
               r"orden del tiempo. Y los tres estimados de adentro se quedan cortos: "
               r"para M2, (b) subestima el error real en un 39\,\%.")

    d = leer("p5_trampa")
    filas = []
    for m in ["M2", "M2t"]:
        fila = [r"\textbf{" + ("M2 + MA7 (ilegal)" if m == "M2t" else "M2") + "}"]
        for c in ["(a) CV aleatoria", "(b) origen movil en entrenamiento",
                  "(c) origen movil en prueba"]:
            sub = d[(d["modelo"] == m) & (d["estimacion"] == c)]
            if sub.empty or pd.isna(sub.iloc[0]["MAE"]):
                fila += [r"\textbf{imposible}" if m == "M2t" and c.startswith("(c)")
                         else "---", "---"]
            else:
                fila += [num(sub.iloc[0]["MAE"]), num(sub.iloc[0]["RMSE"])]
        filas.append(fila)
    tabla("p5_trampa",
          ["modelo", "MAE (a)", "RMSE (a)", "MAE (b)", "RMSE (b)", "MAE (c)", "RMSE (c)"],
          filas, "lrrrrrr",
          r"\textbf{5.4} La trampa: M2 con la media m\'ovil \emph{centrada} de 7 d\'ias "
          r"agregada como predictor.",
          nota=r"\textbf{Qu\'e leer:} el predictor ilegal mejora el MAE un 9.8\,\% en "
               r"(a) y un 9.4\,\% en (b), y ninguna de las dos validaciones protesta. "
               r"La \'unica se\~nal de alarma es que (c) no se puede calcular: "
               r"$\mathrm{MA}_t$ necesita $y_{t+1}$, $y_{t+2}$, $y_{t+3}$ y adem\'as "
               r"contiene $y_t/7$, que es la propia respuesta.")

    # ------------------------------------------------------------ Parte 6 --
    d = leer("p6_rfm_resumen")
    tabla("p6_resumen", ["concepto", "valor"],
          [[esc(r["concepto"]), esc(r["valor"])] for _, r in d.iterrows()],
          "lr", r"\textbf{6.1--6.5} Resumen del an\'alisis RFM.",
          nota=r"\textbf{Qu\'e leer:} todo se calcula con transacciones anteriores al "
               r"1 de octubre; la ventana del 1 de octubre al 8 de diciembre solo se "
               r"usa para definir la variable a predecir.")

    d = leer("p6_fraccion_por_puntaje")
    piv = d.pivot(index="puntaje", columns="dimension", values="fraccion_compro")
    npiv = d.pivot(index="puntaje", columns="dimension", values="n_clientes")
    tabla("p6_fraccion",
          ["puntaje", "R", "F", "M", r"$n$ por quintil"],
          [[str(int(p)), num(piv.loc[p, "R"] * 100, 1, False) + r"\,\%",
            num(piv.loc[p, "F"] * 100, 1, False) + r"\,\%",
            num(piv.loc[p, "M"] * 100, 1, False) + r"\,\%",
            str(int(npiv.loc[p, "R"]))] for p in piv.index],
          "ccccc",
          r"\textbf{6.3} Fracci\'on de clientes de cada puntaje que compr\'o entre el "
          r"1 de octubre y el 8 de diciembre. La tasa base es 50.44\,\%.",
          nota=r"\textbf{Qu\'e leer:} F es la que m\'as separa: va de 31.7\,\% a "
               r"85.7\,\%. R satura arriba (65.7\,\% y 66.9\,\% en los puntajes 4 y 5) "
               r"pero funciona muy bien abajo: su quintil menos reciente compra con "
               r"27.3\,\%, el valor m\'as bajo de la tabla. R sirve para descartar, "
               r"F para seleccionar.")

    d = leer("p6_auc")
    tabla("p6_auc", ["clasificador", "AUC"],
          [[esc(r["clasificador"]), num(r["AUC"], 4, False)] for _, r in d.iterrows()],
          "lc",
          r"\textbf{6.3} AUC de cada dimensi\'on, con los valores crudos (R con el "
          r"signo invertido), y de la suma de los tres puntajes, como clasificadores "
          r"de compra$_i$.",
          nota=r"\textbf{Qu\'e leer:} la suma le gana a las tres por separado, que es "
               r"la justificaci\'on emp\'irica de usar tres n\'umeros y no uno. La "
               r"mejora es modesta porque las tres dimensiones est\'an correlacionadas "
               r"entre s\'i.")

    d = leer("p6_campeones").iloc[0]
    tabla("p6_campeones", ["concepto", "valor"],
          [["clientes campe\\'ones (puntaje 4 o 5 en R, F y M)", num(d["n_campeones"])],
           ["fracci\\'on de la base de clientes",
            num(d["fraccion_de_clientes"] * 100, 2, False) + r"\,\%"],
           [r"venta que generaron (\pounds)", num(d["venta_campeones_GBP"], 2)],
           [r"venta total de la ventana, todos los clientes con ID (\pounds)",
            num(d["venta_total_ventana_GBP"], 2)],
           [r"\textbf{fracci\'on de la venta de la ventana}",
            r"\textbf{" + num(d["fraccion_de_la_venta"] * 100, 2, False) + r"\,\%}"]],
          "lr",
          r"\textbf{6.4} Los campeones: clientes con puntaje 4 o 5 en las tres "
          r"dimensiones.",
          nota=r"\textbf{Qu\'e leer:} el 21.4\,\% de los clientes genera el 53.5\,\% "
               r"de la venta del trimestre. Es casi exactamente la ley de Pareto, y es "
               r"el argumento comercial entero del RFM en una l\'inea.")

    d = leer("p6_nuevos").iloc[0]
    tabla("p6_nuevos", ["concepto", "valor"],
          [["clientes nuevos en la ventana", num(d["clientes_nuevos"])],
           [r"venta que generaron (\pounds)", num(d["venta_nuevos_GBP"], 2)],
           [r"\textbf{fracci\'on de la venta de la ventana}",
            r"\textbf{" + num(d["fraccion_de_la_venta"] * 100, 2, False) + r"\,\%}"]],
          "lr",
          r"\textbf{6.5} Clientes sin ninguna factura de venta antes del 1 de octubre.",
          nota=r"\textbf{Qu\'e leer:} casi una sexta parte de la venta del trimestre "
               r"viene de gente que el 1 de octubre no exist\'ia en la base. Ning\'un "
               r"modelo cliente por cliente puede verlos, porque no tienen fila. El "
               r"pron\'ostico agregado de la Parte~5 s\'i los incluye, sin saberlo.")

    print(f"Listo: {len(os.listdir(OUT))} archivos en tablas/tex/")


if __name__ == "__main__":
    main()
