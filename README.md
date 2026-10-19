# Business Intelligence — Tarea 4: Introducción a Series de Tiempo

Hugo Roberto Tobías Mejía · Facultad de Ingeniería, Universidad del Istmo de Guatemala

Análisis de la serie de **venta neta diaria** del conjunto de datos
[Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) (UCI): todas las
transacciones de una tienda en línea del Reino Unido entre el 1 de diciembre de 2010 y el
9 de diciembre de 2011.

**El entregable principal es [`informe/tarea4.pdf`](informe/tarea4.pdf)** (35 páginas), con
su fuente en `informe/tarea4.tex`.

---

## Cómo correr el código

```bash
git clone <este-repositorio>
cd <este-repositorio>
pip install -r requirements.txt
python src/run_all.py
```

Corre de principio a fin a partir del archivo original `data/Online Retail.xlsx`, que viene
incluido en el repositorio, y reproduce **todas** las cifras, figuras y tablas del documento.
Tarda alrededor de 90 segundos. No hace falta ninguna preparación previa ni descargar nada.

Para regenerar también las tablas de LaTeX y el PDF:

```bash
python src/make_tables.py          # tablas/*.csv  ->  tablas/tex/*.tex
cd informe && pdflatex tarea4.tex && pdflatex tarea4.tex
```

### Semillas

Todas las semillas están fijas y documentadas en el código:

| Dónde | Semilla |
|---|---|
| Parte 0, AR(1) con φ = 0, 0.5, 0.9, 1 | 2026, 2027, 2028, 2029 |
| Parte 0, tendencia lineal | 2030 |
| Parte 0, patrón semanal | 2031 |
| Parte 5, validación cruzada (10 repeticiones) | 2026 … 2035 |
| Parte 6, desempate aleatorio de F | 2026 |

Correr `run_all.py` dos veces da exactamente los mismos números.

---

## Qué hay en cada archivo

```
.
├── informe/
│   ├── tarea4.tex              documento fuente
│   └── tarea4.pdf              ENTREGABLE PRINCIPAL
├── src/
│   ├── limpieza.py             la función limpiar(), en su propio archivo (inciso 1.5)
│   ├── utils_ts.py             r_k, Ljung-Box y SARIMA, implementados a mano
│   ├── test_sarima.py          24 pruebas que verifican utils_ts.py
│   ├── estilo.py               estilo común de las figuras
│   ├── parte0.py               Parte 0: procesos simulados
│   ├── parte1_2.py             Partes 1 y 2: limpieza y calendario
│   ├── parte3.py               Parte 3: tendencia, estacionalidad, autocorrelación
│   ├── parte4.py               Parte 4: del AR(1) al SARIMA
│   ├── parte5.py               Parte 5: pronóstico y fuga
│   ├── parte6.py               Parte 6: análisis RFM
│   ├── make_tables.py          convierte tablas/*.csv en tablas de LaTeX
│   └── run_all.py              corre todo de principio a fin
├── data/Online Retail.xlsx     el archivo original de UCI, sin modificar
├── datos_derivados/            series intermedias que genera el código
├── figuras/                    las 12 figuras del documento (PNG)
├── tablas/                     las 26 tablas en CSV, y tablas/tex/ en LaTeX
├── tarea4.ipynb                notebook que corre todo y muestra los resultados
└── requirements.txt
```

### La función `limpiar`

Vive en `src/limpieza.py`, su propio archivo, como pide el enunciado. Se reusa tal cual en
la Parte 6 y queda lista para el Proyecto 1.

```python
from limpieza import cargar, limpiar, serie_venta_neta_diaria

df = cargar("data/Online Retail.xlsx")
limpio, bitacora = limpiar(df, devolver_bitacora=True)
serie = serie_venta_neta_diaria(limpio)
```

---

## Nota importante sobre `statsmodels`

**El entorno de cómputo donde se ejecutó esta tarea no tenía acceso a PyPI**, así que
`statsmodels` no se pudo instalar. En lugar de omitir la Parte 4, el modelo SARIMA se
implementó desde cero en `src/utils_ts.py`:

- representación de espacio de estados de Harvey para un ARMA(p, q);
- filtro de Kalman para la verosimilitud gaussiana exacta;
- σ² concentrada fuera de la optimización;
- L-BFGS-B para maximizar, con restricciones de estacionariedad e invertibilidad;
- la diferenciación (1−B)^d (1−Bˢ)^D se aplica antes de filtrar, que es lo mismo que
  `SARIMAX` hace con `simple_differencing=True`.

La interfaz imita a la de `statsmodels`:

```python
from utils_ts import SARIMA
mod = SARIMA(np.log(y), order=(0,1,1), seasonal_order=(0,1,1,6))
res = mod.fit(maxiter=500)
res.params, res.resid, res.llf, res.mle_retvals["converged"], res.forecast(7)
```

La prueba de Ljung–Box y la función `r_k` también están implementadas a mano con `numpy`,
como pide el enunciado.

**La implementación está verificada.** `src/test_sarima.py` corre 24 pruebas que simulan
procesos con parámetros conocidos y comprueban que el ajuste los recupera: AR(1), MA(1),
SARIMA(1,0,0)(1,0,0)₆, SARIMA(0,1,1)(0,1,1)₆, el conteo de observaciones que se pierden al
diferenciar, y los dos límites del pronóstico (converge a la media si es estacionario, se
queda en el último valor si es una caminata aleatoria).

```bash
python src/test_sarima.py      # Las 24 pruebas pasaron.
```

Si en otro entorno se quiere contrastar contra `statsmodels`, basta con instalarlo y
comparar; los parámetros deberían coincidir hasta la tolerancia del optimizador.

---

## Resumen de los resultados

| | |
|---|---|
| Filas originales / después de limpiar | 541,909 → 533,979 (se eliminan 7,930) |
| Días calendario / con transacciones | 374 / 305 |
| Día que la tienda no abre | **sábado** (53 de los 69 días sin ventas) |
| Período estacional usado | **s = 6**, porque la semana comercial tiene seis días |
| Pico de la ACF | rezago 6 (r₆ = 0.42), que son 7 días calendario |
| Modelo con residuos de ruido blanco | **solo S2**, SARIMA(0,1,1)(0,1,1)₆ |
| Ganador en prueba | **S2**, MAE £12,817 y RMSE £16,217 |
| Peor modelo en prueba | B1 (media histórica), MAE £23,446 |
| La trampa (media móvil centrada) | "mejora" el MAE 9.8 % en (a) y 9.4 % en (b) |
| Campeones RFM | 21.4 % de los clientes generan 53.5 % de la venta |
| Clientes nuevos | 15.7 % de la venta del trimestre, invisibles al RFM |

Las decisiones de limpieza y el razonamiento detrás de cada número están en el PDF; el
enunciado advierte que no hay una sola limpieza correcta, así que estas cifras no tienen
por qué coincidir exactamente con las de otros compañeros.

---

## Requisitos

Python 3.11 o superior, y los paquetes de `requirements.txt`: `numpy`, `pandas`, `scipy`,
`matplotlib`, `openpyxl` y `scikit-learn`. **No hace falta `statsmodels`.**

Para recompilar el PDF, una distribución de LaTeX con `booktabs`, `siunitx`, `enumitem`,
`xcolor` e `hyperref`.

## Fuentes

- D. Chen. *Online Retail* [conjunto de datos]. UCI Machine Learning Repository, 2015.
- D. Chen, S. L. Sain y K. Guo. "Data mining for the online retail industry: A case study
  of RFM model-based customer segmentation using data mining". *Journal of Database
  Marketing & Customer Strategy Management*, 19(3):197–208, 2012.
- R. J. Hyndman y G. Athanasopoulos. *Forecasting: Principles and Practice*, 2ª ed.
  OTexts, 2018. <https://otexts.com/fpp2/>
- R. S. Tsay. *Analysis of Financial Time Series*, 2ª ed. Wiley, 2005.
- A. C. Harvey. *Forecasting, Structural Time Series Models and the Kalman Filter*.
  Cambridge University Press, 1989.
