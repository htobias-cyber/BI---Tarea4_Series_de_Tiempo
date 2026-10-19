"""
run_all.py -- corre la tarea completa de principio a fin.

Parte del archivo original data/Online Retail.xlsx y reproduce todas las
cifras, figuras y tablas del documento. Las semillas estan documentadas en
cada modulo (2026-2031 en la Parte 0, 2026 en la Parte 6).

    python src/run_all.py

Tiempo aproximado: dos minutos, casi todos leyendo el .xlsx.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parte0, parte1_2, parte3, parte4, parte5, parte6  # noqa: E402

PASOS = [
    ("Parte 0 -- procesos simulados", parte0.main),
    ("Partes 1 y 2 -- limpieza y calendario", parte1_2.main),
    ("Parte 3 -- tendencia, estacionalidad y autocorrelacion", parte3.main),
    ("Parte 4 -- del AR(1) al SARIMA", parte4.main),
    ("Parte 5 -- pronostico y fuga", parte5.main),
    ("Parte 6 -- analisis RFM", parte6.main),
]


def main():
    t0 = time.time()
    for titulo, fn in PASOS:
        print("\n" + "=" * 74)
        print(titulo)
        print("=" * 74)
        t = time.time()
        fn()
        print(f"  ({time.time() - t:.1f} s)")
    print("\n" + "=" * 74)
    print(f"Listo en {time.time() - t0:.1f} s. "
          "Revise las carpetas figuras/ y tablas/.")


if __name__ == "__main__":
    main()
