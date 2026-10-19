"""Estilo comun para todas las figuras de la tarea."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta sobria, legible en impresion a blanco y negro por el contraste de tono.
AZUL = "#2F5C8F"
NARANJA = "#D97A2B"
VERDE = "#3F7D54"
ROJO = "#B5453A"
GRIS = "#6E7781"
MORADO = "#6B5B95"
CICLO = [AZUL, NARANJA, VERDE, ROJO, MORADO, GRIS]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "axes.prop_cycle": plt.cycler(color=CICLO),
})


def guardar(fig, ruta):
    fig.savefig(ruta)
    plt.close(fig)
    print(f"    figura -> {ruta}")
