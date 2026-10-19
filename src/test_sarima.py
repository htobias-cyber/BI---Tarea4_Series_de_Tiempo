"""
test_sarima.py
==============
Verificacion de las herramientas de utils_ts.py.

Como statsmodels no estaba disponible en el entorno donde se corrio esta
tarea, el SARIMA se implemento a mano. Estas pruebas son la evidencia de que
la implementacion esta bien: se simulan procesos con parametros conocidos y
se comprueba que el ajuste los recupera.

Correr con:  python src/test_sarima.py
"""
import numpy as np

from utils_ts import (SARIMA, acf_manual, diferenciar, expandir_poly,
                      ljung_box, simular_ar1)

OK, FALLOS = 0, []


def check(nombre, cond, detalle=""):
    global OK
    if cond:
        OK += 1
        print(f"  [OK]    {nombre} {detalle}")
    else:
        FALLOS.append(nombre)
        print(f"  [FALLA] {nombre} {detalle}")


print("1. acf_manual contra np.correlate")
rng = np.random.default_rng(7)
x = rng.normal(size=500)
r = acf_manual(x, 10)
d = x - x.mean()
ref = np.array([np.sum(d[k:] * d[:-k]) / np.sum(d * d) for k in range(1, 11)])
check("r_k coincide con la definicion", np.allclose(r, ref))
check("r_k de ruido blanco es chico", np.max(np.abs(r)) < 3 / np.sqrt(500),
      f"max|r_k| = {np.max(np.abs(r)):.3f}")

print("\n2. acf_manual sobre un AR(1) conocido: r_k ~ phi^k")
y = simular_ar1(20000, phi=0.7, rng=np.random.default_rng(1))
r = acf_manual(y, 5)
teo = 0.7 ** np.arange(1, 6)
check("r_k ~ phi^k", np.max(np.abs(r - teo)) < 0.03,
      f"max dif = {np.max(np.abs(r - teo)):.4f}")

print("\n3. ljung_box contra el calculo directo")
n, m = 300, 10
rr = acf_manual(rng.normal(size=n), m)
Q, gl, p = ljung_box(rr, n, m)
k = np.arange(1, m + 1)
Qref = n * (n + 2) * np.sum(rr ** 2 / (n - k))
check("Q(m) coincide con la formula", np.isclose(Q, Qref))
check("ruido blanco no se rechaza", p > 0.01, f"p = {p:.3f}")

print("\n4. expandir_poly: (1-phi B)(1-Phi B^6)")
phi, Phi = 0.5, 0.3
coef = -expandir_poly(np.array([-phi]), np.array([-Phi]), 6)
esp = np.zeros(7)
esp[0] = phi
esp[5] = Phi
esp[6] = -phi * Phi
check("coeficientes del producto AR", np.allclose(coef, esp), f"{np.round(coef, 3)}")

print("\n5. diferenciar: (1-B)(1-B^6) sobre una serie conocida")
t = np.arange(40, dtype=float)
serie = 0.5 * t + np.sin(2 * np.pi * t / 6)
z = diferenciar(serie, d=1, D=1, s=6)
check("pierde d + D*s = 7 observaciones", z.size == serie.size - 7, f"n={z.size}")
check("elimina tendencia y estacionalidad", np.max(np.abs(z)) < 1e-10,
      f"max|z| = {np.max(np.abs(z)):.2e}")

print("\n6. SARIMA recupera un AR(1) simulado (phi = 0.6, mu = 2)")
y = 2.0 + simular_ar1(4000, phi=0.6, rng=np.random.default_rng(11))
res = SARIMA(y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 6), trend="c").fit(maxiter=500)
check("convergio", res.converged)
check("phi recuperado", abs(res.phi[0] - 0.6) < 0.05, f"phi_hat = {res.phi[0]:.4f}")
check("mu recuperado", abs(res.mu - 2.0) < 0.15, f"mu_hat = {res.mu:.4f}")
check("sigma^2 recuperado", abs(res.sigma2 - 1.0) < 0.1, f"s2_hat = {res.sigma2:.4f}")

print("\n7. SARIMA recupera un MA(1) simulado (theta = 0.6)")
rng = np.random.default_rng(13)
e = rng.normal(size=5001)
y = e[1:] + 0.6 * e[:-1]
res = SARIMA(y, order=(0, 0, 1), seasonal_order=(0, 0, 0, 6)).fit(maxiter=500)
check("convergio", res.converged)
check("theta recuperado", abs(res.theta[0] - 0.6) < 0.05, f"theta_hat = {res.theta[0]:.4f}")

print("\n8. SARIMA(1,0,0)(1,0,0)_6 recupera phi = 0.5, Phi = 0.4")
rng = np.random.default_rng(17)
n, burn = 6000, 300
e = rng.normal(size=n + burn)
y = np.zeros(n + burn)
ph, Ph = 0.5, 0.4
for t in range(7, n + burn):
    y[t] = ph * y[t - 1] + Ph * y[t - 6] - ph * Ph * y[t - 7] + e[t]
y = y[burn:]
res = SARIMA(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 6), trend="c").fit(maxiter=500)
check("convergio", res.converged)
check("phi recuperado", abs(res.phi[0] - ph) < 0.05, f"phi_hat = {res.phi[0]:.4f}")
check("Phi recuperado", abs(res.Phi[0] - Ph) < 0.05, f"Phi_hat = {res.Phi[0]:.4f}")

print("\n9. SARIMA(0,1,1)(0,1,1)_6 recupera theta = -0.5, Theta = -0.4")
rng = np.random.default_rng(19)
n = 6000
e = rng.normal(size=n + 10)
th, Th = -0.5, -0.4
z = np.zeros(n)
for t in range(n):
    i = t + 10
    z[t] = e[i] + th * e[i - 1] + Th * e[i - 6] + th * Th * e[i - 7]
# integrar para construir la serie en niveles
w = np.zeros(n + 7)
for t in range(7, n + 7):
    w[t] = w[t - 1] + w[t - 6] - w[t - 7] + z[t - 7]
res = SARIMA(w, order=(0, 1, 1), seasonal_order=(0, 1, 1, 6)).fit(maxiter=500)
check("convergio", res.converged)
check("se pierden 7 observaciones", res.nobs == w.size - 7, f"nobs = {res.nobs}")
check("theta recuperado", abs(res.theta[0] - th) < 0.06, f"theta_hat = {res.theta[0]:.4f}")
check("Theta recuperado", abs(res.Theta[0] - Th) < 0.06, f"Theta_hat = {res.Theta[0]:.4f}")

print("\n10. Los residuos del modelo bien especificado son ruido blanco")
r = acf_manual(res.resid, 12)
Q, gl, p = ljung_box(r, res.resid.size, 12, dof_ajuste=2)
check("Ljung-Box no rechaza los residuos", p > 0.01, f"Q(12) = {Q:.2f}, p = {p:.3f}")

print("\n11. El pronostico de un AR(1) converge a la media")
y = 5.0 + simular_ar1(2000, phi=0.6, rng=np.random.default_rng(23))
res = SARIMA(y, order=(1, 0, 0), seasonal_order=(0, 0, 0, 6), trend="c").fit(maxiter=500)
f = res.forecast(60)
check("el pronostico converge a mu", abs(f[-1] - res.mu) < 0.02, f"f[60] = {f[-1]:.4f}, mu = {res.mu:.4f}")

print("\n12. El pronostico de una caminata aleatoria es el ultimo valor")
rng = np.random.default_rng(29)
y = np.cumsum(rng.normal(size=500))
res = SARIMA(y, order=(0, 1, 0), seasonal_order=(0, 0, 0, 6)).fit(maxiter=200)
f = res.forecast(10)
check("pronostico plano = ultimo valor", np.allclose(f, y[-1], atol=0.05),
      f"f = {np.round(f[:3], 3)}, y_n = {y[-1]:.3f}")

print("\n" + "=" * 60)
if FALLOS:
    print(f"FALLARON {len(FALLOS)} pruebas: {FALLOS}")
    raise SystemExit(1)
print(f"Las {OK} pruebas pasaron.")
