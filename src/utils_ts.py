"""
utils_ts.py
===========
Herramientas de series de tiempo implementadas a mano para la Tarea 4.

Contiene:
  - acf_manual      : r_k implementado con numpy (Parte 0 y Parte 3.3).
  - ljung_box       : estadistico Q(m) de Ljung-Box y su valor-p.
  - simular_ar1     : simulador de AR(1) (Parte 0.2).
  - ols             : minimos cuadrados con numpy.
  - SARIMA          : SARIMA(p,d,q)(P,D,Q)_s por maxima verosimilitud exacta
                      usando un filtro de Kalman (reemplazo de
                      statsmodels.tsa.statespace.sarimax.SARIMAX).
  - mae, rmse       : metricas de error.

Nota sobre SARIMA
-----------------
El entorno de computo donde se corrio esta tarea no tenia acceso a PyPI, asi
que statsmodels no estaba disponible. En lugar de omitir la Parte 4 se
implemento el modelo completo: representacion de espacio de estados de Harvey
para un ARMA(p,q), filtro de Kalman para la verosimilitud gaussiana exacta,
sigma^2 concentrada fuera de la optimizacion y L-BFGS-B para maximizar.
La diferenciacion (1-B)^d (1-B^s)^D se aplica antes de filtrar, que es lo
mismo que SARIMAX hace con simple_differencing=True; por eso el modelo S2
pierde d + D*s = 7 observaciones al inicio, exactamente los 7 residuos que
el enunciado pide descartar.

El modulo test_sarima.py verifica la implementacion recuperando parametros
de procesos simulados con valores conocidos.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy import stats


# ----------------------------------------------------------------------------
# Autocorrelacion
# ----------------------------------------------------------------------------
def acf_manual(x, kmax):
    """Autocorrelacion muestral r_k para k = 1..kmax, implementada a mano.

        r_k = sum_{t=k+1}^{n} (y_t - ybar)(y_{t-k} - ybar) / sum_{t=1}^{n} (y_t - ybar)^2

    Devuelve un arreglo de largo kmax (r_1 en la posicion 0).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    d = x - x.mean()
    denom = np.sum(d * d)
    out = np.empty(kmax, dtype=float)
    for k in range(1, kmax + 1):
        out[k - 1] = np.sum(d[k:] * d[:-k]) / denom
    return out


def banda_ruido_blanco(n, alpha=0.05):
    """Semiancho de la banda +- z_{1-alpha/2}/sqrt(n) de la ACF bajo ruido blanco."""
    return stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)


def ljung_box(r, n, m, dof_ajuste=0):
    """Prueba portmanteau de Ljung-Box.

        Q(m) = n (n + 2) sum_{k=1}^{m} r_k^2 / (n - k)

    Parametros
    ----------
    r         : arreglo con r_1, r_2, ... (al menos m elementos)
    n         : numero de observaciones usadas para estimar las r_k
    m         : numero de rezagos a juntar
    dof_ajuste: parametros estimados del modelo (K). Los grados de libertad
                bajan a m - K cuando la prueba se aplica a residuos.

    Devuelve (Q, gl, valor_p).
    """
    r = np.asarray(r, dtype=float)
    k = np.arange(1, m + 1)
    Q = n * (n + 2) * np.sum(r[:m] ** 2 / (n - k))
    gl = m - dof_ajuste
    if gl <= 0:
        return Q, gl, np.nan
    p = 1.0 - stats.chi2.cdf(Q, gl)
    return Q, gl, p


# ----------------------------------------------------------------------------
# Simulacion
# ----------------------------------------------------------------------------
def simular_ar1(n, phi, c=0.0, sigma=1.0, burn=200, rng=None):
    """Simula n pasos de un AR(1): Y_t = c + phi Y_{t-1} + eps_t.

    Se descartan los primeros `burn` pasos para que el valor inicial no
    importe (esto funciona solo si |phi| < 1; con phi = 1 la caminata
    aleatoria no olvida nunca su punto de partida).
    """
    rng = np.random.default_rng() if rng is None else rng
    eps = rng.normal(0.0, sigma, size=n + burn)
    y = np.empty(n + burn, dtype=float)
    y[0] = eps[0]
    for t in range(1, n + burn):
        y[t] = c + phi * y[t - 1] + eps[t]
    return y[burn:]


# ----------------------------------------------------------------------------
# Minimos cuadrados
# ----------------------------------------------------------------------------
def ols(X, y):
    """OLS por minimos cuadrados. Devuelve (beta, residuos, ajustados)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    return beta, y - fit, fit


# ----------------------------------------------------------------------------
# Polinomios estacionales
# ----------------------------------------------------------------------------
def expandir_poly(coefs, coefs_est, s):
    """Expande el producto de un polinomio regular y uno estacional.

    Recibe los coeficientes en la convencion "signo positivo":
        regular  : 1 + a_1 B + ... + a_p B^p
        estacional: 1 + A_1 B^s + ... + A_P B^{Ps}
    y devuelve los coeficientes [a*_1, ..., a*_L] del producto, sin el 1.

    Para la parte AR, el modelo se escribe (1 - phi B)(1 - Phi B^s) y_t = ...
    asi que quien llama pasa -phi y -Phi.
    """
    p = len(coefs)
    P = len(coefs_est)
    L = p + P * s
    if L == 0:
        return np.zeros(0)
    a = np.zeros(p + 1)
    a[0] = 1.0
    a[1:] = coefs
    A = np.zeros(P * s + 1)
    A[0] = 1.0
    for j, v in enumerate(coefs_est, start=1):
        A[j * s] = v
    prod = np.convolve(a, A)
    return prod[1:L + 1]


def diferenciar(y, d=0, D=0, s=1):
    """Aplica (1-B)^d (1-B^s)^D a la serie y."""
    z = np.asarray(y, dtype=float).copy()
    for _ in range(D):
        z = z[s:] - z[:-s]
    for _ in range(d):
        z = z[1:] - z[:-1]
    return z


# ----------------------------------------------------------------------------
# ARMA en espacio de estados + filtro de Kalman
# ----------------------------------------------------------------------------
def _matrices_arma(ar, ma):
    """Forma de Harvey para un ARMA(p,q).

        alpha_t = T alpha_{t-1} + R eps_t
        y_t     = Z alpha_t

    con m = max(p, q+1).
    """
    p, q = len(ar), len(ma)
    m = max(p, q + 1)
    T = np.zeros((m, m))
    if p > 0:
        T[:p, 0] = ar
    if m > 1:
        T[:m - 1, 1:] = np.eye(m - 1)
    R = np.zeros((m, 1))
    R[0, 0] = 1.0
    if q > 0:
        R[1:q + 1, 0] = ma
    Z = np.zeros((1, m))
    Z[0, 0] = 1.0
    return T, R, Z


def _P0_estacionaria(T, R):
    """Varianza incondicional del estado: vec(P0) = (I - T (x) T)^-1 vec(R R')."""
    m = T.shape[0]
    RR = R @ R.T
    A = np.eye(m * m) - np.kron(T, T)
    try:
        vecP = np.linalg.solve(A, RR.flatten(order="F"))
        P0 = vecP.reshape((m, m), order="F")
        # simetrizar y exigir que sea definida positiva
        P0 = 0.5 * (P0 + P0.T)
        if np.any(np.diag(P0) <= 0) or not np.all(np.isfinite(P0)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        P0 = np.eye(m) * 1e6
    return P0


def _kalman_arma(y, ar, ma):
    """Filtro de Kalman con sigma^2 concentrada.

    Devuelve (loglik, sigma2_hat, v) donde v son los errores de prediccion
    a un paso (los residuos del modelo).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    T, R, Z = _matrices_arma(ar, ma)
    m = T.shape[0]
    a = np.zeros((m, 1))
    P = _P0_estacionaria(T, R)
    RR = R @ R.T

    v = np.empty(n)
    Fstar = np.empty(n)
    for t in range(n):
        f = (Z @ P @ Z.T).item()
        if not np.isfinite(f) or f <= 1e-12:
            return -np.inf, np.nan, v
        e = y[t] - (Z @ a).item()
        v[t] = e
        Fstar[t] = f
        K = (P @ Z.T) / f
        a = a + K * e
        P = P - K @ (Z @ P)
        a = T @ a
        P = T @ P @ T.T + RR

    sigma2 = float(np.mean(v ** 2 / Fstar))
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return -np.inf, np.nan, v
    loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1.0) - 0.5 * np.sum(np.log(Fstar))
    return loglik, sigma2, v


def _estacionario(ar):
    """True si el AR es estacionario.

    phi(B) = 1 - ar_1 B - ... - ar_p B^p es estacionario cuando sus raices en B
    caen fuera del circulo unitario, lo que equivale a que las raices de
    z^p - ar_1 z^{p-1} - ... - ar_p caigan dentro.
    """
    if len(ar) == 0:
        return True
    raices = np.roots(np.r_[1.0, -np.asarray(ar, dtype=float)])
    return bool(np.all(np.abs(raices) < 0.999)) if raices.size else True


def _invertible(ma):
    """True si el MA 1 + ma_1 B + ... es invertible (mismo criterio)."""
    if len(ma) == 0:
        return True
    raices = np.roots(np.r_[1.0, np.asarray(ma, dtype=float)])
    return bool(np.all(np.abs(raices) < 0.999)) if raices.size else True


class SARIMA:
    """SARIMA(p,d,q)(P,D,Q)_s ajustado por maxima verosimilitud exacta.

    Uso
    ---
        mod = SARIMA(y, order=(1,0,0), seasonal_order=(1,0,0,6), trend='c')
        res = mod.fit(maxiter=500)
        res.params, res.resid, res.converged
        res.forecast(h)

    `y` debe ser un arreglo de valores (sin indice de fechas), tal como pide
    el enunciado: las fechas de la serie tienen huecos y pasarlas rompe el
    supuesto de espaciamiento regular.
    """

    def __init__(self, y, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1), trend=None):
        self.y = np.asarray(y, dtype=float)
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal_order
        self.trend = trend  # None o 'c'
        self.n_dif = self.d + self.D * self.s

    # -- verosimilitud -------------------------------------------------------
    def _desempacar(self, theta):
        i = 0
        phi = theta[i:i + self.p]; i += self.p
        Phi = theta[i:i + self.P]; i += self.P
        th = theta[i:i + self.q]; i += self.q
        Th = theta[i:i + self.Q]; i += self.Q
        mu = theta[i] if self.trend == "c" else 0.0
        return phi, Phi, th, Th, mu

    def _ar_ma(self, phi, Phi, th, Th):
        ar = -expandir_poly(-np.asarray(phi), -np.asarray(Phi), self.s)
        ma = expandir_poly(np.asarray(th), np.asarray(Th), self.s)
        return ar, ma

    def _neg_loglik(self, theta):
        phi, Phi, th, Th, mu = self._desempacar(theta)
        ar, ma = self._ar_ma(phi, Phi, th, Th)
        if not _estacionario(ar) or not _invertible(ma):
            return 1e10
        z = diferenciar(self.y, self.d, self.D, self.s) - mu
        ll, s2, _ = _kalman_arma(z, ar, ma)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    # -- ajuste --------------------------------------------------------------
    def fit(self, maxiter=50, start=None):
        k = self.p + self.P + self.q + self.Q + (1 if self.trend == "c" else 0)
        if k == 0:
            # Modelo sin parametros (por ejemplo una caminata aleatoria pura):
            # no hay nada que optimizar.
            vacio = optimize.OptimizeResult(
                x=np.zeros(0), fun=self._neg_loglik(np.zeros(0)), success=True,
                nit=0, message="sin parametros libres")
            return SARIMAResultado(self, vacio)
        if start is None:
            x0 = np.full(k, 0.1)
            if self.trend == "c":
                x0[-1] = float(np.mean(diferenciar(self.y, self.d, self.D, self.s)))
        else:
            x0 = np.asarray(start, dtype=float)

        bnd = [(-0.98, 0.98)] * (self.p + self.P + self.q + self.Q)
        if self.trend == "c":
            z = diferenciar(self.y, self.d, self.D, self.s)
            amp = 10 * (np.std(z) + abs(np.mean(z)) + 1.0)
            bnd.append((-amp, amp))

        res = optimize.minimize(
            self._neg_loglik, x0, method="L-BFGS-B", bounds=bnd,
            options={"maxiter": maxiter, "maxfun": maxiter * 20, "ftol": 1e-10},
        )
        return SARIMAResultado(self, res)


class SARIMAResultado:
    """Resultado de un ajuste SARIMA. Imita lo que se usa de statsmodels."""

    def __init__(self, modelo, opt):
        self.modelo = modelo
        self.opt = opt
        self.params = opt.x
        phi, Phi, th, Th, mu = modelo._desempacar(opt.x)
        self.phi, self.Phi, self.theta, self.Theta, self.mu = phi, Phi, th, Th, mu
        self.ar, self.ma = modelo._ar_ma(phi, Phi, th, Th)
        z = diferenciar(modelo.y, modelo.d, modelo.D, modelo.s) - mu
        self.z = z
        ll, s2, v = _kalman_arma(z, self.ar, self.ma)
        self.llf = ll
        self.sigma2 = s2
        self.resid = v
        self.nobs = z.size
        self.k_params = len(self.params)
        # el analogo de mle_retvals['converged'] de statsmodels
        self.mle_retvals = {
            "converged": bool(opt.success),
            "iterations": int(opt.nit),
            "fopt": float(opt.fun),
            "message": str(opt.message),
        }

    @property
    def converged(self):
        return self.mle_retvals["converged"]

    @property
    def aic(self):
        return -2 * self.llf + 2 * (self.k_params + 1)

    # -- pronostico ----------------------------------------------------------
    def forecast(self, h):
        """Pronostico a h pasos (en la escala en que se ajusto el modelo)."""
        m_ = self.modelo
        z = self.z
        ar, ma = self.ar, self.ma

        # Re-filtrar para quedarse con el estado final.
        T, R, Z = _matrices_arma(ar, ma)
        a = np.zeros((T.shape[0], 1))
        P = _P0_estacionaria(T, R)
        RR = R @ R.T
        for t in range(z.size):
            f = (Z @ P @ Z.T).item()
            e = z[t] - (Z @ a).item()
            K = (P @ Z.T) / f
            a = a + K * e
            P = P - K @ (Z @ P)
            a = T @ a
            P = T @ P @ T.T + RR

        # Propagar el estado sin observaciones nuevas.
        zf = np.empty(h)
        for i in range(h):
            zf[i] = (Z @ a).item()
            a = T @ a
        zf = zf + self.mu

        # Deshacer la diferenciacion (1-B)^d (1-B^s)^D.
        return _integrar(m_.y, zf, m_.d, m_.D, m_.s)


def _integrar(y_hist, z_fut, d, D, s):
    """Deshace (1-B)^d (1-B^s)^D sobre un pronostico de la serie diferenciada."""
    if d == 0 and D == 0:
        return z_fut.copy()
    # Se reconstruye nivel por nivel, de la diferencia mas interna hacia afuera.
    # Orden de aplicacion en `diferenciar`: primero D estacionales, luego d regulares.
    niveles = []
    cur = np.asarray(y_hist, dtype=float).copy()
    for _ in range(D):
        niveles.append(("s", cur))
        cur = cur[s:] - cur[:-s]
    for _ in range(d):
        niveles.append(("r", cur))
        cur = cur[1:] - cur[:-1]

    fut = z_fut.copy()
    for tipo, serie in reversed(niveles):
        lag = 1 if tipo == "r" else s
        ext = np.r_[serie, np.empty(fut.size)]
        base = serie.size
        for i in range(fut.size):
            ext[base + i] = ext[base + i - lag] + fut[i]
        fut = ext[base:]
    return fut


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------
def mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))
