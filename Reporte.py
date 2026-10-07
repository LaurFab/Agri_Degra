# =========================================================================

# Relación de los costes de degradación ambiental en el sector agrícola con las actividades económicas del sector en México durante el periodo 2003-2024

# =========================================================================

# Preambulo

# Librerias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, adfuller, pacf  # noqa: F401

# Base de datos
df = pd.read_excel("FIDECOAGUA/Data/Millah_small.xlsx", sheet_name= "Data", index_col= 0, parse_dates = True)

# Definición del gráfico del circulo unitario 
def plot_unit_circle(results, title="ARMAX Model - Inverse Roots"):
    """
    Plots the inverse roots of AR and MA processes on the complex plane 
    to visually inspect model stability against the unit circle (|z| = 1).
    """
    _fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Draw the unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Unit Circle')
    
    # 2. Extract and invert the roots (using state-space model attributes)
    # The statsmodels result object exposes `.arroots` and `.maroots`
    if hasattr(results, 'arroots') and results.arroots is not None:
        ar_inv_roots = 1.0 / results.arroots
        ax.scatter(ar_inv_roots.real, ar_inv_roots.imag, color='blue', marker='o', s=60, label='AR Inverse Roots')
        
    if hasattr(results, 'maroots') and results.maroots is not None:
        ma_inv_roots = 1.0 / results.maroots
        ax.scatter(ma_inv_roots.real, ma_inv_roots.imag, color='red', marker='x', s=60, label='MA Inverse Roots')

    # 3. Highlight the origin and grid
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    # Configure axes limits and aspect ratio (needs to be equal to look circular)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Labeling
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Real Part (\\Re)')
    ax.set_ylabel('Imaginary Part (\\Im)')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    
    # Add stability verification check watermark
    all_stable = True
    if hasattr(results, 'arroots') and results.arroots is not None:
        all_stable &= np.all(np.abs(1.0 / results.arroots) < 1.0)
    if hasattr(results, 'maroots') and results.maroots is not None:
        all_stable &= np.all(np.abs(1.0 / results.maroots) < 1.0)
        
    status_text = "Stable (All roots inside)" if all_stable else "Unstable (Root(s) on/outside)"
    bbox_color = "lightgreen" if all_stable else "lightcoral"
    ax.text(-1.4, -1.4, status_text, fontsize=10, bbox={"facecolor": bbox_color, "alpha": 0.5})

    plt.tight_layout()
    plt.show()

# Transformación de las variables

df["ce_ag"] = df["ce_ag"]*1000
df["ce_ag"] = df["ce_ag"]*1000

# =========================================================================

# Trasformación de los datos a través de un filtro Hodrick-Prescott 

# Valor total de la producción agrícola no orgánica 
Vp_Ciclo, Vp_Tendencia = sm.tsa.filters.hpfilter(df["vp_to"], lamb= 6.25)
df["vp_to_c"] = Vp_Ciclo
df["vp_to_t"] = Vp_Tendencia

# Valor de la producción agrícola orgánica
Vo_Ciclo, Vo_Tendencia = sm.tsa.filters.hpfilter(df["vo_to"], lamb= 6.25)
df["vo_to_c"] = Vo_Ciclo
df["vo_to_t"] = Vo_Tendencia

# Consumo aparente de productos carnicos
Ap_Ciclo, Ap_Tendencia = sm.tsa.filters.hpfilter(df["ap_to"], lamb= 6.25)
df["ap_to_c"] = Ap_Ciclo
df["ap_to_t"] = Ap_Tendencia

# Costo de degradación ambiental
Ce_Ciclo, Ce_Tendencia = sm.tsa.filters.hpfilter(df["ce_ag"], lamb= 6.25)
df["ce_ag_c"] = Ce_Ciclo
df["ce_ag_t"] = Ce_Tendencia

# Costo de agotamiento ambiental
Cn_Ciclo, Cn_Tendencia = sm.tsa.filters.hpfilter(df["cn_ag"], lamb= 6.25)
df["cn_ag_c"] = Cn_Ciclo
df["cn_ag_t"] = Cn_Tendencia

# =========================================================================

# Gráficas

# Valor total de la producción agrícola no orgánica 
plt.figure(figsize=(10, 5), layout="constrained", dpi= 600)
plt.plot(df.index, df["vp_to"], label="Sin Filtro")
plt.plot(df.index, df["vp_to_t"], label="Componente Tendencia")
plt.plot(df.index, df["vp_to_c"], label="Componente Ciclo")
plt.xlabel("Año")
plt.ylabel("Miles de pesos")
plt.title("Valor total de la producción agrícola no orgánica (Filtro Hodrick-Prescott)")
plt.legend()

# Valor total de la producción agrícola orgánica 
plt.figure(figsize=(10, 5), layout="constrained", dpi= 600)
plt.plot(df.index, df["vo_to"], label="Sin Filtro")
plt.plot(df.index, df["vo_to_t"], label="Componente Tendencia")
plt.plot(df.index, df["vo_to_c"], label="Componente Ciclo")
plt.xlabel("Año")
plt.ylabel("Miles de pesos")
plt.title("Valor total de la producción agrícola orgánica (Filtro Hodrick-Prescott)")
plt.legend()

# Consumo aparente de productos pecuarios
plt.figure(figsize=(10, 5), layout="constrained", dpi= 600)
plt.plot(df.index, df["ap_to"], label="Sin Filtro")
plt.plot(df.index, df["ap_to_t"], label="Componente Tendencia")
plt.plot(df.index, df["ap_to_c"], label="Componente Ciclo")
plt.xlabel("Año")
plt.ylabel("Toneladas")
plt.title("Consumo aparente de productos pecuarios (Filtro Hodrick-Prescott)")
plt.legend()

# Costos de la degradación ambiental Agrícola
plt.figure(figsize=(10, 5), layout="constrained", dpi= 600)
plt.plot(df.index, df["ce_ag"], label="Sin Filtro")
plt.plot(df.index, df["ce_ag_t"], label="Componente Tendencia")
plt.plot(df.index, df["ce_ag_c"], label="Componente Ciclo")
plt.xlabel("Año")
plt.ylabel("Millones de pesos")
plt.title("Costos de la degradación ambiental Agrícola (Filtro Hodrick-Prescott)")
plt.legend()

# Costos del agotamiento de recursos ambientales Agrícolas
plt.figure(figsize=(10, 5), layout="constrained", dpi= 600)
plt.plot(df.index, df["cn_ag"], label="Sin Filtro")
plt.plot(df.index, df["cn_ag_t"], label="Componente Tendencia")
plt.plot(df.index, df["cn_ag_c"], label="Componente Ciclo")
plt.xlabel("Año")
plt.ylabel("Millones de pesos")
plt.title("Costos del agotamiento de recursos ambientales Agrícolas (Filtro Hodrick-Prescott)")
plt.legend()

# =========================================================================

# Modelo de regresión lineal (MCO) - Costos de degradación ambiental
Y1 = np.log(df["ce_ag_t"])
X1 = np.log(df[["vp_to_t", "vo_to_t", "ap_to_t"]])
X1 = sm.add_constant(X1)
mod1_OLS = sm.OLS(endog = Y1, exog = X1)
res1 = mod1_OLS.fit()
print(res1.summary())

# Prueba de Breusch-Pagan

bp_test= sm.stats.diagnostic.het_breuschpagan(res1.resid, exog_het= X1)

labels = [
    "Lagrange Multiplier statistic",
    "LM p-value",
    "F-statistic",
    "F p-value",
]

print("=== Resultados de la Prueba de Breusch-Pagan ===")
for label, value in zip(labels, bp_test):
    print(f"{label:<30}: {value:.4f}")

# Construcción del modelo de regresión lineal con errores ARMA

# Funciones de autocorrelación y autocorrelación parcial
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

plot_acf(res1.resid, lags=10, ax=ax1, zero= False)
ax1.set_ylabel("Autocorrelation")
ax1.set_title("Autocorrelation Function (ACF)")

plot_pacf(res1.resid, lags=10, ax=ax2, method='ywm', zero= False)
ax2.set_xlabel("Lag")
ax2.set_ylabel("Partial Autocorrelation")
ax2.set_title("Partial Autocorrelation Function (PACF)")

# Modelo ARMAX (1,0,0)
X1 = np.log(df[["vp_to_t", "vo_to_t", "ap_to_t"]])
mod_ARMAX1 = sm.tsa.arima.ARIMA(endog = Y1, exog = X1, order=(1, 0, 0), freq= "YS-JAN")
res_ARMAX1 = mod_ARMAX1.fit()
print(res_ARMAX1.summary())

# Validación del Modelo

# Parsimonia
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

plot_acf(res_ARMAX1.resid, lags=10, ax=ax1, zero= False)
ax1.set_ylabel("Autocorrelation")
ax1.set_title("Autocorrelation Function (ACF)")

plot_pacf(res_ARMAX1.resid, lags=10, ax=ax2, method='ywm', zero= False)
ax2.set_xlabel("Lag")
ax2.set_ylabel("Partial Autocorrelation")
ax2.set_title("Partial Autocorrelation Function (PACF)")

# Estacionariedad
adf1 = adfuller(res_ARMAX1.resid, maxlag= 4, result_object=True, autolag= None, regression= "ct")
print(adf1)

# Independencia de los errores 
LB1 = sm.stats.diagnostic.acorr_ljungbox(res_ARMAX1.resid, lags=[1,2,3,4], return_df=True)
print(LB1)

# Estabilidad de sus parametros
ar_roots = np.abs(res_ARMAX1.arroots)
ma_roots = np.abs(res_ARMAX1.maroots)

print("AR Inverse Root Magnitudes:", 1 / ar_roots)
print("MA Inverse Root Magnitudes:", 1 / ma_roots)

# Gráfico 

plot_unit_circle(res_ARMAX1)

# =========================================================================

# Modelo de regresión lineal (MCO) - Costos de agotamiento de los recursos ambientales
Y2 = df["cn_ag_t"]
X2 = df[["vp_to_t", "vo_to_t", "ap_to_t"]]
X2 = sm.add_constant(X2)
mod2_OLS = sm.OLS(endog = Y2, exog = X2)
res2 = mod2_OLS.fit()
print(res2.summary())

# Prueba de Breusch-Pagan

bp_test= sm.stats.diagnostic.het_breuschpagan(res2.resid, exog_het= X2)

labels = [
    "Lagrange Multiplier statistic",
    "LM p-value",
    "F-statistic",
    "F p-value",
]

print("=== Resultados de la Prueba de Breusch-Pagan ===")
for label, value in zip(labels, bp_test):
    print(f"{label:<30}: {value:.4f}")

# Construcción del modelo de regresión lineal con errores ARMA

# Funciones de autocorrelación y autocorrelación parcial
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

plot_acf(res2.resid, lags=10, ax=ax1, zero= False)
ax1.set_ylabel("Autocorrelation")
ax1.set_title("Autocorrelation Function (ACF)")

plot_pacf(res2.resid, lags=10, ax=ax2, method='ywm', zero= False)
ax2.set_xlabel("Lag")
ax2.set_ylabel("Partial Autocorrelation")
ax2.set_title("Partial Autocorrelation Function (PACF)")

# Modelo ARMAX (1,0,0)
X2 = df[["vp_to_t", "vo_to_t", "ap_to_t"]]
mod_ARMAX2 = sm.tsa.arima.ARIMA(endog = Y2, exog = X2, order=(1, 0, 0), freq= "YS-JAN")
res_ARMAX2 = mod_ARMAX2.fit()
print(res_ARMAX2.summary())

# Validación del Modelo

# Parsimonia
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

plot_acf(res_ARMAX2.resid, lags=10, ax=ax1, zero= False)
ax1.set_ylabel("Autocorrelation")
ax1.set_title("Autocorrelation Function (ACF)")

plot_pacf(res_ARMAX2.resid, lags=10, ax=ax2, method='ywm', zero= False)
ax2.set_xlabel("Lag")
ax2.set_ylabel("Partial Autocorrelation")
ax2.set_title("Partial Autocorrelation Function (PACF)")


# Estacionariedad
adf2 = adfuller(res_ARMAX2.resid, maxlag= 8, result_object=True, autolag= None, regression= "ct")
print(adf2)

# Independencia de los errores 
LB2 = sm.stats.diagnostic.acorr_ljungbox(res_ARMAX2.resid, lags=[1,2,3,4,5,6,7,8], return_df=True)
print(LB2)

# Estabilidad de sus parametros
ar_roots = np.abs(res_ARMAX2.arroots)
ma_roots = np.abs(res_ARMAX2.maroots)

print("AR Inverse Root Magnitudes:", 1 / ar_roots)
print("MA Inverse Root Magnitudes:", 1 / ma_roots)

# Gráfico 

plot_unit_circle(res_ARMAX2)

