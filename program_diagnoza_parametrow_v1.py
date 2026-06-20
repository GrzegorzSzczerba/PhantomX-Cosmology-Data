import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# WCZYTYWANIE DANYCH 
# =============================================================================
z_bao_list, d_bao_list, types_bao = [], [], []
with open('desi_gaussian_bao_ALL_GCcomb_mean.txt', 'r') as f:
    for linia in f:
        if linia.startswith('#') or not linia.strip(): continue
        czesci = linia.split()
        z_bao_list.append(float(czesci[0]))
        d_bao_list.append(float(czesci[1]))
        types_bao.append(czesci[2])

# Kluczowa naprawa: konwersja na wektory!
z_bao = np.array(z_bao_list)
d_bao = np.array(d_bao_list)

cov_bao = np.loadtxt('desi_gaussian_bao_ALL_GCcomb_cov.txt')
inv_cov_bao = np.linalg.inv(cov_bao)

z_sne_list, mu_sne_list, err_sne_list = [], [], []
with open('DES-Dovekie_HD.csv', 'r', encoding='utf-8') as f:
    kolumny = []
    for linia in f:
        tokens = linia.strip().split()
        if not tokens or tokens[0].startswith('#'): continue
        if tokens[0] == 'VARNAMES:': kolumny = tokens[1:]; continue
        if kolumny and len(tokens) > 1:
            wartosci = tokens[1:] if tokens[0] in ['SN:', 'ROW:', 'OBS:', 'GAL:'] else tokens
            if len(wartosci) >= len(kolumny):
                try:
                    z_sne_list.append(float(wartosci[kolumny.index('zHD')]))
                    mu_sne_list.append(float(wartosci[kolumny.index('MU')]))
                    err_sne_list.append(float(wartosci[kolumny.index('MUERR')])**2 + float(wartosci[kolumny.index('MUERR_SYS')])**2)
                except ValueError: pass

z_sne = np.array(z_sne_list)
mu_sne = np.array(mu_sne_list)
inv_cov_sne = 1.0 / np.array(err_sne_list)

v_obs_cmb = np.array([1.74963, 301.471, 0.02237])
sigma_cmb = np.array([0.00392, 0.089, 0.00015])
rho_cmb = np.array([[1.0, 0.43, -0.66], [0.43, 1.0, -0.27], [-0.66, -0.27, 1.0]])
inv_cov_cmb = np.linalg.inv(np.outer(sigma_cmb, sigma_cmb) * rho_cmb)

omega_r = 4.183e-5  

# =============================================================================
# FIZYKA TŁA I LIKELIHOODY
# =============================================================================
def E_z_lcdm(z, Om_m0, h):
    Om_r0 = omega_r / h**2
    return np.sqrt(Om_r0 * (1.0+z)**4 + Om_m0 * (1.0+z)**3 + (1.0 - Om_m0 - Om_r0))

def E_z_phantom(z, Om_m0, h):
    Om_r0 = omega_r / h**2
    a = 1.0 / (1.0 + z)
    Om_M_a = Om_m0 * a**(-3) + Om_r0 * a**(-4)
    term_sqrt = np.sqrt(1.0 + (4.0 * (1.0 - Om_m0 - Om_r0) * a**(-2)) / (Om_M_a**2))
    return np.sqrt((Om_M_a / 2.0) * (1.0 + term_sqrt))

def D_M_integral(z, Om_m0, model, h):
    if model == 'phantom': return integrate.quad(lambda x: 1.0 / E_z_phantom(x, Om_m0, h), 0, z)[0]
    else: return integrate.quad(lambda x: 1.0 / E_z_lcdm(x, Om_m0, h), 0, z)[0]

def get_z_star(omega_m, omega_b):
    g1 = 0.0783 * (omega_b**-0.238) / (1.0 + 39.5 * omega_b**0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b**1.81)
    return 1048.0 * (1.0 + 0.00124 * omega_b**-0.738) * (1.0 + g1 * omega_m**g2)

def get_z_d(omega_m, omega_b):
    b1 = 0.313 * (omega_m**-0.419) * (1.0 + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    return 1345.0 * (omega_m**0.251) / (1.0 + 0.659 * omega_m**0.828) * (1.0 + b1 * omega_b**b2)

def rs_integral(z_end, Om_m0, omega_b, model, h):
    def integrand(a):
        if a == 0: return (1.0 / np.sqrt(3.0)) / np.sqrt(omega_r / h**2)
        coeff = 3.0 * omega_b / (4.0 * 2.469e-5)
        c_s = 1.0 / np.sqrt(3.0 * (1.0 + coeff * a))
        z = 1.0/a - 1.0
        Ez = E_z_phantom(z, Om_m0, h) if model == 'phantom' else E_z_lcdm(z, Om_m0, h)
        return c_s / (a**2 * Ez)
    a_end = 1.0 / (1.0 + z_end)
    return integrate.quad(integrand, 0, a_end)[0]

def chi2_cmb(p, model):
    Om_m0, h, omega_b = p
    dm_star = D_M_integral(get_z_star(Om_m0*h**2, omega_b), Om_m0, model, h)
    rs_star = rs_integral(get_z_star(Om_m0*h**2, omega_b), Om_m0, omega_b, model, h)
    v_model = np.array([np.sqrt(Om_m0) * dm_star, np.pi * dm_star / rs_star, omega_b])
    return np.dot(v_model - v_obs_cmb, np.dot(inv_cov_cmb, v_model - v_obs_cmb))

def chi2_bao(p, model):
    Om_m0, h, omega_b = p
    rs_d = rs_integral(get_z_d(Om_m0*h**2, omega_b), Om_m0, omega_b, model, h)
    d_model = []
    for i in range(len(z_bao)):
        z, typ = z_bao[i], types_bao[i]
        Ez = E_z_phantom(z, Om_m0, h) if model == 'phantom' else E_z_lcdm(z, Om_m0, h)
        if typ == 'DM_over_rs': d_model.append(D_M_integral(z, Om_m0, model, h) / rs_d)
        elif typ == 'DH_over_rs': d_model.append((1.0 / Ez) / rs_d)
        elif typ == 'DV_over_rs': d_model.append(((z * (D_M_integral(z, Om_m0, model, h)**2) * (1.0/Ez))**(1.0/3.0)) / rs_d)
    
    diff = d_bao - np.array(d_model)
    return np.dot(diff, np.dot(inv_cov_bao, diff))

def chi2_sne(p, model):
    Om_m0, h, _ = p
    z_max = np.max(z_sne)
    z_grid = np.linspace(0.0, z_max, 40)
    dm_grid = np.array([D_M_integral(z, Om_m0, model, h) for z in z_grid])
    dm_interp = interp1d(z_grid, dm_grid, kind='cubic', fill_value="extrapolate")
    dl_model = (1.0 + z_sne) * dm_interp(z_sne)
    delta = mu_sne - (5.0 * np.log10(np.clip(dl_model, 1e-10, None)))
    S0, S1, S2 = np.sum(inv_cov_sne), np.sum(delta * inv_cov_sne), np.sum(delta**2 * inv_cov_sne)
    return S2 - (S1**2 / S0) + np.log(S0 / (2*np.pi))

# =============================================================================
# OPTYMALIZACJA 
# =============================================================================
def get_best_fit_1D(probe, model):
    h_fid, wb_fid = 0.6736, 0.02237
    def loss(om_arr):
        p = [om_arr[0], h_fid, wb_fid]
        if probe == 'bao': return chi2_bao(p, model)
        elif probe == 'sne': return chi2_sne(p, model)
        elif probe == 'cmb': return chi2_cmb(p, model)
    res = minimize(loss, [0.3], bounds=[(0.1, 0.45)], method='L-BFGS-B')
    return res.x[0]

def get_best_fit_3D(model):
    def loss(p):
        return chi2_bao(p, model) + chi2_sne(p, model) + chi2_cmb(p, model)
    bnds = ((0.2, 0.4), (0.6, 0.75), (0.021, 0.024))
    res = minimize(loss, [0.3, 0.67, 0.022], bounds=bnds, method='L-BFGS-B')
    return res.fun

print("\n--- PHANTOMX DIAGNOSTYKA ---")
print(f"BAO Only (1D) : Om_m0 = {get_best_fit_1D('bao', 'phantom'):.4f}")
print(f"SNe Only (1D) : Om_m0 = {get_best_fit_1D('sne', 'phantom'):.4f}")
print(f"CMB Only (1D) : Om_m0 = {get_best_fit_1D('cmb', 'phantom'):.4f}")
print(f"JOINT (3D)    : Total Chi^2 = {get_best_fit_3D('phantom'):.2f}")

print("\n--- LCDM DIAGNOSTYKA ---")
print(f"BAO Only (1D) : Om_m0 = {get_best_fit_1D('bao', 'lcdm'):.4f}")
print(f"SNe Only (1D) : Om_m0 = {get_best_fit_1D('sne', 'lcdm'):.4f}")
print(f"CMB Only (1D) : Om_m0 = {get_best_fit_1D('cmb', 'lcdm'):.4f}")
print(f"JOINT (3D)    : Total Chi^2 = {get_best_fit_3D('lcdm'):.2f}")