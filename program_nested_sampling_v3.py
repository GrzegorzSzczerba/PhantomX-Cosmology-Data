import os
import urllib.request
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

# Ignorujemy ostrzeżenia o limitach całkowania (bezpieczne dla wczesnego Wszechświata)
warnings.filterwarnings("ignore")

try:
    import dynesty
    from dynesty.utils import resample_equal
except ImportError:
    print("Błąd: Biblioteka 'dynesty' nie jest zainstalowana. Uruchom: pip install dynesty")
    exit()

np.random.seed(42)

# =============================================================================
# AUTOMATYCZNE POBIERANIE DANYCH 
# =============================================================================

def pobierz_dane():
    print("Sprawdzanie i pobieranie danych...")
    base_url = "https://raw.githubusercontent.com/GrzegorzSzczerba/PhantomX-Cosmology-Data/main/"
    pliki_do_pobrania = {
        "desi_gaussian_bao_ALL_GCcomb_mean.txt": base_url + "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "desi_gaussian_bao_ALL_GCcomb_cov.txt": base_url + "desi_gaussian_bao_ALL_GCcomb_cov.txt",
        "DES-Dovekie_HD.csv": base_url + "DES-Dovekie_HD.csv"
    }
    
    for nazwa, url in pliki_do_pobrania.items():
        if not os.path.exists(nazwa):
            urllib.request.urlretrieve(url, nazwa)

# =============================================================================
# DANE CMB - FULL DISTANCE PRIOR (PLANCK 2018 TT,TE,EE+lowE)
# =============================================================================
v_obs_cmb = np.array([1.74963, 301.471, 0.02237])
sigma_cmb = np.array([0.00392, 0.089, 0.00015])
rho_cmb = np.array([
    [1.0, 0.43, -0.66],
    [0.43, 1.0, -0.27],
    [-0.66, -0.27, 1.0]
])
cov_cmb = np.outer(sigma_cmb, sigma_cmb) * rho_cmb
inv_cov_cmb = np.linalg.inv(cov_cmb)

# Stała gęstości promieniowania (fotony + neutrina) dla T_cmb = 2.7255 K
omega_r = 4.183e-5  

# =============================================================================
# DEFINICJE DYNAMIKI MODELI Z UWZGLĘDNIENIEM PROMIENIOWANIA
# =============================================================================

def E_z_lcdm(z, Om_m0, h):
    Om_r0 = omega_r / h**2
    Om_L = 1.0 - Om_m0 - Om_r0
    return np.sqrt(Om_r0 * (1.0+z)**4 + Om_m0 * (1.0+z)**3 + Om_L)

def E_z_phantom(z, Om_m0, h):
    Om_r0 = omega_r / h**2
    Om_L = 1.0 - Om_m0 - Om_r0
    a = 1.0 / (1.0 + z)
    # Gęstość materii i promieniowania w danym a
    Om_M_a = Om_m0 * a**(-3) + Om_r0 * a**(-4)
    # Rozwiązanie kwadratowe z ansatzu rho_DE ~ a_dot^-2 dla tła z promieniowaniem
    term_sqrt = np.sqrt(1.0 + (4.0 * Om_L * a**(-2)) / (Om_M_a**2))
    return np.sqrt((Om_M_a / 2.0) * (1.0 + term_sqrt))

def D_M_integral(z, Om_m0, model, h):
    if model == 'phantom':
        return integrate.quad(lambda x: 1.0 / E_z_phantom(x, Om_m0, h), 0, z)[0]
    else:
        return integrate.quad(lambda x: 1.0 / E_z_lcdm(x, Om_m0, h), 0, z)[0]

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
        if a == 0:
            Om_r0 = omega_r / h**2
            return (1.0 / np.sqrt(3.0)) / np.sqrt(Om_r0)
        coeff = 3.0 * omega_b / (4.0 * 2.469e-5)
        c_s = 1.0 / np.sqrt(3.0 * (1.0 + coeff * a))
        z = 1.0/a - 1.0
        Ez = E_z_phantom(z, Om_m0, h) if model == 'phantom' else E_z_lcdm(z, Om_m0, h)
        return c_s / (a**2 * Ez)
    
    a_end = 1.0 / (1.0 + z_end)
    val, _ = integrate.quad(integrand, 0, a_end)
    return val

# =============================================================================
# LIKELIHOOD DLA SOND
# =============================================================================

def get_chi2_cmb(Om_m0, h, omega_b, model):
    omega_m = Om_m0 * h**2
    z_star = get_z_star(omega_m, omega_b)
    
    dm_star = D_M_integral(z_star, Om_m0, model, h)
    rs_star = rs_integral(z_star, Om_m0, omega_b, model, h)
    
    R_model = np.sqrt(Om_m0) * dm_star  # POPRAWIONE R
    lA_model = np.pi * dm_star / rs_star
    
    v_model = np.array([R_model, lA_model, omega_b])
    delta = v_model - v_obs_cmb
    return np.dot(delta.T, np.dot(inv_cov_cmb, delta))

def get_chi2_bao(Om_m0, h, omega_b, z_bao, d_bao, inv_cov_bao, types_bao, model):
    omega_m = Om_m0 * h**2
    z_d = get_z_d(omega_m, omega_b)
    rs_d = rs_integral(z_d, Om_m0, omega_b, model, h)
    
    d_model = np.zeros(len(z_bao))
    for i in range(len(z_bao)):
        z, typ = z_bao[i], types_bao[i]
        if typ == 'DM_over_rs':
            d_model[i] = D_M_integral(z, Om_m0, model, h) / rs_d
        elif typ == 'DH_over_rs':
            Ez = E_z_phantom(z, Om_m0, h) if model == 'phantom' else E_z_lcdm(z, Om_m0, h)
            d_model[i] = (1.0 / Ez) / rs_d
        elif typ == 'DV_over_rs':
            dm = D_M_integral(z, Om_m0, model, h)
            Ez = E_z_phantom(z, Om_m0, h) if model == 'phantom' else E_z_lcdm(z, Om_m0, h)
            dh = 1.0 / Ez
            d_model[i] = ((z * (dm**2) * dh)**(1.0/3.0)) / rs_d
            
    diff = d_bao - d_model
    return np.dot(diff, np.dot(inv_cov_bao, diff))

def get_chi2_sne(Om_m0, h, z_sne, mu_obs, inv_cov_sne, model):
    z_max = np.max(z_sne)
    z_grid = np.linspace(0.0, z_max, 50)
    dm_grid = np.array([D_M_integral(z, Om_m0, model, h) for z in z_grid])
    dm_interp = interp1d(z_grid, dm_grid, kind='cubic', fill_value="extrapolate")
    
    dl_model = (1.0 + z_sne) * dm_interp(z_sne)
    dl_model[dl_model <= 0] = 1e-10 
    mu_model = 5.0 * np.log10(dl_model)
    delta = mu_obs - mu_model
    
    if inv_cov_sne.ndim == 1:
        S0, S1, S2 = np.sum(inv_cov_sne), np.sum(delta * inv_cov_sne), np.sum(delta**2 * inv_cov_sne)
    else:
        S0, S1 = np.sum(inv_cov_sne), np.sum(np.dot(inv_cov_sne, delta))
        S2 = np.dot(delta.T, np.dot(inv_cov_sne, delta))
        
    return S2 - (S1**2 / S0) + np.log(S0 / (2*np.pi))

# =============================================================================
# WCZYTYWANIE DANYCH 
# =============================================================================

def wczytaj_wszystkie_dane():
    # BAO
    z_bao, d_bao, types_bao = [], [], []
    with open('desi_gaussian_bao_ALL_GCcomb_mean.txt', 'r') as f:
        for linia in f:
            if linia.startswith('#') or not linia.strip(): continue
            czesci = linia.split()
            z_bao.append(float(czesci[0])); d_bao.append(float(czesci[1])); types_bao.append(czesci[2])
    cov_bao = np.loadtxt('desi_gaussian_bao_ALL_GCcomb_cov.txt')
    
    # SNe
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
                    
    return (np.array(z_bao), np.array(d_bao), np.linalg.inv(cov_bao), types_bao), \
           (np.array(z_sne_list), np.array(mu_sne_list), 1.0 / np.array(err_sne_list))

# =============================================================================
# LIKELIHOOD I PRIOR DLA DYNESTY (3 PARAMETRY)
# =============================================================================

def prior_transform(u):
    Om_m0 = 0.20 + 0.20 * u[0]      # 0.20 do 0.40
    h = 0.60 + 0.15 * u[1]          # 0.60 do 0.75
    omega_b = 0.021 + 0.003 * u[2]  # 0.021 do 0.024
    return np.array([Om_m0, h, omega_b])

def make_loglike_joint(args_bao, args_sne, model):
    def loglike(theta):
        Om_m0, h, omega_b = theta[0], theta[1], theta[2]
        try:
            l_bao = get_chi2_bao(Om_m0, h, omega_b, *args_bao, model=model)
            l_sne = get_chi2_sne(Om_m0, h, *args_sne, model=model)
            l_cmb = get_chi2_cmb(Om_m0, h, omega_b, model=model)
            total_chi2 = l_bao + l_sne + l_cmb
            if np.isnan(total_chi2) or total_chi2 > 1e6:
                return -1e30
            return -0.5 * total_chi2
        except Exception:
            return -1e30
    return loglike

# =============================================================================
# GŁÓWNA PROCEDURA
# =============================================================================

if __name__ == "__main__":
    pobierz_dane()
    args_bao, args_sne = wczytaj_wszystkie_dane()

    print("\n" + "="*60)
    print("URUCHAMIAM NESTED SAMPLING: MODEL PHANTOMX (3-PARAM)")
    print("="*60)
    loglike_phantom = make_loglike_joint(args_bao, args_sne, model='phantom')
    sampler_phantom = dynesty.NestedSampler(loglike_phantom, prior_transform, ndim=3, nlive=250)
    sampler_phantom.run_nested(dlogz=0.5)
    res_phantom = sampler_phantom.results

    print("\n" + "="*60)
    print("URUCHAMIAM NESTED SAMPLING: MODEL LCDM (3-PARAM)")
    print("="*60)
    loglike_lcdm = make_loglike_joint(args_bao, args_sne, model='lcdm')
    sampler_lcdm = dynesty.NestedSampler(loglike_lcdm, prior_transform, ndim=3, nlive=250)
    sampler_lcdm.run_nested(dlogz=0.5)
    res_lcdm = sampler_lcdm.results

    logZ_phantom = res_phantom.logz[-1]
    logZ_lcdm = res_lcdm.logz[-1]
    delta_logZ = logZ_phantom - logZ_lcdm

    print("\n" + "#"*60)
    print("WYNIKI PORÓWNANIA BAYESOWSKIEGO (FULL CMB PRIOR + BAO + SNe)")
    print("#"*60)
    print(f"Ewidencja (ln Z) PhantomX : {logZ_phantom:.2f}")
    print(f"Ewidencja (ln Z) LCDM     : {logZ_lcdm:.2f}")
    print(f"Czynnik Bayesa (\u0394 ln Z) : {delta_logZ:.2f}")
    print("#"*60)

    # Rysowanie wykresu (wersja ENG)
    weights_p = np.exp(res_phantom.logwt - logZ_phantom)
    samples_p_eq = resample_equal(res_phantom.samples, weights_p)
    
    weights_l = np.exp(res_lcdm.logwt - logZ_lcdm)
    samples_l_eq = resample_equal(res_lcdm.samples, weights_l)

    plt.figure(figsize=(10, 6))
    
    plt.hist(samples_l_eq[:, 0], bins=30, density=True, histtype='stepfilled', alpha=0.5, color='#1f77b4', 
             label=f'$\\Lambda$CDM ($\\Omega_{{m0}}$={np.percentile(samples_l_eq[:,0], 50):.3f})\n$\\ln \\mathcal{{Z}} = {logZ_lcdm:.1f}$')
    plt.hist(samples_l_eq[:, 0], bins=30, density=True, histtype='step', linewidth=2, color='#0b5394')

    plt.hist(samples_p_eq[:, 0], bins=30, density=True, histtype='stepfilled', alpha=0.5, color='#d62728', 
             label=f'PhantomX ($\\Omega_{{m0}}$={np.percentile(samples_p_eq[:,0], 50):.3f})\n$\\ln \\mathcal{{Z}} = {logZ_phantom:.1f}$')
    plt.hist(samples_p_eq[:, 0], bins=30, density=True, histtype='step', linewidth=2, color='#990000')

    plt.axvline(np.percentile(samples_l_eq[:,0], 50), color='#1f77b4', linestyle='dashed', linewidth=1.5)
    plt.axvline(np.percentile(samples_p_eq[:,0], 50), color='#d62728', linestyle='dashed', linewidth=1.5)

    plt.title(f'Model Comparison: $\\Lambda$CDM vs PhantomX (BAO+SNe+Full CMB)\nBayes Factor ($\\Delta \\ln \\mathcal{{Z}}$): {delta_logZ:.2f}', fontsize=14, fontweight='bold')
    plt.xlabel(r'$\Omega_{m0}$', fontsize=14)
    plt.ylabel('Posterior Density', fontsize=12)
    plt.xlim(0.25, 0.35)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig('Model_Comparison_Dynesty_ENG.png', dpi=300)
    
    print("\n-> Wygenerowano wykres w wersji angielskiej: Model_Comparison_Dynesty_ENG.png")