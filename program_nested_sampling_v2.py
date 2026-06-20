import os
import urllib.request
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

try:
    import dynesty
    from dynesty import plotting as dyplot
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
            print(f" -> Pobieranie {nazwa}...")
            urllib.request.urlretrieve(url, nazwa)

    cmb_file = "planck_2018_cmb_prior.txt"
    if not os.path.exists(cmb_file):
        with open(cmb_file, 'w') as f:
            f.write("# Planck 2018 TT,TE,EE+lowE+lensing Distance Priors\n")
            f.write("1089.92  1.74963  0.00392\n")

# =============================================================================
# DEFINICJE DYNAMIKI MODELI (PHANTOMX vs LCDM)
# =============================================================================

def E_z_phantom(z, Om_m0):
    a = 1.0 / (1.0 + z)
    K = 4.0 * (1.0 - Om_m0) / (Om_m0**2)
    term_sqrt = np.sqrt(1.0 + K * (a**4))
    return np.sqrt((Om_m0 / (2 * a**3)) * (1 + term_sqrt))

def E_z_lcdm(z, Om_m0):
    return np.sqrt(Om_m0 * (1.0 + z)**3 + (1.0 - Om_m0))

def D_M_integral(z, Om_m0, model='lcdm'):
    if model == 'phantom':
        return integrate.quad(lambda x: 1.0 / E_z_phantom(x, Om_m0), 0, z)[0]
    else:
        return integrate.quad(lambda x: 1.0 / E_z_lcdm(x, Om_m0), 0, z)[0]

def D_H_val(z, Om_m0, model='lcdm'):
    if model == 'phantom':
        return 1.0 / E_z_phantom(z, Om_m0)
    else:
        return 1.0 / E_z_lcdm(z, Om_m0)

# =============================================================================
# WYLICZANIE CHI^2 DLA POSZCZEGÓLNYCH SOND
# =============================================================================

def get_chi2_bao(Om_m0, z_bao, d_bao, inv_cov_bao, types_bao, model):
    rd_Mpc, H0, c_km_s = 147.05, 67.36, 299792.458  
    skala_rd = c_km_s / (H0 * rd_Mpc) 
    
    d_model = np.zeros(len(z_bao))
    for i in range(len(z_bao)):
        z, typ = z_bao[i], types_bao[i]
        if typ == 'DM_over_rs':
            d_model[i] = skala_rd * D_M_integral(z, Om_m0, model)
        elif typ == 'DH_over_rs':
            d_model[i] = skala_rd * D_H_val(z, Om_m0, model)
        elif typ == 'DV_over_rs':
            dm, dh = D_M_integral(z, Om_m0, model), D_H_val(z, Om_m0, model)
            d_model[i] = skala_rd * (z * (dm**2) * dh)**(1.0/3.0)
            
    diff = d_bao - d_model
    return np.dot(diff, np.dot(inv_cov_bao, diff))

def get_chi2_sne(Om_m0, z_sne, mu_obs, inv_cov_sne, model):
    if Om_m0 <= 0 or Om_m0 >= 1: return np.inf
    z_max = np.max(z_sne)
    z_grid = np.linspace(0.0, z_max, 50)
    dm_grid = np.array([D_M_integral(z, Om_m0, model) for z in z_grid])
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

def get_chi2_cmb(Om_m0, z_star, R_obs, sigma_R, model):
    if Om_m0 <= 0 or Om_m0 >= 1: return np.inf
    dm_star = D_M_integral(z_star, Om_m0, model)
    R_model = np.sqrt(Om_m0) * dm_star
    return ((R_model - R_obs) / sigma_R)**2

# =============================================================================
# LIKELIHOOD I PRIOR DLA DYNESTY
# =============================================================================

def prior_transform(u):
    """
    Transformacja Priora dla dynesty. 
    Mapuje zmienną u z przedziału [0, 1] na fizyczny przedział Om_m0 [0.05, 0.95].
    """
    return np.array([0.05 + 0.90 * u[0]])

def make_loglike_joint(args_bao, args_sne, args_cmb, model):
    def loglike(theta):
        Om_m0 = theta[0]
        if not (0.05 < Om_m0 < 0.95): return -1e30
        l_bao = get_chi2_bao(Om_m0, *args_bao, model=model)
        l_sne = get_chi2_sne(Om_m0, *args_sne, model=model)
        l_cmb = get_chi2_cmb(Om_m0, *args_cmb, model=model)
        return -0.5 * (l_bao + l_sne + l_cmb)
    return loglike

# =============================================================================
# WCZYTYWANIE DANYCH 
# =============================================================================

def wczytaj_wszystkie_dane():
    # 1. BAO
    z_bao, d_bao, types_bao = [], [], []
    with open('desi_gaussian_bao_ALL_GCcomb_mean.txt', 'r') as f:
        for linia in f:
            if linia.startswith('#') or not linia.strip(): continue
            czesci = linia.split()
            z_bao.append(float(czesci[0])); d_bao.append(float(czesci[1])); types_bao.append(czesci[2])
    cov_bao = np.loadtxt('desi_gaussian_bao_ALL_GCcomb_cov.txt')
    
    # 2. SNe
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
                    
    # 3. CMB
    with open('planck_2018_cmb_prior.txt', 'r') as f:
        linie = f.readlines()
        z_star, R_obs, sigma_R = map(float, linie[2].split())

    return (np.array(z_bao), np.array(d_bao), np.linalg.inv(cov_bao), types_bao), \
           (np.array(z_sne_list), np.array(mu_sne_list), 1.0 / np.array(err_sne_list)), \
           (z_star, R_obs, sigma_R)

# =============================================================================
# GŁÓWNA PROCEDURA WYKONAWCZA (NESTED SAMPLING)
# =============================================================================

if __name__ == "__main__":
    pobierz_dane()
    args_bao, args_sne, args_cmb = wczytaj_wszystkie_dane()

    print("\n" + "="*60)
    print("URUCHAMIAM NESTED SAMPLING: MODEL PHANTOMX (6W-ANRA)")
    print("="*60)
    loglike_phantom = make_loglike_joint(args_bao, args_sne, args_cmb, model='phantom')
    sampler_phantom = dynesty.NestedSampler(loglike_phantom, prior_transform, ndim=1, nlive=500, bound='single')
    sampler_phantom.run_nested(dlogz=0.01)
    res_phantom = sampler_phantom.results

    print("\n" + "="*60)
    print("URUCHAMIAM NESTED SAMPLING: MODEL STANDARDOWY LCDM")
    print("="*60)
    loglike_lcdm = make_loglike_joint(args_bao, args_sne, args_cmb, model='lcdm')
    sampler_lcdm = dynesty.NestedSampler(loglike_lcdm, prior_transform, ndim=1, nlive=500, bound='single')
    sampler_lcdm.run_nested(dlogz=0.01)
    res_lcdm = sampler_lcdm.results

    # =============================================================================
    # ANALIZA EWIDENCJI BAYESOWSKIEJ (BAYES FACTOR)
    # =============================================================================
    logZ_phantom = res_phantom.logz[-1]
    logZ_err_phantom = res_phantom.logzerr[-1]
    
    logZ_lcdm = res_lcdm.logz[-1]
    logZ_err_lcdm = res_lcdm.logzerr[-1]
    
    # Czynnik Bayesa
    delta_logZ = logZ_phantom - logZ_lcdm

    print("\n" + "#"*60)
    print("WYNIKI PORÓWNANIA BAYESOWSKIEGO (JOINT: BAO + SNe + CMB)")
    print("#"*60)
    print(f"Ewidencja (ln Z) PhantomX : {logZ_phantom:.2f} ± {logZ_err_phantom:.2f}")
    print(f"Ewidencja (ln Z) LCDM     : {logZ_lcdm:.2f} ± {logZ_err_lcdm:.2f}")
    print("-"*60)
    print(f"Czynnik Bayesa (\u0394 ln Z) : {delta_logZ:.2f}")
    
    if delta_logZ > 0:
        print(f"-> Wynik faworyzuje model PhantomX (z przewagą {delta_logZ:.2f}).")
    else:
        print(f"-> Wynik faworyzuje model LCDM (z przewagą {abs(delta_logZ):.2f}).")
    print("#"*60)

    # =============================================================================
    # GENEROWANIE WYKRESÓW WSPÓLNYCH
    # =============================================================================
    # Dynesty zwraca próbki z wagami. Przeliczamy je na równe wagi dla łatwego rysowania.
    weights_p = np.exp(res_phantom.logwt - logZ_phantom)
    samples_p_eq = resample_equal(res_phantom.samples, weights_p)
    
    weights_l = np.exp(res_lcdm.logwt - logZ_lcdm)
    samples_l_eq = resample_equal(res_lcdm.samples, weights_l)

    plt.figure(figsize=(10, 6))
    
    plt.hist(samples_l_eq[:, 0], bins=40, density=True, histtype='stepfilled', alpha=0.5, color='#1f77b4', 
             label=f'$\\Lambda$CDM ($\\Omega_{{m0}}$={np.percentile(samples_l_eq[:,0], 50):.3f})\n$\\ln \\mathcal{{Z}} = {logZ_lcdm:.1f}$')
    plt.hist(samples_l_eq[:, 0], bins=40, density=True, histtype='step', linewidth=2, color='#0b5394')

    plt.hist(samples_p_eq[:, 0], bins=40, density=True, histtype='stepfilled', alpha=0.5, color='#d62728', 
             label=f'PhantomX ($\\Omega_{{m0}}$={np.percentile(samples_p_eq[:,0], 50):.3f})\n$\\ln \\mathcal{{Z}} = {logZ_phantom:.1f}$')
    plt.hist(samples_p_eq[:, 0], bins=40, density=True, histtype='step', linewidth=2, color='#990000')

    plt.axvline(np.percentile(samples_l_eq[:,0], 50), color='#1f77b4', linestyle='dashed', linewidth=1.5)
    plt.axvline(np.percentile(samples_p_eq[:,0], 50), color='#d62728', linestyle='dashed', linewidth=1.5)

    plt.title(f'Model Comparison: $\\Lambda$CDM vs PhantomX (BAO+SNe+CMB)\nBayes Factor ($\\Delta \\ln \\mathcal{{Z}}$): {delta_logZ:.2f}', fontsize=14, fontweight='bold')
    plt.xlabel(r'$\Omega_{m0}$', fontsize=14)
    plt.ylabel('Posterior Density', fontsize=12)
    plt.xlim(0.2, 0.4)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig('Model_Comparison_Dynesty.png', dpi=300)
    
    print("\n-> Wygenerowano wykres porównawczy: Model_Comparison_Dynesty.png")