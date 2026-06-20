import os
import urllib.request
import numpy as np
import emcee
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.random.seed(42)

# =============================================================================
# AUTOMATYCZNE POBIERANIE DANYCH Z INTERNETU (REPOZYTORIUM PHANTOMX)
# =============================================================================

def pobierz_dane():
    print("Sprawdzanie i pobieranie danych z Internetu...")
    base_url = "https://raw.githubusercontent.com/GrzegorzSzczerba/PhantomX-Cosmology-Data/main/"
    
    pliki_do_pobrania = {
        "desi_gaussian_bao_ALL_GCcomb_mean.txt": base_url + "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        "desi_gaussian_bao_ALL_GCcomb_cov.txt": base_url + "desi_gaussian_bao_ALL_GCcomb_cov.txt",
        "DES-Dovekie_HD.csv": base_url + "DES-Dovekie_HD.csv"
    }
    
    for nazwa, url in pliki_do_pobrania.items():
        if not os.path.exists(nazwa):
            try:
                print(f" -> Pobieranie {nazwa}...")
                urllib.request.urlretrieve(url, nazwa)
            except Exception as e:
                print(f" [!] Błąd pobierania {nazwa}. Błąd: {e}")
        else:
            print(f" -> Plik {nazwa} już istnieje lokalnie.")

    cmb_file = "planck_2018_cmb_prior.txt"
    if not os.path.exists(cmb_file):
        print(f" -> Pobieranie/Generowanie priors CMB ({cmb_file})...")
        with open(cmb_file, 'w') as f:
            f.write("# Planck 2018 TT,TE,EE+lowE+lensing Distance Priors\n")
            f.write("# z_star  R_obs  sigma_R\n")
            f.write("1089.92  1.74963  0.00392\n")

pobierz_dane()

# =============================================================================
# DEFINICJE DYNAMIKI MODELU PŁASKIEGO LCDM
# =============================================================================

def E_z(z, Om_m0):
    """
    Bezwymiarowy parametr Hubble'a E(z) = H(z)/H0 dla płaskiego modelu Lambda-CDM.
    Omega_Lambda = 1.0 - Omega_m0
    """
    return np.sqrt(Om_m0 * (1.0 + z)**3 + (1.0 - Om_m0))

def D_M_integral(z, Om_m0):
    return integrate.quad(lambda x: 1.0 / E_z(x, Om_m0), 0, z)[0]

def D_H_val(z, Om_m0):
    return 1.0 / E_z(z, Om_m0)

# =============================================================================
# WYLICZANIE CHI^2 (LIKELIHOOD) DLA POSZCZEGÓLNYCH SOND
# =============================================================================

def get_chi2_bao(Om_m0, z_data_bao, d_data_bao, inv_cov_bao, data_types_bao):
    rd_Mpc = 147.05      
    c_km_s = 299792.458  
    H0 = 67.36           
    skala_rd = c_km_s / (H0 * rd_Mpc) 
    
    d_model_bao = np.zeros(len(z_data_bao))
    for i in range(len(z_data_bao)):
        z = z_data_bao[i]
        typ = data_types_bao[i]
        if typ == 'DM_over_rs':
            d_model_bao[i] = skala_rd * D_M_integral(z, Om_m0)
        elif typ == 'DH_over_rs':
            d_model_bao[i] = skala_rd * D_H_val(z, Om_m0)
        elif typ == 'DV_over_rs':
            dm = D_M_integral(z, Om_m0)
            dh = D_H_val(z, Om_m0)
            dv_bez_skali = (z * (dm**2) * dh)**(1.0/3.0)
            d_model_bao[i] = skala_rd * dv_bez_skali
            
    diff_bao = d_data_bao - d_model_bao
    return np.dot(diff_bao, np.dot(inv_cov_bao, diff_bao))

def get_chi2_sne(Om_m0, z_sne, mu_obs, inv_cov_sne):
    if Om_m0 <= 0 or Om_m0 >= 1:
        return np.inf
    z_max = np.max(z_sne)
    z_grid = np.linspace(0.0, z_max, 50)
    dm_grid = np.array([D_M_integral(z, Om_m0) for z in z_grid])
    dm_interp = interp1d(z_grid, dm_grid, kind='cubic', fill_value="extrapolate")
    dl_model = (1.0 + z_sne) * dm_interp(z_sne)
    dl_model[dl_model <= 0] = 1e-10 
    mu_model = 5.0 * np.log10(dl_model)
    delta = mu_obs - mu_model
    
    if inv_cov_sne.ndim == 1:
        S0 = np.sum(inv_cov_sne)
        S1 = np.sum(delta * inv_cov_sne)
        S2 = np.sum(delta**2 * inv_cov_sne)
    else:
        S0 = np.sum(inv_cov_sne)
        S1 = np.sum(np.dot(inv_cov_sne, delta))
        S2 = np.dot(delta.T, np.dot(inv_cov_sne, delta))
        
    return S2 - (S1**2 / S0) + np.log(S0 / (2*np.pi))

def get_chi2_cmb(Om_m0, z_star, R_obs, sigma_R):
    if Om_m0 <= 0 or Om_m0 >= 1:
        return np.inf
    dm_star = D_M_integral(z_star, Om_m0)
    R_model = np.sqrt(Om_m0) * dm_star
    return ((R_model - R_obs) / sigma_R)**2

# =============================================================================
# LIKELIHOOD DLA MCMC
# =============================================================================

def log_likelihood_bao(theta, args):
    z_bao, d_bao, inv_cov_bao, types_bao = args
    if not (0.05 < theta[0] < 0.95): return -np.inf 
    return -0.5 * get_chi2_bao(theta[0], z_bao, d_bao, inv_cov_bao, types_bao)

def log_likelihood_sne(theta, args):
    z_sne, mu_sne, inv_cov_sne = args
    if not (0.05 < theta[0] < 0.95): return -np.inf 
    return -0.5 * get_chi2_sne(theta[0], z_sne, mu_sne, inv_cov_sne)

def log_likelihood_cmb(theta, args):
    z_star, R_obs, sigma_R = args
    if not (0.05 < theta[0] < 0.95): return -np.inf 
    return -0.5 * get_chi2_cmb(theta[0], z_star, R_obs, sigma_R)

def log_likelihood_joint(theta, args_bao, args_sne, args_cmb):
    if not (0.05 < theta[0] < 0.95): return -np.inf 
    l_bao = get_chi2_bao(theta[0], *args_bao)
    l_sne = get_chi2_sne(theta[0], *args_sne)
    l_cmb = get_chi2_cmb(theta[0], *args_cmb)
    return -0.5 * (l_bao + l_sne + l_cmb)

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
    inv_cov_bao = np.linalg.inv(cov_bao)
    
    # 2. SNe
    z_sne_list, mu_sne_list, err_sne_list = [], [], []
    with open('DES-Dovekie_HD.csv', 'r', encoding='utf-8') as f:
        kolumny = []
        for linia in f:
            tokens = linia.strip().split()
            if not tokens or tokens[0].startswith('#'): continue
            if tokens[0] == 'VARNAMES:': kolumny = tokens[1:]; continue
            if kolumny:
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

    return (np.array(z_bao), np.array(d_bao), inv_cov_bao, types_bao), \
           (np.array(z_sne_list), np.array(mu_sne_list), 1.0 / np.array(err_sne_list)), \
           (z_star, R_obs, sigma_R)

# =============================================================================
# URUCHAMIANIE MCMC I ANALIZA 
# =============================================================================

def run_mcmc(log_prob_fn, args, steps=1500, nwalkers=32):
    pos = [0.30] + 1e-2 * np.random.randn(nwalkers, 1)
    sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob_fn, args=args)
    sampler.run_mcmc(pos, steps, progress=True)
    return sampler.get_chain(discard=500, thin=15, flat=True)

if __name__ == "__main__":
    args_bao, args_sne, args_cmb = wczytaj_wszystkie_dane()

    print("\n--- MCMC (LCDM): BAO ONLY ---")
    samples_bao = run_mcmc(log_likelihood_bao, (args_bao,))
    
    print("--- MCMC (LCDM): SNe ONLY ---")
    samples_sne = run_mcmc(log_likelihood_sne, (args_sne,))
    
    print("--- MCMC (LCDM): CMB ONLY ---")
    samples_cmb = run_mcmc(log_likelihood_cmb, (args_cmb,))
    
    print("--- MCMC (LCDM): JOINT (BAO + SNe + CMB) ---")
    samples_joint = run_mcmc(log_likelihood_joint, (args_bao, args_sne, args_cmb))
    
    # Obliczanie minimów
    Om_joint = np.percentile(samples_joint, 50)
    chi2_total = get_chi2_bao(Om_joint, *args_bao) + get_chi2_sne(Om_joint, *args_sne) + get_chi2_cmb(Om_joint, *args_cmb)

    print("\n" + "="*50)
    print("RAPORT DIAGNOSTYCZNY: MODEL STANDARDOWY LCDM")
    print("="*50)
    print(f"BAO Only: Om_m0 = {np.percentile(samples_bao, 50):.4f}")
    print(f"SNe Only: Om_m0 = {np.percentile(samples_sne, 50):.4f}")
    print(f"CMB Only: Om_m0 = {np.percentile(samples_cmb, 50):.4f}")
    print(f"JOINT FIT: Om_m0 = {Om_joint:.4f}")
    print(f"Total Chi^2: {chi2_total:.2f}")
    print("="*50)

    # Generowanie wykresu
    plt.figure(figsize=(10, 6))
    plt.hist(samples_bao, bins=30, density=True, histtype='step', linewidth=2.5, color='#1f77b4', label=f'BAO ($\Omega_{{m0}}$={np.percentile(samples_bao, 50):.3f})')
    plt.hist(samples_sne, bins=30, density=True, histtype='step', linewidth=2.5, color='#ff7f0e', label=f'SNe ($\Omega_{{m0}}$={np.percentile(samples_sne, 50):.3f})')
    plt.hist(samples_cmb, bins=30, density=True, histtype='step', linewidth=2.5, color='#2ca02c', label=f'CMB ($\Omega_{{m0}}$={np.percentile(samples_cmb, 50):.3f})')
    plt.hist(samples_joint, bins=30, density=True, histtype='stepfilled', alpha=0.3, color='#d62728', label=f'Joint Fit ($\Omega_{{m0}}$={Om_joint:.3f})')
    
    plt.axvline(Om_joint, color='black', linestyle='dashed', linewidth=2, label='Joint Minimum')
    
    plt.xlabel(r'$\Omega_{m0}$', fontsize=14)
    plt.ylabel('Prawdopodobieństwo (Posterior Density)', fontsize=12)
    plt.title('Zgodność sond (BAO + SNe + CMB) dla modelu standardowego $\Lambda$CDM', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    plt.xlim(0.2, 0.4) 
    plt.tight_layout()
    plt.savefig('MCMC_Tension_Plot_LCDM.png', dpi=300)
    print("-> WYKRES ZAPISANO JAKO: MCMC_Tension_Plot_LCDM.png")