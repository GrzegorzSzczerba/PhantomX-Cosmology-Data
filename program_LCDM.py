
import numpy as np
import emcee
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import corner
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

np.random.seed(42)

# =============================================================================
# DEFINICJE DYNAMIKI MODELU 6W-ANRA
# =============================================================================

# WKLEJ STANDARDOWE LCDM:
def E_z(z, Om_m0):
    """Klasyczny model Lambda-CDM (dla plaskiego Wszechswiata)"""
    return np.sqrt(Om_m0 * (1.0 + z)**3 + (1.0 - Om_m0))

def D_M_integral(z, Om_m0):
    return integrate.quad(lambda x: 1.0 / E_z(x, Om_m0), 0, z)[0]

def D_H_val(z, Om_m0):
    return 1.0 / E_z(z, Om_m0)

def D_L_model(z, Om_m0):
    dm = D_M_integral(z, Om_m0)
    return (1.0 + z) * dm

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
    chi2_bao = np.dot(diff_bao, np.dot(inv_cov_bao, diff_bao))
    return chi2_bao

def get_chi2_sne(Om_m0, z_sne, mu_obs, inv_cov_sne):
    # Zabezpieczenie przed bledami marginalizacji dla bardzo dziwnych parametrow
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
    
    # Pelny wzor na analityczna marginalizacje stalej M (w tym log z det macierzy jesli trzeba)
    # Wzor Conleya/Gullana: chi^2_marg = A - (B^2 / C)
    # A = delta.T * C^-1 * delta
    # B = sum(C^-1 * delta)
    # C = sum(C^-1)
    
    if inv_cov_sne.ndim == 1:
        S0 = np.sum(inv_cov_sne)
        S1 = np.sum(delta * inv_cov_sne)
        S2 = np.sum(delta**2 * inv_cov_sne)
    else:
        # Macierz 2D
        S0 = np.sum(inv_cov_sne)
        S1 = np.sum(np.dot(inv_cov_sne, delta))
        S2 = np.dot(delta.T, np.dot(inv_cov_sne, delta))
        
    # Marginalized chi2 + normalization term (czesto pomijany, ale dodajemy dla pewnosci)
    # ln(C / 2pi) jest stale wzgledem modeli, wiec nie wplywa mocno na ksztalt, 
    # ale zachowujemy poprawny wzor analityczny
    chi2_marg = S2 - (S1**2 / S0) + np.log(S0 / (2*np.pi))
    return chi2_marg

# =============================================================================
# DEFINICJE FUNKCJI PRAWDOPODOBIENSTWA DLA MCMC
# =============================================================================

# Likelihood dla BAO-only
def log_likelihood_bao(theta, z_bao, d_bao, inv_cov_bao, types_bao):
    Om_m0 = theta[0]
    if not (0.05 < Om_m0 < 0.95): 
        return -np.inf 
    chi2 = get_chi2_bao(Om_m0, z_bao, d_bao, inv_cov_bao, types_bao)
    return -0.5 * chi2

# Likelihood dla SNe-only
def log_likelihood_sne(theta, z_sne, mu_sne, inv_cov_sne):
    Om_m0 = theta[0]
    if not (0.05 < Om_m0 < 0.95): 
        return -np.inf 
    chi2 = get_chi2_sne(Om_m0, z_sne, mu_sne, inv_cov_sne)
    return -0.5 * chi2

# Likelihood laczony BAO+SNe
def log_likelihood_joint(theta, z_bao, d_bao, inv_cov_bao, types_bao, z_sne, mu_sne, inv_cov_sne):
    Om_m0 = theta[0]
    if not (0.05 < Om_m0 < 0.95): 
        return -np.inf 
    chi2_bao = get_chi2_bao(Om_m0, z_bao, d_bao, inv_cov_bao, types_bao)
    chi2_sne = get_chi2_sne(Om_m0, z_sne, mu_sne, inv_cov_sne)
    return -0.5 * (chi2_bao + chi2_sne)

# =============================================================================
# WCZYTYWANIE DANYCH 
# =============================================================================

def wczytaj_dane_bao(sciezka_plik_txt, sciezka_macierz_txt):
    print("Wczytywanie danych DESI DR2 BAO...")
    z_data, d_data, data_types = [], [], []
    with open(sciezka_plik_txt, 'r') as f:
        for linia in f:
            if linia.startswith('#') or not linia.strip():
                continue
            czesci = linia.split()
            z_data.append(float(czesci[0]))
            d_data.append(float(czesci[1]))
            data_types.append(czesci[2])
            
    z_data = np.array(z_data)
    d_data = np.array(d_data)
    cov_matrix = np.loadtxt(sciezka_macierz_txt)
    inv_cov = np.linalg.inv(cov_matrix)
    print(f"-> Znaleziono {len(z_data)} punktow BAO.")
    return z_data, d_data, inv_cov, data_types

def wczytaj_dane_sne(sciezka_hd):
    print("Wczytywanie danych DES-Dovekie SNe Ia (z wbudowana macierza diagonalna)...")
    with open(sciezka_hd, 'r', encoding='utf-8') as f:
        linie = f.readlines()
        
    z_list, mu_list, err_list = [], [], []
    kolumny = []
    
    for linia in linie:
        linia = linia.strip()
        if not linia or linia.startswith('#'):
            continue
            
        tokens = linia.split()
        if tokens[0] == 'VARNAMES:':
            kolumny = tokens[1:] 
            continue
            
        if kolumny:
            wartosci = tokens[1:] if tokens[0] in ['SN:', 'ROW:', 'OBS:', 'GAL:'] else tokens
            if len(wartosci) >= len(kolumny):
                try:
                    idx_z = kolumny.index('zHD')
                    idx_mu = kolumny.index('MU')
                    idx_err1 = kolumny.index('MUERR')
                    idx_err2 = kolumny.index('MUERR_SYS')
                    
                    z_list.append(float(wartosci[idx_z]))
                    mu_list.append(float(wartosci[idx_mu]))
                    # Suma w kwadraturze
                    total_err2 = float(wartosci[idx_err1])**2 + float(wartosci[idx_err2])**2
                    err_list.append(total_err2)
                except (ValueError, IndexError):
                    pass
                    
    z_sne = np.array(z_list)
    mu_sne = np.array(mu_list)
    inv_cov_sne = 1.0 / np.array(err_list) 
    print(f"-> Pomyślnie wyekstrahowano {len(z_sne)} Supernowych.")
    return z_sne, mu_sne, inv_cov_sne

# =============================================================================
# URUCHAMIANIE MCMC I ANALIZA 
# =============================================================================

def run_mcmc_analysis(log_prob_fn, args, steps=1500, nwalkers=32):
    pos = [0.30] + 1e-2 * np.random.randn(nwalkers, 1)
    ndim = 1
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, args=args)
    sampler.run_mcmc(pos, steps, progress=True)
    return sampler

def calculate_best_fit_and_chi2(flat_samples, get_chi2_fn, args):
    best_Om = np.percentile(flat_samples, 50)
    lower = best_Om - np.percentile(flat_samples, 16)
    upper = np.percentile(flat_samples, 84) - best_Om
    best_chi2 = get_chi2_fn(best_Om, *args)
    return best_Om, lower, upper, best_chi2

# --- GŁÓWNA PROCEDURA WYKONAWCZA ---
if __name__ == "__main__":
    
    # 1. Wczytanie danych
    try:
        z_bao, d_bao, inv_cov_bao, types_bao = wczytaj_dane_bao('desi_gaussian_bao_ALL_GCcomb_mean.txt', 'desi_gaussian_bao_ALL_GCcomb_cov.txt')
        z_sne, mu_sne, inv_cov_sne = wczytaj_dane_sne('DES-Dovekie_HD.csv')
    except FileNotFoundError as e:
        print(f"Brak pliku z danymi! Oczekiwane pliki to: desi_gaussian_bao_ALL_GCcomb_mean.txt, desi_gaussian_bao_ALL_GCcomb_cov.txt, DES-Dovekie_HD.csv")
        print(e)
        exit()

    STEPS = 1500
    DISCARD = 500
    THIN = 15
    
    # 2. Bieg BAO-only
    print("--- URUCHAMIAM MCMC: BAO ONLY ---")
    sampler_bao = run_mcmc_analysis(log_likelihood_bao, (z_bao, d_bao, inv_cov_bao, types_bao), steps=STEPS)
    samples_bao = sampler_bao.get_chain(discard=DISCARD, thin=THIN, flat=True)
    Om_bao, l_bao, u_bao, chi2_bao_only = calculate_best_fit_and_chi2(samples_bao, get_chi2_bao, (z_bao, d_bao, inv_cov_bao, types_bao))
    
    # 3. Bieg SNe-only
    print("--- URUCHAMIAM MCMC: SNe ONLY ---")
    sampler_sne = run_mcmc_analysis(log_likelihood_sne, (z_sne, mu_sne, inv_cov_sne), steps=STEPS)
    samples_sne = sampler_sne.get_chain(discard=DISCARD, thin=THIN, flat=True)
    Om_sne, l_sne, u_sne, chi2_sne_only = calculate_best_fit_and_chi2(samples_sne, get_chi2_sne, (z_sne, mu_sne, inv_cov_sne))
    
    # 4. Bieg Laczony (Joint)
    print("--- URUCHAMIAM MCMC: JOINT (BAO + SNe) ---")
    sampler_joint = run_mcmc_analysis(log_likelihood_joint, (z_bao, d_bao, inv_cov_bao, types_bao, z_sne, mu_sne, inv_cov_sne), steps=STEPS)
    samples_joint = sampler_joint.get_chain(discard=DISCARD, thin=THIN, flat=True)
    
    Om_joint = np.percentile(samples_joint, 50)
    l_joint = Om_joint - np.percentile(samples_joint, 16)
    u_joint = np.percentile(samples_joint, 84) - Om_joint
    
    # Wyliczanie dokladnych skladowych chi2 dla minimum połączonego
    chi2_joint_bao_part = get_chi2_bao(Om_joint, z_bao, d_bao, inv_cov_bao, types_bao)
    chi2_joint_sne_part = get_chi2_sne(Om_joint, z_sne, mu_sne, inv_cov_sne)
    chi2_joint_total = chi2_joint_bao_part + chi2_joint_sne_part
    
    # =========================================================================
    # RAPORT DLA RECENZENTOW
    # =========================================================================
    print("" + "="*50)
    print("RAPORT DIAGNOSTYCZNY DLA RECENZENTÓW (REPRODUCIBILITY SHEET)")
    print("="*50)
    print(f"Liczba wędrowców (walkers): 32")
    print(f"Liczba kroków: {STEPS} (Burn-in: {DISCARD}, Thinning: {THIN})")
    print(f"Priors: Om_m0 in (0.05, 0.95), H0=67.36 (stale), r_d=147.05 (stale)")
    
    print("1. WYNIKI BAO-ONLY:")
    print(f"   Om_m0 = {Om_bao:.4f} +{u_bao:.4f} / -{l_bao:.4f}")
    print(f"   chi^2 = {chi2_bao_only:.2f} (dla {len(z_bao)} punktow)")
    
    print("2. WYNIKI SNe-ONLY:")
    print(f"   Om_m0 = {Om_sne:.4f} +{u_sne:.4f} / -{l_sne:.4f}")
    print(f"   chi^2 = {chi2_sne_only:.2f} (dla {len(z_sne)} punktow)")
    
    print("3. WYNIKI POŁĄCZONE (JOINT BAO+SNe):")
    print(f"   Om_m0 = {Om_joint:.4f} +{u_joint:.4f} / -{l_joint:.4f}")
    print(f"   chi^2(total) = {chi2_joint_total:.2f}")
    print(f"      w tym chi^2(BAO) = {chi2_joint_bao_part:.2f}")
    print(f"      w tym chi^2(SNe) = {chi2_joint_sne_part:.2f}")
    print("="*50)
    
    # =========================================================================
    # GENEROWANIE WYKRESOW
    # =========================================================================
    try:
        plt.figure(figsize=(9, 6))
        
        # Generowanie nakładających się dzwonów
        plt.hist(samples_bao, bins=30, density=True, histtype='step', linewidth=2.5, color='#1f77b4', label=f'BAO Only ($\Omega_{{m0}}$={Om_bao:.3f})')
        plt.hist(samples_sne, bins=30, density=True, histtype='step', linewidth=2.5, color='#ff7f0e', label=f'SNe Only ($\Omega_{{m0}}$={Om_sne:.3f})')
        plt.hist(samples_joint, bins=30, density=True, histtype='stepfilled', alpha=0.4, color='#9467bd', label=f'Joint Fit ($\Omega_{{m0}}$={Om_joint:.3f})')
        
        plt.axvline(Om_joint, color='black', linestyle='dashed', linewidth=2, alpha=0.8, label=f'Joint Minimum')
        
        plt.xlabel(r'$\Omega_{m0}$', fontsize=14)
        plt.ylabel('Prawdopodobienstwo posterior (Posterior Density)', fontsize=12)
        plt.title('Zgodnosc MCMC (Tension-Free Check) dla Modelu LCDM', fontsize=14, fontweight='bold')
        
        # Kosmetyka wykresu
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11, loc='upper left')
        plt.xlim(0.2, 0.4) # Zawężamy oś X dla lepszej widoczności!
        plt.tight_layout() # POPRAWIONE!
        
        plt.savefig('LCDM_MCMC_Tension_Plot.png', dpi=300)
        print("\n-> ZAPISANO WYKRES DO PLIKU: LCDM_MCMC_Tension_Plot.png")
        print("-> TEN WYKRES JEST KLUCZOWYM DOWODEM DO PUBLIKACJI!")
        
    except Exception as e:
        print("Nie udalo sie wygenerowac wykresu:", e)
        
    