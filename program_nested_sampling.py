import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys

# Pakiety do Ewidencji Bayesowskiej
import dynesty
from dynesty import plotting as dyplot

np.random.seed(42)

# =============================================================================
# DEFINICJE DYNAMIKI MODELI (6W-ANRA / PhantomX oraz LCDM)
# =============================================================================

def E_z_phantom(z, Om_m0):
    a = 1.0 / (1.0 + z)
    K = 4.0 * (1.0 - Om_m0) / (Om_m0**2)
    term_sqrt = np.sqrt(1.0 + K * (a**4))
    return np.sqrt((Om_m0 / (2 * a**3)) * (1.0 + term_sqrt))

def E_z_lcdm(z, Om_m0):
    return np.sqrt(Om_m0 * (1.0 + z)**3 + (1.0 - Om_m0))

def D_M_phantom(z, Om_m0):
    return integrate.quad(lambda x: 1.0 / E_z_phantom(x, Om_m0), 0, z)[0]

def D_M_lcdm(z, Om_m0):
    return integrate.quad(lambda x: 1.0 / E_z_lcdm(x, Om_m0), 0, z)[0]

# =============================================================================
# WYLICZANIE CHI^2 Z UWZGLĘDNIENIEM ZMIENNEGO H0
# =============================================================================

def get_chi2_bao(Om_m0, H0, z_data_bao, d_data_bao, inv_cov_bao, data_types_bao, model='phantom'):
    rd_Mpc = 147.05      # Fix rd according to Planck (standard distance ladder approach)
    c_km_s = 299792.458  
    skala_rd = c_km_s / (H0 * rd_Mpc) 
    
    d_model_bao = np.zeros(len(z_data_bao))
    
    for i in range(len(z_data_bao)):
        z = z_data_bao[i]
        typ = data_types_bao[i]
        
        if model == 'phantom':
            dm = D_M_phantom(z, Om_m0)
            dh = 1.0 / E_z_phantom(z, Om_m0)
        else:
            dm = D_M_lcdm(z, Om_m0)
            dh = 1.0 / E_z_lcdm(z, Om_m0)
            
        if typ == 'DM_over_rs':
            d_model_bao[i] = skala_rd * dm
        elif typ == 'DH_over_rs':
            d_model_bao[i] = skala_rd * dh
        elif typ == 'DV_over_rs':
            dv_bez_skali = (z * (dm**2) * dh)**(1.0/3.0)
            d_model_bao[i] = skala_rd * dv_bez_skali
            
    diff_bao = d_data_bao - d_model_bao
    chi2_bao = np.dot(diff_bao, np.dot(inv_cov_bao, diff_bao))
    return chi2_bao

def get_chi2_sne(Om_m0, z_sne, mu_obs, inv_cov_sne, model='phantom'):
    if Om_m0 <= 0.01 or Om_m0 >= 0.99:
        return np.inf
        
    z_max = np.max(z_sne)
    z_grid = np.linspace(0.0, z_max, 50)
    
    if model == 'phantom':
        dm_grid = np.array([D_M_phantom(z, Om_m0) for z in z_grid])
    else:
        dm_grid = np.array([D_M_lcdm(z, Om_m0) for z in z_grid])
    
    dm_interp = interp1d(z_grid, dm_grid, kind='cubic', fill_value="extrapolate")
    dl_model = (1.0 + z_sne) * dm_interp(z_sne)
    dl_model[dl_model <= 0] = 1e-10 
    mu_model = 5.0 * np.log10(dl_model)
    
    delta = mu_obs - mu_model
    
    # Analityczna marginalizacja
    if inv_cov_sne.ndim == 1:
        S0 = np.sum(inv_cov_sne)
        S1 = np.sum(delta * inv_cov_sne)
        S2 = np.sum(delta**2 * inv_cov_sne)
    else:
        S0 = np.sum(inv_cov_sne)
        S1 = np.sum(np.dot(inv_cov_sne, delta))
        S2 = np.dot(delta.T, np.dot(inv_cov_sne, delta))
        
    chi2_marg = S2 - (S1**2 / S0) + np.log(S0 / (2*np.pi))
    return chi2_marg

# =============================================================================
# DEFINICJE DLA NESTED SAMPLING (DYNESTY)
# =============================================================================

# Transformacja priorów (od płaskiej kostki [0,1] do rzeczywistych wartości)
def prior_transform(u):
    x = np.array(u)
    x[0] = 0.05 + 0.90 * u[0]   # Om_m0 uniform prior: [0.05, 0.95]
    x[1] = 60.0 + 20.0 * u[1]   # H0 uniform prior: [60.0, 80.0]
    return x

# Likelihood dla wybranego modelu
def loglike(theta, model, z_bao, d_bao, inv_cov_bao, types_bao, z_sne, mu_sne, inv_cov_sne):
    Om_m0, H0 = theta
    
    chi2_bao = get_chi2_bao(Om_m0, H0, z_bao, d_bao, inv_cov_bao, types_bao, model=model)
    chi2_sne = get_chi2_sne(Om_m0, z_sne, mu_sne, inv_cov_sne, model=model)
    
    return -0.5 * (chi2_bao + chi2_sne)

# Wrappery dla poszczególnych modeli (dla dynesty)
def loglike_phantom(theta):
    return loglike(theta, 'phantom', z_bao, d_bao, inv_cov_bao, types_bao, z_sne, mu_sne, inv_cov_sne)

def loglike_lcdm(theta):
    return loglike(theta, 'lcdm', z_bao, d_bao, inv_cov_bao, types_bao, z_sne, mu_sne, inv_cov_sne)


# =============================================================================
# WCZYTYWANIE DANYCH 
# =============================================================================

def wczytaj_dane_bao(sciezka_plik_txt, sciezka_macierz_txt):
    z_data, d_data, data_types = [], [], []
    with open(sciezka_plik_txt, 'r') as f:
        for linia in f:
            if linia.startswith('#') or not linia.strip(): continue
            czesci = linia.split()
            z_data.append(float(czesci[0]))
            d_data.append(float(czesci[1]))
            data_types.append(czesci[2])
    cov_matrix = np.loadtxt(sciezka_macierz_txt)
    return np.array(z_data), np.array(d_data), np.linalg.inv(cov_matrix), data_types

def wczytaj_dane_sne(sciezka_hd):
    with open(sciezka_hd, 'r', encoding='utf-8') as f:
        linie = f.readlines()
    z_list, mu_list, err_list, kolumny = [], [], [], []
    for linia in linie:
        linia = linia.strip()
        if not linia or linia.startswith('#'): continue
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
                    err_list.append(float(wartosci[idx_err1])**2 + float(wartosci[idx_err2])**2)
                except (ValueError, IndexError): pass
    return np.array(z_list), np.array(mu_list), 1.0 / np.array(err_list)

# =============================================================================
# GŁÓWNA PROCEDURA WYKONAWCZA
# =============================================================================

if __name__ == "__main__":
    print("Wczytywanie danych...")
    try:
        z_bao, d_bao, inv_cov_bao, types_bao = wczytaj_dane_bao('desi_gaussian_bao_ALL_GCcomb_mean.txt', 'desi_gaussian_bao_ALL_GCcomb_cov.txt')
        z_sne, mu_sne, inv_cov_sne = wczytaj_dane_sne('DES-Dovekie_HD.csv')
    except Exception as e:
        print("Błąd plików:", e)
        sys.exit()

    # Parametry Nested Samplera
    NLIVE = 500  # Liczba tzw. live points. 500 to wystarczająco dużo dla dwóch wymiarów
    NDIM = 2

    print("\n" + "="*60)
    print("1/2 URUCHAMIAM NESTED SAMPLING DLA MODELU: PhantomX (6W-ANRA)")
    print("="*60)
    sampler_ph = dynesty.NestedSampler(loglike_phantom, prior_transform, ndim=NDIM, nlive=NLIVE)
    sampler_ph.run_nested(print_progress=True)
    res_ph = sampler_ph.results
    logZ_ph = res_ph.logz[-1]
    logZerr_ph = res_ph.logzerr[-1]

    print("\n" + "="*60)
    print("2/2 URUCHAMIAM NESTED SAMPLING DLA MODELU: LambdaCDM")
    print("="*60)
    sampler_lcdm = dynesty.NestedSampler(loglike_lcdm, prior_transform, ndim=NDIM, nlive=NLIVE)
    sampler_lcdm.run_nested(print_progress=True)
    res_lcdm = sampler_lcdm.results
    logZ_lcdm = res_lcdm.logz[-1]
    logZerr_lcdm = res_lcdm.logzerr[-1]

    # =========================================================================
    # RAPORT BAYESOWSKI
    # =========================================================================
    
    Bayes_Factor = logZ_ph - logZ_lcdm
    
    print("\n" + "#"*60)
    print("RAPORT EWIDENCJI BAYESOWSKIEJ (DO PUBLIKACJI)")
    print("#"*60)
    print(f"Ewidencja PhantomX (ln Z) :  {logZ_ph:.4f} +/- {logZerr_ph:.4f}")
    print(f"Ewidencja LambdaCDM (ln Z):  {logZ_lcdm:.4f} +/- {logZerr_lcdm:.4f}")
    print("-" * 60)
    print(f"Delta ln Z (Bayes Factor) =  {Bayes_Factor:.4f}")
    
    print("\nINTERPRETACJA SKALI JEFFREYSA:")
    if Bayes_Factor > 0:
        print("-> Twój model (PhantomX) jest BARDZIEJ PRAWDOPODOBNY od LambdaCDM.")
        if Bayes_Factor < 1.0: print("   Dowód: Słaby / Bezwartościowy (Inconclusive)")
        elif Bayes_Factor < 2.5: print("   Dowód: Umiarkowany (Substantial)")
        elif Bayes_Factor < 5.0: print("   Dowód: Silny (Strong evidence)")
        else: print("   Dowód: Decydujący (Decisive evidence)!")
    else:
        print("-> LambdaCDM jest BARDZIEJ PRAWDOPODOBNY od Twojego modelu.")
        print(f"   Przewaga LambdaCDM wynosi: {abs(Bayes_Factor):.4f} na skali logarytmicznej.")
    print("#"*60)

    # =========================================================================
    # GENEROWANIE PROFESJONALNYCH WYKRESÓW (CORNER PLOTS)
    # =========================================================================
    try:
        # Generowanie wykresu dla PhantomX
        fig, axes = dyplot.cornerplot(res_ph, labels=[r'$\Omega_{m0}$', r'$H_0$'],
                                      color='darkblue', show_titles=True,
                                      title_kwargs={'fontsize': 14})
        fig.suptitle('PhantomX (6W-ANRA) - Posterior Distribution', fontsize=16, y=1.02)
        plt.savefig('PhantomX_Nested_Corner.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Generowanie wykresu dla LCDM
        fig, axes = dyplot.cornerplot(res_lcdm, labels=[r'$\Omega_{m0}$', r'$H_0$'],
                                      color='darkorange', show_titles=True,
                                      title_kwargs={'fontsize': 14})
        fig.suptitle(r'$\Lambda$CDM - Posterior Distribution', fontsize=16, y=1.02)
        plt.savefig('LCDM_Nested_Corner.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("\n-> Wygenerowano profesjonalne wykresy: 'PhantomX_Nested_Corner.png' i 'LCDM_Nested_Corner.png'")
        print("-> Możesz je bezpośrednio osadzić w pliku PDF swojej pracy badawczej.")
    except Exception as e:
        print("\nNie udało się wygenerować wykresów Dynesty:", e)