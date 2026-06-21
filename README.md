Jednoparametrowy model ciemnej energii ρDE ∝ (aH)−2:
trajektoria tła i dopasowanie do danych BAO, SNe oraz CMB
DES-Dovekie HD
Autor: Grzegorz Szczerba

Pochodzenie danych:
https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT
https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2
Dla CMB:
### Cosmic Microwave Background (CMB) Data
The CMB constraints used in this analysis are based on the compressed distance priors from the **Planck 2018 Final Release**. 
The observational vector $v = (R, l_A, \omega_b)$, the standard deviations, and the correlation matrix used in the code are derived exactly from Table 1 and Table 2 of the following paper:
* **Reference:** Chen, L., Huang, Q.-G., & Wang, K. (2019). *Distance Priors from Planck Final Release*. JCAP 02(2019)028.
* **arXiv link:** [https://arxiv.org/abs/1808.05724](https://arxiv.org/abs/1808.05724)

3 główne programy:
program_nested_sampling_v3.py
program_diagnoza_parametrow_v1.py
program_nested_sampling_v3_testing_wide_priors.py

Uruchomienie bez żadnych parametrów
Użyty (zapisany w kodzie) seed = 42 dla pierwszego i trzeciego programu.
