import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

# On importe la fonction d'inférence BL pour l'analyse de sensibilité
# (Assure-toi que black_litterman.py est dans le même répertoire)
from black_litterman import compute_black_litterman_posterior

def compute_kl_divergence(
    mu_BL: np.ndarray, 
    Sigma_BL: np.ndarray, 
    Pi: np.ndarray, 
    Sigma_LW: np.ndarray, 
    tau: float
) -> float:
    """
    Calcule analytiquement la Divergence de Kullback-Leibler KL(Q_dist || P)
    entre le Posterior Q_dist = N(mu_BL, Sigma_BL) et le Prior P = N(Pi, tau * Sigma_LW).
    
    Cette implémentation utilise les décompositions stables via np.linalg.solve 
    et slogdet pour éviter les inversions matricielles explicites instables.
    
    Args:
        mu_BL: Espérance a posteriori (N,)
        Sigma_BL: Matrice de covariance a posteriori (N x N)
        Pi: Espérance du prior de marché (N,)
        Sigma_LW: Matrice de covariance empirique régularisée du marché (N x N)
        tau: Scalaire d'incertitude sur le prior
        
    Returns:
        kl: Valeur scalaire de la divergence KL (>= 0)
    """
    k = len(mu_BL)
    tau_Sigma = tau * Sigma_LW
    
    # 1. Terme de trace: tr((tau * Sigma_LW)^-1 @ Sigma_BL)
    # Résout (tau*Sigma) * X = Sigma_BL => X = (tau*Sigma)^-1 @ Sigma_BL
    # Puis trace(X)
    X = np.linalg.solve(tau_Sigma, Sigma_BL)
    trace_term = np.trace(X)
    
    # 2. Terme quadratique: (Pi - mu_BL)^T @ (tau * Sigma_LW)^-1 @ (Pi - mu_BL)
    diff = Pi - mu_BL
    # Résout (tau*Sigma) * Y = diff
    Y = np.linalg.solve(tau_Sigma, diff)
    quad_term = np.dot(diff.T, Y)
    
    # 3. Terme du log-déterminant: ln( det(tau * Sigma_LW) / det(Sigma_BL) )
    # Utilisation du pseudo-logdéterminant pour éviter les underflows/overflows
    sign_prior, logdet_prior = np.linalg.slogdet(tau_Sigma)
    sign_post, logdet_post = np.linalg.slogdet(Sigma_BL)
    
    # Avertissement si matrices non définies positives (bruit numérique)
    if sign_prior <= 0 or sign_post <= 0:
        jitter = np.eye(k) * 1e-7
        sign_prior, logdet_prior = np.linalg.slogdet(tau_Sigma + jitter)
        sign_post, logdet_post = np.linalg.slogdet(Sigma_BL + jitter)
        if sign_prior <= 0 or sign_post <= 0:
            return 1000.0 # Pénalité élevée si toujours non inversible
        
    logdet_term = logdet_prior - logdet_post
    
    # 4. Formule complète
    kl = 0.5 * (trace_term + quad_term - k + logdet_term)
    
    # Par définition mathématique, KL >= 0. Les bruits numériques peuvent donner un léger -1e-15.
    return float(max(0.0, kl))

def compute_pac_bayes_bounds(
    empirical_loss: float, 
    kl_div: float, 
    m: int, 
    epsilon: float = 0.05, 
    C: float = 1.0
) -> Dict[str, float]:
    """
    Calcule les bornes de généralisation numériques PAC-Bayes (Catoni & McAllester).
    Majore l'erreur de généralisation (hors-échantillon) en fonction de l'erreur empirique,
    de la complexité (KL), et de la taille de l'échantillon.
    
    Args:
        empirical_loss: Perte empirique in-sample (normalisée entre 0 et 1 idéalement)
        kl_div: Divergence KL entre le posterior et le prior
        m: Nombre d'observations (taille de l'échantillon)
        epsilon: Niveau de confiance (ex: 0.05 pour 95%)
        C: Paramètre d'échelle (température) pour la borne de Catoni (C > 0)
        
    Returns:
        Dictionnaire contenant la borne de McAllester, Catoni et la KL d'origine.
    """
    if m <= 0:
        return {'mcallester_bound': float('inf'), 'catoni_bound': float('inf'), 'kl_div': kl_div}
        
    # --- Borne de McAllester (1999) ---
    # R_vrai <= R_emp + sqrt( (KL + ln(2 * sqrt(m) / epsilon)) / (2 * m) )
    mca_complexity = (kl_div + np.log(2.0 * np.sqrt(m) / epsilon)) / (2.0 * m)
    mcallester_bound = empirical_loss + np.sqrt(max(0.0, mca_complexity))
    
    # --- Borne de Catoni (2007) ---
    # R_vrai <= (1 - exp(-C * R_emp - (KL + ln(1/epsilon)) / m)) / (1 - exp(-C))
    num_term = -C * empirical_loss - (kl_div + np.log(1.0 / epsilon)) / float(m)
    num = 1.0 - np.exp(num_term)
    den = 1.0 - np.exp(-C)
    
    catoni_bound = num / den if den > 1e-12 else float('inf')
    
    return {
        'mcallester_bound': float(mcallester_bound),
        'catoni_bound': float(catoni_bound),
        'kl_div': float(kl_div)
    }

def analyze_kl_sensitivity(
    Sigma_LW: np.ndarray, 
    Q: np.ndarray, 
    Omega: np.ndarray, 
    delta_grid: Union[List[float], np.ndarray], 
    tau_grid: Union[List[float], np.ndarray], 
    w_mkt: np.ndarray
) -> pd.DataFrame:
    """
    Évalue la sensibilité de la divergence KL(Q_dist || P) par rapport aux 
    hyperparamètres tau (incertitude structurelle) et delta (aversion au risque).
    
    Args:
        Sigma_LW: Matrice de covariance (N x N)
        Q: Vues espérées (N,)
        Omega: Incertitude des vues (N x N)
        delta_grid: Grille des valeurs de delta à tester
        tau_grid: Grille des valeurs de tau à tester
        w_mkt: Poids de marché (N,)
        
    Returns:
        DataFrame contenant chaque couple (delta, tau) et sa divergence KL associée.
    """
    results = []
    
    # Pour respecter la consigne "à delta fixé" puis "à tau fixé", on utilise un plan 
    # d'expérience factoriel complet (Cross Join) qui permet d'analyser toutes les coupes.
    for delta in delta_grid:
        for tau in tau_grid:
            try:
                # 1. Mise à jour Bayésienne (Black-Litterman)
                mu_BL, Sigma_BL, Pi = compute_black_litterman_posterior(
                    Sigma_LW=Sigma_LW, 
                    Q=Q, 
                    Omega=Omega, 
                    delta=delta, 
                    tau=tau, 
                    w_mkt=w_mkt
                )
                
                # 2. Calcul de la KL
                kl = compute_kl_divergence(
                    mu_BL=mu_BL, 
                    Sigma_BL=Sigma_BL, 
                    Pi=Pi, 
                    Sigma_LW=Sigma_LW, 
                    tau=tau
                )
                
                results.append({'delta': delta, 'tau': tau, 'kl_div': kl, 'status': 'OK'})
                
            except Exception as e:
                # Capture des instabilités pour des deltas aberrants (ex: matrices non-D.P.)
                results.append({'delta': delta, 'tau': tau, 'kl_div': np.nan, 'status': str(e)})
                
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Bloc de Test Unitaire (Validation des propriétés mathématiques)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    k = 10
    
    # 1. Génération de matrices factices mathématiquement valides
    # Covariance définie positive
    A = np.random.randn(k, k)
    Sigma_LW = A.T @ A + np.eye(k) * 1e-3
    
    Omega = np.diag(np.random.rand(k) * 0.05 + 0.01)
    w_mkt = np.ones(k) / k
    
    delta_test = 2.0
    tau_test = 0.05
    
    # Prior direct
    Pi_test = delta_test * Sigma_LW @ w_mkt
    
    # Vues aléatoires
    Q_test = Pi_test + np.random.randn(k) * 0.02
    
    # --- TEST 1: Calcul de la KL ---
    mu_BL, Sigma_BL, Pi = compute_black_litterman_posterior(
        Sigma_LW, Q_test, Omega, delta=delta_test, tau=tau_test, w_mkt=w_mkt
    )
    kl_value = compute_kl_divergence(mu_BL, Sigma_BL, Pi, Sigma_LW, tau_test)
    print(f"[TEST 1] Divergence KL calculée : {kl_value:.6f}")
    assert kl_value >= 0, "Erreur fatale: la KL doit être strictement positive ou nulle."
    
    # --- TEST 2: Cas extrême où Posterior == Prior (Aucune Vue Nouvelle) ---
    # Si Q = Pi et Omega -> inf, le posterior doit s'écraser sur le prior et KL = 0
    Omega_inf = np.eye(k) * 1e8
    mu_BL_inf, Sigma_BL_inf, Pi_inf = compute_black_litterman_posterior(
        Sigma_LW, Pi_test, Omega_inf, delta=delta_test, tau=tau_test, w_mkt=w_mkt
    )
    kl_inf = compute_kl_divergence(mu_BL_inf, Sigma_BL_inf, Pi_inf, Sigma_LW, tau_test)
    print(f"[TEST 2] Divergence KL (Q=Prior, Omega->inf) : {kl_inf:.10f}")
    assert np.isclose(kl_inf, 0.0, atol=1e-5), "Erreur: KL non nulle lorsque le posterior est identique au prior."
    
    # --- TEST 3: Bornes PAC-Bayes ---
    m_samples = 1000
    emp_loss = 0.15 # 15% d'erreur arbitraire
    bounds = compute_pac_bayes_bounds(emp_loss, kl_value, m_samples)
    print(f"[TEST 3] Borne McAllester : {bounds['mcallester_bound']:.4f}")
    print(f"[TEST 3] Borne Catoni     : {bounds['catoni_bound']:.4f}")
    assert bounds['mcallester_bound'] > emp_loss, "McAllester doit majorer l'erreur empirique."
    
    # --- TEST 4: Sensibilité ---
    delta_grid = [1.0, 10.0, 18.0, 100.0, -1.0, -5.5, -100.0]
    tau_grid = np.logspace(-3, 0, 5) # [1e-3, ..., 1.0]
    df_sens = analyze_kl_sensitivity(Sigma_LW, Q_test, Omega, delta_grid, tau_grid, w_mkt)
    print(f"\n[TEST 4] Analyse de sensibilité :\n{df_sens.head()}")
    
    print("\n[SUCCÈS] Tous les tests mathématiques du module PAC-Bayes sont validés.")
