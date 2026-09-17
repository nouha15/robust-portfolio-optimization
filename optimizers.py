import numpy as np
from scipy.optimize import minimize
import warnings

def _get_constraints_and_bounds(N: int):
    """
    Fournit les contraintes standard du projet :
    - Somme des poids = 1.0 (Full investment)
    - Long-only (w >= 0)
    - Concentration max par actif = 30% (w <= 0.30)
    """
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 0.30) for _ in range(N))
    w0 = np.ones(N) / N
    return constraints, bounds, w0

def optimize_markowitz(mu_hist: np.ndarray, S_empirique: np.ndarray, delta: float) -> np.ndarray:
    """
    Solveur Markowitz Classique Moyenne-Variance.
    Objectif: max w.T @ mu - (delta/2) * w.T @ S @ w
    Ce qui équivaut à minimiser l'opposé.
    """
    N = len(mu_hist)
    constraints, bounds, w0 = _get_constraints_and_bounds(N)
    
    def objective(w):
        port_ret = w.T @ mu_hist
        port_var = w.T @ S_empirique @ w
        return -(port_ret - (delta / 2.0) * port_var)
        
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not res.success:
        warnings.warn(f"Markowitz SLSQP n'a pas convergé: {res.message}. Retour du 1/N.")
        return w0
        
    return res.x

def optimize_black_litterman(mu_BL: np.ndarray, Sigma_BL: np.ndarray, delta: float) -> np.ndarray:
    """
    Solveur Black-Litterman Contraint.
    Objectif: max w.T @ mu_BL - (delta/2) * w.T @ Sigma_BL @ w
    Gère gracieusement le cas où delta < 0 (filtre sélectif de signaux)
    en gardant la pénalité de variance ou en forçant une fonction convexe localement si nécessaire.
    (SLSQP le gère naturellement si Sigma_BL est définie positive et l'optimum existe sous contraintes).
    """
    N = len(mu_BL)
    constraints, bounds, w0 = _get_constraints_and_bounds(N)
    
    def objective(w):
        port_ret = w.T @ mu_BL
        port_var = w.T @ Sigma_BL @ w
        return -(port_ret - (delta / 2.0) * port_var)
        
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not res.success:
        warnings.warn(f"BL SLSQP n'a pas convergé: {res.message}. Retour du 1/N.")
        return w0
        
    return res.x

def optimize_minimax_robust(mu_BL: np.ndarray, Sigma_BL: np.ndarray, Omega: np.ndarray, delta: float, kappa_rob: float = 1.0) -> np.ndarray:
    """
    Solveur Robuste Minimax (Hansen-Sargent).
    L'incertitude porte sur la moyenne (mu_BL) via la matrice d'incertitude des vues (Omega).
    Objectif dual exact:
    max w.T @ mu_BL - kappa_rob * sqrt(w.T @ Omega @ w) - (delta/2) * w.T @ Sigma_BL @ w
    """
    N = len(mu_BL)
    constraints, bounds, w0 = _get_constraints_and_bounds(N)
    
    def objective(w):
        port_ret = w.T @ mu_BL
        port_var = w.T @ Sigma_BL @ w
        
        # Pénalité robuste (worst-case shift bound)
        # max(0, ...) pour éviter des NaN lors d'erreurs d'arrondis numériques (sqrt de négatif)
        omega_var = max(0.0, w.T @ Omega @ w)
        robust_penalty = kappa_rob * np.sqrt(omega_var)
        
        return -(port_ret - robust_penalty - (delta / 2.0) * port_var)
        
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not res.success:
        warnings.warn(f"Minimax Robuste SLSQP n'a pas convergé: {res.message}. Retour du 1/N.")
        return w0
        
    return res.x
