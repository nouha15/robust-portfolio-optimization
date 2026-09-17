import numpy as np
import pandas as pd
from typing import Tuple

def compute_ledoit_wolf_covariance(returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Implémente l'estimateur de matrice de covariance régularisée de Ledoit & Wolf (2004)
    "Honey, I Shrunk the Sample Covariance Matrix" avec une cible à corrélation constante.
    
    Args:
        returns: DataFrame des rendements historiques de taille (T, N)
        
    Returns:
        Sigma_LW: Matrice de covariance régularisée (N x N)
        lambda_star: Le coefficient de shrinkage optimal estimé (scalaire)
    """
    X = returns.values
    T, N = X.shape
    
    # Matrice de covariance empirique S
    S = np.cov(X, rowvar=False)
    
    # Si N=1, retourner la variance simple
    if N == 1:
        return S, 0.0
        
    # Calcul de la matrice de corrélation empirique
    # On gère les variances nulles pour éviter les divisions par zéro
    v = np.diag(S)
    std = np.sqrt(np.maximum(v, 1e-8))
    R = S / np.outer(std, std)
    
    # Calcul de la corrélation moyenne r_bar (hors diagonale)
    r_bar = (np.sum(R) - N) / (N * (N - 1))
    
    # Construction de la cible à corrélation constante T_target
    T_target = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                T_target[i, j] = S[i, i]
            else:
                T_target[i, j] = r_bar * std[i] * std[j]
                
    # Calcul non-paramétrique du paramètre de shrinkage lambda_star
    # Formule asymptotique de Ledoit & Wolf
    # Pi_mat: matrice de la variance asymptotique des entrées de S
    X_centered = X - np.mean(X, axis=0)
    
    # Calcul de pi_hat
    pi_hat = 0.0
    for t in range(T):
        x_t = X_centered[t, :]
        # (x_t x_t' - S)^2
        diff = np.outer(x_t, x_t) - S
        pi_hat += np.sum(diff ** 2)
    pi_hat = pi_hat / T
    
    # Calcul de rho_hat
    rho_hat = 0.0
    
    # Diagonale
    for i in range(N):
        rho_hat += pi_hat_ii(X_centered, S, i, i, T)
        
    # Hors diagonale
    for i in range(N):
        for j in range(N):
            if i != j:
                # Approximation de l'élément i, j de la covariance entre S et T_target
                term1 = r_bar * 0.5 * (std[j] / std[i] * pi_hat_ii(X_centered, S, i, i, T) + 
                                       std[i] / std[j] * pi_hat_ii(X_centered, S, j, j, T))
                rho_hat += term1
                
    # Gamma_hat: norme de Frobenius au carré entre S et T_target
    gamma_hat = np.sum((S - T_target) ** 2)
    
    # kappa_hat
    kappa_hat = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    
    # Delta_star
    delta_star = max(0.0, min(kappa_hat / T, 1.0))
    lambda_star = delta_star
    
    # Matrice finale projetée
    Sigma_LW = lambda_star * T_target + (1 - lambda_star) * S
    
    return Sigma_LW, lambda_star

def pi_hat_ii(X_centered, S, i, j, T):
    """Fonction utilitaire pour le terme Pi"""
    res = 0.0
    for t in range(T):
        res += (X_centered[t, i] * X_centered[t, j] - S[i, j]) ** 2
    return res / T
