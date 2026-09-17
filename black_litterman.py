import numpy as np
from typing import Tuple
import pandas as pd

def compute_black_litterman_posterior(
    Sigma_LW: np.ndarray, 
    Q: np.ndarray, 
    Omega: np.ndarray, 
    delta: float, 
    tau: float = 0.05, 
    w_mkt: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inférence Bayésienne Multivariée (Black-Litterman).
    Calcule le posterior sans inversion explicite instable en utilisant
    les identités matricielles.
    
    Args:
        Sigma_LW: Matrice de covariance (N x N)
        Q: Vecteur des vues EGARCH-M (N,)
        Omega: Matrice d'incertitude des vues (N x N)
        delta: Aversion au risque scalaire
        tau: Scalaire d'incertitude sur le prior
        w_mkt: Poids du marché (N,). Si None, équipondéré (1/N)
        
    Returns:
        mu_BL: Rendement espéré a posteriori (N,)
        Sigma_BL: Covariance a posteriori de la moyenne (N x N)
        Pi: Prior de marché (N,)
    """
    N = Sigma_LW.shape[0]
    
    # 1. Vecteur de marché équipondéré par défaut
    if w_mkt is None:
        w_mkt = np.ones(N) / N
        
    # 2. Prior d'équilibre Pi
    Pi = delta * Sigma_LW @ w_mkt
    
    # 3. Calculs robustes (sans inverse explicite)
    # L'identité de Woodbury donne : 
    # (tau*Sigma)^-1 + Omega^-1)^-1 = tau*Sigma - tau*Sigma @ (tau*Sigma + Omega)^-1 @ tau*Sigma
    # Et mu_BL = Pi + tau*Sigma @ (tau*Sigma + Omega)^-1 @ (Q - Pi)
    
    tau_Sigma = tau * Sigma_LW
    # Matrice à inverser / résoudre
    M = tau_Sigma + Omega
    
    # Sigma_BL = tau_Sigma - tau_Sigma @ M^-1 @ tau_Sigma
    # On résout M X = tau_Sigma -> X = M^-1 tau_Sigma
    X = np.linalg.solve(M, tau_Sigma)
    Sigma_BL = tau_Sigma - tau_Sigma @ X
    
    # mu_BL = Pi + tau_Sigma @ M^-1 @ (Q - Pi)
    # On résout M Y = (Q - Pi) -> Y = M^-1 (Q - Pi)
    Y = np.linalg.solve(M, Q - Pi)
    mu_BL = Pi + tau_Sigma @ Y
    
    return mu_BL, Sigma_BL, Pi
