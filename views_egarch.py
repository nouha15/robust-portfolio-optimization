import warnings
import numpy as np
import pandas as pd
from typing import Tuple
from arch import arch_model

# Désactivation des warnings d'optimisation
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='arch')

def generate_egarch_views(returns_window: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère les vues endogènes Q et l'incertitude Omega via un modèle EGARCH-M(1,1)
    sous loi de Student, avec fallback robuste GARCH(1,1) ou empirique.
    
    L'effet in-mean est estimé par une régression deux étapes (Two-Step OLS) sur la variance
    conditionnelle puisque la librairie `arch` ne l'expose pas nativement dans `arch_model`.
    
    Args:
        returns_window: DataFrame des rendements sur la fenêtre de calibration (T x N)
        
    Returns:
        Q: Vecteur (N,) des rendements projetés à T+1
        Omega: Matrice (N, N) diagonale des incertitudes projetées à T+1
    """
    N = returns_window.shape[1]
    Q = np.zeros(N)
    Omega = np.zeros((N, N))
    
    for i, col in enumerate(returns_window.columns):
        # Mise à l'échelle (x100) recommandée par arch pour la stabilité du gradient
        series = returns_window[col].values * 100.0
        
        # Validation stationnarité et fallback
        mu_fallback = np.mean(series) / 100.0
        var_fallback = np.var(series) / 10000.0
        
        try:
            # 1. Tentative EGARCH(1,1) asymétrique (p=1, o=1, q=1) sous loi de Student
            model = arch_model(series, mean='Constant', vol='EGARCH', p=1, o=1, q=1, dist='t')
            res = model.fit(disp='off', show_warning=False)
            
            # Vérification de la stationnarité / persistance
            beta = res.params.get('beta[1]', 0.0)
            if abs(beta) >= 1.0 or res.conditional_volatility is None:
                raise ValueError("EGARCH Non stationnaire")
                
            # Estimation Two-Step pour le paramètre lambda_m (effet in-mean)
            # r_t = mu + lambda_m * sigma_t^2 + e_t
            cond_var = res.conditional_volatility ** 2
            
            # Régression simple (r - mean) sur cond_var pour isoler lambda_m
            covariance = np.cov(series, cond_var)[0, 1]
            variance_vol = np.var(cond_var)
            lambda_m = covariance / variance_vol if variance_vol > 1e-8 else 0.0
            
            # Projection 1 pas en avant (T+1)
            forecast = res.forecast(horizon=1, align='origin')
            sigma2_t1 = forecast.variance.iloc[-1, 0]
            
            # Re-scaling inverse ( / 100 pour la moyenne, / 10000 pour la variance)
            sigma2_t1_scaled = sigma2_t1 / 10000.0
            
            # Calcul du rendement espéré Q_i
            # res.params['mu'] est déjà la moyenne de base
            mu_i = res.params['mu'] / 100.0
            q_i = mu_i + lambda_m * sigma2_t1_scaled
            
            Q[i] = q_i
            Omega[i, i] = sigma2_t1_scaled
            
        except Exception as e_egarch:
            # 2. Fallback GARCH(1,1) standard sous loi de Student
            try:
                model_fallback = arch_model(series, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
                res_fallback = model_fallback.fit(disp='off', show_warning=False)
                
                beta_garch = res_fallback.params.get('beta[1]', 0.0)
                if abs(beta_garch) >= 1.0:
                    raise ValueError("GARCH Non stationnaire")
                    
                forecast_fb = res_fallback.forecast(horizon=1, align='origin')
                sigma2_t1 = forecast_fb.variance.iloc[-1, 0] / 10000.0
                mu_i = res_fallback.params['mu'] / 100.0
                
                # Pas d'in-mean robuste dans le fallback simple
                Q[i] = mu_i
                Omega[i, i] = sigma2_t1
                
            except Exception as e_garch:
                # 3. Fallback Empirique Strict
                Q[i] = mu_fallback
                # Majoration d'incertitude 1.5x comme spécifié
                Omega[i, i] = var_fallback * 1.5

    return Q, Omega
