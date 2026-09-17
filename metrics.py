import numpy as np
import pandas as pd
from typing import Dict, Tuple

def compute_financial_metrics(returns: pd.Series, risk_free_rate: float = 0.03) -> Dict[str, float]:
    """
    Calcule l'ensemble des métriques de performance et de risque.
    
    Args:
        returns: Série pandas des rendements (quotidiens, nets de frais).
        risk_free_rate: Taux sans risque annualisé (3% par défaut).
        
    Returns:
        Dictionnaire des métriques (Annualisé, Volatilité, Sharpe, Sortino, MDD).
    """
    # 1. Rendement et Volatilité Annualisés
    # R_ann = prod(1 + R_t)^(252 / T) - 1
    T = len(returns)
    if T == 0:
        return {'Ann_Ret': 0.0, 'Ann_Vol': 0.0, 'Sharpe': 0.0, 'Sortino': 0.0, 'Max_DD': 0.0}
        
    # La pipeline exporte désormais des rendements arithmétiques corrigés
    cum_returns = (1.0 + returns).cumprod()
    
    total_return = cum_returns.iloc[-1]
    ann_ret = (total_return ** (252.0 / T)) - 1.0 if total_return > 0 else -1.0
    
    # La volatilité se mesure habituellement sur les log-rendements pour la symétrie, ou sur l'arithmétique
    ann_vol = returns.std() * np.sqrt(252)
    
    # 2. Ratio de Sharpe
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 1e-8 else 0.0
    
    # 3. Ratio de Sortino (Volatilité à la baisse uniquement)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (ann_ret - risk_free_rate) / downside_vol if downside_vol > 1e-8 else 0.0
    
    # 4. Maximum Drawdown (MDD) exact
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = drawdowns.min()
    
    return {
        'Ann_Ret': float(ann_ret),
        'Ann_Vol': float(ann_vol),
        'Sharpe': float(sharpe),
        'Sortino': float(sortino),
        'Max_DD': float(max_dd)
    }

def calculate_turnover_and_costs(weights_matrix: pd.DataFrame, returns_matrix: pd.DataFrame, kappa_cost: float = 0.0020) -> Tuple[pd.Series, float]:
    """
    Applique le Modèle de Coûts de Transaction et calcule le Turnover Moyen (Section 8.1.2).
    Cost_k = kappa_cost * sum_i |w_{k,i} - w_{k-1,i}^drifted|
    
    Args:
        weights_matrix: DataFrame (T_rebal x N) des allocations cibles.
        returns_matrix: DataFrame (T_daily x N) des rendements quotidiens des actifs.
        kappa_cost: Taux de frais de transaction (20 bps).
        
    Returns:
        Série temporelle des rendements quotidiens NETS du portefeuille.
        Turnover moyen par rebalancement.
    """
    daily_portfolio_returns = []
    
    # Aligner les indices de rebalancement
    rebal_dates = weights_matrix.index
    
    total_turnover = 0.0
    n_rebals = len(rebal_dates)
    
    w_current = None
    
    # Nous parcourons chaque jour où le portefeuille est investi
    start_date = rebal_dates[0]
    end_date = returns_matrix.index[-1]
    
    active_returns = returns_matrix.loc[start_date:end_date]
    
    # Initialisation de la série de rendements
    net_returns = pd.Series(index=active_returns.index, dtype=float)
    
    for date, r_day in active_returns.iterrows():
        # Si c'est un jour de rebalancement
        if date in weights_matrix.index:
            w_target = weights_matrix.loc[date].values
            
            if w_current is not None:
                # Calcul du turnover: | w_cible - w_drifted |
                turnover = np.sum(np.abs(w_target - w_current))
                total_turnover += turnover
                
                # Le rendement du jour subit le choc des coûts de transaction
                cost = turnover * kappa_cost
            else:
                # Allocation initiale : on paie les frais sur tout le capital
                turnover = 1.0
                cost = 1.0 * kappa_cost
                
            w_current = w_target
            
            # Rendement brut de la journée moins les coûts
            ret_net = np.dot(w_current, r_day.values) - cost
        else:
            # Jour normal sans rebalancement
            ret_net = np.dot(w_current, r_day.values)
            
            # Drift des poids
            w_drifted = w_current * (1.0 + r_day.values)
            sum_drifted = np.sum(w_drifted)
            w_current = w_drifted / sum_drifted if sum_drifted > 0 else w_current
            
        net_returns.loc[date] = ret_net
        
    avg_turnover = total_turnover / (n_rebals - 1) if n_rebals > 1 else 0.0
    
    return net_returns, avg_turnover

def block_bootstrap_stability(returns_series: pd.Series, risk_free_rate: float = 0.03, block_size: int = 21, n_boot: int = 500) -> Dict[str, float]:
    """
    Diagnostic de Surapprentissage (Section 8.5.2) par Block Bootstrap Stationnaire.
    Rééchantillonne les blocs de 21 jours pour obtenir une distribution de Sharpe empirique.
    
    Returns:
        Moyenne, 5e percentile et 95e percentile des ratios de Sharpe bootstrappés.
    """
    n = len(returns_series)
    returns_arr = returns_series.values
    sharpes = []
    
    n_blocks = n // block_size
    
    for _ in range(n_boot):
        # Tirage aléatoire de blocs avec remise
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        boot_sample = np.concatenate([returns_arr[s:s+block_size] for s in starts])
        
        # Calcul du Sharpe
        vol = np.std(boot_sample) * np.sqrt(252)
        if vol > 1e-8:
            mean_ret = np.mean(boot_sample) * 252 # Approximation simple
            sharpes.append((mean_ret - risk_free_rate) / vol)
            
    if not sharpes:
        return {'Mean_Sharpe': 0.0, 'CI_5': 0.0, 'CI_95': 0.0}
        
    return {
        'Mean_Sharpe': np.mean(sharpes),
        'CI_5': np.percentile(sharpes, 5),
        'CI_95': np.percentile(sharpes, 95)
    }
