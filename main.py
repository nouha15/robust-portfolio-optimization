import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Importer tous les modules construits
from data_pipeline import load_and_prepare_data
from covariance import compute_ledoit_wolf_covariance
from views_egarch import generate_egarch_views
from black_litterman import compute_black_litterman_posterior
from optimizers import optimize_markowitz, optimize_black_litterman, optimize_minimax_robust
from pac_bayes import compute_kl_divergence, compute_pac_bayes_bounds
from online_ftrl import FTRLRiskAversionOptimizer
from metrics import compute_financial_metrics, calculate_turnover_and_costs, block_bootstrap_stability

def main():
    print("=========================================================")
    print("BACKTEST GLOBAL : OPTIMISATION DE PORTEFEUILLE ROBUSTE")
    print("=========================================================\n")
    
    # 1. Création du dossier de résultats
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # 2. Chargement des données MASI (via data_pipeline en mémoire)
    # df_train, df_val, df_test = load_and_prepare_data()
    # Pour un backtest temporel fluide, on concatène tout pour générer l'historique glissant
    try:
        df_train, df_val, df_test = load_and_prepare_data()
        df_full = pd.concat([df_train, df_val, df_test])
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return
        
    if df_full.empty:
        print("Erreur: Les données MASI n'ont pas pu être chargées.")
        return
        
    # Paramètres de simulation
    WINDOW_SIZE = 756 # 3 ans d'historique (In-Sample)
    REBAL_FREQ = 21   # Rebalancement mensuel (~21 jours ouvrés)
    N_ASSETS = df_full.shape[1]
    
    # Indices pour le parcours glissant
    indices = np.arange(WINDOW_SIZE, len(df_full), REBAL_FREQ)
    rebal_dates = df_full.index[indices]
    
    print(f"\n[INFO] Lancement du backtest sur {len(rebal_dates)} rebalancements (Pas = {REBAL_FREQ} jours)")
    
    # 3. Initialisation des stratégies à comparer
    # Dict pour stocker les allocations choisies: date -> w
    weights_history = {
        '1/N': [],
        'Markowitz': [],
        'BL_d18': [],
        'BL_d-5.5': [],
        'Minimax': [],
        'FTRL': []
    }
    
    # Moteur FTRL pour la stratégie adaptative
    # delta_init=18.0 est inclus nativement dans linspace(1.0, 50.0, 50) avec un pas exact de 1.0
    ftrl_optimizer = FTRLRiskAversionOptimizer(delta_bounds=(1.0, 50.0), delta_init=18.0, r_reg=1e-5, grid_size=50)
    
    # Tracking PAC-Bayes et FTRL
    tracking_data = {'date': [], 'delta_ftrl': [], 'kl_div': [], 'pac_mcallester': []}
    
    # 4. Boucle Principale
    for i, idx in enumerate(tqdm(indices, desc="Rebalancement", unit="mois")):
        date = df_full.index[idx]
        
        # Fenêtre glissante
        window_returns = df_full.iloc[idx - WINDOW_SIZE : idx]
        
        # Rendements de la Période OOS qui va suivre (pour mettre à jour le FTRL)
        # On regarde 21 jours en avant (ou jusqu'à la fin)
        next_idx = min(idx + REBAL_FREQ, len(df_full))
        oos_returns_matrix = df_full.iloc[idx : next_idx]
        # Rendement composé sur la période pour évaluer l'allocation
        oos_period_return = np.exp(oos_returns_matrix.sum()) - 1.0
        
        # --- Modélisation Quantitative ---
        # Covariance
        Sigma_LW, _ = compute_ledoit_wolf_covariance(window_returns)
        
        # Vues
        Q, Omega = generate_egarch_views(window_returns)
        
        # Espérance Historique Moyenne (pour Markowitz)
        mu_hist = window_returns.mean().values * 252 # Annualisé ou sur la fenêtre. On laisse brut pour cohérence d'échelle
        mu_hist = window_returns.mean().values # Echelle journalière ou total return
        
        # --- Générations des Allocations ---
        # 1/N
        w_eq = np.ones(N_ASSETS) / N_ASSETS
        weights_history['1/N'].append(w_eq)
        
        # Markowitz (delta = 18 standard)
        w_mark = optimize_markowitz(mu_hist, Sigma_LW, delta=18.0)
        weights_history['Markowitz'].append(w_mark)
        
        # Black-Litterman statiques
        mu_BL_18, Sigma_BL_18, Pi_18 = compute_black_litterman_posterior(Sigma_LW, Q, Omega, delta=18.0)
        w_bl_18 = optimize_black_litterman(mu_BL_18, Sigma_BL_18, delta=18.0)
        weights_history['BL_d18'].append(w_bl_18)
        
        mu_BL_neg, Sigma_BL_neg, Pi_neg = compute_black_litterman_posterior(Sigma_LW, Q, Omega, delta=-5.5)
        w_bl_neg = optimize_black_litterman(mu_BL_neg, Sigma_BL_neg, delta=-5.5)
        weights_history['BL_d-5.5'].append(w_bl_neg)
        
        # Minimax Robuste
        w_minimax = optimize_minimax_robust(mu_BL_18, Sigma_BL_18, Omega, delta=18.0, kappa_rob=1.0)
        weights_history['Minimax'].append(w_minimax)
        
        # --- Apprentissage en Ligne (FTRL) ---
        delta_ftrl = ftrl_optimizer.get_current_delta()
        mu_BL_ftrl, Sigma_BL_ftrl, Pi_ftrl = compute_black_litterman_posterior(Sigma_LW, Q, Omega, delta=delta_ftrl)
        w_ftrl = optimize_black_litterman(mu_BL_ftrl, Sigma_BL_ftrl, delta=delta_ftrl)
        weights_history['FTRL'].append(w_ftrl)
        
        # Mettre à jour FTRL en fonction des rendements OOS qui viennent de se réaliser
        # Le FTRL a besoin d'une fonction simulant l'allocation pour n'importe quel delta
        def ftrl_weight_function(d: float):
            mu_d, Sigma_d, _ = compute_black_litterman_posterior(Sigma_LW, Q, Omega, delta=d)
            return optimize_black_litterman(mu_d, Sigma_d, delta=d)
            
        ftrl_optimizer.record_step(oos_period_return.values, ftrl_weight_function)
        ftrl_optimizer.update_delta() # Prépare delta pour le prochain mois
        
        # --- Mesures PAC-Bayes ---
        kl_div = compute_kl_divergence(mu_BL_18, Sigma_BL_18, Pi_18, Sigma_LW, tau=0.05)
        # Erreur empirique in-sample approximative pour la borne
        is_port_returns = np.dot(window_returns.values, w_bl_18)
        is_loss = max(0.0, 1.0 - np.exp(np.sum(is_port_returns))) # Prox d'erreur bornée [0,1]
        bounds = compute_pac_bayes_bounds(empirical_loss=is_loss, kl_div=kl_div, m=WINDOW_SIZE)
        
        tracking_data['date'].append(date)
        tracking_data['delta_ftrl'].append(delta_ftrl)
        tracking_data['kl_div'].append(kl_div)
        tracking_data['pac_mcallester'].append(bounds['mcallester_bound'])
        
    # 5. Conversion en DataFrames d'allocations
    dfs_weights = {}
    for strategy, w_list in weights_history.items():
        dfs_weights[strategy] = pd.DataFrame(w_list, index=rebal_dates, columns=df_full.columns)
        
    df_tracking = pd.DataFrame(tracking_data).set_index('date')
    
    # Identification de la dynamique exacte de delta_ftrl
    print("\n[ANALYSE FTRL] Séquence complète des 84 valeurs de delta_t :")
    print(df_tracking['delta_ftrl'].values.tolist())
    
    # Identification du pic KL aberrant
    max_kl_idx = df_tracking['kl_div'].idxmax()
    max_kl_val = df_tracking.loc[max_kl_idx, 'kl_div']
    print(f"\n[ANALYSE PAC-BAYES] Pic de divergence KL détecté le : {max_kl_idx.date()} (Valeur : {max_kl_val:.2f})")
    
    
    # 6. Évaluation post-trade STRICTE OOS (Coûts de transaction et Métriques)
    print("\n[INFO] Application du filtre STRICT Out-Of-Sample (>= 2023-01-01) et du modèle de coûts (20 bps)...")
    
    OOS_START_DATE = pd.to_datetime('2023-01-01')
    
    results_summary = []
    daily_net_curves = {}
    
    for strategy, w_df in dfs_weights.items():
        # Filtre OOS pour les poids
        w_df_oos = w_df[w_df.index >= OOS_START_DATE]
        if w_df_oos.empty:
            continue
            
        # Rendements Nets et Turnover STRICTEMENT sur OOS
        net_returns_oos, turnover_oos = calculate_turnover_and_costs(w_df_oos, df_full, kappa_cost=0.0020)
        daily_net_curves[strategy] = net_returns_oos
        
        # Métriques Financières
        metrics = compute_financial_metrics(net_returns_oos)
        metrics['Strategy'] = strategy
        metrics['Turnover_%'] = turnover_oos * 100
        results_summary.append(metrics)
        
    df_results = pd.DataFrame(results_summary).set_index('Strategy')
    # Ordonner colonnes
    df_results = df_results[['Ann_Ret', 'Ann_Vol', 'Sharpe', 'Sortino', 'Max_DD', 'Turnover_%']]
    
    # 7. Affichage Console (Tableau 1 de la thèse)
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF DES PERFORMANCES (OOS, Nets de frais)")
    print("="*80)
    # Formatage propre
    df_display = df_results.copy()
    df_display['Ann_Ret'] = (df_display['Ann_Ret'] * 100).map("{:.2f}%".format)
    df_display['Ann_Vol'] = (df_display['Ann_Vol'] * 100).map("{:.2f}%".format)
    df_display['Sharpe'] = df_display['Sharpe'].map("{:.3f}".format)
    df_display['Sortino'] = df_display['Sortino'].map("{:.3f}".format)
    df_display['Max_DD'] = (df_display['Max_DD'] * 100).map("{:.2f}%".format)
    df_display['Turnover_%'] = df_display['Turnover_%'].map("{:.2f}%".format)
    print(df_display.to_string())
    print("="*80)
    
    # 8. Bootstrap Diagnostic sur le régime spéculatif
    print("\n[DIAGNOSTIC] Lancement du Block Bootstrap sur BL (delta = -5.5)...")
    boot_stats = block_bootstrap_stability(daily_net_curves['BL_d-5.5'])
    print(f"  -> Mean Sharpe: {boot_stats['Mean_Sharpe']:.3f} | 90% CI: [{boot_stats['CI_5']:.3f}, {boot_stats['CI_95']:.3f}]")
    if boot_stats['CI_5'] < 0:
        print("  -> ALERTE: L'intervalle inclut zéro, signe d'instabilité structurelle (Overfitting confirmant la Section 8.5.2).")
    
    # Filtrer le tracking PAC-Bayes et les dates de rebalancement pour l'OOS pur
    df_tracking_oos = df_tracking[df_tracking.index >= OOS_START_DATE]
    rebal_dates_oos = df_tracking_oos.index

    # 9. Génération des Graphiques
    print("\n[INFO] Génération des graphiques dans 'results/'...")
    
    # A. Wealth Curves
    plt.figure(figsize=(12, 6))
    for strat, rets in daily_net_curves.items():
        wealth = (1.0 + rets).cumprod()
        plt.plot(wealth.index, wealth.values, label=strat, linewidth=1.5 if strat != 'FTRL' else 2.5)
    plt.title("Évolution de la Richesse Cumulée (OOS Net de frais, 2023-2026)")
    plt.ylabel("Richesse (Base 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/wealth_curves.png", dpi=300)
    
    # B. Stacked Area pour FTRL (Stabilité d'allocation)
    plt.figure(figsize=(12, 6))
    w_ftrl_plot = dfs_weights['FTRL'][dfs_weights['FTRL'].index >= OOS_START_DATE].copy()
    plt.stackplot(w_ftrl_plot.index, w_ftrl_plot.T, labels=w_ftrl_plot.columns)
    plt.title("Allocations Dynamiques FTRL (OOS)")
    plt.ylabel("Poids")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("results/allocations_weights.png", dpi=300)
    
    # Évaluation du Regret (STRICT OOS)
    # Les steps OOS correspondent aux index de df_tracking_oos par rapport à la liste complète
    oos_start_idx = len(df_tracking) - len(df_tracking_oos)
    
    if len(df_tracking_oos) > 0:
        realized_loss_oos = ftrl_optimizer.history_realized_loss[oos_start_idx:]
        cum_realized_oos = np.cumsum(realized_loss_oos)
        
        # Meilleur delta statique sur la période OOS
        best_delta_oos = 18.0
        best_cum_loss_oos = float('inf')
        
        for d in ftrl_optimizer.history_losses.keys():
            losses_oos = ftrl_optimizer.history_losses[d][oos_start_idx:]
            c_loss = np.sum(losses_oos)
            if c_loss < best_cum_loss_oos:
                best_cum_loss_oos = c_loss
                best_delta_oos = d
                
        # Calcul de la série de regret OOS
        best_loss_series_oos = ftrl_optimizer.history_losses[best_delta_oos][oos_start_idx:]
        regret_series_oos = cum_realized_oos - np.cumsum(best_loss_series_oos)
        final_regret_oos = regret_series_oos[-1]
    else:
        final_regret_oos, best_delta_oos, regret_series_oos = 0.0, 18.0, []
        
    print(f"\n[REGRET STRICT OOS] Final Regret: {final_regret_oos:.6f}, Best Hindsight Delta: {best_delta_oos}")
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Delta Adaptatif (FTRL)', color=color)
    ax1.plot(df_tracking_oos.index, df_tracking_oos['delta_ftrl'], color=color, linewidth=2, label='$\delta_t$')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Regret Cumulé OOS', color=color)
    ax2.plot(rebal_dates_oos, regret_series_oos, color=color, linestyle='--', label='Regret vs Meilleur Statique')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title(f"Dynamique FTRL OOS (Regret final: {final_regret_oos:.4f}, Meilleur $\delta^*$={best_delta_oos:.2f})")
    plt.savefig("results/ftrl_regret_and_delta.png", dpi=300)
    
    # D. PAC-Bayes Bounds
    plt.figure(figsize=(10, 5))
    plt.plot(df_tracking_oos.index, df_tracking_oos['kl_div'], label='Divergence KL(Q||P)', color='purple')
    plt.plot(df_tracking_oos.index, df_tracking_oos['pac_mcallester'], label='Borne McAllester', color='orange', linestyle=':')
    plt.title("Complexité Informationnelle (PAC-Bayes) OOS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/pac_bayes_bounds.png", dpi=300)
    
    print("[SUCCÈS] Exécution terminée ! Les graphiques sont disponibles dans le dossier 'results/'.")

if __name__ == "__main__":
    main()
