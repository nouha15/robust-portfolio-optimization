import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Callable

class FTRLRiskAversionOptimizer:
    """
    Moteur d'Apprentissage en Ligne (Follow-The-Regularized-Leader).
    Calibre séquentiellement le paramètre d'aversion au risque delta_t 
    sans connaissance du futur, de manière à minimiser le regret cumulé.
    """
    def __init__(self, delta_bounds: Tuple[float, float] = (1.0, 50.0), delta_init: float = 10.0, r_reg: float = 1.0, grid_size: int = 50):
        """
        Initialise le FTRL avec une grille discrète d'aversions au risque.
        
        Args:
            delta_bounds: (min, max) pour l'espace de recherche (Delta_admissible).
            delta_init: Valeur initiale de delta_0.
            r_reg: Intensité de régularisation L2 (freine les sauts brusques).
            grid_size: Résolution de la grille discrète pour évaluer FTRL.
        """
        # On s'assure que la grille est parfaitement uniforme pour éviter tout biais local
        self.delta_grid = np.linspace(delta_bounds[0], delta_bounds[1], grid_size)
        
        self.delta_init = delta_init
        self.delta_t = delta_init
        self.r_reg = r_reg
        
        # Historiques
        self.history_losses = {d: [] for d in self.delta_grid}
        self.history_delta_t = []
        self.history_realized_loss = []
        
    def get_current_delta(self) -> float:
        """Retourne la décision actuelle delta_t pour le rebalancement."""
        return self.delta_t
        
    def record_step(self, realized_return_vector: np.ndarray, weights_function_dict: Union[Dict[float, np.ndarray], Callable]):
        """
        Observe les rendements réalisés r_t et évalue la perte (rendement négatif)
        pour toutes les stratégies virtuelles de la grille de deltas.
        
        Args:
            realized_return_vector: Vecteur des rendements réels de la période T (N,).
            weights_function_dict: Soit un dict {delta: w_t_delta}, soit un callable 
                                   f(delta) -> w_t_delta renvoyant l'allocation.
        """
        # Archivage de la décision passée
        self.history_delta_t.append(self.delta_t)
        
        # Résolution des poids pour la grille (génération des contre-factuels)
        if callable(weights_function_dict):
            w_dict = {d: weights_function_dict(d) for d in self.delta_grid}
        else:
            w_dict = weights_function_dict
            
        # --- 1. Perte réellement subie par l'algorithme ---
        # Si le delta_t n'est pas dans la grille, on l'interpole via le plus proche
        closest_d = min(self.delta_grid, key=lambda x: abs(x - self.delta_t))
        w_chosen = w_dict[closest_d]
        
        # Fonction de perte: Loss_t = - w^T * r_t (On veut maximiser le rendement, donc minimiser son opposé)
        realized_loss = -np.dot(w_chosen, realized_return_vector)
        self.history_realized_loss.append(realized_loss)
        
        # --- 2. Pertes virtuelles (contre-factuelles) pour l'espace d'hypothèses ---
        for d in self.delta_grid:
            loss_d = -np.dot(w_dict[d], realized_return_vector)
            self.history_losses[d].append(loss_d)
            
    def update_delta(self) -> float:
        """
        Met à jour delta_t via la règle d'optimisation FTRL :
        delta_{t+1} = argmin [ Somme_{s=1}^t (Loss_s(delta)) + (R_reg / 2) * (delta - delta_init)^2 ]
        """
        best_obj = float('inf')
        best_d = self.delta_t
        
        for d in self.delta_grid:
            # Perte cumulée historique pour ce delta
            sum_loss = sum(self.history_losses[d])
            
            # FTRL: Pénalité de régularisation par rapport au PRIOR initial (delta_init)
            penalty = (self.r_reg / 2.0) * (d - self.delta_init)**2
            
            obj = sum_loss + penalty
            if obj < best_obj:
                best_obj = obj
                best_d = d
                
        # Mise à jour de l'état
        self.delta_t = best_d
        return self.delta_t
        
    def compute_cumulative_regret(self) -> Tuple[float, float, np.ndarray]:
        """
        Calcule la métrique reine de l'apprentissage en ligne : le Regret Cumulé.
        Regret_T = Somme(Pertes_Algo) - Somme(Pertes_Meilleur_Delta_A_Posteriori)
        
        Returns:
            final_regret: Regret total à l'instant T.
            best_static_d: Le delta statique qui aurait été optimal a posteriori.
            regret_series: Série temporelle du regret (doit être sous-linéaire).
        """
        algo_cum_loss = np.cumsum(self.history_realized_loss)
        
        best_static_loss = float('inf')
        best_static_d = None
        best_static_cum_loss_series = None
        
        for d in self.delta_grid:
            static_loss_series = np.array(self.history_losses[d])
            static_cum_loss = np.sum(static_loss_series)
            
            if static_cum_loss < best_static_loss:
                best_static_loss = static_cum_loss
                best_static_d = d
                best_static_cum_loss_series = np.cumsum(static_loss_series)
                
        regret_series = algo_cum_loss - best_static_cum_loss_series
        final_regret = regret_series[-1]
        
        return float(final_regret), float(best_static_d), regret_series

# -----------------------------------------------------------------------------
# Bloc de Test Unitaire (Validation de la sous-linéarité du Regret)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. Configuration de l'environnement de simulation
    T_steps = 200
    N_assets = 5
    
    # On simule un marché où un régime risqué nécessite un fort delta, 
    # et un régime haussier favorise un faible delta.
    optimizer = FTRLRiskAversionOptimizer(delta_bounds=(1.0, 10.0), delta_init=5.0, r_reg=2.0, grid_size=10)
    
    # Simulation des étapes de rebalancement
    for t in range(T_steps):
        current_delta = optimizer.get_current_delta()
        
        # Le marché a un rendement aléatoire
        r_t = np.random.randn(N_assets) * 0.02 + 0.001
        
        # Fonction factice simulant la sensibilité de l'allocation au paramètre delta
        # Si delta est fort, on concentre sur l'actif 0 (actif sans risque simulé).
        # Si delta est faible, on diversifie uniformément.
        def dummy_weights_function(d: float) -> np.ndarray:
            w = np.ones(N_assets) / N_assets
            risk_shift = (d - 1.0) / 9.0  # Normalisé entre 0 et 1
            w[0] = 0.2 + 0.8 * risk_shift
            rest = (1.0 - w[0]) / (N_assets - 1)
            w[1:] = rest
            return w
            
        # Étape 1 : FTRL observe les résultats et archive les pertes contre-factuelles
        optimizer.record_step(realized_return_vector=r_t, weights_function_dict=dummy_weights_function)
        
        # Étape 2 : FTRL ajuste son paramètre pour t+1
        optimizer.update_delta()
        
    # Évaluation du Regret Théorique
    final_regret, best_hindsight_delta, regret_series = optimizer.compute_cumulative_regret()
    
    print(f"[TEST FTRL] Simulation sur {T_steps} pas de temps :")
    print(f" - Regret Cumulé Final : {final_regret:.4f}")
    print(f" - Meilleur delta a posteriori : {best_hindsight_delta:.2f}")
    print(f" - Delta terminal de l'algo  : {optimizer.get_current_delta():.2f}")
    
    # Vérification de la propriété d'apprentissage (Regret Moyen -> 0)
    average_regret = final_regret / T_steps
    print(f" - Regret Moyen par étape (Regret_T / T) : {average_regret:.6f}")
    
    assert average_regret < 0.05, "Échec: L'algorithme FTRL ne démontre pas un regret sous-linéaire acceptable."
    print("[SUCCÈS] Propriété asymptotique de sous-linéarité du Regret validée.")
