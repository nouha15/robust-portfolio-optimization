import os
import glob
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Configuration
K_DIV = 0.03
N_JOURS = 252

# Dictionnaire de correspondance Investing.com -> Ticker court
TICKERS_MASI = {
    'attijariwafa': 'ATW',
    'boa': 'BOA',
    'cma': 'CMA',
    'csr': 'CSR',
    'addoha': 'ADH',
    'itissalat': 'IAM',
    'mng': 'MNG',
    'exploitation': 'MSA',
    'sot': 'SOT',
    'wafa': 'WAA'
}

def parse_investing_number(val) -> float:
    """Convertit un format texte d'Investing.com en float."""
    if pd.isna(val) or val == '-':
        return np.nan
        
    val = str(val).replace(',', '')
    multiplier = 1
    if val.endswith('K'):
        multiplier = 1e3
        val = val[:-1]
    elif val.endswith('M'):
        multiplier = 1e6
        val = val[:-1]
    elif val.endswith('B'):
        multiplier = 1e9
        val = val[:-1]
    elif val.endswith('%'):
        multiplier = 0.01
        val = val[:-1]
        
    try:
        return float(val) * multiplier
    except ValueError:
        return np.nan

def load_local_csvs(data_dir: str = "data_masi") -> pd.DataFrame:
    """
    Force la lecture des 10 historiques MASI à partir de fichiers CSV locaux
    sans aucun basculement synthétique.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Le dossier {data_dir} n'existe pas. Veuillez le créer et y déposer les CSV.")
        
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"Aucun fichier CSV trouvé dans le dossier {data_dir}.")
        
    all_prices = {}
    print(f"[INFO] Lecture forcée des fichiers CSV locaux depuis le dossier '{data_dir}'...")
    
    for file in csv_files:
        filename = os.path.basename(file).lower()
        
        # Reconnaissance automatique du ticker par mots-clés
        ticker = None
        for key, val in TICKERS_MASI.items():
            if key in filename:
                ticker = val
                break
                
        if not ticker:
            print(f"  [Ignoré] Fichier non reconnu : {filename}")
            continue
            
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Repérage de la colonne de prix ('Price' ou 'Dernier')
            price_col = 'Price' if 'Price' in df.columns else ('Dernier' if 'Dernier' in df.columns else df.columns[0])
            
            # Nettoyage
            df[price_col] = df[price_col].astype(str).apply(parse_investing_number)
            prices = df[price_col].sort_index()
            
            # Vérification de doublons de date potentiels
            prices = prices[~prices.index.duplicated(keep='last')]
            
            all_prices[ticker] = prices
            obs = len(prices)
            status = "OK" if obs > 2000 else "ATTENTION (< 2000 obs)"
            print(f"  - Chargement réussi : {ticker} ({obs} observations) [{status}]")
            
        except Exception as e:
            print(f"  [Erreur] Échec de la lecture de {filename} : {e}")
            
    if not all_prices:
        raise ValueError("Aucun fichier valide n'a pu être extrait. Vérifiez les noms et le format.")
        
    df_combined = pd.DataFrame(all_prices)
    
    # --- CORRECTION DES DONNÉES CORROMPUES ---
    # Détection des baisses/hausses > 40% (impossible au Maroc hors split non ajusté)
    for col in df_combined.columns:
        pct_change = df_combined[col].pct_change()
        corrupt_mask = (pct_change < -0.40) | (pct_change > 0.40)
        if corrupt_mask.any():
            # Remplacer les prix aberrants par NaN pour qu'ils soient interpolés
            df_combined.loc[corrupt_mask, col] = np.nan
    
    # Remplissage des jours fériés/sans cotation et des erreurs interpolées
    df_combined = df_combined.ffill().dropna()
    print(f"[SUCCÈS] Matrice finale de prix alignée (Intersection stricte) : {df_combined.shape}")
    
    return df_combined

def calculate_adjusted_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le rendement Total Return (arithmétique) :
    R_t = (P_t / P_{t-1}) - 1 + (0.03 / 252)
    L'agrégation linéaire w^T R_t n'est mathématiquement valide que sur les rendements simples.
    """
    simple_returns = df_prices.pct_change()
    adj_factor = K_DIV / N_JOURS
    total_returns = simple_returns + adj_factor
    return total_returns.dropna()

def load_and_prepare_data(data_dir: str = "data_masi") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestre le chargement local, l'ajustement kappa_div et le découpage temporel.
    """
    # 1. Chargement EXCLUSIF des CSV locaux
    df_prices = load_local_csvs(data_dir)
    
    # 2. Calcul des rendements nets + Total Return (kappa_div = 3%)
    df_returns = calculate_adjusted_returns(df_prices)
    
    # 3. Découpage temporel out-of-sample strict
    # Warm-up initial : 2015-01-01 à 2018-12-31
    mask_train = (df_returns.index >= '2015-01-01') & (df_returns.index <= '2018-12-31')
    df_returns_train = df_returns.loc[mask_train]
    
    # Calibration (in-sample) : 2019-01-01 à 2022-12-31
    mask_val = (df_returns.index >= '2019-01-01') & (df_returns.index <= '2022-12-31')
    df_returns_val = df_returns.loc[mask_val]
    
    # Évaluation aveugle (out-of-sample pur) : 2023-01-01 à 2026-09-14
    mask_test = (df_returns.index >= '2023-01-01') & (df_returns.index <= '2026-09-14')
    df_returns_test = df_returns.loc[mask_test]
    
    print(f"\n[DÉCOUPAGE] Structure temporelle :")
    print(f" - Train (Warm-up)   : {df_returns_train.shape[0]} jours de bourse")
    print(f" - Validation (Cal.) : {df_returns_val.shape[0]} jours de bourse")
    print(f" - Test (OOS)        : {df_returns_test.shape[0]} jours de bourse")
    
    return df_returns_train, df_returns_val, df_returns_test

if __name__ == "__main__":
    df_train, df_val, df_test = load_and_prepare_data()
