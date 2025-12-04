
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Ruta al archivo Excel dentro del repositorio
DATA_PATH = os.path.join("data", "customer_lookalike_raw_100k.xlsx")


# ============================================================
# Funciones de limpieza
# ============================================================

def parse_age_from_dirty(col: pd.Series) -> pd.Series:
    """Convierte age_str_dirty a numérico cuando sea necesario."""
    def _parse(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x in ["", "N/A", "na", "NA", "None"]:
            return np.nan
        x = x.replace("years", "").strip()
        try:
            return float(x)
        except ValueError:
            return np.nan
    return col.apply(_parse)


def parse_total_spent(col: pd.Series) -> pd.Series:
    """Convierte total_spent_dirty a float (quita $, comas, etc.)."""
    def _parse(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x == "":
            return np.nan
        x = x.replace("$", "").replace(",", "")
        try:
            return float(x)
        except ValueError:
            return np.nan
    return col.apply(_parse)


def parse_pct(col: pd.Series) -> pd.Series:
    """Convierte porcentajes tipo '45.2%' o 45.2 a proporción [0,1]."""
    def _parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        x = str(x).strip()
        if x.endswith("%"):
            try:
                return float(x[:-1]) / 100.0
            except ValueError:
                return np.nan
        try:
            val = float(x)
            if val > 1:
                val = val / 100.0
            return val
        except ValueError:
            return np.nan
    return col.apply(_parse)


def clean_region(col: pd.Series) -> pd.Series:
    """Limpia región sucia: espacios, minúsculas, typo 'Nortee'."""
    def _clean(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x.lower() == "nortee":
            return "Norte"
        return x.title()
    return col.apply(_clean)


def parse_days_since_last_purchase(col: pd.Series) -> pd.Series:
    """Convierte días desde última compra, manejando 'never' y valores extremos."""
    def _parse(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            val = int(x)
            if val > 999:
                val = 999
            return val
        x = str(x).strip().lower()
        if x == "never":
            return 999
        try:
            val = int(float(x))
            if val > 999:
                val = 999
            return val
        except ValueError:
            return np.nan
    return col.apply(_parse)


# ============================================================
# Feature engineering y matriz de features
# ============================================================

def build_feature_matrix(df: pd.DataFrame):
    """
    Limpia columnas sucias, crea columnas *_clean y construye:
    - df limpio
    - X_scaled (features normalizadas)
    - nombres de features, scaler y encoder
    """
    df = df.copy()

    # Edad limpia
    df["age_clean"] = df["age"]
    mask_age_missing = df["age_clean"].isna()
    if "age_str_dirty" in df.columns:
        df.loc[mask_age_missing, "age_clean"] = parse_age_from_dirty(
            df.loc[mask_age_missing, "age_str_dirty"]
        )

    # Total gastado limpio
    df["total_spent"] = parse_total_spent(df["total_spent_dirty"])

    # Porcentaje reordered limpio
    df["pct_reordered_clean"] = parse_pct(df["pct_reordered_dirty"])

    # Región limpia
    df["region_clean"] = clean_region(df["region_dirty"])

    # Días desde última compra limpio
    df["days_since_last_purchase_clean"] = parse_days_since_last_purchase(
        df["days_since_last_purchase_dirty"]
    )

    # Rellenar NA numéricos
    numeric_cols_to_fill = [
        "age_clean",
        "total_spent",
        "pct_reordered_clean",
        "days_since_last_purchase_clean",
        "income",
        "tenure_months",
        "total_orders",
        "avg_items_per_order",
    ]
    for col in numeric_cols_to_fill:
        df[col] = df[col].fillna(df[col].median())

    # Rellenar categóricas
    cat_cols_to_fill = ["gender", "region_clean", "fav_category", "signup_channel"]
    for col in cat_cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Features numéricas y categóricas
    num_features = [
        "age_clean",
        "income",
        "tenure_months",
        "total_orders",
        "days_since_last_purchase_clean",
        "pct_reordered_clean",
        "avg_items_per_order",
        "total_spent",
    ]
    cat_features = ["region_clean", "fav_category", "signup_channel"]

    X_num = df[num_features].copy()

    # One-hot encoding
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(df[cat_features])
    ohe_feature_names = ohe.get_feature_names_out(cat_features)

    # Matriz final de features
    X = np.hstack([X_num.values, X_cat])
    feature_names = num_features + list(ohe_feature_names)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, feature_names, scaler, ohe


# ============================================================
# Modelo kNN y función de lookalikes
# ============================================================

def train_knn(X_scaled: np.ndarray,
              n_neighbors: int = 11) -> NearestNeighbors:
    """Entrena un modelo NearestNeighbors sobre X_scaled."""
    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1
    )
    knn.fit(X_scaled)
    return knn


def find_lookalikes(customer_id: str,
                    k: int,
                    df: pd.DataFrame,
                    X_scaled: np.ndarray,
                    knn: NearestNeighbors) -> pd.DataFrame:
    """
    Devuelve un DataFrame con los k clientes más similares a customer_id,
    incluyendo su distancia en el espacio de features.
    """
    if customer_id not in df["customer_id"].values:
        raise ValueError(f"customer_id {customer_id} no encontrado en el DataFrame.")

    idx = df.index[df["customer_id"] == customer_id][0]

    distances, indices = knn.kneighbors(
        X_scaled[idx].reshape(1, -1),
        n_neighbors=k + 1  # incluye al propio cliente
    )

    neighbor_indices = indices[0]
    neighbor_distances = distances[0]

    # Quitamos al propio cliente
    mask_not_self = neighbor_indices != idx
    neighbor_indices = neighbor_indices[mask_not_self][:k]
    neighbor_distances = neighbor_distances[mask_not_self][:k]

    result = df.iloc[neighbor_indices].copy()
    cols = [
        "customer_id",
        "vip_flag",
        "income",
        "total_orders",
        "region_clean",
        "fav_category",
        "total_spent",
    ]
    existing_cols = [c for c in cols if c in result.columns]
    result = result[existing_cols]
    result["distance"] = neighbor_distances

    return result


# ============================================================
# Visualizaciones (EDA + PCA)
# ============================================================

def plot_basic_distributions(df: pd.DataFrame):
    important_features = [
        "income",
        "total_orders",
        "total_spent",
        "days_since_last_purchase_clean",
    ]
    for col in important_features:
        if col not in df.columns:
            continue
        plt.figure()
        df[col].hist(bins=50)
        plt.title(f"Distribución de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()


def plot_pca_2d(X_scaled: np.ndarray, df: pd.DataFrame):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame({
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
        "vip_flag": df["vip_flag"].values,
    })

    plt.figure(figsize=(8, 6))
    mask_vip = df_pca["vip_flag"] == 1

    plt.scatter(
        df_pca.loc[~mask_vip, "pc1"],
        df_pca.loc[~mask_vip, "pc2"],
        alpha=0.3,
        s=5,
        label="No VIP",
    )
    plt.scatter(
        df_pca.loc[mask_vip, "pc1"],
        df_pca.loc[mask_vip, "pc2"],
        alpha=0.6,
        s=8,
        label="VIP",
    )

    plt.title("Proyección PCA de clientes (VIP vs No VIP)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main: pipeline completo
# ============================================================

def main():
    # 1) Validar que exista el Excel
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en {DATA_PATH}. "
            "Asegúrate de subir customer_lookalike_raw_100k.xlsx a la carpeta data/ del repo."
        )

    # 2) Cargar datos
    df_raw = pd.read_excel(DATA_PATH)
    print("Shape del dataset:", df_raw.shape)
    print(df_raw.head())

    # 3) EDA rápido
    print("\n=== Descripción numérica ===")
    cols_desc = [
        "age",
        "income",
        "tenure_months",
        "total_orders",
        "days_since_last_purchase",
        "pct_reordered",
        "avg_items_per_order",
        "vip_score",
    ]
    print(df_raw[cols_desc].describe())

    print("\n=== Distribución de VIP flag ===")
    print(df_raw["vip_flag"].value_counts(normalize=True))

    print("\n=== Clientes por región (cruda) ===")
    print(df_raw["region"].value_counts().head())

    print("\n=== Clientes por categoría favorita ===")
    print(df_raw["fav_category"].value_counts().head())

    # 4) Features y escalado
    df_clean, X_scaled, feature_names, scaler, ohe = build_feature_matrix(df_raw)
    print("\nMatriz de features escaladas:", X_scaled.shape)
    print("Número de features:", len(feature_names))

    # 5) Entrenar kNN
    knn = train_knn(X_scaled, n_neighbors=11)
    print("Modelo kNN entrenado.")

    # 6) Ejemplo: lookalikes para un cliente VIP
    vip_customers = df_clean[df_clean["vip_flag"] == 1]
    if len(vip_customers) > 0:
        example_vip = vip_customers.sample(1, random_state=42)
        vip_id = example_vip["customer_id"].iloc[0]
        print(f"\nEjemplo de cliente VIP: {vip_id}")

        lookalikes_df = find_lookalikes(
            customer_id=vip_id,
            k=5,
            df=df_clean,
            X_scaled=X_scaled,
            knn=knn,
        )
        print(f"\nLookalikes para el cliente {vip_id}:")
        print(lookalikes_df)
    else:
        print("No se encontraron clientes VIP en el dataset.")

    # 7) Visualizaciones
    plot_basic_distributions(df_clean)
    plot_pca_2d(X_scaled, df_clean)

    # 8) Ejemplo adicional: lookalikes para cualquier cliente
    example_any_id = df_clean["customer_id"].iloc[123]
    print(f"\nEjemplo adicional: lookalikes para {example_any_id}")
    print(
        find_lookalikes(
            customer_id=example_any_id,
            k=10,
            df=df_clean,
            X_scaled=X_scaled,
            knn=knn,
        )
    )

    # Variables globales útiles si importas este módulo desde otro lado
    global df_global, X_scaled_global, knn_global
    df_global = df_clean
    X_scaled_global = X_scaled
    knn_global = knn


if __name__ == "__main__":
    main()
