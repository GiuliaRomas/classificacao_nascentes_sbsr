# Autora: Giulia Katherine Romas Tomazeli

import geopandas as gpd
from sklearn.model_selection import StratifiedGroupKFold

def load_spatial_groups(shapefile_path, dataset_fids):
    gdf = gpd.read_file(shapefile_path)

    gdf["fid"] = gdf["fid"].astype(str)
    gdf["ugrhi_id"] = gdf["ugrhi_id"].astype(int)

    # fid_12345 -> 12345
    dataset_fids = {str(fid).replace("fid_", "") for fid in dataset_fids}

    # manter apenas os FIDs presentes no dataset
    gdf = gdf[gdf["fid"].isin(dataset_fids)].copy()

    if gdf["fid"].duplicated().any():
        raise ValueError("Existem FIDs duplicados no shapefile.")

    return gdf[["fid", "ugrhi_id", "ugrhi_nome"]]


def create_spatial_folds(samples, spatial_groups, n_splits=5, random_state=42):
    # Também normaliza o FID do dataset
    samples = samples.copy()
    samples["fid_original"] = samples["fid"]

    samples["fid"] = (samples["fid"].astype(str).str.replace("fid_", "", regex=False))

    df = samples.merge(spatial_groups, on="fid", how="left", validate="one_to_one")

    if df["ugrhi_id"].isna().any():
        missing = df.loc[df["ugrhi_id"].isna(), "fid_original"].tolist()
        raise ValueError(f"UGRHI não encontrada para os FIDs: {missing[:10]}")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X=df, y=df["label"], groups=df["ugrhi_id"]), start=1):
        folds.append({"fold": fold, "train_idx": train_idx, "val_idx": val_idx})

    return df, folds