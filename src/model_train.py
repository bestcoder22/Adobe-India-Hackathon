import os
import glob
import pandas as pd
import json
import joblib
from lightgbm import LGBMClassifier

def load_all_labeled_data(feat_pattern, lab_pattern):
    """
    Merges features and labels from all PDF samples.

    Args:
      feat_pattern: glob for unlabeled feature CSVs,
                    e.g. "data/labels/*_blocks_unlabeled.csv"
      lab_pattern:  glob for labeled CSVs,
                    e.g. "data/labels/*_blocks_labeled.csv"

    Returns:
      A single pandas.DataFrame with all features + 'label' column.
    """
    feat_files = sorted(glob.glob(feat_pattern))
    lab_files  = sorted(glob.glob(lab_pattern))

    merged_dfs = []
    for feat_path, lab_path in zip(feat_files, lab_files):
        # Ensure matching bases: "sample1_blocks_unlabeled.csv" vs "sample1_blocks_labeled.csv"
        base_feat = os.path.basename(feat_path).replace("_blocks_unlabeled.csv", "")
        base_lab  = os.path.basename(lab_path).replace("_blocks_labeled.csv", "")
        if base_feat != base_lab:
            raise ValueError(f"Filename mismatch: {feat_path} vs {lab_path}")

        df_feat = pd.read_csv(feat_path)
        df_lab  = pd.read_csv(lab_path)[["page_idx", "node_idx", "label"]]

        # Inner merge ensures only labeled rows are kept
        df = df_feat.merge(df_lab, on=["page_idx", "node_idx"], how="inner")
        merged_dfs.append(df)

    # Concatenate all samples into one DataFrame
    return pd.concat(merged_dfs, ignore_index=True)

def train_and_serialize(df, model_path="models/heading_model.txt"):
    # 1. Prepare X and y, drop non‑numeric columns
    df_clean = df.copy()
    df_clean["label"] = df_clean["label"].astype(str)
    y = df_clean["label"]

    # Drop label, text_snippet, and any other non-numeric columns
    to_drop = [
         c for c in df_clean.columns
         if c == "label" or c == "text_snippet" or df_clean[c].dtype == object
         or c in ("page_idx", "node_idx")
    ]
    X = df_clean.drop(columns=to_drop, errors="ignore")

    # 2. Fill missing numeric values
    X = X.fillna(0)

    import re
    X.columns = [
        re.sub(r'[^0-9A-Za-z_]', '_', col)
        for col in X.columns
    ]
    # 3. Train LightGBM
    clf = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X, y)

    # Save raw Booster as before
    clf.booster_.save_model(model_path)
    print(f"✅ Booster saved to {model_path}")

    # ALSO save the full sklearn classifier
    pickle_path = model_path.replace(".txt", ".pkl")
    joblib.dump(clf, pickle_path)
    print(f"✅ Sklearn model saved to {pickle_path}")

    # Save feature names so inference can align exactly
    feat_names_path = os.path.join(os.path.dirname(model_path), "feature_names.json")
    with open(feat_names_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f)
    print(f"✅ Feature names saved to {feat_names_path}")

    # 4. Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    clf.booster_.save_model(model_path)
    print(f"✅ Model trained and saved to {model_path}")


if __name__ == "__main__":
    # 1. Load & merge all labeled data
    df_all = load_all_labeled_data(
        feat_pattern="data/labels/*_blocks_unlabeled.csv",
        lab_pattern="data/labels/*_blocks_labeled.csv"
    )
    print("🔍 Training data shape:", df_all.shape)
    print("🔢 Class distribution:\n", df_all["label"].value_counts(), "\n")

    # 2. Train & serialize
    train_and_serialize(df_all, model_path="models/heading_model.txt")
