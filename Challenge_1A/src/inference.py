import os
import json
import joblib
import re
import numpy as np
import lightgbm as lgb
from ingestion import extract_text_blocks
from graph import build_page_graph
from features import build_feature_dataframe

MODEL_PATH = "models/heading_model.txt"

# Regex patterns for numbering schemes
PATTERN_H1 = re.compile(r"^\d+\.?\s")               # e.g., "1. Intro"
PATTERN_H2 = re.compile(r"^\d+\.\d+\.?\s")         # e.g., "1.1 Background"
PATTERN_H3 = re.compile(r"^(?:\d+\.\d+\.\d+\.?\s|[a-z]\)\s|\([ivx]+\)\s)")

def load_model(model_path="models/heading_model.txt"):
    # Load the sklearn classifier we saved
    pickle_path = model_path.replace(".txt", ".pkl")
    return joblib.load(pickle_path)

def load_feature_names(path="models/feature_names.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_labels(model, X):
    # Model.predict returns exactly the labels you trained on
    return list(model.predict(X))


def extract_block_text(block):
    """Concatenate all spans of a block into full text."""
    spans = [s for line in block["lines"] for s in line["spans"]]
    return "".join(s.get("text", "") for s in spans).strip()


def assign_hierarchy(heading_infos):
    """
    Assign H1/H2/H3 levels to detected headings.
    heading_infos: list of dicts with keys: 'text', 'page', 'font_size', 'numbering_pattern', 'norm_x0'
    """
    # Extract font sizes
    sizes = sorted({info['font_size'] for info in heading_infos}, reverse=True)
    # Create tier mapping: size -> level by index
    tiers = {size: f"H{min(i+1,3)}" for i, size in enumerate(sizes)}

    outline = []
    for info in heading_infos:
        level = None
        txt = info['text']
        # Check numbering override first
        if PATTERN_H3.match(txt):
            level = "H3"
        elif PATTERN_H2.match(txt):
            level = "H2"
        elif PATTERN_H1.match(txt):
            level = "H1"
        else:
            # Fallback to font-size tier
            level = tiers.get(info['font_size'], "H3")
        outline.append({
            "level": level,
            "text": txt,
            # zero-based page number
            "page": info['page']
        })
    return outline


def process_pdf(pdf_path, booster):
    """Run full pipeline on a single PDF and return JSON dict."""
    pages = extract_text_blocks(pdf_path)
    graphs = [build_page_graph(blks) for blks in pages]
    df = build_feature_dataframe(graphs)


        # Prepare numeric feature matrix for inference
    df_in = df.drop(columns=[c for c in df.columns if df[c].dtype == object], errors='ignore')
    df_in = df_in.fillna(0)

    # Align to training features
    feat_names = load_feature_names()
    # reindex will add any missing columns as 0, and drop extras
    df_in = df_in.reindex(columns=feat_names, fill_value=0)

    # Predict labels
    labels = predict_labels(booster, df_in)
    df['pred'] = labels

    # Identify title block (first predicted title)
    title_df = df[df['pred'] == 'title']
    if not title_df.empty:
        # Use the first model‑predicted title
        title_row = title_df.iloc[0]
        title_block = pages[int(title_row['page_idx'])-1][int(title_row['node_idx'])]
    else:
        # FALLBACK: pick the block on page 1 (index 0) with max font size
        first_page_blocks = pages[0]
        # compute font size per block
        max_idx, max_size = 0, 0.0
        for i, blk in enumerate(first_page_blocks):
            sizes = [s['size'] for line in blk['lines'] for s in line['spans']]
            avg = float(np.mean(sizes)) if sizes else 0.0
            if avg > max_size:
                max_size = avg
                max_idx = i
        title_block = first_page_blocks[max_idx]
        print("⚠️  No title predicted—using largest‐font block on page 0 as title.")
    title_text = extract_block_text(title_block)

    # Collect heading infos
    headings = []
    for _, row in df[df['pred'] == 'heading'].iterrows():
        blk = pages[int(row['page_idx'])-1][int(row['node_idx'])]
        font_sizes = [s['size'] for line in blk['lines'] for s in line['spans']]
        headings.append({
            'text': extract_block_text(blk),
            # zero-based page
            'page': int(row['page_idx']) - 1,
            'font_size': float(np.mean(font_sizes)) if font_sizes else 0.0,
            'numbering_pattern': bool(row.get('numbering_pattern', False)),
            'norm_x0': row.get('norm_x0', 0.0)
        })

    # Assign hierarchical levels
    outline = assign_hierarchy(headings)

    # Build final JSON
    result = {
        "title": title_text,
        "outline": outline
    }
    return result


def save_json(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved outline JSON to {output_path}")


if __name__ == '__main__':
    import sys, os

    # Expect either single or batch mode
    path_in, path_out = sys.argv[1], sys.argv[2]
    batch = ("--batch" in sys.argv)

    booster = load_model("models/heading_model.txt")
    # feature names loader already uses models/feature_names.json

    if batch:
        os.makedirs(path_out, exist_ok=True)
        for fname in sorted(os.listdir(path_in)):
            if not fname.lower().endswith(".pdf"):
                continue
            in_pdf  = os.path.join(path_in, fname)
            base    = os.path.splitext(fname)[0]
            out_json= os.path.join(path_out, f"{base}_outline.json")
            print(f"▶ Processing {fname}")
            result = process_pdf(in_pdf, booster)
            save_json(result, out_json)
    else:
        result = process_pdf(path_in, booster)
        save_json(result, path_out)
