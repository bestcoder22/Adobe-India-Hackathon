# from ingestion import extract_text_blocks, PDFParseError
# from graph import build_page_graph
# from features  import build_feature_dataframe

# -- ingestion.py
# if __name__ == "__main__":
#     import sys
#     pdf_path = sys.argv[1]  # e.g., python main.py sample.pdf

#     try:
#         pages = extract_text_blocks(pdf_path)
#         for pg_num, blocks in enumerate(pages, start=1):
#             print(f"Page {pg_num}: {len(blocks)} text blocks")
#     except PDFParseError as e:
#         print("Error:", e)
#         sys.exit(1)

# -- Graph.py
# pdf_path = "data/samples/sample2.pdf"
# pages = extract_text_blocks(pdf_path)

# # Build graphs for all pages
# page_graphs = [build_page_graph(blocks) for blocks in pages]

# # Quick sanity check: print node & edge counts
# for page_num, G in enumerate(page_graphs, start=1):
#     print(f"Page {page_num}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# -- Features.py
# pdf_path = "data/samples/sample2.pdf"
# pages = extract_text_blocks(pdf_path)
# graphs = [build_page_graph(blks) for blks in pages]
# df = build_feature_dataframe(graphs)

# # Quick check
# # print(df.head())
# # print("Total blocks:", len(df))

# # in main.py, just after building df:
# def get_snippet(row):
#     # Cast page_idx and node_idx to int
#     p = int(row["page_idx"]) - 1
#     n = int(row["node_idx"])
#     # Grab the first line’s spans for that block
#     spans = pages[p][n]["lines"][0]["spans"]
#     text = "".join(span["text"] for span in spans)
#     return text[:50]  # first 50 chars

# df["text_snippet"] = df.apply(get_snippet, axis=1)
# df.to_csv("data/labels/unlabeled_blocks.csv", index=False)
# print("Exported feature table for labeling: data/labels/unlabeled_blocks.csv")

import os
from ingestion import extract_text_blocks
from graph     import build_page_graph
from features  import build_feature_dataframe

PDF_DIR    = "data/samples"
OUT_DIR    = "data/labels"

for pdf_file in os.listdir(PDF_DIR):
    if not pdf_file.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_DIR, pdf_file)
    # 1) Ingest → graph → features
    pages = extract_text_blocks(pdf_path)
    graphs = [build_page_graph(blks) for blks in pages]
    df = build_feature_dataframe(graphs)

    # 2) Add snippet so you know which block is which
    def get_snip(row):
        p = int(row.page_idx) - 1
        n = int(row.node_idx)
        spans = pages[p][n]["lines"][0]["spans"]
        return "".join(s["text"] for s in spans)[:50]
    df["text_snippet"] = df.apply(get_snip, axis=1)

    # 3) Build output filename
    base = os.path.splitext(pdf_file)[0]            # e.g. "sample1"
    csv_name = f"{base}_blocks_unlabeled.csv"     # e.g. "sample1_blocks_unlabeled.csv"
    out_path = os.path.join(OUT_DIR, csv_name)

    # 4) Save it
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Exported {out_path}")

