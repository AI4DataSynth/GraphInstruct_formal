#!/usr/bin/env bash
# ============================================================
# Download raw upstream graph datasets used to build the L4
# real-world reference pool. These are NOT shipped with the
# repository because they have separate upstream licenses and
# total ~50 MB.
#
# After running this script, see scripts/build_l4_pipeline.py
# for the deduplication / size-normalization pipeline that
# produces data/reference_pools/l4_real/*.pkl.
#
# Usage:
#   bash scripts/download_raw.sh
# ============================================================

set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "==> Downloading SNAP citation networks (DBLP + Cora)..."
# DBLP citation network (Tang et al., 2008)
[ -f cit-HepPh.txt.gz ] || wget -q https://snap.stanford.edu/data/cit-HepPh.txt.gz
# Cora is bundled with NetworkX; no download needed
echo "    OK: cit-HepPh.txt.gz"

echo "==> Downloading SNAP social networks (Reddit + Facebook ego)..."
# Reddit hyperlink network (Hamilton et al., 2017 — re-released by SNAP)
[ -f redditHyperlinks-body.tsv ] || wget -q https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv
# Facebook ego network (Leskovec & Krevl, 2014)
[ -f facebook_combined.txt.gz ] || wget -q https://snap.stanford.edu/data/facebook_combined.txt.gz
echo "    OK"

echo "==> Downloading SNAP infrastructure + communication..."
# Pennsylvania road network (Leskovec et al., 2009)
[ -f roadNet-PA.txt.gz ] || wget -q https://snap.stanford.edu/data/roadNet-PA.txt.gz
# EU email-core (Leskovec et al., 2007)
[ -f email-Eu-core.txt.gz ] || wget -q https://snap.stanford.edu/data/email-Eu-core.txt.gz
echo "    OK"

echo "==> Downloading TUDataset MUTAG (Morris et al., 2020 — CC-BY)..."
# MUTAG benchmark
[ -f MUTAG.zip ] || wget -q https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip
[ -d MUTAG ] || unzip -q MUTAG.zip
echo "    OK"

echo ""
echo "==> All upstream raw datasets downloaded to: $RAW_DIR"
echo ""
echo "    Note: Cora, ZINC, QM9 are loaded directly via PyTorch Geometric / DeepChem"
echo "    inside scripts/build_l4_pipeline.py — no separate download needed."
echo ""
echo "    Next: run scripts/build_l4_pipeline.py to regenerate the L4 pool from"
echo "    these raw inputs (already cached at data/reference_pools/l4_real/)."
