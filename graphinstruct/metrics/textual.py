"""D2: Text quality metrics — G-BERTScore, G-BLEU, G-ROUGE, Text-F1.

Evaluates semantic correctness of text attributes in generated graphs
(node labels, edge types). Only active for Level 3+ instructions.

Reference: GraphJudge [P4] for G-BERTScore/G-BLEU/G-ROUGE;
           EDC [P6] for Text-F1.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------


def extract_triples(G: nx.Graph) -> list[tuple[str, str, str]]:
    """Extract (head, relation, tail) triples from a graph.

    Uses node 'label' attribute and edge 'label'/'type' attribute.
    Falls back to string representation of node IDs if no label.
    """
    triples = []
    # Graph-level edge type fallback (from serialize optimization)
    default_rel = G.graph.get("edge_type", "connected_to")
    for u, v, data in G.edges(data=True):
        h_label = G.nodes[u].get("label", str(u))
        t_label = G.nodes[v].get("label", str(v))
        rel = data.get("label", data.get("type", default_rel))
        triples.append((h_label, rel, t_label))
    return triples


def triple_to_sentence(triple: tuple[str, str, str]) -> str:
    """Convert a triple to a natural language sentence."""
    h, r, t = triple
    return f"{h} is related to {t} through {r}"


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a list of tokens."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _tokenize_triples(G: nx.Graph) -> list[str]:
    """Extract all triple sentences from a graph and tokenize."""
    triples = extract_triples(G)
    tokens: list[str] = []
    for t in triples:
        tokens.extend(triple_to_sentence(t).lower().split())
    return tokens


# ---------------------------------------------------------------------------
# G-BLEU
# ---------------------------------------------------------------------------


def g_bleu(
    reference: nx.Graph,
    generated: nx.Graph,
    max_n: int = 4,
) -> float:
    """Compute G-BLEU score between reference and generated graphs.

    Modified BLEU-4 computed on triple sentences.
    Precision-oriented: measures how much of the generated text
    appears in the reference.

    Args:
        reference: Reference graph with text attributes.
        generated: Generated graph with text attributes.
        max_n: Maximum n-gram order (default 4 for BLEU-4).

    Returns:
        BLEU score in [0.0, 1.0].
    """
    ref_tokens = _tokenize_triples(reference)
    gen_tokens = _tokenize_triples(generated)

    if not gen_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        gen_ngrams = Counter(_ngrams(gen_tokens, n))

        if not gen_ngrams:
            precisions.append(0.0)
            continue

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in gen_ngrams.items())
        total = sum(gen_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    if any(p == 0.0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)

    # Brevity penalty
    bp = 1.0
    if len(gen_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(gen_tokens))

    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# G-ROUGE
# ---------------------------------------------------------------------------


def g_rouge(
    reference: nx.Graph,
    generated: nx.Graph,
) -> float:
    """Compute G-ROUGE-2 (bigram recall) between reference and generated.

    Recall-oriented: measures how much of the reference text
    is covered by the generated text.

    Returns:
        ROUGE-2 score in [0.0, 1.0].
    """
    ref_tokens = _tokenize_triples(reference)
    gen_tokens = _tokenize_triples(generated)

    if not ref_tokens or len(ref_tokens) < 2:
        return 0.0

    ref_bigrams = Counter(_ngrams(ref_tokens, 2))
    gen_bigrams = Counter(_ngrams(gen_tokens, 2))

    overlap = sum(min(ref_bigrams[bg], gen_bigrams[bg]) for bg in ref_bigrams)
    total_ref = sum(ref_bigrams.values())

    return overlap / total_ref if total_ref > 0 else 0.0


# ---------------------------------------------------------------------------
# Text-F1
# ---------------------------------------------------------------------------


def text_f1(
    reference: nx.Graph,
    generated: nx.Graph,
) -> float:
    """Compute Text-F1 on node/edge text attributes.

    Precision-Recall F1 based on exact match of text attribute values.

    Returns:
        F1 score in [0.0, 1.0].
    """
    ref_labels = _collect_labels(reference)
    gen_labels = _collect_labels(generated)

    if not ref_labels and not gen_labels:
        return 1.0  # Both empty = perfect match
    if not ref_labels or not gen_labels:
        return 0.0

    overlap = ref_labels & gen_labels
    precision = len(overlap) / len(gen_labels)
    recall = len(overlap) / len(ref_labels)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _collect_labels(G: nx.Graph) -> set[tuple[str, str]]:
    """Collect all text attribute labels from a graph."""
    labels: set[tuple[str, str]] = set()
    for _, data in G.nodes(data=True):
        if "label" in data:
            labels.add(("node", data["label"]))
    for _, _, data in G.edges(data=True):
        for key in ("label", "type"):
            if key in data:
                labels.add(("edge", data[key]))
    return labels


# ---------------------------------------------------------------------------
# G-BERTScore
# ---------------------------------------------------------------------------

# Module-level cache: load BERT model/tokenizer once, reuse across calls
_BERT_CACHE: dict[str, tuple] = {}


def _get_bert_model(model_name: str):
    """Get cached BERT model and tokenizer (loaded once per model_name).

    Automatically uses CUDA if available.
    """
    if model_name not in _BERT_CACHE:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        _BERT_CACHE[model_name] = (tokenizer, model, torch, device)
    return _BERT_CACHE[model_name]


def g_bertscore(
    reference: nx.Graph,
    generated: nx.Graph,
    model_name: str = "bert-base-uncased",
) -> float:
    """Compute G-BERTScore between reference and generated graphs.

    Converts triples to sentences, then computes BERT cosine similarity.
    Requires ``transformers`` and ``torch`` packages.
    Falls back to word-overlap approximation if unavailable.

    Returns:
        F1 score in [0.0, 1.0].
    """
    ref_triples = extract_triples(reference)
    gen_triples = extract_triples(generated)

    if not ref_triples or not gen_triples:
        return 0.0

    try:
        tokenizer, model, torch, device = _get_bert_model(model_name)
    except ImportError:
        return _fallback_bertscore(ref_triples, gen_triples)

    ref_sentences = [triple_to_sentence(t) for t in ref_triples]
    gen_sentences = [triple_to_sentence(t) for t in gen_triples]

    def encode(sentences: list[str]) -> "torch.Tensor":
        inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    ref_embs = encode(ref_sentences)
    gen_embs = encode(gen_sentences)

    # Normalize
    ref_embs = ref_embs / ref_embs.norm(dim=1, keepdim=True)
    gen_embs = gen_embs / gen_embs.norm(dim=1, keepdim=True)

    # Cosine similarity matrix
    sim_matrix = torch.mm(gen_embs, ref_embs.t())

    recall = sim_matrix.max(dim=0).values.mean().item()
    precision = sim_matrix.max(dim=1).values.mean().item()

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _fallback_bertscore(
    ref_triples: list[tuple[str, str, str]],
    gen_triples: list[tuple[str, str, str]],
) -> float:
    """Fallback BERTScore using simple word overlap (no BERT model)."""
    ref_sentences = [triple_to_sentence(t).lower().split() for t in ref_triples]
    gen_sentences = [triple_to_sentence(t).lower().split() for t in gen_triples]

    if not ref_sentences or not gen_sentences:
        return 0.0

    def word_sim(s1: list[str], s2: list[str]) -> float:
        set1, set2 = set(s1), set(s2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    recall_scores = [max(word_sim(r, g) for g in gen_sentences) for r in ref_sentences]
    recall = sum(recall_scores) / len(recall_scores)

    precision_scores = [
        max(word_sim(g, r) for r in ref_sentences) for g in gen_sentences
    ]
    precision = sum(precision_scores) / len(precision_scores)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Aggregate D2 score
# ---------------------------------------------------------------------------


def d2_score(
    reference: Optional[nx.Graph],
    generated: Optional[nx.Graph],
    use_bert: bool = False,
) -> dict[str, float]:
    """Compute all D2 metrics and return as a dict.

    Args:
        reference: Reference graph (with text attributes).
        generated: Generated graph.
        use_bert: If True, use real BERT for G-BERTScore.

    Returns:
        Dict with keys: g_bertscore, g_bleu, g_rouge, text_f1, aggregate.
        Returns all zeros if either graph is None or has no edges.
    """
    if reference is None or generated is None:
        return _zero_d2()

    if reference.number_of_edges() == 0 and generated.number_of_edges() == 0:
        tp = text_presence(generated)
        ts = text_similarity(generated, reference, use_bert=use_bert)
        return {
            "g_bertscore": 1.0,
            "g_bleu": 1.0,
            "g_rouge": 1.0,
            "text_f1": 1.0,
            "text_presence": tp,
            "text_similarity": ts,
            "aggregate": 0.5 * tp + 0.5 * ts,
        }

    ref_triples = extract_triples(reference)
    gen_triples = extract_triples(generated)

    if not ref_triples or not gen_triples:
        # One graph has edges, the other doesn't — partial mismatch
        bleu = g_bleu(reference, generated)
        rouge = g_rouge(reference, generated)
        tf1 = text_f1(reference, generated)
        tp = text_presence(generated)
        ts = text_similarity(generated, reference, use_bert=use_bert)
        return {
            "g_bertscore": 0.0,
            "g_bleu": bleu,
            "g_rouge": rouge,
            "text_f1": tf1,
            "text_presence": tp,
            "text_similarity": ts,
            "aggregate": 0.5 * tp + 0.5 * ts,
        }

    scores = {
        "g_bertscore": (
            g_bertscore(reference, generated)
            if use_bert
            else _fallback_bertscore(ref_triples, gen_triples)
        ),
        "g_bleu": g_bleu(reference, generated),
        "g_rouge": g_rouge(reference, generated),
        "text_f1": text_f1(reference, generated),
    }

    # v2 D2: text_presence + text_similarity replaces naive average.
    # Old sub-metrics are retained for diagnostic but don't drive aggregate.
    tp = text_presence(generated)
    ts = text_similarity(generated, reference, use_bert=use_bert)
    scores["text_presence"] = tp
    scores["text_similarity"] = ts
    scores["aggregate"] = 0.5 * tp + 0.5 * ts
    return scores


# ---------------------------------------------------------------------------
# v2 D2: text_presence + text_similarity
# ---------------------------------------------------------------------------


def text_presence(G: nx.Graph) -> float:
    """Score how much meaningful text the generated graph contains.

    Does not require reference alignment — purely measures whether
    the model outputs domain-appropriate text attributes.

    Scoring:
    - Node: label attribute = 1.0, semantic ID = 0.7, numeric ID = 0.0
    - Edge: explicit edge_type = 1.0, default "connected_to" = 0.0

    Returns:
        Score in [0.0, 1.0].
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0

    # Node label quality (0.6 weight)
    has_label_attr = 0
    has_semantic_id = 0
    for nd in G.nodes():
        if "label" in G.nodes[nd]:
            has_label_attr += 1
        elif not str(nd).isdigit() and len(str(nd)) > 1:
            has_semantic_id += 1
    node_score = (has_label_attr * 1.0 + has_semantic_id * 0.7) / n

    # Edge type quality (0.4 weight)
    graph_etype = G.graph.get("edge_type", "connected_to")
    has_explicit_etype = graph_etype != "connected_to"
    has_per_edge_type = any(
        "label" in d or "type" in d for _, _, d in G.edges(data=True)
    )
    edge_score = 1.0 if (has_explicit_etype or has_per_edge_type) else 0.0

    return 0.6 * node_score + 0.4 * edge_score


def text_similarity(
    G: nx.Graph,
    reference: nx.Graph,
    use_bert: bool = False,
) -> float:
    """Set-level semantic similarity between generated and reference labels.

    Compares the vocabulary of node labels (not aligned triples).
    Numeric-only IDs with no label attribute yield 0 (no text to compare).

    Returns:
        Score in [0.0, 1.0].
    """

    def _get_meaningful_labels(graph: nx.Graph) -> list[str]:
        labels = []
        for nd in graph.nodes():
            if "label" in graph.nodes[nd]:
                # Explicit label attribute is always meaningful
                labels.append(str(graph.nodes[nd]["label"]).lower())
            else:
                # Node ID: only meaningful if non-numeric and non-trivial
                sid = str(nd)
                if not sid.isdigit() and len(sid) > 1:
                    labels.append(sid.lower())
        return labels

    ref_labels = _get_meaningful_labels(reference)
    gen_labels = _get_meaningful_labels(G)

    if not ref_labels or not gen_labels:
        return 0.0

    if use_bert:
        return _bert_set_similarity(ref_labels, gen_labels)
    return _word_overlap_set_similarity(ref_labels, gen_labels)


def _word_overlap_set_similarity(
    ref_labels: list[str],
    gen_labels: list[str],
) -> float:
    """Fallback set similarity using word/character overlap."""
    ref_set = set(ref_labels)
    gen_set = set(gen_labels)
    if not ref_set or not gen_set:
        return 0.0
    # Jaccard on the label vocabulary
    intersection = ref_set & gen_set
    union = ref_set | gen_set
    return len(intersection) / len(union) if union else 0.0


def _bert_set_similarity(
    ref_labels: list[str],
    gen_labels: list[str],
) -> float:
    """BERT-based set similarity: encode label sets, compute cosine similarity."""
    try:
        tokenizer, model, torch, device = _get_bert_model("bert-base-uncased")
    except (ImportError, Exception):
        return _word_overlap_set_similarity(ref_labels, gen_labels)

    ref_text = " ".join(sorted(set(ref_labels)))
    gen_text = " ".join(sorted(set(gen_labels)))

    def encode_single(text: str):
        inputs = tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    ref_emb = encode_single(ref_text)
    gen_emb = encode_single(gen_text)

    ref_emb = ref_emb / ref_emb.norm(dim=1, keepdim=True)
    gen_emb = gen_emb / gen_emb.norm(dim=1, keepdim=True)

    sim = torch.mm(ref_emb, gen_emb.t()).item()
    return max(0.0, sim)


def _zero_d2() -> dict[str, float]:
    return {
        "g_bertscore": 0.0,
        "g_bleu": 0.0,
        "g_rouge": 0.0,
        "text_f1": 0.0,
        "text_presence": 0.0,
        "text_similarity": 0.0,
        "aggregate": 0.0,
    }
