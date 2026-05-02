"""D3: Embedding quality metrics — Grassmann Coherence, Node Clf. Gap, Emb. MMD.

Evaluates generated graph quality in embedding space, directly relating
to downstream GNN task usability. Only active for Level 3+ instructions.

Sub-metric weights (empirically tuned via SNR analysis):
  Grassmann=0.15, Node Clf. Gap=0.50, Emb MMD=0.35

NOTE: Full implementations require PyTorch + GNN libraries.
This module provides:
- Pure-Python stubs that return 0.0 when dependencies are missing
- Full implementations when torch/torch_geometric are available
- A spectral fallback for Grassmann Coherence using numpy/scipy

References:
- [P1] GAG (ACL 2025): Node Clf. Gap
- [P13] GRAPHMASTER: Grassmann Coherence
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

# Try importing numpy/scipy for spectral methods
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_HAS_PYG = False
if _HAS_TORCH:
    try:
        import torch_geometric  # noqa: F401

        _HAS_PYG = True
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# D3 sub-metric weights (re-tuned 2026-04-04 based on discriminative power analysis).
# Previous SNR-based weights (0.15/0.50/0.35) gave NCG too much weight — NCG has
# near-zero discriminative power on 10-100 node graphs (spread=0.014 on big graphs,
# all models cluster at 0.97+). Grassmann provides 2x better model separation.
# New weights tested across 7 models: spread 0.0126 → 0.0252 (2x improvement).
# ---------------------------------------------------------------------------
_W_GRASSMANN: float = 0.50
_W_NODE_CLF_GAP: float = 0.15
_W_EMB_MMD: float = 0.35


def _weighted_aggregate(grassmann: float, node_clf_gap: float, emb_mmd: float) -> float:
    """Compute weighted D3 aggregate from sub-metrics."""
    if _HAS_NUMPY:
        return (
            _W_GRASSMANN * grassmann
            + _W_NODE_CLF_GAP * node_clf_gap
            + _W_EMB_MMD * emb_mmd
        )
    # Without numpy, NCG is unavailable — re-normalise Grassmann + EmbMMD
    total = _W_GRASSMANN + _W_EMB_MMD
    return (_W_GRASSMANN * grassmann + _W_EMB_MMD * emb_mmd) / total if total else 0.0


# ---------------------------------------------------------------------------
# Grassmann Coherence
# ---------------------------------------------------------------------------


def grassmann_coherence(
    generated: nx.Graph,
    reference: Optional[nx.Graph] = None,
    k: int = 4,
) -> float:
    """Compute Grassmann Coherence of generated graph embeddings.

    Measures alignment of node embeddings with principal semantic
    directions on the Grassmann manifold. Range [0, 1], higher is better.

    Uses spectral embeddings (eigenvectors of the Laplacian) as a
    proxy when GNN embeddings are not available.

    Args:
        generated: Generated graph.
        reference: Reference graph (optional; if provided, measures
            subspace alignment between the two).
        k: Number of leading eigenvectors to use.

    Returns:
        Coherence score in [0.0, 1.0].
    """
    if not _HAS_NUMPY:
        return 0.0

    if generated.number_of_nodes() < 2:
        return 0.0

    gen_emb = _spectral_embedding(generated, k)
    if gen_emb is None:
        return 0.0

    if reference is not None and reference.number_of_nodes() >= 2:
        ref_emb = _spectral_embedding(reference, k)
        if ref_emb is None:
            return 0.0
        return _subspace_alignment(gen_emb, ref_emb)

    # Self-coherence: how well-structured are the embeddings
    # (ratio of variance explained by top-k components)
    singular_values = np.linalg.svd(gen_emb, compute_uv=False)
    total_var = np.sum(singular_values**2)
    if total_var < 1e-12:
        return 0.0
    return float(np.sum(singular_values[:k] ** 2) / total_var)


def _spectral_embedding(
    G: nx.Graph,
    k: int,
) -> Optional["np.ndarray"]:
    """Compute spectral embedding of a graph using Laplacian eigenvectors."""
    if not _HAS_NUMPY:
        return None

    n = G.number_of_nodes()
    if n < 2:
        return None

    # Use at most k eigenvectors, but don't exceed n-1
    actual_k = min(k, n - 1)

    try:
        # Strip edge attributes to avoid scipy.sparse dtype=object error
        H = type(G)()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges())
        L = nx.laplacian_matrix(H).toarray().astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Skip the first eigenvector (constant, eigenvalue ≈ 0)
        embedding = eigenvectors[:, 1 : actual_k + 1]
        return embedding
    except Exception:
        return None


def _subspace_alignment(
    A: "np.ndarray",
    B: "np.ndarray",
) -> float:
    """Compute subspace alignment between two sets of eigenvectors.

    Uses the Grassmann distance: cos^2 of principal angles.
    """
    # Ensure same number of columns
    min_cols = min(A.shape[1], B.shape[1])
    A = A[:, :min_cols]
    B = B[:, :min_cols]

    # Align dimensions (pad shorter matrix with zeros)
    max_rows = max(A.shape[0], B.shape[0])
    if A.shape[0] < max_rows:
        A = np.vstack([A, np.zeros((max_rows - A.shape[0], A.shape[1]))])
    if B.shape[0] < max_rows:
        B = np.vstack([B, np.zeros((max_rows - B.shape[0], B.shape[1]))])

    # Orthonormalize
    Q_A, _ = np.linalg.qr(A)
    Q_B, _ = np.linalg.qr(B)

    # Principal angles via SVD of Q_A^T Q_B
    M = Q_A[:, :min_cols].T @ Q_B[:, :min_cols]
    singular_values = np.linalg.svd(M, compute_uv=False)

    # Coherence = mean of cos^2(principal angles)
    cos2 = np.clip(singular_values**2, 0.0, 1.0)
    return float(np.mean(cos2))


# ---------------------------------------------------------------------------
# Node Classification Gap
# ---------------------------------------------------------------------------


def node_clf_gap(
    generated: nx.Graph,
    reference: nx.Graph,
    labels: Optional[dict] = None,
    epochs: int = 100,
) -> float:
    """Compute Node Classification Gap.

    Gap = Accuracy(train_on_real) - Accuracy(train_on_generated).
    Lower gap means the generated graph better preserves learning signals.

    Uses a 2-layer GCN backbone. When labels are not provided, uses
    community detection to assign synthetic labels.

    Args:
        generated: Generated graph to evaluate.
        reference: Real/reference graph.
        labels: Node label mapping {node -> int} (optional).
        epochs: GCN training epochs.

    Returns:
        Score = max(0.0, 1.0 - |gap|) in [0, 1], higher is better.
        Returns 0.0 when dependencies are missing or graphs are too small.
    """
    if not _HAS_NUMPY:
        return 0.0

    n_ref = reference.number_of_nodes()
    n_gen = generated.number_of_nodes()

    # Need enough nodes for train/test split and meaningful classification
    if n_ref < 10 or n_gen < 4:
        return 0.0

    # Use provided labels, or fall back to community detection
    if labels is None:
        labels = _community_labels(reference)
    if labels is None or len(set(labels.values())) < 2:
        return 0.0
    # Ensure labels cover reference nodes; filter to intersection
    ref_nodes = set(reference.nodes())
    labels = {k: v for k, v in labels.items() if k in ref_nodes}

    try:
        if _HAS_PYG:
            return _run_node_clf_gap(generated, reference, labels, epochs)
        return _run_node_clf_gap_fallback(generated, reference, labels, epochs)
    except Exception:
        return 0.0


def _community_labels(G: nx.Graph) -> Optional[dict]:
    """Assign synthetic node labels via greedy modularity communities."""
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        if len(communities) < 2:
            return None
        label_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                label_map[node] = idx
        return label_map
    except Exception:
        return None


def node_clf_gap_batch(
    generated_list: list[nx.Graph],
    reference: nx.Graph,
    labels: Optional[dict] = None,
    epochs: int = 100,
) -> list[float]:
    """Batch Node Clf. Gap: train reference GCN once, eval each gen graph.

    Equivalent to ``[node_clf_gap(g, reference) for g in generated_list]``
    but reuses Experiment A (ref GCN) across all generated graphs, reducing
    GCN training from 2N to N+1 instances.

    Args:
        generated_list: Generated graphs to evaluate.
        reference: Real/reference graph (shared).
        labels: Node label mapping (optional).
        epochs: GCN training epochs.

    Returns:
        List of scores in [0, 1], one per generated graph.
    """
    if not generated_list:
        return []
    if not _HAS_NUMPY:
        return [0.0] * len(generated_list)

    n_ref = reference.number_of_nodes()
    if n_ref < 10:
        return [0.0] * len(generated_list)

    if labels is None:
        labels = _community_labels(reference)
    if labels is None or len(set(labels.values())) < 2:
        return [0.0] * len(generated_list)

    ref_nodes = set(reference.nodes())
    labels = {k: v for k, v in labels.items() if k in ref_nodes}

    try:
        if _HAS_PYG:
            return _run_node_clf_gap_batch(generated_list, reference, labels, epochs)
        return _run_node_clf_gap_fallback_batch(
            generated_list, reference, labels, epochs
        )
    except Exception:
        return [0.0] * len(generated_list)


def _run_node_clf_gap(
    generated: nx.Graph,
    reference: nx.Graph,
    labels: dict,
    epochs: int,
) -> float:
    """Core GCN-based Node Clf. Gap computation.

    Experiment A: Train GCN on reference train nodes → eval on reference test.
    Experiment B: Train GCN on ALL generated nodes → eval on reference test.
    Score = max(0, 1.0 - |acc_A - acc_B|).
    """
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data

    torch.manual_seed(42)

    ref_nodes = list(reference.nodes())
    node_to_idx_ref = {n: i for i, n in enumerate(ref_nodes)}

    # Build features: [normalized_degree, clustering_coeff]
    ref_deg = dict(reference.degree())
    ref_cc = nx.clustering(reference)
    max_deg_ref = max(ref_deg.values()) if ref_deg else 1
    ref_features = torch.tensor(
        [[ref_deg[n] / max(max_deg_ref, 1), ref_cc[n]] for n in ref_nodes],
        dtype=torch.float,
    )

    # Labels for reference nodes
    num_classes = len(set(labels.values()))
    ref_labels = torch.tensor([labels.get(n, 0) for n in ref_nodes], dtype=torch.long)

    # Build edge_index for reference graph (undirected → both directions)
    ref_edges = []
    for u, v in reference.edges():
        i, j = node_to_idx_ref[u], node_to_idx_ref[v]
        ref_edges.extend([[i, j], [j, i]])
    if not ref_edges:
        return 0.0
    ref_edge_index = torch.tensor(ref_edges, dtype=torch.long).t().contiguous()

    ref_data = Data(x=ref_features, edge_index=ref_edge_index, y=ref_labels)

    # Train/test split on reference: 80% train, 20% test
    n_ref = len(ref_nodes)
    perm = torch.randperm(n_ref)
    n_train = max(1, int(0.8 * n_ref))
    train_mask = torch.zeros(n_ref, dtype=torch.bool)
    test_mask = torch.zeros(n_ref, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask[perm[n_train:]] = True

    if test_mask.sum() == 0:
        return 0.0

    ref_data.train_mask = train_mask
    ref_data.test_mask = test_mask

    # --- Experiment A: Train on reference, test on reference ---
    acc_real = _train_and_eval_gcn(ref_data, ref_data, num_classes, epochs)

    # --- Build generated graph data ---
    gen_nodes = list(generated.nodes())
    node_to_idx_gen = {n: i for i, n in enumerate(gen_nodes)}

    gen_deg = dict(generated.degree())
    gen_cc = nx.clustering(generated)
    max_deg_gen = max(gen_deg.values()) if gen_deg else 1
    gen_features = torch.tensor(
        [[gen_deg[n] / max(max_deg_gen, 1), gen_cc[n]] for n in gen_nodes],
        dtype=torch.float,
    )

    # Map labels from reference to generated (by index or shared node IDs)
    gen_labels = torch.tensor(
        [labels.get(n, 0) % num_classes for n in gen_nodes], dtype=torch.long
    )

    gen_edges = []
    for u, v in generated.edges():
        if u in node_to_idx_gen and v in node_to_idx_gen:
            i, j = node_to_idx_gen[u], node_to_idx_gen[v]
            gen_edges.extend([[i, j], [j, i]])
    if not gen_edges:
        return 0.0
    gen_edge_index = torch.tensor(gen_edges, dtype=torch.long).t().contiguous()

    n_gen = len(gen_nodes)
    gen_train_mask = torch.ones(n_gen, dtype=torch.bool)  # Train on ALL gen nodes
    gen_data = Data(
        x=gen_features,
        edge_index=gen_edge_index,
        y=gen_labels,
        train_mask=gen_train_mask,
    )

    # --- Experiment B: Train on generated, test on reference ---
    acc_gen = _train_and_eval_gcn(gen_data, ref_data, num_classes, epochs)

    gap = abs(acc_real - acc_gen)
    return max(0.0, 1.0 - gap)


if _HAS_TORCH:

    class _SimpleGCN(torch.nn.Module):
        """2-layer GCN for node classification."""

        def __init__(self, in_channels: int, hidden: int, out_channels: int):
            super().__init__()
            from torch_geometric.nn import GCNConv

            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, out_channels)

        def forward(self, data: "Data") -> "torch.Tensor":
            x = self.conv1(data.x, data.edge_index)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, data.edge_index)
            return x

    _GCN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_and_eval_gcn(
        train_data: "Data",
        test_data: "Data",
        num_classes: int,
        epochs: int,
    ) -> float:
        """Train a GCN on train_data and evaluate accuracy on test_data."""
        device = _GCN_DEVICE
        in_channels = train_data.x.shape[1]
        model = _SimpleGCN(in_channels, 16, num_classes).to(device)
        train_data = train_data.to(device)
        test_data = test_data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Train
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(train_data)
            loss = torch.nn.functional.cross_entropy(
                out[train_data.train_mask], train_data.y[train_data.train_mask]
            )
            loss.backward()
            optimizer.step()

        # Eval on test_data — soft accuracy (mean probability of correct class)
        model.eval()
        with torch.no_grad():
            out = model(test_data)
            test_out = out[test_data.test_mask]
            test_y = test_data.y[test_data.test_mask]
            total = int(test_data.test_mask.sum().item())
            if total == 0:
                return 0.0
            probs = torch.nn.functional.softmax(test_out, dim=1)
            correct_probs = probs[torch.arange(total), test_y]
        return float(correct_probs.mean().item())


def _run_node_clf_gap_batch(
    generated_list: list[nx.Graph],
    reference: nx.Graph,
    labels: dict,
    epochs: int,
) -> list[float]:
    """Batch PYG NCG: train ref GCN once, eval each generated graph (A2)."""
    from torch_geometric.data import Data

    torch.manual_seed(42)

    # --- Shared reference data (identical to _run_node_clf_gap) ---
    ref_nodes = list(reference.nodes())
    node_to_idx_ref = {n: i for i, n in enumerate(ref_nodes)}

    ref_deg = dict(reference.degree())
    ref_cc = nx.clustering(reference)
    max_deg_ref = max(ref_deg.values()) if ref_deg else 1
    ref_features = torch.tensor(
        [[ref_deg[n] / max(max_deg_ref, 1), ref_cc[n]] for n in ref_nodes],
        dtype=torch.float,
    )

    num_classes = len(set(labels.values()))
    ref_labels = torch.tensor([labels.get(n, 0) for n in ref_nodes], dtype=torch.long)

    ref_edges: list[list[int]] = []
    for u, v in reference.edges():
        i, j = node_to_idx_ref[u], node_to_idx_ref[v]
        ref_edges.extend([[i, j], [j, i]])
    if not ref_edges:
        return [0.0] * len(generated_list)
    ref_edge_index = torch.tensor(ref_edges, dtype=torch.long).t().contiguous()

    ref_data = Data(x=ref_features, edge_index=ref_edge_index, y=ref_labels)

    n_ref = len(ref_nodes)
    perm = torch.randperm(n_ref)
    n_train = max(1, int(0.8 * n_ref))
    train_mask = torch.zeros(n_ref, dtype=torch.bool)
    test_mask = torch.zeros(n_ref, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask[perm[n_train:]] = True

    if test_mask.sum() == 0:
        return [0.0] * len(generated_list)

    ref_data.train_mask = train_mask
    ref_data.test_mask = test_mask

    # --- Shared: Experiment A (train once) ---
    acc_real = _train_and_eval_gcn(ref_data, ref_data, num_classes, epochs)

    # Save RNG state so each Experiment B starts from identical state
    rng_state = torch.random.get_rng_state()

    # --- Per generated graph: Experiment B ---
    results: list[float] = []
    for generated in generated_list:
        n_gen = generated.number_of_nodes()
        if n_gen < 4:
            results.append(0.0)
            continue

        gen_nodes = list(generated.nodes())
        node_to_idx_gen = {n: i for i, n in enumerate(gen_nodes)}

        gen_deg = dict(generated.degree())
        gen_cc = nx.clustering(generated)
        max_deg_gen = max(gen_deg.values()) if gen_deg else 1
        gen_features = torch.tensor(
            [[gen_deg[n] / max(max_deg_gen, 1), gen_cc[n]] for n in gen_nodes],
            dtype=torch.float,
        )

        gen_labels = torch.tensor(
            [labels.get(n, 0) % num_classes for n in gen_nodes], dtype=torch.long
        )

        gen_edges: list[list[int]] = []
        for u, v in generated.edges():
            if u in node_to_idx_gen and v in node_to_idx_gen:
                i, j = node_to_idx_gen[u], node_to_idx_gen[v]
                gen_edges.extend([[i, j], [j, i]])
        if not gen_edges:
            results.append(0.0)
            continue
        gen_edge_index = torch.tensor(gen_edges, dtype=torch.long).t().contiguous()

        gen_train_mask = torch.ones(n_gen, dtype=torch.bool)
        gen_data = Data(
            x=gen_features,
            edge_index=gen_edge_index,
            y=gen_labels,
            train_mask=gen_train_mask,
        )

        # Restore RNG state → same GCN-B init weights as the per-call version
        torch.random.set_rng_state(rng_state)
        acc_gen = _train_and_eval_gcn(gen_data, ref_data, num_classes, epochs)

        gap = abs(acc_real - acc_gen)
        results.append(max(0.0, 1.0 - gap))

    return results


# ---------------------------------------------------------------------------
# Node Clf. Gap — numpy fallback (SGC: Simple Graph Convolution)
# ---------------------------------------------------------------------------


def _node_features_np(
    G: nx.Graph,
    nodes: list,
) -> "np.ndarray":
    """Compute structural features [normalized_degree, clustering_coeff]."""
    deg = dict(G.degree())
    cc = nx.clustering(G)
    max_deg = max(deg.values()) if deg else 1
    return np.array(
        [[deg[n] / max(max_deg, 1), cc[n]] for n in nodes],
        dtype=float,
    )


def _normalized_adjacency_np(
    G: nx.Graph,
    nodes: list,
) -> "np.ndarray":
    """Build D^{-1/2} (A + I) D^{-1/2} normalized adjacency (GCN-style)."""
    n = len(nodes)
    # Strip edge attributes to avoid scipy.sparse dtype=object error
    H = type(G)()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    A = nx.adjacency_matrix(H, nodelist=nodes).toarray().astype(float)
    A_hat = A + np.eye(n)
    d = np.sum(A_hat, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return np.diag(d_inv_sqrt) @ A_hat @ np.diag(d_inv_sqrt)


def _softmax_np(z: "np.ndarray") -> "np.ndarray":
    """Numerically stable softmax."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _train_softmax_clf(
    X: "np.ndarray",
    y: "np.ndarray",
    num_classes: int,
    lr: float = 0.1,
    epochs: int = 100,
) -> tuple:
    """Train softmax (multinomial logistic regression) classifier."""
    n_features = X.shape[1]
    rng = np.random.RandomState(42)
    W = rng.randn(n_features, num_classes) * 0.01
    b = np.zeros(num_classes)
    Y_onehot = np.eye(num_classes)[y]

    for _ in range(epochs):
        logits = X @ W + b
        probs = _softmax_np(logits)
        grad = probs - Y_onehot
        W -= lr * (X.T @ grad / len(y))
        b -= lr * np.mean(grad, axis=0)

    return W, b


def _run_node_clf_gap_fallback(
    generated: nx.Graph,
    reference: nx.Graph,
    labels: dict,
    epochs: int,
) -> float:
    """Node Clf. Gap via SGC (feature propagation + softmax), no PyG needed.

    Uses Simple Graph Convolution (Wu et al., 2019): propagate structural
    features through the normalized adjacency, then classify with softmax.
    Same experimental protocol as the GCN version.
    """
    ref_nodes = list(reference.nodes())
    n_ref = len(ref_nodes)
    num_classes = len(set(labels.values()))

    # Propagated features for reference graph (2-hop SGC)
    ref_raw = _node_features_np(reference, ref_nodes)
    A_ref = _normalized_adjacency_np(reference, ref_nodes)
    ref_feat = A_ref @ A_ref @ ref_raw  # 2-hop propagation

    ref_labels = np.array([labels.get(n, 0) for n in ref_nodes])

    # Train/test split: 80/20
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_ref)
    n_train = max(1, int(0.8 * n_ref))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    if len(test_idx) == 0:
        return 0.0

    # Experiment A: Train on reference, test on reference (soft accuracy)
    W_a, b_a = _train_softmax_clf(
        ref_feat[train_idx], ref_labels[train_idx], num_classes, epochs=epochs
    )
    probs_a = _softmax_np(ref_feat[test_idx] @ W_a + b_a)
    acc_real = float(np.mean(probs_a[np.arange(len(test_idx)), ref_labels[test_idx]]))

    # Propagated features for generated graph
    gen_nodes = list(generated.nodes())
    gen_raw = _node_features_np(generated, gen_nodes)
    A_gen = _normalized_adjacency_np(generated, gen_nodes)
    gen_feat = A_gen @ A_gen @ gen_raw

    gen_labels = np.array([labels.get(n, 0) % num_classes for n in gen_nodes])

    # Experiment B: Train on generated, test on reference (soft accuracy)
    W_b, b_b = _train_softmax_clf(gen_feat, gen_labels, num_classes, epochs=epochs)
    probs_b = _softmax_np(ref_feat[test_idx] @ W_b + b_b)
    acc_gen = float(np.mean(probs_b[np.arange(len(test_idx)), ref_labels[test_idx]]))

    gap = abs(acc_real - acc_gen)
    return max(0.0, 1.0 - gap)


def _run_node_clf_gap_fallback_batch(
    generated_list: list[nx.Graph],
    reference: nx.Graph,
    labels: dict,
    epochs: int,
) -> list[float]:
    """Batch numpy NCG: shared ref softmax training (A2 optimisation)."""
    ref_nodes = list(reference.nodes())
    n_ref = len(ref_nodes)
    num_classes = len(set(labels.values()))

    ref_raw = _node_features_np(reference, ref_nodes)
    A_ref = _normalized_adjacency_np(reference, ref_nodes)
    ref_feat = A_ref @ A_ref @ ref_raw
    ref_labels_arr = np.array([labels.get(n, 0) for n in ref_nodes])

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_ref)
    n_train = max(1, int(0.8 * n_ref))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    if len(test_idx) == 0:
        return [0.0] * len(generated_list)

    # Shared: Experiment A (train on reference, soft accuracy)
    W_a, b_a = _train_softmax_clf(
        ref_feat[train_idx], ref_labels_arr[train_idx], num_classes, epochs=epochs
    )
    probs_a = _softmax_np(ref_feat[test_idx] @ W_a + b_a)
    acc_real = float(
        np.mean(probs_a[np.arange(len(test_idx)), ref_labels_arr[test_idx]])
    )

    # Per gen graph: Experiment B (soft accuracy)
    results: list[float] = []
    for generated in generated_list:
        if generated.number_of_nodes() < 4:
            results.append(0.0)
            continue

        gen_nodes = list(generated.nodes())
        gen_raw = _node_features_np(generated, gen_nodes)
        A_gen = _normalized_adjacency_np(generated, gen_nodes)
        gen_feat = A_gen @ A_gen @ gen_raw
        gen_labels_arr = np.array([labels.get(n, 0) % num_classes for n in gen_nodes])

        W_b, b_b = _train_softmax_clf(
            gen_feat, gen_labels_arr, num_classes, epochs=epochs
        )
        probs_b = _softmax_np(ref_feat[test_idx] @ W_b + b_b)
        acc_gen = float(
            np.mean(probs_b[np.arange(len(test_idx)), ref_labels_arr[test_idx]])
        )

        gap = abs(acc_real - acc_gen)
        results.append(max(0.0, 1.0 - gap))

    return results


# ---------------------------------------------------------------------------
# Embedding MMD
# ---------------------------------------------------------------------------


def embedding_mmd(
    generated: nx.Graph,
    reference: nx.Graph,
    sigma: float = 1.0,
) -> float:
    """Compute MMD between node embeddings of generated and reference graphs.

    Uses spectral embeddings when GNN embeddings are not available.

    Args:
        generated: Generated graph.
        reference: Reference graph.
        sigma: Gaussian kernel bandwidth.

    Returns:
        MMD distance (lower is better). We return 1/(1+MMD)
        as a score in [0, 1] where higher is better.
    """
    if not _HAS_NUMPY:
        return 0.0

    gen_emb = _spectral_embedding(generated, k=4)
    ref_emb = _spectral_embedding(reference, k=4)

    if gen_emb is None or ref_emb is None:
        return 0.0

    # Align dimensions
    min_cols = min(gen_emb.shape[1], ref_emb.shape[1])
    gen_emb = gen_emb[:, :min_cols]
    ref_emb = ref_emb[:, :min_cols]

    mmd_val = _compute_mmd(gen_emb, ref_emb, sigma)
    # Convert to score: higher is better
    return 1.0 / (1.0 + mmd_val)


def _compute_mmd(
    X: "np.ndarray",
    Y: "np.ndarray",
    sigma: float = 1.0,
) -> float:
    """Compute MMD^2 between two sets of samples using Gaussian kernel."""

    def kernel(a: "np.ndarray", b: "np.ndarray") -> float:
        diff = a - b
        return float(np.exp(-np.dot(diff, diff) / (2 * sigma**2)))

    n = X.shape[0]
    m = Y.shape[0]

    if n == 0 or m == 0:
        return 0.0

    # E[k(x,x')]
    xx = sum(kernel(X[i], X[j]) for i in range(n) for j in range(n)) / (n * n)
    # E[k(y,y')]
    yy = sum(kernel(Y[i], Y[j]) for i in range(m) for j in range(m)) / (m * m)
    # E[k(x,y)]
    xy = sum(kernel(X[i], Y[j]) for i in range(n) for j in range(m)) / (n * m)

    mmd2 = xx + yy - 2 * xy
    return float(max(0.0, mmd2) ** 0.5)


# ---------------------------------------------------------------------------
# Aggregate D3 score
# ---------------------------------------------------------------------------


def d3_score(
    reference: Optional[nx.Graph],
    generated: Optional[nx.Graph],
) -> dict[str, float]:
    """Compute all D3 metrics and return as a dict.

    Args:
        reference: Reference graph.
        generated: Generated graph.

    Returns:
        Dict with keys: grassmann, node_clf_gap, emb_mmd, aggregate.
    """
    if reference is None or generated is None:
        return _zero_d3()

    if generated.number_of_nodes() < 2:
        return _zero_d3()

    gc = grassmann_coherence(generated, reference)
    ncg = node_clf_gap(generated, reference)
    emmd = embedding_mmd(generated, reference)

    scores = {
        "grassmann": gc,
        "node_clf_gap": ncg,
        "emb_mmd": emmd,
    }

    scores["aggregate"] = _weighted_aggregate(gc, ncg, emmd)

    return scores


def d3_score_batch(
    reference: Optional[nx.Graph],
    generated_list: list[nx.Graph],
) -> list[dict[str, float]]:
    """Batch D3 metrics: reuses ref GCN training across generated graphs.

    Equivalent to ``[d3_score(reference, g) for g in generated_list]`` but
    trains the reference GCN only once for Node Clf. Gap (A2 optimisation).

    Args:
        reference: Reference graph (shared).
        generated_list: List of generated graphs.

    Returns:
        List of score dicts with keys: grassmann, node_clf_gap, emb_mmd, aggregate.
    """
    if reference is None or not generated_list:
        return [_zero_d3() for _ in (generated_list or [])]

    # Separate graphs with >= 2 nodes (valid for D3) from too-small ones
    valid_mask = [g.number_of_nodes() >= 2 for g in generated_list]
    valid_graphs = [g for g, v in zip(generated_list, valid_mask) if v]

    if not valid_graphs:
        return [_zero_d3() for _ in generated_list]

    # Per-graph metrics (fast, no caching benefit)
    gc_scores = [grassmann_coherence(g, reference) for g in valid_graphs]
    emmd_scores = [embedding_mmd(g, reference) for g in valid_graphs]

    # NCG in batch — ref GCN trained once
    ncg_scores = node_clf_gap_batch(valid_graphs, reference)

    # Build results for valid graphs
    valid_results: list[dict[str, float]] = []
    for gc, ncg, emmd in zip(gc_scores, ncg_scores, emmd_scores):
        scores = {
            "grassmann": gc,
            "node_clf_gap": ncg,
            "emb_mmd": emmd,
        }
        scores["aggregate"] = _weighted_aggregate(gc, ncg, emmd)
        valid_results.append(scores)

    # Reconstruct full list (zeros for too-small graphs)
    results: list[dict[str, float]] = []
    vi = 0
    for v in valid_mask:
        if v:
            results.append(valid_results[vi])
            vi += 1
        else:
            results.append(_zero_d3())
    return results


def _zero_d3() -> dict[str, float]:
    return {
        "grassmann": 0.0,
        "node_clf_gap": 0.0,
        "emb_mmd": 0.0,
        "aggregate": 0.0,
    }
