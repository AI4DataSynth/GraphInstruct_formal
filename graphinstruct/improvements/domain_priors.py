"""Domain structural priors for L4 instruction augmentation.

Each prior captures the expected value ranges for key structural properties
(degree, clustering, density, degree distribution) of a real-world graph
domain. CAAP injects the rendered text into the prompt for L4 tasks.

Initial ranges are based on literature and v4 L4 reference-solution analysis.
Day 2+ work may recalibrate ranges by computing ground-truth statistics from
``data/instructions/level_4.json`` reference solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DomainPrior:
    """Expected structural statistics for a graph domain."""

    name: str
    avg_degree: tuple[float, float]
    clustering: tuple[float, float]
    density: tuple[float, float]
    degree_dist: str  # "power-law" | "roughly-uniform" | "bimodal"
    hints: tuple[str, ...] = ()


DOMAIN_PRIORS: dict[str, DomainPrior] = {
    "social": DomainPrior(
        name="social",
        avg_degree=(4.0, 6.0),
        clustering=(0.40, 0.70),
        density=(0.30, 0.50),
        degree_dist="power-law",
        hints=(
            "People form tight friend circles, producing high clustering.",
            "A few hubs have many friends; most nodes have few (power-law).",
        ),
    ),
    "citation": DomainPrior(
        name="citation",
        avg_degree=(2.5, 4.0),
        clustering=(0.05, 0.25),
        density=(0.05, 0.15),
        degree_dist="power-law",
        hints=(
            "Citations form a directed acyclic graph (new papers cite old).",
            "A few seminal papers accumulate many citations (hub papers).",
        ),
    ),
    "biological": DomainPrior(
        name="biological",
        avg_degree=(2.0, 5.0),
        clustering=(0.10, 0.40),
        density=(0.10, 0.30),
        degree_dist="power-law",
        hints=(
            "Protein interaction or regulatory networks are sparse.",
            "Functional modules form weakly connected clusters.",
        ),
    ),
    "ecological": DomainPrior(
        name="ecological",
        avg_degree=(2.0, 4.0),
        clustering=(0.15, 0.35),
        density=(0.10, 0.25),
        degree_dist="roughly-uniform",
        hints=("Food-web producer-consumer links form a shallow hierarchy.",),
    ),
    "communication": DomainPrior(
        name="communication",
        avg_degree=(3.0, 8.0),
        clustering=(0.20, 0.50),
        density=(0.30, 0.50),
        degree_dist="power-law",
        hints=("Messaging networks have active hubs and bursty interactions.",),
    ),
    "infrastructure": DomainPrior(
        name="infrastructure",
        avg_degree=(2.0, 4.0),
        clustering=(0.05, 0.20),
        density=(0.05, 0.20),
        degree_dist="roughly-uniform",
        hints=(
            "Transport or utility networks have regular, grid-like topology.",
            "High betweenness on backbone nodes; most nodes have low degree.",
        ),
    ),
    "knowledge_graph": DomainPrior(
        name="knowledge_graph",
        avg_degree=(2.0, 6.0),
        clustering=(0.05, 0.30),
        density=(0.05, 0.20),
        degree_dist="power-law",
        hints=(
            "Entities are connected by typed relations (directed edges).",
            "A few hub entities (e.g., countries, broad concepts) have many relations.",
        ),
    ),
    "molecular": DomainPrior(
        name="molecular",
        avg_degree=(1.5, 3.5),
        clustering=(0.00, 0.15),
        density=(0.05, 0.25),
        degree_dist="roughly-uniform",
        hints=(
            "Atoms have valence-limited degree (C<=4, N<=3, O<=2).",
            "Molecules are typically sparse, planar, and acyclic or have small rings.",
        ),
    ),
}


def get_prior(domain: Optional[str]) -> Optional[DomainPrior]:
    """Case-insensitive lookup. Returns None for missing/unknown domains."""
    if domain is None:
        return None
    key = domain.strip().lower()
    return DOMAIN_PRIORS.get(key)


def build_domain_text(domain: Optional[str], num_nodes: Optional[int] = None) -> str:
    """Produce the prompt-injection text for a domain prior.

    Returns an empty string when the domain is unknown or None, so callers can
    unconditionally append the result to the base prompt.
    """
    prior = get_prior(domain)
    if prior is None:
        return ""
    size_hint = f" ({num_nodes} nodes)" if num_nodes else ""
    lines = [
        f"Domain context: {prior.name} network{size_hint}.",
        "Expected structural properties for this domain:",
        f"- Average degree: {prior.avg_degree[0]:.1f} - {prior.avg_degree[1]:.1f}",
        f"- Clustering coefficient: "
        f"{prior.clustering[0]:.2f} - {prior.clustering[1]:.2f}",
        f"- Density: {prior.density[0]:.2f} - {prior.density[1]:.2f}",
        f"- Degree distribution: {prior.degree_dist}",
    ]
    for hint in prior.hints:
        lines.append(f"- {hint}")
    lines.append(
        "Generate a graph that satisfies BOTH the explicit constraints "
        "AND these domain expectations."
    )
    return "\n".join(lines)
