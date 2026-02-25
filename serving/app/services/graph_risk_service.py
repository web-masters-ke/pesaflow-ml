"""Graph Risk Engine — Network analysis for merchant and user risk propagation."""

import uuid
from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger


class GraphRiskService:
    """Graph-based risk analysis using PageRank, community detection, and risk propagation."""

    def __init__(self, db_pool: Any = None, damping_factor: float = 0.85, max_iterations: int = 100):
        self._db = db_pool
        self._damping = damping_factor
        self._max_iter = max_iterations

    async def build_merchant_graph(self) -> dict:
        """Build merchant graph from database edges."""
        if not self._db:
            return {"nodes": [], "edges": []}

        try:
            async with self._db.acquire() as conn:
                edges = await conn.fetch(
                    """
                    SELECT source_merchant_id, target_merchant_id, edge_type, weight, shared_entity_count
                    FROM merchant_graph_edges
                    """
                )

                nodes = set()
                edge_list = []
                for row in edges:
                    src = str(row["source_merchant_id"])
                    tgt = str(row["target_merchant_id"])
                    nodes.add(src)
                    nodes.add(tgt)
                    edge_list.append(
                        {
                            "source": src,
                            "target": tgt,
                            "type": row["edge_type"],
                            "weight": float(row["weight"]),
                            "shared_count": row["shared_entity_count"],
                        }
                    )

                return {"nodes": list(nodes), "edges": edge_list}
        except Exception as e:
            logger.error(f"Failed to build merchant graph: {e}")
            return {"nodes": [], "edges": []}

    def compute_pagerank(self, nodes: list[str], edges: list[dict]) -> dict[str, float]:
        """Compute PageRank scores for all nodes in the graph."""
        if not nodes:
            return {}

        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}

        # Build adjacency matrix
        adj = np.zeros((n, n))
        for edge in edges:
            src_idx = node_idx.get(edge["source"])
            tgt_idx = node_idx.get(edge["target"])
            if src_idx is not None and tgt_idx is not None:
                adj[src_idx][tgt_idx] = edge.get("weight", 1.0)
                adj[tgt_idx][src_idx] = edge.get("weight", 1.0)  # Undirected

        # Normalize columns
        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1
        transition = adj / col_sums

        # Power iteration
        scores = np.ones(n) / n
        for _ in range(self._max_iter):
            new_scores = (1 - self._damping) / n + self._damping * transition @ scores
            if np.abs(new_scores - scores).sum() < 1e-8:
                break
            scores = new_scores

        return {nodes[i]: float(scores[i]) for i in range(n)}

    def detect_communities(self, nodes: list[str], edges: list[dict]) -> dict[str, int]:
        """Detect communities using label propagation algorithm."""
        if not nodes:
            return {}

        # Build adjacency list
        neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in edges:
            src, tgt = edge["source"], edge["target"]
            w = edge.get("weight", 1.0)
            neighbors[src].append((tgt, w))
            neighbors[tgt].append((src, w))

        # Initialize each node with its own community
        labels = {node: i for i, node in enumerate(nodes)}

        for _ in range(self._max_iter):
            changed = False
            for node in nodes:
                if not neighbors[node]:
                    continue

                # Count weighted votes from neighbors
                votes: dict[int, float] = defaultdict(float)
                for neighbor, weight in neighbors[node]:
                    votes[labels[neighbor]] += weight

                best_label = max(votes, key=lambda k: votes[k])
                if labels[node] != best_label:
                    labels[node] = best_label
                    changed = True

            if not changed:
                break

        return labels

    def propagate_risk(
        self,
        nodes: list[str],
        edges: list[dict],
        known_risk_scores: dict[str, float],
        propagation_decay: float = 0.7,
    ) -> dict[str, float]:
        """Propagate risk scores through the graph from known risky nodes."""
        risk_scores = {node: known_risk_scores.get(node, 0.0) for node in nodes}

        # Build adjacency list
        neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in edges:
            src, tgt = edge["source"], edge["target"]
            w = edge.get("weight", 1.0)
            neighbors[src].append((tgt, w))
            neighbors[tgt].append((src, w))

        # Multi-hop risk propagation
        for _ in range(3):  # 3 hops
            new_scores = dict(risk_scores)
            for node in nodes:
                if not neighbors[node]:
                    continue

                neighbor_risk = sum(risk_scores.get(n, 0.0) * w * propagation_decay for n, w in neighbors[node])
                max_neighbor_risk = max(
                    (risk_scores.get(n, 0.0) * w for n, w in neighbors[node]),
                    default=0.0,
                )

                # Take max of own score and propagated risk
                propagated = max(neighbor_risk / max(len(neighbors[node]), 1), max_neighbor_risk * propagation_decay)
                new_scores[node] = max(risk_scores[node], min(1.0, propagated))

            risk_scores = new_scores

        return risk_scores

    async def compute_graph_metrics(self) -> dict:
        """Compute and store all graph metrics for merchants."""
        graph = await self.build_merchant_graph()
        nodes = graph["nodes"]
        edges = graph["edges"]

        if not nodes:
            return {"status": "no_data", "nodes_processed": 0}

        # Compute all metrics
        pagerank = self.compute_pagerank(nodes, edges)
        communities = self.detect_communities(nodes, edges)

        # Degree centrality
        degree: dict[str, int] = defaultdict(int)
        for edge in edges:
            degree[edge["source"]] += 1
            degree[edge["target"]] += 1
        max_degree = max(degree.values()) if degree else 1
        degree_centrality = {n: degree.get(n, 0) / max_degree for n in nodes}

        # Fetch known risk scores for propagation
        known_risks = await self._get_known_risk_scores(nodes)
        propagated_risk = self.propagate_risk(nodes, edges, known_risks)

        # Store metrics
        await self._store_graph_metrics(nodes, pagerank, communities, degree_centrality, propagated_risk)

        return {
            "status": "completed",
            "nodes_processed": len(nodes),
            "edges_processed": len(edges),
            "communities_found": len(set(communities.values())),
        }

    async def get_merchant_graph_features(self, merchant_id: str) -> dict:
        """Get graph-based features for a specific merchant (fed into ML model)."""
        if not self._db:
            return {"pagerank_score": 0.0, "community_risk": 0.0, "degree_centrality": 0.0}

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT pagerank_score, community_id, degree_centrality,
                           betweenness_centrality, cluster_risk_score
                    FROM merchant_graph_metrics
                    WHERE merchant_id = $1
                    ORDER BY computed_at DESC LIMIT 1
                    """,
                    merchant_id,
                )

                if row:
                    return {
                        "pagerank_score": float(row["pagerank_score"]),
                        "community_id": row["community_id"],
                        "degree_centrality": float(row["degree_centrality"]),
                        "betweenness_centrality": float(row["betweenness_centrality"]),
                        "cluster_risk_score": float(row["cluster_risk_score"]),
                    }
        except Exception as e:
            logger.error(f"Failed to get graph features for merchant {merchant_id}: {e}")

        return {"pagerank_score": 0.0, "community_risk": 0.0, "degree_centrality": 0.0}

    async def _get_known_risk_scores(self, merchant_ids: list[str]) -> dict[str, float]:
        """Fetch known merchant risk scores from DB."""
        if not self._db or not merchant_ids:
            return {}

        try:
            async with self._db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT merchant_id::text, risk_score
                    FROM merchant_risk_profiles
                    WHERE merchant_id::text = ANY($1)
                    """,
                    merchant_ids,
                )
                return {str(row["merchant_id"]): float(row["risk_score"]) for row in rows}
        except Exception:
            return {}

    async def _store_graph_metrics(
        self,
        nodes: list[str],
        pagerank: dict[str, float],
        communities: dict[str, int],
        degree_centrality: dict[str, float],
        propagated_risk: dict[str, float],
    ) -> None:
        """Persist graph metrics to database."""
        if not self._db:
            return

        try:
            async with self._db.acquire() as conn:
                for node in nodes:
                    await conn.execute(
                        """
                        INSERT INTO merchant_graph_metrics
                        (id, merchant_id, pagerank_score, community_id, degree_centrality,
                         betweenness_centrality, cluster_risk_score)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        str(uuid.uuid4()),
                        node,
                        pagerank.get(node, 0.0),
                        communities.get(node, 0),
                        degree_centrality.get(node, 0.0),
                        0.0,  # Betweenness computed separately for large graphs
                        propagated_risk.get(node, 0.0),
                    )

            logger.info(f"Stored graph metrics for {len(nodes)} merchants")
        except Exception as e:
            logger.error(f"Failed to store graph metrics: {e}")
