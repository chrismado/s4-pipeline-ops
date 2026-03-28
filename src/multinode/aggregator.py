"""
Multi-node aggregator — polls agents on remote GPU nodes and combines metrics.

The aggregator maintains a registry of known nodes and periodically polls
each node's /agent/metrics endpoint to build a unified view of the cluster.

Usage:
    aggregator = NodeAggregator(["http://node1:8101", "http://node2:8101"])
    cluster = aggregator.collect_all()

Nodes can also self-register via POST /nodes/register.
"""

from datetime import UTC, datetime, timedelta
from typing import Optional

import httpx
from loguru import logger



class NodeInfo:
    """Tracked state for a single remote node."""

    def __init__(self, url: str, node_id: str = ""):
        self.url = url.rstrip("/")
        self.node_id = node_id or url
        self.last_seen: Optional[datetime] = None
        self.last_metrics: Optional[dict] = None
        self.healthy: bool = False
        self.gpu_count: int = 0

    @property
    def is_stale(self) -> bool:
        if self.last_seen is None:
            return True
        return (datetime.now(UTC) - self.last_seen) > timedelta(minutes=2)


class NodeAggregator:
    """
    Collects metrics from all registered agent nodes.

    Nodes can be added statically (config) or dynamically (register endpoint).
    Each poll cycle fetches /agent/metrics from every node and merges the results.
    """

    def __init__(self, node_urls: list[str] | None = None):
        self.nodes: dict[str, NodeInfo] = {}
        for url in (node_urls or []):
            self.add_node(url)

    def add_node(self, url: str, node_id: str = "") -> NodeInfo:
        """Register a new node."""
        node = NodeInfo(url=url, node_id=node_id or url)
        self.nodes[node.url] = node
        logger.info(f"Registered node: {node.url}")
        return node

    def remove_node(self, url: str) -> bool:
        """Unregister a node."""
        url = url.rstrip("/")
        if url in self.nodes:
            del self.nodes[url]
            logger.info(f"Removed node: {url}")
            return True
        return False

    def collect_all(self, timeout: float = 5.0) -> dict:
        """
        Poll all nodes and return aggregated cluster metrics.

        Returns:
            {
                "nodes": [...],
                "cluster_gpu_count": N,
                "cluster_gpus": [...],
                "healthy_nodes": N,
                "total_nodes": N,
                "timestamp": "..."
            }
        """
        results = []
        all_gpus = []

        for url, node in self.nodes.items():
            data = self._poll_node(node, timeout)
            if data:
                results.append(data)
                # Extract GPU metrics from the node
                node_metrics = data.get("metrics", {})
                for gpu_data in node_metrics.get("gpus", []):
                    # Tag GPU with node info
                    gpu_data["node_id"] = data.get("node_id", url)
                    gpu_data["node_url"] = url
                    all_gpus.append(gpu_data)

        healthy = sum(1 for n in self.nodes.values() if n.healthy)

        return {
            "nodes": results,
            "cluster_gpu_count": len(all_gpus),
            "cluster_gpus": all_gpus,
            "healthy_nodes": healthy,
            "total_nodes": len(self.nodes),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_node_status(self) -> list[dict]:
        """Return status of all registered nodes."""
        return [
            {
                "url": n.url,
                "node_id": n.node_id,
                "healthy": n.healthy,
                "gpu_count": n.gpu_count,
                "last_seen": n.last_seen.isoformat() if n.last_seen else None,
                "stale": n.is_stale,
            }
            for n in self.nodes.values()
        ]

    def _poll_node(self, node: NodeInfo, timeout: float) -> Optional[dict]:
        """Fetch metrics from a single node."""
        try:
            resp = httpx.get(f"{node.url}/agent/metrics", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            node.last_seen = datetime.now(UTC)
            node.last_metrics = data
            node.healthy = True
            node.node_id = data.get("node_id", node.url)
            node.gpu_count = data.get("gpu_count", 0)
            return data
        except Exception as e:
            node.healthy = False
            logger.warning(f"Failed to poll node {node.url}: {e}")
            return None
