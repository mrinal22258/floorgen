"""
Graph Store for FloorGen: Neo4j & NetworkX dual backend.
Stores bubble diagrams and supports structural topological queries
(e.g., finding floorplans where a master bedroom is directly connected to a bathroom).
"""

from typing import List, Dict, Any, Optional
import networkx as nx


class FloorplanGraphStore:
    """
    Graph store supporting both Neo4j (via Cypher queries) and
    in-memory NetworkX graphs for zero-dependency local execution.
    """

    def __init__(self, uri: Optional[str] = None, auth: Optional[tuple] = None):
        self.uri = uri
        self.auth = auth
        self.driver = None
        self.networkx_graphs: Dict[str, nx.Graph] = {}
        self._try_connect_neo4j()

    def _try_connect_neo4j(self):
        if self.uri and self.auth:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
                # Verify connectivity
                self.driver.verify_connectivity()
                print(f"[FloorGen GraphStore] Connected to Neo4j at {self.uri}")
            except Exception as e:
                print(f"[FloorGen GraphStore] Neo4j connection not available ({e}), using in-memory NetworkX engine.")
                self.driver = None
        else:
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def add_plan_graph(self, plan: Dict[str, Any]):
        """Inserts a floorplan's room nodes and adjacency edges into the graph store."""
        plan_id = plan["id"]
        rooms = plan.get("rooms", [])
        adj = plan.get("adjacency", [])

        # Store in NetworkX
        G = nx.Graph()
        G.graph["id"] = plan_id
        G.graph["archetype"] = plan.get("archetype", "")
        
        for r in rooms:
            G.add_node(
                r["id"],
                category=r.get("category", "room"),
                area=r.get("area", 0.0),
                centroid=r.get("centroid", [0, 0])
            )

        for u, v in adj:
            G.add_edge(u, v)

        self.networkx_graphs[plan_id] = G

        # If Neo4j is connected, write to Neo4j
        if self.driver:
            try:
                with self.driver.session() as session:
                    session.execute_write(self._write_neo4j_plan, plan)
            except Exception as e:
                print(f"Failed to write plan {plan_id} to Neo4j: {e}")

    @staticmethod
    def _write_neo4j_plan(tx, plan: Dict[str, Any]):
        plan_id = plan["id"]
        tx.run("MERGE (p:Floorplan {id: $id, archetype: $archetype})",
               id=plan_id, archetype=plan.get("archetype", ""))
        
        for r in plan.get("rooms", []):
            tx.run("""
                MERGE (rm:Room {id: $room_uid, plan_id: $plan_id, category: $cat, local_id: $loc_id, area: $area})
                WITH rm
                MATCH (p:Floorplan {id: $plan_id})
                MERGE (p)-[:CONTAINS]->(rm)
            """, room_uid=f"{plan_id}_{r['id']}", plan_id=plan_id, cat=r["category"], loc_id=r["id"], area=r.get("area", 0.0))

        for u, v in plan.get("adjacency", []):
            tx.run("""
                MATCH (r1:Room {id: $r1_id}), (r2:Room {id: $r2_id})
                MERGE (r1)-[:ADJACENT_TO]-(r2)
            """, r1_id=f"{plan_id}_{u}", r2_id=f"{plan_id}_{v}")

    def query_by_subgraph(
        self,
        required_rooms: List[str],
        required_adjacencies: Optional[List[tuple]] = None
    ) -> List[str]:
        """
        Queries the graph store for plan IDs matching room categories and topological adjacencies.
        Example: required_rooms=["master_bedroom", "bathroom"], required_adjacencies=[("master_bedroom", "bathroom")]
        """
        matching_ids = []

        for plan_id, G in self.networkx_graphs.items():
            # Check room types
            categories = [G.nodes[n].get("category", "") for n in G.nodes]
            has_all_rooms = True
            for req in required_rooms:
                if req not in categories:
                    has_all_rooms = False
                    break
            if not has_all_rooms:
                continue

            # Check required adjacencies
            if required_adjacencies:
                has_all_adj = True
                for cat_a, cat_b in required_adjacencies:
                    adj_found = False
                    for u, v in G.edges():
                        cu = G.nodes[u].get("category", "")
                        cv = G.nodes[v].get("category", "")
                        if (cu == cat_a and cv == cat_b) or (cu == cat_b and cv == cat_a):
                            adj_found = True
                            break
                    if not adj_found:
                        has_all_adj = False
                        break
                if not has_all_adj:
                    continue

            matching_ids.append(plan_id)

        return matching_ids
