import json
from mcp.server.fastmcp import FastMCP
from locus_sdk import LocusMemory

mcp = FastMCP("Locus Bi-Temporal Memory")
memory = LocusMemory(db_path="locus_mcp.db", scope_id="global_user")


@mcp.tool()
def remember_episode(content: str, scope_id: str = "global_user") -> str:
    """Ingest a user message into the Hippocampal Episodic Buffer (<1ms)."""
    ep_id = memory.engine.ingest_episode(scope_id, content, min_density=0.35)
    if ep_id is None:
        return json.dumps({"status": "REJECTED_LOW_DENSITY_NOISE"})
    return json.dumps({"status": "ADMITTED_TO_WORKING_MEMORY", "episode_id": ep_id})


@mcp.tool()
def store_atomic_fact(
    subject: str,
    predicate: str,
    object_val: str,
    cardinality: str = "functional",
    is_negated: bool = False,
    scope_id: str = "global_user",
) -> str:
    """Store or update a structured (subject, predicate, object) fact with deterministic 1:1 conflict resolution."""
    ep_id = memory.engine.ingest_episode(
        scope_id, f"{subject} {predicate} {object_val}", min_density=0.0
    )
    return memory.engine.consolidate_triple(
        scope_id=scope_id,
        subject=subject,
        predicate=predicate,
        object=object_val,
        cardinality=cardinality,
        confidence=0.95,
        parent_episode_id=ep_id or 0,
        is_negated=is_negated,
    )


@mcp.tool()
def recall_context(
    query: str,
    focal_entity: str = "USER",
    as_of_timestamp: int | None = None,
    scope_id: str = "global_user",
) -> str:
    """Retrieve ACT-R ranked memories, 2-hop entity subgraphs, or time-travel ('as_of') state."""
    if as_of_timestamp is not None:
        return memory.engine.retrieve(scope_id, query, as_of=as_of_timestamp)
    memory.scope_id = scope_id
    return memory.compile_context(query, focal_entity=focal_entity)


@mcp.tool()
def hard_forget_topic(target_keyword: str, scope_id: str = "global_user") -> str:
    """Cascade-delete a secret or topic across graph edges, episodes, FTS5, and core blocks with SHA-256 audit proof."""
    return memory.engine.hard_forget(scope_id, target_keyword)


@mcp.tool()
def inspect_telemetry(scope_id: str = "global_user") -> str:
    """Return memory observability metrics: contradiction turnover rate, active/invalidated edges, and audit counts."""
    return memory.engine.get_telemetry(scope_id)


if __name__ == "__main__":
    mcp.run()