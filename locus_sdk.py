import json
import queue
import threading
import urllib.request
from typing import Callable, Optional
from locus import LocusEngine


class LocusMemory:
    """High-level Python SDK wrapping the Rust LocusEngine."""

    def __init__(
        self,
        db_path: str = "locus_memory.db",
        scope_id: str = "default_user",
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5:3b",
        embed_model: Optional[str] = None,
        auto_sleep_worker: bool = False,
    ):
        self.engine = LocusEngine(db_path)
        self.scope_id = scope_id
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.embed_model = embed_model
        self._sleep_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        if auto_sleep_worker:
            self._worker = threading.Thread(target=self._background_sleep_loop, daemon=True)
            self._worker.start()

    # -------------------- 1. FAST-PATH WRITE --------------------
    def remember(
        self,
        text: str,
        min_density: float = 0.35,
        timestamp_override: Optional[int] = None,
    ) -> Optional[int]:
        """<1ms synchronous episodic admission. Returns episode_id or None if noise."""
        ep_id = self.engine.ingest_episode(
            self.scope_id, text, min_density=min_density, timestamp_override=timestamp_override
        )
        if ep_id is not None:
            self._sleep_queue.put(ep_id)
        return ep_id

    # -------------------- 2. PROMPT CONTEXT COMPILER --------------------
    def compile_context(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        focal_entity: str = "USER",
        top_k: int = 5,
    ) -> str:
        """Compiles Core Memory + ACT-R Graph Facts + Working Memory + 2-Hop Subgraph into a System Prompt block."""
        core_blocks = self.engine.get_core_blocks(self.scope_id)
        recalled = json.loads(
            self.engine.retrieve(
                self.scope_id,
                query,
                query_embedding=query_embedding,
                top_k=top_k,
                include_working_memory=True,
            )
        )
        subgraph = json.loads(
            self.engine.traverse_subgraph(self.scope_id, focal_entity, max_hops=2)
        )

        sections = ["<locus_memory_context>"]

        if core_blocks:
            sections.append("  [PINNED CORE MEMORY]")
            for key, val in core_blocks:
                sections.append(f"  - {key}: {val}")

        if recalled:
            sections.append("  [RELEVANT RECALLED MEMORIES (ACT-R RANKED)]")
            for item in recalled:
                sections.append(
                    f"  - {item['fact']} (score: {item['act_r_score']}, conf: {item['confidence']:.2f})"
                )

        if subgraph:
            sections.append(f"  [ENTITY SUBGRAPH ({focal_entity} 2-HOP)]")
            for hop in subgraph[:6]:
                sections.append(
                    f"  - (Hop {hop['hop']}) {hop['subject']} -> {hop['predicate']} -> {hop['object']}"
                )

        sections.append("</locus_memory_context>")
        return "\n".join(sections)

    # -------------------- 3. LETTA-STYLE AGENT CONTROL TOOLS --------------------
    def get_agent_tools(self) -> list[dict]:
        """Returns OpenAI/Ollama-compatible tool schemas so the agent can self-edit memory."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "locus_upsert_fact",
                    "description": "Explicitly store or update an atomic fact in the bi-temporal knowledge graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "cardinality": {"type": "string", "enum": ["functional", "multi"]},
                            "is_negated": {"type": "boolean"},
                        },
                        "required": ["subject", "predicate", "object", "cardinality"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "locus_hard_forget",
                    "description": "Cryptographically purge a topic/secret from all graph edges, episodes, and core blocks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_keyword": {"type": "string"}
                        },
                        "required": ["target_keyword"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "locus_time_travel_query",
                    "description": "Query what facts were true at a specific past Unix timestamp.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "as_of_timestamp": {"type": "integer"},
                        },
                        "required": ["query", "as_of_timestamp"],
                    },
                },
            },
        ]

    def execute_agent_tool(self, tool_name: str, arguments: dict) -> str:
        """Executes a tool call invoked by the LLM agent."""
        if tool_name == "locus_upsert_fact":
            ep_id = self.remember(
                f"Agent tool write: {arguments['subject']} {arguments['predicate']} {arguments['object']}",
                min_density=0.0,
            )
            return self.engine.consolidate_triple(
                scope_id=self.scope_id,
                subject=arguments["subject"],
                predicate=arguments["predicate"],
                object=arguments["object"],
                cardinality=arguments.get("cardinality", "functional"),
                confidence=0.98,
                parent_episode_id=ep_id or 0,
                is_negated=arguments.get("is_negated", False),
            )
        elif tool_name == "locus_hard_forget":
            return self.engine.hard_forget(self.scope_id, arguments["target_keyword"])
        elif tool_name == "locus_time_travel_query":
            return self.engine.retrieve(
                self.scope_id,
                arguments["query"],
                as_of=int(arguments["as_of_timestamp"]),
            )
        raise ValueError(f"Unknown Locus tool: {tool_name}")

    # -------------------- 4. BACKGROUND SLEEP CONSOLIDATION --------------------
    def _background_sleep_loop(self):
        while not self._stop_event.is_set():
            try:
                _ = self._sleep_queue.get(timeout=1.0)
                self._consolidate_pending_with_ollama()
            except queue.Empty:
                continue

    def _consolidate_pending_with_ollama(self):
        pending = self.engine.fetch_unconsolidated_episodes(self.scope_id, 5)
        for ep_id, content, created_at in pending:
            try:
                triples = self._extract_ollama(content)
                for t in triples:
                    self.engine.consolidate_triple(
                        scope_id=self.scope_id,
                        subject=t["subject"],
                        predicate=t["predicate"],
                        object=t["object"],
                        cardinality=t["cardinality"],
                        confidence=float(t.get("confidence", 0.9)),
                        parent_episode_id=ep_id,
                        is_negated=bool(t.get("is_negated", False)),
                        valid_from_override=created_at,
                    )
            except Exception:
                # If Ollama is offline, episode remains safely in Working Memory (consolidated=0)
                pass

    def _extract_ollama(self, text: str) -> list[dict]:
        schema = {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "cardinality": {"type": "string", "enum": ["functional", "multi"]},
                            "confidence": {"type": "number"},
                            "is_negated": {"type": "boolean"},
                        },
                        "required": ["subject", "predicate", "object", "cardinality", "confidence", "is_negated"],
                    },
                }
            },
            "required": ["triples"],
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract atomic facts as triples. Use 'USER' for the speaker. "
                        "Use 'functional' for 1:1 facts (LIVES_IN, CURRENT_COLLEGE, AGE) and 'multi' for 1:N facts."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return json.loads(body["message"]["content"]).get("triples", [])