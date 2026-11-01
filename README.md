# 🧠 Locus: Bi-Temporal Cognitive Memory Engine for AI Agents

[![Rust Core](https://img.shields.io/badge/Core-Rust%20%2B%20PyO3-orange?logo=rust)](https://github.com/crimsonOmoon/Locus)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://github.com/crimsonOmoon/Locus)
[![Storage](https://img.shields.io/badge/Storage-Single--File%20SQLite%20WAL-003B57?logo=sqlite)](https://github.com/crimsonOmoon/Locus)
[![MCP Ready](https://img.shields.io/badge/MCP-Compatible-success)](https://modelcontextprotocol.io)

**Locus** is a high-performance, embedded AI memory engine written in **Rust with native Python bindings**. Modeled after human neurobiology (Sensory Gatekeeper $\rightarrow$ Hippocampal Episodic Buffer $\rightarrow$ Neocortical Bi-Temporal Graph), Locus solves the hardest problems in agentic memory—**synchronous LLM write latency, contradiction resolution, historical time-travel (`as_of`), activation decay, and verifiable hard-forgetting**—inside a single local SQLite file with zero external database containers.

---

## ⚡ Why Locus? (How It Beats Existing Engines)

Existing memory frameworks force painful trade-offs between latency, infrastructure complexity, and accuracy:

| Feature / Capability | Mem0 (OSS) | Zep / Graphiti | Letta | Claude Memory | **Locus (This)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Structure** | Vector store (Graph paywalled) | Bi-temporal KG (Requires Neo4j/FalkorDB) | Agent-editable blocks + archival | Topic list + chat search | **Embedded Bi-Temporal KG + Vector + FTS5 + Core Blocks** |
| **Write-Path Latency** | Slow (LLM calls on every write) | Slow (LLM graph extraction per turn) | Moderate (Tool call overhead) | Automatic / Manual | **`~2.0 ms` per turn (Zero-LLM Fast-Path + Async Sleep Worker)** |
| **Contradiction Detection** | Shallow top-$k$ LLM prompt | Invalidates old edges via LLM | Manual agent tool calls | Manual + basic auto | **Deterministic Predicate Cardinality (`1:1` vs `1:N`)** |
| **Time-Travel (`as_of`)** | Moderate | Excellent (`valid_at` / `invalid_at`) | Weak (Agent managed) | Limited | **Native Sub-Millisecond `as_of` Queries (`1.36 ms`)** |
| **Memory Decay & Promotion** | Static recency | Recency / confidence filters | Manual block edits | None | **ACT-R Cognitive Decay + Hebbian Core Auto-Promotion** |
| **Hard Forgetting (`GDPR`)** | Soft/partial delete | Edge invalidation | Manual block edit | Manual delete | **Lineage DAG Cascade Purge + SHA-256 Cryptographic Proof** |
| **Self-Hosted Infra** | Vector DB required | External Graph DB required | Full agent server | Closed SaaS | **Zero-Server Single SQLite File (`pip install`)** |

---

## 🏗️ Cognitive Architecture

```text
User Turn ──► [1. Sensory Gatekeeper (<0.2ms)]
               ├─ Low Entropy / Noise ("ok", "thanks") ──► DROPPED
               └─ High Information Density
                   │
                   ▼
        [2. Hippocampal Working Buffer (Fast-Path SQLite + FTS5, ~2ms)]
          │        └─ Immediately searchable via Working-Memory Fallback
          │
          ▼ (Async Background "Sleep Consolidation" Worker via Local Ollama / LLM)
        [3. Neocortical Bi-Temporal Knowledge Graph]
          ├─ Functional (1:1) Predicate? ──► Auto-Invalidates Old Fact (valid_to = NOW)
          ├─ Repeated Fact (N >= 3)?     ──► Hebbian Promotion to Pinned Core Memory Block
          └─ Explicit Negation?          ──► Sets valid_to = NOW on Target Edge
                   │
                   ▼
        [4. Multi-Signal ACT-R Retrieval & 2-Hop Subgraph Compiler]
          └─ Ranks via Vector Cosine + Lexical Overlap + ACT-R Decay + Associative Cue Gate
```

### Key Technical Innovations

1. **Dual-Speed Write Pipeline (Fast Hippocampus / Slow Neocortex):**
   Instead of blocking live chat for 5–15 seconds to run LLM conflict checks, Locus admits high-density turns into an FTS5 working buffer in **`~2 ms`** and consolidates structured `(Subject, Predicate, Object)` triples asynchronously in the background.
2. **Deterministic Predicate Cardinality (`functional` vs `multi`):**
   Relationships like `LIVES_IN`, `PRIMARY_OS`, or `AGE` are marked `functional` (`1:1`). When a user moves from `Siliguri` to `Kolkata`, the Rust engine deterministically invalidates the old edge (`valid_to = timestamp`) and inserts the new one **with zero LLM conflict-detection tokens**.
3. **ACT-R Cognitive Activation & Associative Cue Gate:**
   Memories are scored using the cognitive activation equation:
   $$S_i = w_1 \cdot \text{CosineSim}(Q, M_i) + w_2 \cdot \text{Lexical}(Q, M_i) + w_3 \cdot \max\left(0, \ln(1 + N_i) - d \ln(1 + t_{\text{hours}})\right) + w_4 \cdot C_i$$
   Frequently used memories automatically promote into Letta-style **Pinned Core Memory Blocks**, while unused trivia naturally decays.
4. **Lineage DAG Cascade Forgetting + SHA-256 Audit Proof:**
   Every graph edge and core block tracks its `parent_episode_id`. Calling `hard_forget("keyword")` traces the lineage upward and downward—wiping the graph edges, raw episodic text, FTS5 tokens, and core blocks—and logs a **SHA-256 cryptographic receipt**.

---

## 📊 Reproducible Benchmark Scorecard

Run `python eval_suite.py` to verify Locus on your hardware:

| Metric | Locus Result | Target |
| :--- | :--- | :--- |
| **Fast-Path Write Latency (2 turns)** | **`4.18 ms`** (`2.09 ms/turn`) | `< 5.00 ms` |
| **Full Context Retrieval Latency** | **`1.36 ms`** | `< 5.00 ms` |
| **Sensory Noise Gatekeeper** | **`PASS`** | `PASS` |
| **1:1 Contradiction Resolution** | **`100%`** | `100%` |
| **Bi-Temporal Time-Travel (`as_of`)** | **`100%`** | `100%` |
| **Hebbian Core Memory Auto-Promotion** | **`PASS`** | `PASS` |
| **Hard-Forget Zero-Leakage Rate** | **`100%`** | `100%` |

---

## 🚀 Quickstart & Installation

### Prerequisites
* **Rust toolchain** (`cargo` / `rustc`)
* **Python 3.10+**

### Build from Source
```bash
git clone [https://github.com/crimsonOmoon/Locus.git](https://github.com/crimsonOmoon/Locus.git)
cd Locus

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install maturin mcp
maturin develop --release
```

### Run the Verification & Benchmark Suites
```bash
python test_brain.py
python eval_suite.py
```

---

## 💻 Python SDK Usage (`LocusMemory`)

```python
import json
from locus_sdk import LocusMemory

# Initialize embedded single-file engine (optional: enable background Ollama worker)
mem = LocusMemory(db_path="agent_brain.db", scope_id="user_crimson", auto_sleep_worker=False)

# 1. Fast-Path Ingestion (<2ms, filters out low-entropy noise automatically)
mem.remember("ok thanks")  # -> Returns None (Filtered by Sensory Gatekeeper)
ep1 = mem.remember("I currently live in Siliguri.", timestamp_override=1000)
ep2 = mem.remember("I moved to Kolkata for my B.Tech at IEM.", timestamp_override=2000)

# 2. Deterministic Bi-Temporal Consolidation
mem.engine.consolidate_triple("user_crimson", "USER", "LIVES_IN", "Siliguri", "functional", 0.92, ep1, valid_from_override=1000)
mem.engine.consolidate_triple("user_crimson", "USER", "LIVES_IN", "Kolkata", "functional", 0.96, ep2, valid_from_override=2000)

# 3. Time-Travel Query ("Where did the user live at t=1500?")
past = json.loads(mem.engine.retrieve("user_crimson", "Where does user live?", as_of=1500))
print("At t=1500:", past[0]["fact"])  # -> USER LIVES_IN Siliguri

# 4. Compile Full System Prompt Context (Core Blocks + ACT-R Ranked Facts + 2-Hop Subgraph)
print(mem.compile_context("Where does the user live now?"))

# 5. Cryptographic Hard-Forget
receipt = json.loads(mem.engine.hard_forget("user_crimson", "Siliguri"))
print("SHA-256 Purge Proof:", receipt["cryptographic_proof_sha256"])
```

### Sample Compiled Prompt Context Output
```xml
<locus_memory_context>
  [PINNED CORE MEMORY]
  - USER:STUDIES_AT: USER STUDIES_AT IEM Kolkata
  [RELEVANT RECALLED MEMORIES (ACT-R RANKED)]
  - USER STUDIES_AT IEM Kolkata (score: 0.445, conf: 0.99)
  - USER PRIMARY_OS Kali Linux (score: 0.408, conf: 0.95)
  [ENTITY SUBGRAPH (USER 2-HOP)]
  - (Hop 1) USER -> PRIMARY_OS -> Kali Linux
  - (Hop 1) USER -> STUDIES_AT -> IEM Kolkata
</locus_memory_context>
```

---

## 🔌 Model Context Protocol (MCP) Server

Locus ships with a ready-to-run MCP server (`locus_mcp.py`) for **Claude Desktop, Cursor, Windsurf, and Cline**:

```bash
python locus_mcp.py
```

Exposed MCP Tools:
* `remember_episode(content, scope_id)` — Fast-path hippocampal buffer write.
* `store_atomic_fact(subject, predicate, object_val, cardinality, is_negated)` — Bi-temporal graph write with `1:1` conflict resolution.
* `recall_context(query, focal_entity, as_of_timestamp)` — Multi-signal ACT-R retrieval, 2-hop subgraph, and time-travel queries.
* `hard_forget_topic(target_keyword)` — Cascade lineage purge with SHA-256 verification.
* `inspect_telemetry(scope_id)` — Live contradiction turnover rate and memory observability stats.

---

## 📜 License

MIT License © 2026 [crimsonOmoon](https://github.com/crimsonOmoon)
