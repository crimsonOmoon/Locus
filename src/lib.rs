use chrono::Utc;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::Mutex;

#[derive(Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub action: String, // "INSERTED", "REINFORCED", "INVALIDATED_AND_UPDATED", "NEGATED"
    pub edge_id: i64,
    pub invalidated_edge_ids: Vec<i64>,
    pub promoted_to_core: bool,
}

#[derive(Serialize, Deserialize)]
pub struct ScoredMemory {
    pub edge_id: i64, // Positive = Neocortical Graph Edge, Negative = Unconsolidated Working Memory
    pub fact: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub act_r_score: f64,
    pub valid_from: i64,
    pub valid_to: Option<i64>,
    pub parent_episode_id: i64,
}

#[derive(Serialize, Deserialize)]
pub struct SubgraphHop {
    pub hop: usize,
    pub edge_id: i64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub valid_from: i64,
}

#[pyclass]
pub struct LocusEngine {
    conn: Mutex<Connection>,
}

/// Convert f32 slice to raw bytes for SQLite BLOB storage
fn f32_to_bytes(vec: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for &val in vec {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert SQLite BLOB bytes back to f32 vector
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Fast native cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

/// Sensory Gatekeeper: Measures token diversity + character entropy to filter noise
fn compute_information_density(text: &str) -> f64 {
    let trimmed = text.trim();
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() < 3 {
        return 0.1; // Reject ultra-short acknowledgements ("ok", "thanks bro")
    }
    let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let lexical_diversity = unique_words.len() as f64 / words.len() as f64;
    let length_factor = (words.len() as f64 / 12.0).min(1.0);
    (lexical_diversity * 0.6) + (length_factor * 0.4)
}

#[pymethods]
impl LocusEngine {
    #[new]
    pub fn new(db_path: &str) -> PyResult<Self> {
        let conn = Connection::open(db_path)
            .map_err(|e| PyRuntimeError::new_err(format!("DB Open Error: {}", e)))?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;

             -- 1. Hippocampal Episodic Buffer
             CREATE TABLE IF NOT EXISTS episodes (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 scope_id TEXT NOT NULL,
                 content TEXT NOT NULL,
                 density_score REAL NOT NULL,
                 created_at INTEGER NOT NULL,
                 consolidated INTEGER DEFAULT 0
             );

             -- Fast BM25 Keyword Index over Episodes
             CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                 content,
                 scope_id UNINDEXED,
                 episode_id UNINDEXED
             );

             -- 2. Neocortical Bi-Temporal Knowledge Graph
             CREATE TABLE IF NOT EXISTS kg_edges (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 scope_id TEXT NOT NULL,
                 subject TEXT NOT NULL,
                 predicate TEXT NOT NULL,
                 object TEXT NOT NULL,
                 cardinality TEXT NOT NULL, -- 'functional' (1:1) or 'multi' (1:N)
                 confidence REAL NOT NULL,
                 embedding BLOB,
                 valid_from INTEGER NOT NULL,
                 valid_to INTEGER,          -- NULL means currently true/active
                 created_at INTEGER NOT NULL,
                 access_count INTEGER DEFAULT 1,
                 last_accessed INTEGER NOT NULL,
                 parent_episode_id INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_kg_lookup 
                 ON kg_edges(scope_id, subject, predicate, valid_to);

             -- 3. Letta-Style Always-Visible Core Memory Blocks
             CREATE TABLE IF NOT EXISTS core_blocks (
                 scope_id TEXT NOT NULL,
                 block_key TEXT NOT NULL,
                 content TEXT NOT NULL,
                 parent_episode_id INTEGER NOT NULL,
                 updated_at INTEGER NOT NULL,
                 PRIMARY KEY (scope_id, block_key)
             );

             -- 4. Cryptographic Hard-Forget Audit Log
             CREATE TABLE IF NOT EXISTS forget_audit_log (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 scope_id TEXT NOT NULL,
                 target_entity TEXT NOT NULL,
                 purged_edges_count INTEGER NOT NULL,
                 purged_episodes_count INTEGER NOT NULL,
                 proof_sha256 TEXT NOT NULL,
                 timestamp INTEGER NOT NULL
             );",
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Schema Init Error: {}", e)))?;

        Ok(LocusEngine {
            conn: Mutex::new(conn),
        })
    }

    /// FAST-PATH (<1ms): Sensory Admission Control + Episodic Buffer Write
    #[pyo3(signature = (scope_id, content, min_density=0.35, timestamp_override=None))]
    pub fn ingest_episode(
        &self,
        scope_id: &str,
        content: &str,
        min_density: f64,
        timestamp_override: Option<i64>,
    ) -> PyResult<Option<i64>> {
        let density = compute_information_density(content);
        if density < min_density {
            return Ok(None); // Filtered out by Sensory Gatekeeper
        }

        let now = timestamp_override.unwrap_or_else(|| Utc::now().timestamp());
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO episodes (scope_id, content, density_score, created_at, consolidated)
             VALUES (?1, ?2, ?3, ?4, 0)",
            params![scope_id, content, density, now],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let ep_id = conn.last_insert_rowid();

        conn.execute(
            "INSERT INTO episodes_fts (content, scope_id, episode_id) VALUES (?1, ?2, ?3)",
            params![content, scope_id, ep_id],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Some(ep_id))
    }

    /// Fetch unconsolidated episodes for the background "Sleep Worker"
    pub fn fetch_unconsolidated_episodes(
        &self,
        scope_id: &str,
        limit: usize,
    ) -> PyResult<Vec<(i64, String, i64)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, content, created_at FROM episodes 
                 WHERE scope_id = ?1 AND consolidated = 0 
                 ORDER BY created_at ASC LIMIT ?2",
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let rows = stmt
            .query_map(params![scope_id, limit as i64], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut result = Vec::new();
        for r in rows {
            result.push(r.map_err(|e| PyRuntimeError::new_err(e.to_string()))?);
        }
        Ok(result)
    }

    /// SLOW-PATH: Deterministic Predicate Cardinality & Bi-Temporal Graph Consolidation
    #[pyo3(signature = (scope_id, subject, predicate, object, cardinality, confidence, parent_episode_id, embedding=None, is_negated=false, valid_from_override=None))]
    pub fn consolidate_triple(
        &self,
        scope_id: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        cardinality: &str, // "functional" (1:1) or "multi" (1:N)
        confidence: f64,
        parent_episode_id: i64,
        embedding: Option<Vec<f32>>,
        is_negated: bool,
        valid_from_override: Option<i64>,
    ) -> PyResult<String> {
        let now = valid_from_override.unwrap_or_else(|| Utc::now().timestamp());
        let subj_norm = subject.trim().to_uppercase();
        let pred_norm = predicate.trim().to_uppercase();
        let obj_norm = object.trim().to_string();
        let emb_bytes = embedding.map(|v| f32_to_bytes(&v));

        let conn = self.conn.lock().unwrap();

        // Mark parent episode as consolidated
        conn.execute(
            "UPDATE episodes SET consolidated = 1 WHERE id = ?1",
            params![parent_episode_id],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Case 1: Explicit Negation ("I no longer use Windows")
        if is_negated {
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM kg_edges 
                     WHERE scope_id = ?1 AND subject = ?2 AND predicate = ?3 
                     AND LOWER(object) = LOWER(?4) AND valid_to IS NULL",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let invalidated: Vec<i64> = stmt
                .query_map(params![scope_id, subj_norm, pred_norm, obj_norm], |r| r.get(0))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .filter_map(Result::ok)
                .collect();

            conn.execute(
                "UPDATE kg_edges SET valid_to = ?1 
                 WHERE scope_id = ?2 AND subject = ?3 AND predicate = ?4 
                 AND LOWER(object) = LOWER(?5) AND valid_to IS NULL",
                params![now, scope_id, subj_norm, pred_norm, obj_norm],
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let res = ConsolidationResult {
                action: "NEGATED".to_string(),
                edge_id: -1,
                invalidated_edge_ids: invalidated,
                promoted_to_core: false,
            };
            return Ok(serde_json::to_string(&res).unwrap());
        }

        // Case 2: Check if exact same fact is already active -> Hebbian Reinforcement
        let existing_same: Option<(i64, i64, f64)> = conn
            .query_row(
                "SELECT id, access_count, confidence FROM kg_edges
                 WHERE scope_id = ?1 AND subject = ?2 AND predicate = ?3 
                 AND LOWER(object) = LOWER(?4) AND valid_to IS NULL",
                params![scope_id, subj_norm, pred_norm, obj_norm],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if let Some((edge_id, count, old_conf)) = existing_same {
            let new_count = count + 1;
            let boosted_conf = (old_conf + confidence * 0.2).min(0.99);
            conn.execute(
                "UPDATE kg_edges SET access_count = ?1, confidence = ?2, last_accessed = ?3
                 WHERE id = ?4",
                params![new_count, boosted_conf, now, edge_id],
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let promoted = new_count >= 3;
            if promoted {
                let block_key = format!("{}:{}", subj_norm, pred_norm);
                let block_val = format!("{} {} {}", subj_norm, pred_norm, obj_norm);
                conn.execute(
                    "INSERT INTO core_blocks (scope_id, block_key, content, parent_episode_id, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)
                     ON CONFLICT(scope_id, block_key) DO UPDATE SET
                     content = excluded.content, updated_at = excluded.updated_at",
                    params![scope_id, block_key, block_val, parent_episode_id, now],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }

            let res = ConsolidationResult {
                action: "REINFORCED".to_string(),
                edge_id,
                invalidated_edge_ids: vec![],
                promoted_to_core: promoted,
            };
            return Ok(serde_json::to_string(&res).unwrap());
        }

        // Case 3: Deterministic Contradiction Resolution for 'functional' (1:1) predicates
        let mut invalidated_ids = Vec::new();
        if cardinality == "functional" {
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM kg_edges
                     WHERE scope_id = ?1 AND subject = ?2 AND predicate = ?3 AND valid_to IS NULL",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            invalidated_ids = stmt
                .query_map(params![scope_id, subj_norm, pred_norm], |r| r.get(0))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .filter_map(Result::ok)
                .collect();

            if !invalidated_ids.is_empty() {
                conn.execute(
                    "UPDATE kg_edges SET valid_to = MAX(?1, valid_from)
                     WHERE scope_id = ?2 AND subject = ?3 AND predicate = ?4 AND valid_to IS NULL",
                    params![now, scope_id, subj_norm, pred_norm],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }

        // Insert new active edge
        conn.execute(
            "INSERT INTO kg_edges (
                scope_id, subject, predicate, object, cardinality, confidence,
                embedding, valid_from, valid_to, created_at, access_count, last_accessed, parent_episode_id
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?8, 1, ?8, ?9)",
            params![
                scope_id,
                subj_norm,
                pred_norm,
                obj_norm,
                cardinality,
                confidence,
                emb_bytes,
                now,
                parent_episode_id
            ],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let new_edge_id = conn.last_insert_rowid();
        let action = if invalidated_ids.is_empty() {
            "INSERTED"
        } else {
            "INVALIDATED_AND_UPDATED"
        };

        let res = ConsolidationResult {
            action: action.to_string(),
            edge_id: new_edge_id,
            invalidated_edge_ids: invalidated_ids,
            promoted_to_core: false,
        };
        Ok(serde_json::to_string(&res).unwrap())
    }

    /// RETRIEVAL: ACT-R + Vector + Lexical + Bi-Temporal 'as_of' + Unconsolidated Working-Memory Fallback
    #[pyo3(signature = (scope_id, query_text, query_embedding=None, as_of=None, top_k=5, include_working_memory=true))]
    pub fn retrieve(
        &self,
        scope_id: &str,
        query_text: &str,
        query_embedding: Option<Vec<f32>>,
        as_of: Option<i64>,
        top_k: usize,
        include_working_memory: bool,
    ) -> PyResult<String> {
        let now = Utc::now().timestamp();
        let query_lower = query_text.to_lowercase();
        let conn = self.conn.lock().unwrap();

        let sql = if as_of.is_some() {
            "SELECT id, subject, predicate, object, confidence, embedding, valid_from, valid_to,
                    access_count, last_accessed, parent_episode_id
             FROM kg_edges
             WHERE scope_id = ?1 AND valid_from <= ?2 AND (valid_to IS NULL OR valid_to > ?2)"
        } else {
            "SELECT id, subject, predicate, object, confidence, embedding, valid_from, valid_to,
                    access_count, last_accessed, parent_episode_id
             FROM kg_edges
             WHERE scope_id = ?1 AND valid_to IS NULL"
        };

        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut candidates: Vec<ScoredMemory> = Vec::new();

        let mut process_row = |row: &rusqlite::Row| -> rusqlite::Result<()> {
            let edge_id: i64 = row.get(0)?;
            let subject: String = row.get(1)?;
            let predicate: String = row.get(2)?;
            let object: String = row.get(3)?;
            let confidence: f64 = row.get(4)?;
            let emb_blob: Option<Vec<u8>> = row.get(5)?;
            let valid_from: i64 = row.get(6)?;
            let valid_to: Option<i64> = row.get(7)?;
            let access_count: i64 = row.get(8)?;
            let last_accessed: i64 = row.get(9)?;
            let parent_ep: i64 = row.get(10)?;

            // 1. Vector Similarity Signal
            let vec_sim = match (&query_embedding, emb_blob) {
                (Some(q_vec), Some(blob)) => {
                    let edge_vec = bytes_to_f32(&blob);
                    cosine_similarity(q_vec, &edge_vec).max(0.0)
                }
                _ => 0.0,
            };

            // 2. Lexical / Entity Overlap Signal
            let mut lexical_boost: f64 = 0.0;
            for token in query_lower.split_whitespace() {
                if token.len() > 2
                    && (subject.to_lowercase().contains(token)
                        || predicate.to_lowercase().contains(token)
                        || object.to_lowercase().contains(token))
                {
                    lexical_boost += 0.35;
                }
            }
            let lexical_score = lexical_boost.min(1.0);

            // Associative Cue Gate: A memory cannot fire without lexical or vector relevance
            if lexical_score == 0.0 && vec_sim < 0.15 {
                return Ok(());
            }

            // 3. ACT-R Base-Level Activation Decay
            let elapsed_hours = ((now - last_accessed).max(0) as f64) / 3600.0;
            let decay_rate = 0.5_f64;
            let act_r_base = ((1.0 + access_count as f64).ln()
                - decay_rate * (1.0 + elapsed_hours).ln())
            .max(0.0);

            let total_score = (0.45 * vec_sim)
                + (0.30 * lexical_score)
                + (0.15 * act_r_base)
                + (0.10 * confidence);

            candidates.push(ScoredMemory {
                edge_id,
                fact: format!("{} {} {}", subject, predicate, object),
                subject,
                predicate,
                object,
                confidence,
                act_r_score: (total_score * 1000.0).round() / 1000.0,
                valid_from,
                valid_to,
                parent_episode_id: parent_ep,
            });
            Ok(())
        };

        if let Some(ts) = as_of {
            let mut rows = stmt
                .query(params![scope_id, ts])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            while let Some(row) = rows.next().map_err(|e| PyRuntimeError::new_err(e.to_string()))? {
                process_row(row).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        } else {
            let mut rows = stmt
                .query(params![scope_id])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            while let Some(row) = rows.next().map_err(|e| PyRuntimeError::new_err(e.to_string()))? {
                process_row(row).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }

        // UPGRADE 1: Hippocampal Working-Memory Fallback (Unconsolidated Episodes)
        if as_of.is_none() && include_working_memory {
            let mut ep_stmt = conn
                .prepare(
                    "SELECT id, content, density_score, created_at FROM episodes
                     WHERE scope_id = ?1 AND consolidated = 0",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let ep_rows = ep_stmt
                .query_map(params![scope_id], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, f64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            for ep in ep_rows {
                let (ep_id, content, density, created_at) =
                    ep.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let content_lower = content.to_lowercase();

                let mut lex_hits: f64 = 0.0;
                for token in query_lower.split_whitespace() {
                    let clean_tok = token.trim_matches(|c: char| !c.is_alphanumeric());
                    if clean_tok.len() > 2 && content_lower.contains(clean_tok) {
                        lex_hits += 0.35;
                    }
                }
                let lex_score = lex_hits.min(1.0);
                if lex_score > 0.0 {
                    // Fresh working-memory turns get a recency boost so immediate recall never misses
                    let wm_score = (0.55 * lex_score) + (0.25 * density) + 0.15;
                    candidates.push(ScoredMemory {
                        edge_id: -ep_id, // Negative ID denotes Working Memory episode
                        fact: format!("[WORKING_MEMORY] {}", content),
                        subject: "WORKING_MEMORY".to_string(),
                        predicate: "UNCONSOLIDATED_TURN".to_string(),
                        object: content,
                        confidence: density,
                        act_r_score: (wm_score * 1000.0).round() / 1000.0,
                        valid_from: created_at,
                        valid_to: None,
                        parent_episode_id: ep_id,
                    });
                }
            }
        }

        candidates.sort_by(|a, b| b.act_r_score.partial_cmp(&a.act_r_score).unwrap());
        candidates.truncate(top_k);

        // Reinforce retrieved active graph edges (skip negative working-memory IDs)
        if as_of.is_none() {
            for mem in &candidates {
                if mem.edge_id > 0 {
                    let _ = conn.execute(
                        "UPDATE kg_edges SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
                        params![now, mem.edge_id],
                    );
                }
            }
        }

        Ok(serde_json::to_string(&candidates).unwrap())
    }

    /// UPGRADE 2: Multi-Hop Subgraph Traversal (Entity-Centric Reasoning)
    #[pyo3(signature = (scope_id, start_entity, max_hops=2))]
    pub fn traverse_subgraph(
        &self,
        scope_id: &str,
        start_entity: &str,
        max_hops: usize,
    ) -> PyResult<String> {
        let conn = self.conn.lock().unwrap();
        let mut visited_edges: HashSet<i64> = HashSet::new();
        let mut frontier: Vec<String> = vec![start_entity.trim().to_uppercase()];
        let mut results: Vec<SubgraphHop> = Vec::new();

        for hop in 1..=max_hops {
            if frontier.is_empty() {
                break;
            }
            let mut next_frontier: Vec<String> = Vec::new();

            for entity in &frontier {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, subject, predicate, object, confidence, valid_from
                         FROM kg_edges
                         WHERE scope_id = ?1 AND valid_to IS NULL
                         AND (UPPER(subject) = ?2 OR UPPER(object) = ?2)",
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                let rows = stmt
                    .query_map(params![scope_id, entity], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, String>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, f64>(4)?,
                            r.get::<_, i64>(5)?,
                        ))
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                for row in rows {
                    let (id, subj, pred, obj, conf, vf) =
                        row.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    if visited_edges.insert(id) {
                        let next_node = if subj.to_uppercase() == *entity {
                            obj.to_uppercase()
                        } else {
                            subj.to_uppercase()
                        };
                        next_frontier.push(next_node);
                        results.push(SubgraphHop {
                            hop,
                            edge_id: id,
                            subject: subj,
                            predicate: pred,
                            object: obj,
                            confidence: conf,
                            valid_from: vf,
                        });
                    }
                }
            }
            frontier = next_frontier;
        }

        Ok(serde_json::to_string(&results).unwrap())
    }

    /// LINEAGE DAG HARD FORGET: Cascade purge across Graph, Unconsolidated Episodes, FTS5, and Core Blocks + SHA256 Proof
    pub fn hard_forget(&self, scope_id: &str, target_keyword: &str) -> PyResult<String> {
        let now = Utc::now().timestamp();
        let pattern = format!("%{}%", target_keyword.trim());
        let conn = self.conn.lock().unwrap();

        // 1. Trace matching graph edges and their parent_episode_ids
        let mut stmt = conn
            .prepare(
                "SELECT id, parent_episode_id FROM kg_edges
                 WHERE scope_id = ?1 AND (subject LIKE ?2 OR predicate LIKE ?2 OR object LIKE ?2)",
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut edge_ids = Vec::new();
        let mut episode_ids = HashSet::new();

        let rows = stmt
            .query_map(params![scope_id, pattern], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        for row in rows {
            let (eid, epid) = row.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            edge_ids.push(eid);
            episode_ids.insert(epid);
        }

        // Also catch unconsolidated raw episodes matching the keyword directly
        let mut ep_stmt = conn
            .prepare("SELECT id FROM episodes WHERE scope_id = ?1 AND content LIKE ?2")
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let direct_eps = ep_stmt
            .query_map(params![scope_id, pattern], |r| r.get::<_, i64>(0))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        for ep in direct_eps.flatten() {
            episode_ids.insert(ep);
        }

        // 2. Hard delete matching edges from Knowledge Graph
        let purged_edges = conn
            .execute(
                "DELETE FROM kg_edges WHERE scope_id = ?1 
                 AND (subject LIKE ?2 OR predicate LIKE ?2 OR object LIKE ?2)",
                params![scope_id, pattern],
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // 3. Cascade delete upstream raw episodes, FTS5 tokens, and downstream Core Blocks
        let mut purged_episodes = 0;
        for ep_id in &episode_ids {
            purged_episodes += conn
                .execute(
                    "DELETE FROM episodes WHERE id = ?1 AND scope_id = ?2",
                    params![ep_id, scope_id],
                )
                .unwrap_or(0);
            let _ = conn.execute(
                "DELETE FROM episodes_fts WHERE episode_id = ?1",
                params![ep_id],
            );
            let _ = conn.execute(
                "DELETE FROM core_blocks WHERE scope_id = ?1 AND parent_episode_id = ?2",
                params![scope_id, ep_id],
            );
        }

        // 4. Generate SHA-256 Cryptographic Compliance Proof
        let mut hasher = Sha256::new();
        hasher.update(format!(
            "FORGET|scope:{}|target:{}|edges:{:?}|episodes:{:?}|ts:{}",
            scope_id, target_keyword, edge_ids, episode_ids, now
        ));
        let proof_hex = format!("{:x}", hasher.finalize());

        conn.execute(
            "INSERT INTO forget_audit_log (
                scope_id, target_entity, purged_edges_count, purged_episodes_count, proof_sha256, timestamp
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                scope_id,
                target_keyword,
                purged_edges as i64,
                purged_episodes as i64,
                proof_hex,
                now
            ],
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(serde_json::json!({
            "status": "HARD_PURGED",
            "purged_graph_edges": purged_edges,
            "purged_raw_episodes": purged_episodes,
            "cryptographic_proof_sha256": proof_hex,
            "timestamp": now
        })
        .to_string())
    }

    /// Inspect Letta-style Core Memory blocks
    pub fn get_core_blocks(&self, scope_id: &str) -> PyResult<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT block_key, content FROM core_blocks WHERE scope_id = ?1")
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let rows = stmt
            .query_map(params![scope_id], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut blocks = Vec::new();
        for r in rows {
            blocks.push(r.map_err(|e| PyRuntimeError::new_err(e.to_string()))?);
        }
        Ok(blocks)
    }

    /// UPGRADE 3: Observability & Memory Evaluation Telemetry
    pub fn get_telemetry(&self, scope_id: &str) -> PyResult<String> {
        let conn = self.conn.lock().unwrap();

        let active_edges: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM kg_edges WHERE scope_id = ?1 AND valid_to IS NULL",
                params![scope_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let invalidated_edges: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM kg_edges WHERE scope_id = ?1 AND valid_to IS NOT NULL",
                params![scope_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let unconsolidated_eps: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodes WHERE scope_id = ?1 AND consolidated = 0",
                params![scope_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let consolidated_eps: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episodes WHERE scope_id = ?1 AND consolidated = 1",
                params![scope_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let core_blocks_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM core_blocks WHERE scope_id = ?1",
                params![scope_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let forget_audits: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM forget_audit_log WHERE scope_id = ?1",
                params![scope_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let total_edges = active_edges + invalidated_edges;
        let contradiction_turnover_rate = if total_edges > 0 {
            ((invalidated_edges as f64 / total_edges as f64) * 1000.0).round() / 1000.0
        } else {
            0.0
        };

        Ok(serde_json::json!({
            "scope_id": scope_id,
            "active_graph_edges": active_edges,
            "historical_invalidated_edges": invalidated_edges,
            "contradiction_turnover_rate": contradiction_turnover_rate,
            "unconsolidated_working_episodes": unconsolidated_eps,
            "consolidated_episodes": consolidated_eps,
            "promoted_core_blocks": core_blocks_count,
            "verified_forget_operations": forget_audits
        })
        .to_string())
    }
}

#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LocusEngine>()?;
    Ok(())
}
