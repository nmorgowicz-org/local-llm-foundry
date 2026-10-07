use anyhow::{Context, Result};
use rusqlite::backup::Backup;
use rusqlite::{Connection, params, params_from_iter};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

const MAX_ADMIN_QUERY_ROWS: usize = 1_000;
const MAX_ADMIN_QUERY_RESULT_BYTES: usize = 1_048_576;

const SCHEMA_SQL: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA wal_autocheckpoint=200;

CREATE TABLE IF NOT EXISTS tabs (
    id                     TEXT    PRIMARY KEY,
    name                   TEXT    NOT NULL,
    system_prompt          TEXT    NOT NULL DEFAULT '',
    ai_name                TEXT,
    user_name              TEXT,
    explicit_level         INTEGER NOT NULL DEFAULT 0,
    active_template_id     TEXT,
    auto_compact           INTEGER NOT NULL DEFAULT 1,
    auto_compact_summarize INTEGER NOT NULL DEFAULT 0,
    compact_mode           TEXT    NOT NULL DEFAULT 'percent',
    compact_threshold      REAL    NOT NULL DEFAULT 0.8,
    model_params           TEXT    NOT NULL DEFAULT '{}',
    context_notes          TEXT    NOT NULL DEFAULT '[]',
    sidebar_width          INTEGER NOT NULL DEFAULT 280,
    tab_order              INTEGER NOT NULL DEFAULT 0,
    pinned                 INTEGER NOT NULL DEFAULT 0,
    last_ctx_pct           REAL,
    total_input_tokens     INTEGER NOT NULL DEFAULT 0,
    total_output_tokens    INTEGER NOT NULL DEFAULT 0,
    created_at             INTEGER NOT NULL,
    updated_at             INTEGER NOT NULL,
    composer_draft         TEXT    NOT NULL DEFAULT '',
    ai_gender              TEXT,
    template_version_or_hash TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    tab_id                    TEXT    NOT NULL REFERENCES tabs(id) ON DELETE CASCADE,
    role                      TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
    content                   TEXT    NOT NULL,
    thinking_content          TEXT,
    timestamp_ms              INTEGER NOT NULL DEFAULT 0,
    input_tokens              INTEGER,
    output_tokens             INTEGER,
    cumulative_input_tokens   INTEGER,
    cumulative_output_tokens  INTEGER,
    compaction_marker         INTEGER NOT NULL DEFAULT 0,
    variants                  TEXT,
    variant_index             INTEGER,
    context_fingerprint       TEXT,
    seq                       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_tab ON messages(tab_id, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"#;

// ── Data types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TabVisibility {
    #[default]
    Active,
    Archived,
    Hidden,
}

impl fmt::Display for TabVisibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TabVisibility::Active => write!(f, "active"),
            TabVisibility::Archived => write!(f, "archived"),
            TabVisibility::Hidden => write!(f, "hidden"),
        }
    }
}

impl FromStr for TabVisibility {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "archived" => Ok(TabVisibility::Archived),
            "hidden" => Ok(TabVisibility::Hidden),
            _ => Ok(TabVisibility::Active),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TabMeta {
    pub id: String,
    pub name: String,
    pub explicit_level: u8,
    pub active_template_id: Option<String>,
    pub pinned: bool,
    pub tab_order: i64,
    pub last_ctx_pct: Option<f64>,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub message_count: i64,
    pub notes_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
    /// Timestamp of the most-recent message in this tab; None if the tab has no messages.
    #[serde(default)]
    pub last_message_at: Option<i64>,
    #[serde(default)]
    pub visibility: String,
    #[serde(default)]
    pub composer_draft: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatTabRow {
    pub id: String,
    pub name: String,
    pub system_prompt: String,
    pub ai_name: Option<String>,
    pub user_name: Option<String>,
    pub explicit_level: u8,
    pub active_template_id: Option<String>,
    pub auto_compact: bool,
    pub auto_compact_summarize: bool,
    pub compact_mode: String,
    pub compact_threshold: f64,
    pub model_params: serde_json::Value,
    pub context_notes: serde_json::Value,
    pub sidebar_width: u32,
    pub tab_order: i64,
    pub pinned: bool,
    pub last_ctx_pct: Option<f64>,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default = "default_visibility")]
    pub visibility: String,
    #[serde(default)]
    pub composer_draft: String,
    #[serde(default)]
    pub ai_gender: Option<String>,
    #[serde(default)]
    pub template_version_or_hash: Option<String>,
    #[serde(default)]
    pub messages: Vec<MessageRow>,
}

fn default_visibility() -> String {
    "active".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MessageRow {
    #[serde(default)]
    pub id: i64,
    #[serde(default)]
    pub tab_id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub thinking_content: Option<String>,
    #[serde(default)]
    pub timestamp_ms: i64,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cumulative_input_tokens: Option<i64>,
    pub cumulative_output_tokens: Option<i64>,
    /// None denotes legacy usage; an empty fingerprint explicitly invalidates it.
    #[serde(default)]
    pub context_fingerprint: Option<String>,
    #[serde(default)]
    pub compaction_marker: bool,
    pub variants: Option<serde_json::Value>,
    pub variant_index: Option<i64>,
    #[serde(default)]
    pub seq: i64,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub tab_id: String,
    pub tab_name: String,
    pub message_id: i64,
    pub role: String,
    pub snippet: String,
    pub timestamp_ms: Option<i64>,
    #[serde(default)]
    pub visibility: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResultsPage {
    pub results: Vec<SearchResult>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    pub has_more: bool,
}

// ── Storage ───────────────────────────────────────────────────────────────────

impl Default for ChatStorage {
    fn default() -> Self {
        let random_id: u32 = rand::random();
        let path = std::env::temp_dir().join(format!("llama_monitor_test_{}.db", random_id));
        Self::open(&path).expect("Failed to open test chat database")
    }
}

pub struct ChatStorage {
    // Option so restore_from_path can atomically close and reopen the connection
    // while holding the mutex exclusively. It is None only during that brief window.
    conn: std::sync::Mutex<Option<Connection>>,
    db_path: PathBuf,
}

impl ChatStorage {
    pub fn open(db_path: &PathBuf) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("opening chat.db at {}", db_path.display()))?;
        conn.execute_batch(SCHEMA_SQL)?;
        run_schema_migrations(&conn)?;
        Ok(Self {
            conn: std::sync::Mutex::new(Some(conn)),
            db_path: db_path.clone(),
        })
    }

    /// Return the path this storage was opened from.
    #[allow(dead_code)]
    pub fn get_db_path(&self) -> PathBuf {
        self.db_path.clone()
    }

    // ── Migration ─────────────────────────────────────────────────────────────

    pub fn migrate_from_legacy(&self, legacy_path: &PathBuf) -> Result<()> {
        if !legacy_path.exists() {
            return Ok(());
        }
        let raw = std::fs::read_to_string(legacy_path)?;
        let tabs: Vec<serde_json::Value> =
            serde_json::from_str(&raw).context("parsing legacy chat-tabs.json")?;

        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let tx = conn.unchecked_transaction()?;

        let mut migrated_tabs = 0u64;
        let mut migrated_msgs = 0u64;

        for (order, tab) in tabs.iter().enumerate() {
            let id = tab["id"].as_str().unwrap_or_default().to_string();
            if id.is_empty() {
                continue;
            }

            tx.execute(
                "INSERT OR REPLACE INTO tabs (
                    id, name, system_prompt, ai_name, user_name,
                    explicit_level, active_template_id,
                    auto_compact, auto_compact_summarize, compact_mode, compact_threshold,
                    model_params, context_notes, sidebar_width,
                    tab_order, pinned, last_ctx_pct,
                    total_input_tokens, total_output_tokens,
                    created_at, updated_at
                ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21)",
                params![
                    id,
                    tab["name"].as_str().unwrap_or("Untitled"),
                    tab["system_prompt"].as_str().unwrap_or(""),
                    tab["ai_name"].as_str(),
                    tab["user_name"].as_str(),
                    tab["explicit_level"].as_i64().or_else(|| {
                        tab["explicit_mode"].as_bool().map(|b| if b { 1 } else { 0 })
                    }).unwrap_or(0),
                    tab["active_template_id"].as_str(),
                    tab["auto_compact"].as_bool().unwrap_or(true) as i64,
                    tab["auto_compact_summarize"].as_bool().unwrap_or(false) as i64,
                    tab["compact_mode"].as_str().unwrap_or("percent"),
                    tab["compact_threshold"].as_f64().unwrap_or(0.8),
                    tab["model_params"].to_string(),
                    tab["context_notes"].to_string(),
                    tab["sidebar_width"].as_i64().unwrap_or(280),
                    order as i64,
                    tab["pinned"].as_bool().unwrap_or(false) as i64,
                    tab["lastCtxPct"].as_f64(),
                    tab["totalInputTokens"].as_i64().unwrap_or(0),
                    tab["totalOutputTokens"].as_i64().unwrap_or(0),
                    tab["created_at"].as_i64().unwrap_or(0),
                    tab["updated_at"].as_i64().unwrap_or(0),
                ],
            )?;
            migrated_tabs += 1;

            if let Some(msgs) = tab["messages"].as_array() {
                for (seq, msg) in msgs.iter().enumerate() {
                    tx.execute(
                        "INSERT OR IGNORE INTO messages (tab_id, role, content, thinking_content, timestamp_ms,
                             input_tokens, output_tokens, compaction_marker, seq)
                          VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                        params![
                            id,
                            msg["role"].as_str().unwrap_or("user"),
                            msg["content"].as_str().unwrap_or(""),
                            msg["thinking_content"].as_str(),
                            msg["timestamp_ms"].as_i64().unwrap_or(0),
                            msg["input_tokens"].as_i64(),
                            msg["output_tokens"].as_i64(),
                            msg["compaction_marker"].as_bool().unwrap_or(false) as i64,
                            seq as i64,
                        ],
                    )?;
                    migrated_msgs += 1;
                }
            }
        }
        tx.commit()?;
        let tab_count = migrated_tabs;
        let msg_count = migrated_msgs;
        std::fs::rename(legacy_path, legacy_path.with_extension("json.bak"))?;
        eprintln!(
            "[info] Migrated {} tabs with {} messages from chat-tabs.json",
            tab_count, msg_count
        );
        Ok(())
    }

    // ── Tab CRUD ──────────────────────────────────────────────────────────────

    pub fn list_tabs(&self, visibilities: &[TabVisibility]) -> Result<Vec<TabMeta>> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");

        let (where_clause, params_vec): (String, Vec<Box<dyn rusqlite::ToSql>>) =
            if visibilities.is_empty() {
                (String::new(), Vec::new())
            } else {
                let placeholders: Vec<_> = visibilities.iter().map(|_| "?").collect();
                (
                    format!("WHERE t.visibility IN ({})", placeholders.join(", ")),
                    visibilities
                        .iter()
                        .map(|v| Box::new(v.to_string()) as Box<dyn rusqlite::ToSql>)
                        .collect(),
                )
            };

        let sql = format!(
            "SELECT t.id, t.name, t.explicit_level, t.active_template_id,
                    t.pinned, t.tab_order, t.last_ctx_pct,
                    t.total_input_tokens, t.total_output_tokens,
                    COUNT(m.id) as message_count,
                    COALESCE(json_array_length(t.context_notes), 0) as notes_count,
                    t.created_at, t.updated_at, t.visibility, t.composer_draft,
                    MAX(m.timestamp_ms) as last_message_at
             FROM tabs t
             LEFT JOIN messages m ON m.tab_id = t.id AND m.compaction_marker = 0
             {}
             GROUP BY t.id
             ORDER BY t.tab_order ASC",
            where_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(params_vec), |row| {
            Ok(TabMeta {
                id: row.get(0)?,
                name: row.get(1)?,
                explicit_level: row.get::<_, i64>(2)? as u8,
                active_template_id: row.get(3)?,
                pinned: row.get::<_, i64>(4)? != 0,
                tab_order: row.get(5)?,
                last_ctx_pct: row.get(6)?,
                total_input_tokens: row.get(7)?,
                total_output_tokens: row.get(8)?,
                message_count: row.get(9)?,
                notes_count: row.get(10)?,
                created_at: row.get(11)?,
                updated_at: row.get(12)?,
                visibility: row.get(13)?,
                composer_draft: row.get(14)?,
                last_message_at: row.get(15)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn get_tab(&self, id: &str) -> Result<ChatTabRow> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let mut stmt = conn.prepare(
            "SELECT id, name, system_prompt, ai_name, user_name,
                    explicit_level, active_template_id,
                    auto_compact, auto_compact_summarize, compact_mode, compact_threshold,
                    model_params, context_notes, sidebar_width,
                    tab_order, pinned, last_ctx_pct,
                    total_input_tokens, total_output_tokens,
                    created_at, updated_at, visibility, composer_draft, ai_gender, template_version_or_hash
             FROM tabs WHERE id = ?1",
        )?;
        let mut tab = stmt.query_row(params![id], |row| {
            Ok(ChatTabRow {
                id: row.get(0)?,
                name: row.get(1)?,
                system_prompt: row.get(2)?,
                ai_name: row.get(3)?,
                user_name: row.get(4)?,
                explicit_level: row.get::<_, i64>(5)? as u8,
                active_template_id: row.get(6)?,
                auto_compact: row.get::<_, i64>(7)? != 0,
                auto_compact_summarize: row.get::<_, i64>(8)? != 0,
                compact_mode: row.get(9)?,
                compact_threshold: row.get(10)?,
                model_params: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                context_notes: serde_json::from_str(&row.get::<_, String>(12)?).unwrap_or_default(),
                sidebar_width: row.get::<_, i64>(13)? as u32,
                tab_order: row.get(14)?,
                pinned: row.get::<_, i64>(15)? != 0,
                last_ctx_pct: row.get(16)?,
                total_input_tokens: row.get(17)?,
                total_output_tokens: row.get(18)?,
                created_at: row.get(19)?,
                updated_at: row.get(20)?,
                visibility: row.get(21)?,
                composer_draft: row.get(22)?,
                ai_gender: row.get(23)?,
                template_version_or_hash: row.get(24)?,
                messages: vec![],
            })
        })?;

        tab.messages = self._load_messages_locked(conn, id)?;
        Ok(tab)
    }

    pub fn create_tab(&self, tab: &ChatTabRow) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        conn.execute(
            "INSERT INTO tabs (id, name, system_prompt, ai_name, user_name,
                 explicit_level, active_template_id,
                 auto_compact, auto_compact_summarize, compact_mode, compact_threshold,
                 model_params, context_notes, sidebar_width,
                 tab_order, pinned, last_ctx_pct,
                 total_input_tokens, total_output_tokens,
                 created_at, updated_at, visibility, composer_draft, ai_gender, template_version_or_hash)
              VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25)",
            params![
                tab.id,
                tab.name,
                tab.system_prompt,
                tab.ai_name,
                tab.user_name,
                tab.explicit_level as i64,
                tab.active_template_id,
                tab.auto_compact as i64,
                tab.auto_compact_summarize as i64,
                tab.compact_mode,
                tab.compact_threshold,
                serde_json::to_string(&tab.model_params)?,
                serde_json::to_string(&tab.context_notes)?,
                tab.sidebar_width as i64,
                tab.tab_order,
                tab.pinned as i64,
                tab.last_ctx_pct,
                tab.total_input_tokens,
                tab.total_output_tokens,
                tab.created_at,
                tab.updated_at,
                tab.visibility,
                tab.composer_draft,
                tab.ai_gender.as_deref(),
                tab.template_version_or_hash.as_deref(),
            ],
        )?;
        Ok(())
    }

    pub fn update_tab_meta(&self, tab: &ChatTabRow) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        conn.execute(
            "UPDATE tabs SET
                name=?2, system_prompt=?3, ai_name=?4, user_name=?5,
                explicit_level=?6, active_template_id=?7,
                auto_compact=?8, auto_compact_summarize=?9, compact_mode=?10, compact_threshold=?11,
                model_params=?12, context_notes=?13, sidebar_width=?14,
                pinned=?15, last_ctx_pct=?16,
                total_input_tokens=?17, total_output_tokens=?18,
                updated_at=?19, visibility=?20, composer_draft=?21, ai_gender=?22, template_version_or_hash=?23
             WHERE id=?1",
            params![
                tab.id,
                tab.name,
                tab.system_prompt,
                tab.ai_name,
                tab.user_name,
                tab.explicit_level as i64,
                tab.active_template_id,
                tab.auto_compact as i64,
                tab.auto_compact_summarize as i64,
                tab.compact_mode,
                tab.compact_threshold,
                serde_json::to_string(&tab.model_params)?,
                serde_json::to_string(&tab.context_notes)?,
                tab.sidebar_width as i64,
                tab.pinned as i64,
                tab.last_ctx_pct,
                tab.total_input_tokens,
                tab.total_output_tokens,
                tab.updated_at,
                tab.visibility,
                tab.composer_draft,
                tab.ai_gender.as_deref(),
                tab.template_version_or_hash.as_deref(),
            ],
        )?;
        Ok(())
    }

    pub fn delete_tab(&self, id: &str) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        conn.execute("DELETE FROM tabs WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn reorder_tabs(&self, ordered_ids: &[String]) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let tx = conn.unchecked_transaction()?;
        for (i, id) in ordered_ids.iter().enumerate() {
            tx.execute(
                "UPDATE tabs SET tab_order = ?1 WHERE id = ?2",
                params![i as i64, id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn set_visibility(&self, id: &str, visibility: &TabVisibility) -> Result<TabMeta> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let vis_str = visibility.to_string();

        conn.execute(
            "UPDATE tabs SET visibility = ?1, updated_at = ?2 WHERE id = ?3",
            params![
                vis_str,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as i64,
                id
            ],
        )?;

        let tab: ChatTabRow = conn.query_row(
            "SELECT id, name, system_prompt, ai_name, user_name,
                    explicit_level, active_template_id,
                    auto_compact, auto_compact_summarize, compact_mode, compact_threshold,
                    model_params, context_notes, sidebar_width,
                    tab_order, pinned, last_ctx_pct,
                    total_input_tokens, total_output_tokens,
                    created_at, updated_at, visibility, composer_draft, ai_gender, template_version_or_hash
             FROM tabs WHERE id = ?1",
            params![id],
            |row| {
                Ok(ChatTabRow {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    system_prompt: row.get(2)?,
                    ai_name: row.get(3)?,
                    user_name: row.get(4)?,
                    explicit_level: row.get::<_, i64>(5)? as u8,
                    active_template_id: row.get(6)?,
                    auto_compact: row.get::<_, i64>(7)? != 0,
                    auto_compact_summarize: row.get::<_, i64>(8)? != 0,
                    compact_mode: row.get(9)?,
                    compact_threshold: row.get(10)?,
                    model_params: serde_json::from_str(&row.get::<_, String>(11)?)
                        .unwrap_or_default(),
                    context_notes: serde_json::from_str(&row.get::<_, String>(12)?)
                        .unwrap_or_default(),
                    sidebar_width: row.get::<_, i64>(13)? as u32,
                    tab_order: row.get(14)?,
                    pinned: row.get::<_, i64>(15)? != 0,
                    last_ctx_pct: row.get(16)?,
                    total_input_tokens: row.get(17)?,
                    total_output_tokens: row.get(18)?,
                    created_at: row.get(19)?,
                    updated_at: row.get(20)?,
                    visibility: row.get(21)?,
                    composer_draft: row.get(22)?,
                    ai_gender: row.get(23)?,
                    template_version_or_hash: row.get(24)?,
                    messages: vec![],
                })
            },
        )?;

        let message_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM messages WHERE tab_id = ?1 AND compaction_marker = 0",
            params![id],
            |row| row.get(0),
        )?;

        let notes_count = match &tab.context_notes {
            serde_json::Value::Array(arr) => arr.len() as i64,
            _ => 0,
        };

        Ok(TabMeta {
            id: tab.id,
            name: tab.name,
            explicit_level: tab.explicit_level,
            active_template_id: tab.active_template_id,
            pinned: tab.pinned,
            tab_order: tab.tab_order,
            last_ctx_pct: tab.last_ctx_pct,
            total_input_tokens: tab.total_input_tokens,
            total_output_tokens: tab.total_output_tokens,
            message_count,
            notes_count,
            created_at: tab.created_at,
            updated_at: tab.updated_at,
            last_message_at: None,
            visibility: tab.visibility,
            composer_draft: tab.composer_draft,
        })
    }

    // ── Message CRUD ──────────────────────────────────────────────────────────

    fn _load_messages_locked(&self, conn: &Connection, tab_id: &str) -> Result<Vec<MessageRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, tab_id, role, content, thinking_content, timestamp_ms,
                    input_tokens, output_tokens,
                    cumulative_input_tokens, cumulative_output_tokens,
                    compaction_marker, variants, variant_index, seq, context_fingerprint
             FROM messages WHERE tab_id = ?1 ORDER BY seq",
        )?;
        let rows = stmt.query_map(params![tab_id], |row| {
            Ok(MessageRow {
                id: row.get(0)?,
                tab_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                thinking_content: row.get(4)?,
                timestamp_ms: row.get(5)?,
                input_tokens: row.get(6)?,
                output_tokens: row.get(7)?,
                cumulative_input_tokens: row.get(8)?,
                cumulative_output_tokens: row.get(9)?,
                compaction_marker: row.get::<_, i64>(10)? != 0,
                variants: row
                    .get::<_, Option<String>>(11)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
                variant_index: row.get(12)?,
                seq: row.get(13)?,
                context_fingerprint: row.get(14)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn append_message(&self, msg: &MessageRow) -> Result<i64> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        conn.execute(
            "INSERT INTO messages (tab_id, role, content, thinking_content, timestamp_ms,
                 input_tokens, output_tokens,
                 cumulative_input_tokens, cumulative_output_tokens,
                 compaction_marker, variants, variant_index, context_fingerprint, seq)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,
                 COALESCE((SELECT MAX(seq)+1 FROM messages WHERE tab_id=?1), 0))",
            params![
                msg.tab_id,
                msg.role,
                msg.content,
                msg.thinking_content,
                msg.timestamp_ms,
                msg.input_tokens,
                msg.output_tokens,
                msg.cumulative_input_tokens,
                msg.cumulative_output_tokens,
                msg.compaction_marker as i64,
                msg.variants.as_ref().map(|v| v.to_string()),
                msg.variant_index,
                msg.context_fingerprint,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn replace_messages(&self, tab_id: &str, messages: &[MessageRow]) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let tx = conn.unchecked_transaction()?;
        tx.execute("DELETE FROM messages WHERE tab_id = ?1", params![tab_id])?;
        for (seq, msg) in messages.iter().enumerate() {
            tx.execute(
                "INSERT INTO messages (tab_id, role, content, thinking_content, timestamp_ms,
                     input_tokens, output_tokens,
                     cumulative_input_tokens, cumulative_output_tokens,
                     compaction_marker, variants, variant_index, seq, context_fingerprint)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                params![
                    tab_id,
                    msg.role,
                    msg.content,
                    msg.thinking_content,
                    msg.timestamp_ms,
                    msg.input_tokens,
                    msg.output_tokens,
                    msg.cumulative_input_tokens,
                    msg.cumulative_output_tokens,
                    msg.compaction_marker as i64,
                    msg.variants.as_ref().map(|v| v.to_string()),
                    msg.variant_index,
                    seq as i64,
                    msg.context_fingerprint,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    // ── Full-text search ──────────────────────────────────────────────────────

    fn escape_html_except_mark(s: &str) -> String {
        // Escape all HTML, then restore <mark> and </mark> tags.
        let escaped = s
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;");
        escaped
            .replace("&lt;mark&gt;", "<mark>")
            .replace("&lt;/mark&gt;", "</mark>")
    }

    /// Search messages using the FTS index.
    ///
    /// `tab_id`: when `Some`, restricts results to that tab only; when
    /// `None`, searches across all tabs (subject to `visibilities`).
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        offset: usize,
        visibilities: &[TabVisibility],
        tab_id: Option<&str>,
    ) -> Result<SearchResultsPage> {
        let normalized_query = normalize_fts_query(query);
        if normalized_query.is_empty() {
            return Ok(SearchResultsPage {
                results: Vec::new(),
                total: 0,
                limit,
                offset,
                has_more: false,
            });
        }

        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");

        // Build optional WHERE clauses.  Track count so LIMIT/OFFSET positions
        // are computed correctly.
        let mut extra_filters = String::new();
        if !visibilities.is_empty() {
            let placeholders: Vec<_> = visibilities.iter().map(|_| "?").collect();
            extra_filters.push_str(&format!(
                " AND t.visibility IN ({})",
                placeholders.join(", ")
            ));
        }
        if tab_id.is_some() {
            extra_filters.push_str(" AND m.tab_id = ?");
        }
        // ?1 = query, then visibilities, then optional tab_id
        let extra_param_count = visibilities.len() + usize::from(tab_id.is_some());

        // Helper: build the shared base parameter list (query + vis + tab_id).
        let base_params = || -> Vec<Box<dyn rusqlite::ToSql>> {
            let mut p: Vec<Box<dyn rusqlite::ToSql>> =
                vec![Box::new(normalized_query.clone()) as Box<dyn rusqlite::ToSql>];
            for v in visibilities {
                p.push(Box::new(v.to_string()));
            }
            if let Some(tid) = tab_id {
                p.push(Box::new(tid.to_string()));
            }
            p
        };

        let total_sql = format!(
            "SELECT COUNT(*)
             FROM messages_fts
             JOIN messages m ON m.id = messages_fts.rowid
             JOIN tabs t ON t.id = m.tab_id
             WHERE messages_fts MATCH ?1
               AND m.compaction_marker = 0
               {}",
            extra_filters
        );
        let total: i64 = conn.query_row(&total_sql, params_from_iter(base_params()), |row| {
            row.get(0)
        })?;

        let results_sql = format!(
            "SELECT t.id, t.name, m.id, m.role,
                    snippet(messages_fts, 0, '<mark>', '</mark>', '…', 24),
                    m.timestamp_ms, t.visibility
             FROM messages_fts
             JOIN messages m ON m.id = messages_fts.rowid
             JOIN tabs t ON t.id = m.tab_id
             WHERE messages_fts MATCH ?1
               AND m.compaction_marker = 0
               {}
             ORDER BY rank
             LIMIT ?{} OFFSET ?{}",
            extra_filters,
            extra_param_count + 2,
            extra_param_count + 3,
        );
        let mut stmt = conn.prepare(&results_sql)?;
        let mut result_params = base_params();
        result_params.push(Box::new(limit as i64));
        result_params.push(Box::new(offset as i64));

        let rows = stmt.query_map(params_from_iter(result_params), |row| {
            let raw_snippet: String = row.get(4)?;
            let snippet = Self::escape_html_except_mark(&raw_snippet);
            Ok(SearchResult {
                tab_id: row.get(0)?,
                tab_name: row.get(1)?,
                message_id: row.get(2)?,
                role: row.get(3)?,
                snippet,
                timestamp_ms: row.get(5)?,
                visibility: row.get(6)?,
            })
        })?;
        let results = rows.collect::<rusqlite::Result<Vec<_>>>()?;
        let total = total.max(0) as usize;
        let has_more = offset.saturating_add(results.len()) < total;
        Ok(SearchResultsPage {
            results,
            total,
            limit,
            offset,
            has_more,
        })
    }

    // ── Database Health Management ──────────────────────────────────────────────

    /// Run integrity check on the database
    pub fn integrity_check(&self) -> Result<String> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let mut stmt = conn.prepare("PRAGMA integrity_check")?;
        let result: String = stmt.query_row([], |row| row.get(0))?;
        Ok(result)
    }

    /// Check database size and table counts
    pub fn database_stats(&self) -> Result<serde_json::Value> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");

        // Get table counts
        let tab_count: i64 = conn.query_row("SELECT COUNT(*) FROM tabs", [], |row| row.get(0))?;
        let msg_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
        let fts_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM messages_fts", [], |row| row.get(0))?;

        let file_size_bytes = std::fs::metadata(&self.db_path)
            .map(|m| m.len())
            .unwrap_or(0);

        Ok(serde_json::json!({
            "tab_count": tab_count,
            "message_count": msg_count,
            "fts_index_count": fts_count,
            "file_size_bytes": file_size_bytes,
        }))
    }

    /// Rebuild FTS index
    pub fn rebuild_fts_index(&self) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        // The 'rebuild' special command drops and repopulates the index from the
        // content table in one step; no separate DELETE is needed.
        conn.execute_batch("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")?;
        Ok(())
    }

    /// Run vacuum to optimize database
    pub fn vacuum(&self) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        conn.execute_batch("VACUUM")?;
        Ok(())
    }

    /// Checkpoint WAL
    pub fn checkpoint(&self) -> Result<(i64, i64, i64)> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let mut stmt = conn.prepare("PRAGMA wal_checkpoint(TRUNCATE)")?;
        let row = stmt.query_row([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        Ok(row)
    }

    /// Create backup to specified path using SQLite online backup API
    pub fn backup(&self, backup_path: &std::path::Path) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");

        // Checkpoint before backup to ensure consistency
        conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE)")?;

        let mut dst = Connection::open(backup_path)
            .with_context(|| format!("opening backup at {}", backup_path.display()))?;

        let backup = Backup::new(conn, &mut dst).context("creating backup handle")?;
        backup
            .run_to_completion(5, std::time::Duration::from_millis(250), None)
            .context("backup operation")?;

        Ok(())
    }

    /// Analyze database for query optimization
    pub fn analyze(&self) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        conn.execute_batch("ANALYZE")?;
        Ok(())
    }

    /// Get list of indexes
    pub fn list_indexes(&self) -> Result<Vec<serde_json::Value>> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let mut stmt = conn.prepare(
            "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| {
            let name: String = row.get(0)?;
            let tbl_name: String = row.get(1)?;
            let sql: String = row.get(2)?;
            Ok(serde_json::json!({
                "name": name,
                "table": tbl_name,
                "sql": sql,
                "rebuildable": name.contains("fts") || name.contains("search"),
            }))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    /// Execute arbitrary SQL query (admin only, restricted to safe operations)
    pub fn execute_query(&self, sql: &str, is_admin: bool) -> Result<serde_json::Value> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");

        // Normalize for checks
        let trimmed = sql.trim();
        let upper = trimmed.to_uppercase();

        // Reject multi-statement queries — defense-in-depth against semicolon injection.
        if trimmed.contains(';') {
            return Err(anyhow::anyhow!("Multi-statement queries are not allowed"));
        }

        // Blocklist: disallow dangerous operations
        let dangerous_keywords = [
            "ATTACH DATABASE",
            "DETACH DATABASE",
            "LOAD_EXTENSION",
            "CREATE TABLE",
            "DROP TABLE",
            "ALTER TABLE",
            "CREATE INDEX",
            "DROP INDEX",
            "CREATE TRIGGER",
            "DROP TRIGGER",
            "CREATE VIEW",
            "DROP VIEW",
            "INSERT INTO",
            "UPDATE ",
            "DELETE FROM",
            "REPLACE INTO",
            "BEGIN ",
            "COMMIT",
            "ROLLBACK",
            "PRAGMA journal_mode",
            "PRAGMA synchronous",
            "PRAGMA foreign_keys",
            "PRAGMA encoding",
            "PRAGMA page_size",
            "PRAGMA cache_size",
            "PRAGMA temp_store",
            "PRAGMA mmap_size",
            "PRAGMA locking_mode",
            "PRAGMA wal_checkpoint(PASSIVE)",
            "PRAGMA wal_checkpoint(FULL)",
            "PRAGMA wal_checkpoint(TRUNCATE)",
        ];

        if dangerous_keywords.iter().any(|k| upper.contains(k)) {
            return Err(anyhow::anyhow!("Query contains a disallowed operation"));
        }

        // Allowlist: only allow SELECT, VACUUM, ANALYZE, and a restricted PRAGMA subset.
        // Arbitrary PRAGMA is dangerous (e.g. writable_schema).
        if upper.starts_with("SELECT")
            || upper.starts_with("VACUUM")
            || upper.starts_with("ANALYZE")
        {
            // allowed
        } else if upper.starts_with("PRAGMA") {
            let allowed_pragmas = [
                "PRAGMA INTEGRITY_CHECK",
                "PRAGMA QUICK_CHECK",
                "PRAGMA INTEGRITY",
                "PRAGMA PAGE_COUNT",
                "PRAGMA FREELIST_COUNT",
                "PRAGMA FREE_LIST",
                "PRAGMA SCHEMA_VERSION",
                "PRAGMA USER_VERSION",
                "PRAGMA MUTEX_STATUS",
                "PRAGMA INDEX_LIST",
                "PRAGMA INDEX_INFO",
                "PRAGMA TABLE_INFO",
                "PRAGMA TABLE_XINFO",
                "PRAGMA FOREIGN_KEY_LIST",
                "PRAGMA LOCK_LIST",
                "PRAGMA DATABASE_LIST",
                "PRAGMA JOURNAL_MODE",
                "PRAGMA SYNCRONOUS",
                "PRAGMA CACHE_SIZE",
                "PRAGMA CACHE_SPILL",
                "PRAGMA DEFAULT_JOURNAL_SIZE_LIMIT",
                "PRAGMA MAX_PAGE_COUNT",
                "PRAGMA HARD_HEAP_LIMIT",
                "PRAGMA SOFT_HEAP_LIMIT",
                "PRAGMA DATA_JOURNALING_MODE",
                "PRAGMA FULL_COLUMN_NAMES",
                "PRAGMA IGNORE_CHECK_CONSTRAINTS",
                "PRAGMA CASE_SENSITIVE_LIKE",
                "PRAGMA RECURSIVE_TRIGGERS",
                "PRAGMA SECURE_DELETE",
                "PRAGMA WAL_AUTOCHECKPOINT",
                "PRAGMA AUTOVACUUM",
                "PRAGMA INCREMENTAL_VACUUM",
                "PRAGMA TEMP_STORE",
                "PRAGMA MMAP_SIZE",
                "PRAGMA QUERY_ONLY",
                "PRAGMA DEFAULT_CACHE_SIZE",
                "PRAGMA DOUBLE_QUOTED_STRING",
                "PRAGMA LEGACY_FILE_FORMAT",
                "PRAGMA ENCODING",
                "PRAGMA PAGE_SIZE",
                "PRAGMA COUNT_CHANGES",
                "PRAGMA EPOCHMS",
                "PRAGMA FUNCTION_LIST",
                "PRAGMA MODULE_LIST",
                "PRAGMA VDBE_TRACE",
                "PRAGMA WALK_STACK",
                "PRAGMA TRIGGER_LIST",
            ];
            if allowed_pragmas.iter().any(|p| upper.starts_with(p)) {
                // allowed (exact-name + prefix-safe; no writable_schema in this list)
            } else {
                return Err(anyhow::anyhow!("PRAGMA not allowed"));
            }
        } else {
            return Err(anyhow::anyhow!(
                "Only SELECT, PRAGMA, VACUUM, and ANALYZE queries are allowed"
            ));
        }

        // Restricted mode: limit exposure of sensitive columns when not using admin token
        if !is_admin {
            // Non-SELECT operations are allowed as-is (PRAGMA/VACUUM/ANALYZE).
            if upper.starts_with("SELECT") {
                // Check for queries that would expose sensitive content.
                // We block:
                // - SELECT * FROM messages
                // - SELECT ... FROM messages where content is selected
                // - SELECT system_prompt / context_notes / model_params from tabs
                if is_select_exposing_sensitive(&upper) {
                    return Err(anyhow::anyhow!(
                        "Query accesses restricted columns; use admin token for full access"
                    ));
                }
            }
        }

        // Try to execute as a query that returns multiple rows
        let mut stmt = conn.prepare(trimmed)?;
        let column_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

        let cols: Vec<String> = column_names.clone();
        let rows = stmt.query_map([], |row| {
            let mut row_data = serde_json::Map::new();
            for (i, name) in cols.iter().enumerate() {
                // Try to get the value as a string, falling back to null
                let val: Option<String> = row.get_ref(i).ok().and_then(|v| match v {
                    rusqlite::types::ValueRef::Text(b) => {
                        Some(String::from_utf8_lossy(b).to_string())
                    }
                    rusqlite::types::ValueRef::Integer(i) => Some(i.to_string()),
                    rusqlite::types::ValueRef::Real(f) => Some(f.to_string()),
                    rusqlite::types::ValueRef::Blob(b) => Some(format!("[{} bytes]", b.len())),
                    rusqlite::types::ValueRef::Null => None,
                });
                row_data.insert(name.clone(), val.into());
            }
            Ok(serde_json::Value::Object(row_data))
        })?;

        let mut collected = Vec::new();
        let mut result_bytes = 0usize;
        for row in rows {
            if collected.len() >= MAX_ADMIN_QUERY_ROWS {
                return Err(anyhow::anyhow!(
                    "Query result exceeds the {} row limit; add a LIMIT clause",
                    MAX_ADMIN_QUERY_ROWS
                ));
            }
            let row = row?;
            let row_bytes = serde_json::to_vec(&row)?;
            result_bytes = result_bytes.saturating_add(row_bytes.len());
            if result_bytes > MAX_ADMIN_QUERY_RESULT_BYTES {
                return Err(anyhow::anyhow!(
                    "Query result exceeds the {} byte limit; narrow the selected columns or add a LIMIT clause",
                    MAX_ADMIN_QUERY_RESULT_BYTES
                ));
            }
            collected.push(row);
        }

        Ok(serde_json::json!({
            "columns": column_names,
            "rows": collected,
            "row_count": collected.len(),
        }))
    }

    /// Restore the live database from a backup file safely.
    ///
    /// While holding the mutex exclusively: checkpoints WAL, closes the
    /// connection, copies the backup file over chat.db, removes any stale WAL
    /// sidecars, then reopens and applies the schema.
    pub fn restore_from_path(&self, backup_path: &std::path::Path) -> Result<()> {
        let mut guard = self.conn.lock().unwrap();

        // Checkpoint so nothing is stranded in the WAL before we close.
        if let Some(conn) = guard.as_ref() {
            let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
        }

        // Drop the connection (closes the file handle).
        *guard = None;

        // Overwrite chat.db with the chosen backup.
        std::fs::copy(backup_path, &self.db_path).with_context(|| {
            format!(
                "copying backup {} → {}",
                backup_path.display(),
                self.db_path.display()
            )
        })?;

        // Remove stale WAL sidecars so SQLite starts clean.
        let wal = self.db_path.with_extension("db-wal");
        let shm = self.db_path.with_extension("db-shm");
        let _ = std::fs::remove_file(&wal);
        let _ = std::fs::remove_file(&shm);

        // Reopen and ensure schema is current.
        let new_conn = Connection::open(&self.db_path)
            .with_context(|| format!("reopening {} after restore", self.db_path.display()))?;
        new_conn.execute_batch(SCHEMA_SQL)?;
        run_schema_migrations(&new_conn)?;
        *guard = Some(new_conn);

        Ok(())
    }

    /// Repair corrupted FTS index by dropping and recreating it
    pub fn repair_indexes(&self) -> Result<()> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");

        // Drop the FTS virtual table and its associated triggers, then recreate from scratch.
        conn.execute_batch(
            r#"
            DROP TRIGGER IF EXISTS messages_ai;
            DROP TRIGGER IF EXISTS messages_ad;
            DROP TRIGGER IF EXISTS messages_au;
            DROP TABLE IF EXISTS messages_fts;

            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content,
                content='messages',
                content_rowid='id'
            );

            CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
            END;
            CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;

            INSERT INTO messages_fts(messages_fts) VALUES('rebuild');
            "#,
        )?;

        Ok(())
    }

    /// Emergency recovery: run a full integrity check and return any errors found.
    ///
    /// If the database is corrupted beyond what integrity_check can read, restore
    /// from one of the backups in the `backups/` directory beside the database file.
    pub fn emergency_recovery(&self) -> Result<String> {
        let guard = self.conn.lock().unwrap();
        let conn = guard.as_ref().expect("db open");
        let mut stmt = conn.prepare("PRAGMA integrity_check")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let lines: rusqlite::Result<Vec<String>> = rows.collect();
        let report = lines?.join("\n");
        if report.trim() == "ok" {
            Ok("ok".to_string())
        } else {
            Err(anyhow::anyhow!("Integrity issues found:\n{}", report))
        }
    }
}

fn run_schema_migrations(conn: &Connection) -> Result<()> {
    let has_visibility: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tabs') WHERE name = 'visibility'",
        [],
        |row| row.get(0),
    )?;
    if !has_visibility {
        conn.execute(
            "ALTER TABLE tabs ADD COLUMN visibility TEXT NOT NULL DEFAULT 'active' CHECK (visibility IN ('active','archived','hidden'))",
            [],
        )?;
    }
    let has_composer_draft: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tabs') WHERE name = 'composer_draft'",
        [],
        |row| row.get(0),
    )?;
    if !has_composer_draft {
        conn.execute(
            "ALTER TABLE tabs ADD COLUMN composer_draft TEXT NOT NULL DEFAULT ''",
            [],
        )?;
    }
    let has_ai_gender: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tabs') WHERE name = 'ai_gender'",
        [],
        |row| row.get(0),
    )?;
    if !has_ai_gender {
        conn.execute("ALTER TABLE tabs ADD COLUMN ai_gender TEXT", [])?;
    }
    let has_template_hash: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tabs') WHERE name = 'template_version_or_hash'",
        [],
        |row| row.get(0),
    )?;
    if !has_template_hash {
        conn.execute(
            "ALTER TABLE tabs ADD COLUMN template_version_or_hash TEXT",
            [],
        )?;
    }
    let has_thinking_content: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('messages') WHERE name = 'thinking_content'",
        [],
        |row| row.get(0),
    )?;
    if !has_thinking_content {
        conn.execute("ALTER TABLE messages ADD COLUMN thinking_content TEXT", [])?;
    }
    let has_context_fingerprint: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('messages') WHERE name = 'context_fingerprint'",
        [],
        |row| row.get(0),
    )?;
    if !has_context_fingerprint {
        conn.execute(
            "ALTER TABLE messages ADD COLUMN context_fingerprint TEXT",
            [],
        )?;
    }
    Ok(())
}

/// Returns true if this SELECT is likely to expose sensitive chat content or config
/// and should be blocked in restricted (non-admin-token) mode.
fn is_select_exposing_sensitive(upper: &str) -> bool {
    // Quick checks: if there's no reference to sensitive tables, allow.
    if !upper.contains("MESSAGES") && !upper.contains("TABS") {
        return false;
    }

    // Block SELECT * from messages (would expose content).
    if upper.contains("MESSAGES") && (upper.contains("SELECT *") || upper.contains("SELECT  *")) {
        return true;
    }

    // Block if selecting messages.content explicitly.
    if upper.contains("MESSAGES") && upper.contains("MESSAGES.CONTENT") {
        return true;
    }
    if upper.contains("M.CONTENT") {
        return true;
    }

    // Block if selecting tabs.system_prompt, context_notes, or model_params.
    if upper.contains("TABS")
        && (upper.contains("SYSTEM_PROMPT")
            || upper.contains("CONTEXT_NOTES")
            || upper.contains("MODEL_PARAMS"))
    {
        return true;
    }

    false
}

fn normalize_fts_query(query: &str) -> String {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| format!("{token}*"))
        .collect::<Vec<_>>()
        .join(" AND ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_tab(id: &str, name: &str) -> ChatTabRow {
        ChatTabRow {
            id: id.to_string(),
            name: name.to_string(),
            system_prompt: String::new(),
            ai_name: None,
            user_name: None,
            explicit_level: 0,
            active_template_id: None,
            auto_compact: true,
            auto_compact_summarize: false,
            compact_mode: "percent".to_string(),
            compact_threshold: 0.8,
            model_params: serde_json::json!({}),
            context_notes: serde_json::json!([]),
            sidebar_width: 280,
            tab_order: 0,
            pinned: false,
            last_ctx_pct: None,
            total_input_tokens: 0,
            total_output_tokens: 0,
            created_at: 1,
            updated_at: 1,
            visibility: "active".to_string(),
            composer_draft: String::new(),
            ai_gender: None,
            template_version_or_hash: None,
            messages: Vec::new(),
        }
    }

    #[test]
    fn schema_migration_adds_composer_draft_for_existing_databases() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("chat.db");

        let conn = Connection::open(&db_path).expect("open raw db");
        conn.execute_batch(
            r#"
            CREATE TABLE tabs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL DEFAULT '',
                ai_name TEXT,
                user_name TEXT,
                explicit_level INTEGER NOT NULL DEFAULT 0,
                active_template_id TEXT,
                auto_compact INTEGER NOT NULL DEFAULT 1,
                auto_compact_summarize INTEGER NOT NULL DEFAULT 0,
                compact_mode TEXT NOT NULL DEFAULT 'percent',
                compact_threshold REAL NOT NULL DEFAULT 0.8,
                model_params TEXT NOT NULL DEFAULT '{}',
                context_notes TEXT NOT NULL DEFAULT '[]',
                sidebar_width INTEGER NOT NULL DEFAULT 280,
                tab_order INTEGER NOT NULL DEFAULT 0,
                pinned INTEGER NOT NULL DEFAULT 0,
                last_ctx_pct REAL,
                total_input_tokens INTEGER NOT NULL DEFAULT 0,
                total_output_tokens INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'active'
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tab_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                thinking_content TEXT,
                timestamp_ms INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cumulative_input_tokens INTEGER,
                cumulative_output_tokens INTEGER,
                compaction_marker INTEGER NOT NULL DEFAULT 0,
                variants TEXT,
                variant_index INTEGER,
                seq INTEGER NOT NULL
            );
            "#,
        )
        .expect("create legacy schema");
        drop(conn);

        let store = ChatStorage::open(&db_path).expect("open migrated storage");
        let columns: i64 = {
            let guard = store.conn.lock().expect("lock db");
            let conn = guard.as_ref().expect("db open");
            conn.query_row(
                "SELECT COUNT(*) FROM pragma_table_info('tabs') WHERE name = 'composer_draft'",
                [],
                |row| row.get(0),
            )
            .expect("query columns")
        };
        assert_eq!(columns, 1);
    }

    #[test]
    fn schema_migration_adds_thinking_content_for_existing_databases() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("chat.db");

        let conn = Connection::open(&db_path).expect("open raw db");
        conn.execute_batch(
            r#"
            CREATE TABLE tabs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL DEFAULT '',
                ai_name TEXT,
                user_name TEXT,
                explicit_level INTEGER NOT NULL DEFAULT 0,
                active_template_id TEXT,
                auto_compact INTEGER NOT NULL DEFAULT 1,
                auto_compact_summarize INTEGER NOT NULL DEFAULT 0,
                compact_mode TEXT NOT NULL DEFAULT 'percent',
                compact_threshold REAL NOT NULL DEFAULT 0.8,
                model_params TEXT NOT NULL DEFAULT '{}',
                context_notes TEXT NOT NULL DEFAULT '[]',
                sidebar_width INTEGER NOT NULL DEFAULT 280,
                tab_order INTEGER NOT NULL DEFAULT 0,
                pinned INTEGER NOT NULL DEFAULT 0,
                last_ctx_pct REAL,
                total_input_tokens INTEGER NOT NULL DEFAULT 0,
                total_output_tokens INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'active',
                composer_draft TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tab_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cumulative_input_tokens INTEGER,
                cumulative_output_tokens INTEGER,
                compaction_marker INTEGER NOT NULL DEFAULT 0,
                variants TEXT,
                variant_index INTEGER,
                seq INTEGER NOT NULL
            );
            "#,
        )
        .expect("create legacy schema");
        drop(conn);

        let store = ChatStorage::open(&db_path).expect("open migrated storage");
        let columns: i64 = {
            let guard = store.conn.lock().expect("lock db");
            let conn = guard.as_ref().expect("db open");
            conn.query_row(
                "SELECT COUNT(*) FROM pragma_table_info('messages') WHERE name = 'thinking_content'",
                [],
                |row| row.get(0),
            )
            .expect("query columns")
        };
        assert_eq!(columns, 1);
    }

    fn make_message(tab_id: &str, role: &str, content: &str) -> MessageRow {
        MessageRow {
            id: 0,
            tab_id: tab_id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            thinking_content: None,
            timestamp_ms: 1,
            input_tokens: None,
            output_tokens: None,
            cumulative_input_tokens: None,
            cumulative_output_tokens: None,
            context_fingerprint: None,
            compaction_marker: false,
            variants: None,
            variant_index: None,
            seq: 0,
        }
    }

    #[test]
    fn context_fingerprint_round_trips_append_replace_get_with_usage() {
        let dir = tempdir().expect("temp dir");
        let store = ChatStorage::open(&dir.path().join("chat.db")).expect("open storage");
        let mut tab = make_tab("usage", "Usage");
        tab.total_input_tokens = 1_234;
        tab.total_output_tokens = 567;
        store.create_tab(&tab).expect("create tab");
        let mut messages = Vec::new();
        for fingerprint in [Some("context-v1"), Some(""), None] {
            let mut msg = make_message("usage", "assistant", "answer");
            msg.context_fingerprint = fingerprint.map(str::to_string);
            msg.input_tokens = Some(101);
            msg.output_tokens = Some(23);
            msg.cumulative_input_tokens = Some(1_234);
            msg.cumulative_output_tokens = Some(567);
            store.append_message(&msg).expect("append message");
            messages.push(msg);
        }
        for replace in [false, true] {
            if replace {
                store
                    .replace_messages("usage", &messages)
                    .expect("replace messages");
            }
            let loaded = store.get_tab("usage").expect("get tab");
            assert_eq!(loaded.total_input_tokens, 1_234);
            assert_eq!(loaded.total_output_tokens, 567);
            assert_eq!(loaded.messages.len(), messages.len());
            for (seq, (actual, expected)) in loaded.messages.iter().zip(&messages).enumerate() {
                assert_eq!(actual.context_fingerprint, expected.context_fingerprint);
                assert_eq!(actual.input_tokens, expected.input_tokens);
                assert_eq!(actual.output_tokens, expected.output_tokens);
                assert_eq!(
                    actual.cumulative_input_tokens,
                    expected.cumulative_input_tokens
                );
                assert_eq!(
                    actual.cumulative_output_tokens,
                    expected.cumulative_output_tokens
                );
                assert_eq!(actual.seq, seq as i64);
            }
        }
    }

    #[test]
    fn legacy_message_json_defaults_context_fingerprint_to_none() {
        let msg: MessageRow = serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": "legacy answer",
            "input_tokens": 101, "output_tokens": 23,
            "cumulative_input_tokens": 1234, "cumulative_output_tokens": 567
        }))
        .expect("deserialize legacy message");
        assert_eq!(msg.context_fingerprint, None);
        assert_eq!(msg.input_tokens, Some(101));
        assert_eq!(msg.output_tokens, Some(23));
        assert_eq!(msg.cumulative_input_tokens, Some(1_234));
        assert_eq!(msg.cumulative_output_tokens, Some(567));
    }

    #[test]
    fn schema_migration_adds_nullable_context_fingerprint_for_existing_databases() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("chat.db");
        let conn = Connection::open(&db_path).expect("open raw db");
        let legacy_schema = SCHEMA_SQL.replace("    context_fingerprint       TEXT,\n", "");
        conn.execute_batch(&legacy_schema)
            .expect("create legacy schema");
        conn.execute(
            "INSERT INTO tabs (id, name, created_at, updated_at,
                total_input_tokens, total_output_tokens) VALUES ('legacy', 'Legacy', 1, 1, 1234, 567)",
            [],
        ).expect("insert legacy tab");
        conn.execute(
            "INSERT INTO messages (tab_id, role, content, seq,
                input_tokens, output_tokens, cumulative_input_tokens, cumulative_output_tokens)
             VALUES ('legacy', 'assistant', 'old answer', 0, 101, 23, 1234, 567)",
            [],
        )
        .expect("insert legacy message");
        drop(conn);
        // Opening twice also verifies the additive migration is idempotent.
        for _ in 0..2 {
            let store = ChatStorage::open(&db_path).expect("open migrated storage");
            let loaded = store.get_tab("legacy").expect("get legacy tab");
            assert_eq!(loaded.total_input_tokens, 1_234);
            assert_eq!(loaded.total_output_tokens, 567);
            assert_eq!(loaded.messages.len(), 1);
            let msg = &loaded.messages[0];
            assert_eq!(msg.context_fingerprint, None);
            assert_eq!(msg.input_tokens, Some(101));
            assert_eq!(msg.output_tokens, Some(23));
            assert_eq!(msg.cumulative_input_tokens, Some(1_234));
            assert_eq!(msg.cumulative_output_tokens, Some(567));
            let guard = store.conn.lock().expect("lock db");
            let conn = guard.as_ref().expect("db open");
            let nullable_columns: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM pragma_table_info('messages')
                 WHERE name = 'context_fingerprint' AND type = 'TEXT' AND \"notnull\" = 0",
                    [],
                    |row| row.get(0),
                )
                .expect("query nullable column");
            assert_eq!(nullable_columns, 1);
        }
    }

    #[test]
    fn thinking_content_round_trips_through_storage() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("chat.db");
        let store = ChatStorage::open(&db_path).expect("open storage");

        let tab = make_tab("tab-1", "Thinking");
        store.create_tab(&tab).expect("create tab");

        let mut msg = make_message("tab-1", "assistant", "hello");
        msg.thinking_content = Some("step 1 -> step 2".to_string());
        store.append_message(&msg).expect("append message");

        let loaded = store.get_tab("tab-1").expect("load tab");
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(
            loaded.messages[0].thinking_content.as_deref(),
            Some("step 1 -> step 2")
        );
    }

    #[test]
    fn normalize_fts_query_builds_prefix_and_terms() {
        assert_eq!(normalize_fts_query("rain"), "rain*");
        assert_eq!(normalize_fts_query("gpu-43c"), "gpu* AND 43c*");
        assert_eq!(
            normalize_fts_query("slow HTTP endpoint."),
            "slow* AND HTTP* AND endpoint*"
        );
        assert_eq!(normalize_fts_query("..."), "");
    }

    #[test]
    fn search_supports_prefix_matches() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("chat.db");
        let store = ChatStorage::open(&db_path).expect("open storage");

        let tab = make_tab("tab-1", "Noir Scene");
        store.create_tab(&tab).expect("create tab");
        store
            .append_message(&make_message(
                "tab-1",
                "assistant",
                "The rain fell like needles on the pavement.",
            ))
            .expect("append message");

        let results = store
            .search("rai", 10, 0, &[], None)
            .expect("prefix search");
        assert_eq!(results.total, 1);
        assert_eq!(results.results.len(), 1);
        assert_eq!(results.results[0].tab_name, "Noir Scene");
        assert!(results.results[0].snippet.contains("<mark>rain</mark>"));
    }

    #[test]
    fn search_tolerates_free_form_punctuation() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("chat.db");
        let store = ChatStorage::open(&db_path).expect("open storage");

        let tab = make_tab("tab-1", "Debug Session");
        store.create_tab(&tab).expect("create tab");
        store
            .append_message(&make_message(
                "tab-1",
                "assistant",
                "Check gpu-43c status before you debug the slow HTTP endpoint.",
            ))
            .expect("append message");

        let hyphenated = store
            .search("gpu-43c", 10, 0, &[], None)
            .expect("hyphenated search");
        assert_eq!(hyphenated.total, 1);
        assert_eq!(hyphenated.results.len(), 1);

        let punctuated = store
            .search("slow HTTP endpoint.", 10, 0, &[], None)
            .expect("punctuated search");
        assert_eq!(punctuated.total, 1);
        assert_eq!(punctuated.results.len(), 1);
        assert_eq!(punctuated.results[0].tab_name, "Debug Session");
    }

    #[test]
    fn search_scoped_to_tab_excludes_other_tabs() {
        let dir = tempdir().expect("temp dir");
        let db_path = dir.path().join("chat.db");
        let store = ChatStorage::open(&db_path).expect("open storage");

        let tab_a = make_tab("tab-a", "Conversation A");
        let tab_b = make_tab("tab-b", "Conversation B");
        store.create_tab(&tab_a).expect("create tab a");
        store.create_tab(&tab_b).expect("create tab b");

        store
            .append_message(&make_message("tab-a", "user", "The dragon breathed fire."))
            .expect("msg a");
        store
            .append_message(&make_message("tab-b", "user", "The dragon slept quietly."))
            .expect("msg b");

        // Unscoped search returns both tabs.
        let all = store.search("dragon", 10, 0, &[], None).expect("unscoped");
        assert_eq!(all.total, 2);

        // Scoped to tab-a returns only tab-a result.
        let scoped = store
            .search("dragon", 10, 0, &[], Some("tab-a"))
            .expect("scoped");
        assert_eq!(scoped.total, 1);
        assert_eq!(scoped.results[0].tab_id, "tab-a");

        // Scoped to tab-b returns only tab-b result.
        let scoped_b = store
            .search("dragon", 10, 0, &[], Some("tab-b"))
            .expect("scoped b");
        assert_eq!(scoped_b.total, 1);
        assert_eq!(scoped_b.results[0].tab_id, "tab-b");

        // Scoped to a nonexistent tab returns nothing.
        let empty = store
            .search("dragon", 10, 0, &[], Some("tab-x"))
            .expect("empty scope");
        assert_eq!(empty.total, 0);
    }

    #[test]
    fn admin_query_rejects_unbounded_row_results() {
        let dir = tempdir().expect("temp dir");
        let store = ChatStorage::open(&dir.path().join("chat.db")).expect("open storage");
        let error = store
        .execute_query(
            "SELECT n FROM (WITH RECURSIVE rows(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM rows WHERE n < 1001) SELECT n FROM rows)",
            true,
        )
        .expect_err("row limit must be enforced");
        assert!(error.to_string().contains("row limit"));
    }

    #[test]
    fn admin_query_rejects_unbounded_result_bytes() {
        let dir = tempdir().expect("temp dir");
        let store = ChatStorage::open(&dir.path().join("chat.db")).expect("open storage");
        let error = store
        .execute_query(
            "SELECT payload FROM (WITH RECURSIVE rows(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM rows WHERE n < 1000) SELECT printf('%01200d', n) AS payload FROM rows)",
            true,
        )
        .expect_err("byte limit must be enforced");
        assert!(error.to_string().contains("byte limit"));
    }
}
