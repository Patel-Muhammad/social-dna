"""Schema definition with simple version-based migrations.

Each migration is (version, SQL). `schema_version` records what has been applied,
so future changes are added as new entries without touching old ones.
"""
from __future__ import annotations

import sqlite3

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE dataset_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            dataset_source TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            total_users INTEGER NOT NULL CHECK (total_users >= 0),
            total_edges INTEGER NOT NULL CHECK (total_edges >= 0),
            feature_count INTEGER,
            raw_sha256 TEXT NOT NULL UNIQUE,
            processed_path TEXT,
            validation_json TEXT,
            processed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER REFERENCES dataset_metadata(id) ON DELETE CASCADE,
            algorithm TEXT NOT NULL CHECK (algorithm IN ('louvain','greedy_modularity')),
            random_seed INTEGER,
            resolution REAL,
            modularity REAL,
            community_count INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','failed')),
            notes TEXT
        );
        CREATE INDEX idx_runs_lookup ON analysis_runs(algorithm, random_seed, resolution, status);

        CREATE TABLE communities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            community_number INTEGER NOT NULL,
            member_count INTEGER NOT NULL CHECK (member_count > 0),
            percentage_of_network REAL NOT NULL,
            description TEXT,
            description_model TEXT,
            UNIQUE (analysis_run_id, community_number)
        );

        CREATE TABLE community_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
            original_user_id INTEGER NOT NULL,
            UNIQUE (analysis_run_id, original_user_id)
        );
        CREATE INDEX idx_members_comm ON community_members(community_id);

        CREATE TABLE community_interests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
            feature_id INTEGER NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('common','distinctive')),
            rank INTEGER NOT NULL,
            frequency INTEGER NOT NULL,
            prevalence REAL NOT NULL,
            network_prevalence REAL NOT NULL,
            lift REAL,
            distinctiveness_score REAL
        );
        CREATE INDEX idx_interests_comm ON community_interests(community_id, kind, rank);

        CREATE TABLE community_metrics (
            community_id INTEGER PRIMARY KEY REFERENCES communities(id) ON DELETE CASCADE,
            analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            internal_edges INTEGER NOT NULL,
            external_edges INTEGER NOT NULL,
            internal_density REAL,
            avg_internal_degree REAL,
            avg_degree REAL,
            conductance REAL,
            within_cosine REAL,
            within_jaccard REAL,
            cosine_to_network REAL,
            avg_features_per_user REAL
        );

        CREATE TABLE community_similarity (
            analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            community_a INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
            community_b INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
            cosine_similarity REAL NOT NULL,
            jaccard_similarity REAL NOT NULL,
            cross_edges INTEGER NOT NULL,
            PRIMARY KEY (analysis_run_id, community_a, community_b),
            CHECK (community_a < community_b)
        );

        CREATE TABLE network_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            note TEXT,
            UNIQUE (analysis_run_id, metric_name)
        );

        CREATE TABLE user_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
            original_user_id INTEGER NOT NULL,
            degree INTEGER,
            degree_centrality REAL,
            betweenness_centrality REAL,
            closeness_centrality REAL,
            pagerank REAL,
            community_id INTEGER REFERENCES communities(id) ON DELETE CASCADE,
            external_degree INTEGER,
            neighbour_communities INTEGER,
            participation_coefficient REAL,
            UNIQUE (analysis_run_id, original_user_id)
        );
        CREATE INDEX idx_user_metrics_comm ON user_metrics(analysis_run_id, community_id);

        CREATE TABLE chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            title TEXT NOT NULL DEFAULT 'New chat'
        );

        CREATE TABLE chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user','assistant')),
            message TEXT NOT NULL,
            meta_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_messages_session ON chat_messages(session_id);
        """,
    ),
]

LATEST_VERSION = MIGRATIONS[-1][0]


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0] or 0)


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply any missing migrations inside one transaction each; return final version."""
    version = current_version(conn)
    for number, sql in MIGRATIONS:
        if number > version:
            try:
                conn.executescript("BEGIN;\n" + sql + f"\nINSERT INTO schema_version(version) VALUES ({number});\nCOMMIT;")
            except Exception:
                conn.rollback()
                raise
            version = number
    return version
