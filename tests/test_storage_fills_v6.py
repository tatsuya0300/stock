import sqlite3

from jp_signal.storage import SCHEMA_VERSION, Storage

EXPECTED_V6_COLUMNS = {
    "exit_date",
    "exit_price",
    "exit_slippage_bp",
    "holding_days",
    "carry_cost",
    "carry_days",
    "pnl",
}


def test_new_database_has_fills_v6_columns(
    tmp_path,
):
    db_path = tmp_path / "new.sqlite"

    with Storage(str(db_path)) as storage:
        columns = storage._table_columns("fills")

        assert EXPECTED_V6_COLUMNS.issubset(
            columns
        )

        assert storage.get_metadata(
            "schema_version"
        ) == str(SCHEMA_VERSION)


def test_old_fills_table_is_migrated(
    tmp_path,
):
    db_path = tmp_path / "old.sqlite"

    connection = sqlite3.connect(db_path)

    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE fills (
            run_id TEXT,
            trade_date TEXT,
            code TEXT,
            side TEXT,
            qty INTEGER,
            price REAL,
            note TEXT,
            source_file_hash TEXT,
            source_row_number INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_file_hash, source_row_number)
        );
        """
    )

    connection.commit()
    connection.close()

    with Storage(str(db_path)) as storage:
        columns = storage._table_columns("fills")

        assert EXPECTED_V6_COLUMNS.issubset(
            columns
        )

        assert storage.get_metadata(
            "schema_version"
        ) == str(SCHEMA_VERSION)
