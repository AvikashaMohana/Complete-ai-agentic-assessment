"""SQLite-backed persistent chat memory."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")


@asynccontextmanager
async def open_checkpointer():
    db_path = Path(CHECKPOINT_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        await saver.setup()
        yield saver
