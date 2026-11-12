
"""Authenticated per-user notes and tasks MCP server."""

import json
import os
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from app.mcp_server.taskstore import TaskError, TaskStore
from app.users import UserStore

mcp = FastMCP(
    "tasks",
    host="0.0.0.0",
    port=int(os.getenv("MCP_PORT", "9000")),
)

store = TaskStore()
users = UserStore()


def current_user_id(ctx: Context) -> str:
    """Authenticate the request and return the calling user's ID."""
    request = ctx.request_context.request
    authorization = request.headers.get("authorization", "")
    parts = authorization.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ToolError(
            "ERROR[UNAUTHENTICATED]: Missing or malformed bearer token. "
            "Next step: log in and retry."
        )

    user = users.user_for_token(parts[1])

    if user is None:
        raise ToolError(
            "ERROR[UNAUTHENTICATED]: Invalid or expired bearer token. "
            "Next step: log in and retry."
        )

    return user["user_id"]


@mcp.tool()
async def add_note(ctx: Context, title: str, body: str) -> str:
    """Create a private note for the authenticated user.
    Use when the user asks to save or remember information.
    Args:
        title: Short title of the note.
        body: Content of the note.
    """
    user_id = current_user_id(ctx)
    try:
        result = store.add_note(user_id, title, body)
        return json.dumps(result)
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def list_notes(ctx: Context) -> str:
    """List the authenticated user's saved notes.
    Use when the user asks to view or retrieve their notes.
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.list_notes(user_id))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def delete_note(ctx: Context, note_id: int) -> str:
    """Delete one of the authenticated user's notes.
    Use when the user explicitly asks to delete a note.
    Args:
        note_id: ID of the note, obtained from list_notes.
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.delete_note(user_id, note_id))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def add_task(
    ctx: Context,
    title: str,
    details: str = "",
) -> str:
    """Create a task for the authenticated user.
    Use when the user asks to add or plan a task.
    Args:
        title: Short task title.
        details: Optional additional task details.
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.add_task(user_id, title, details))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def list_tasks(ctx: Context, status: str = "") -> str:
    """List the authenticated user's tasks.
    Use when the user asks to view tasks or filter them by status.
    Args:
        status: Optional filter: todo, in_progress, or done. Empty means all tasks.
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.list_tasks(user_id, status))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def set_task_status(
    ctx: Context,
    task_id: int,
    status: str,
) -> str:
    """Change the status of one of the authenticated user's tasks.
    Use when the user asks to progress, complete, or reopen a task.
    Args:
        task_id: ID of the task, obtained from list_tasks.
        status: New status: todo, in_progress, or done.
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.set_task_status(user_id, task_id, status))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def delete_task(ctx: Context, task_id: int) -> str:
    """Delete one of the authenticated user's tasks.
    Use when the user explicitly asks to remove a task.
    Args:
        task_id: ID of the task, obtained from list_tasks.
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.delete_task(user_id, task_id))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def task_summary(ctx: Context) -> str:
    """Summarise the authenticated user's tasks by status.
    Use when the user asks for a task progress overview.
    """
    user_id = current_user_id(ctx)
    try:
        return json.dumps(store.summary(user_id))
    except TaskError as error:
        raise ToolError(str(error)) from error


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http" if "--http" in sys.argv else "stdio"
    )
