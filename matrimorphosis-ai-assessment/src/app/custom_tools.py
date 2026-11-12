"""Tasks 7 and 8: four user-scoped knowledge-base tools."""
import asyncio
import time

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool
from opentelemetry import metrics

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import log


_meter = metrics.get_meter("agent")
KBSTATS_CALLS = _meter.create_counter(
    "kbstats.calls", unit="1", description="Knowledge-base statistics tool calls"
)
OUTLINE_CALLS = _meter.create_counter(
    "outline.calls", unit="1", description="Document outline tool calls"
)


def _unknown_document(doc_name: str, documents: list[dict]) -> ToolException:
    names = ", ".join(f'"{d["filename"]}"' for d in documents) or "(none)"
    return ToolException(
        f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in this user's knowledge base. "
        f"Available documents: {names}. Next step: retry using one of these exact file names."
    )


async def _find_document(kb: KnowledgeBase, user_id: str, doc_name: str):
    """Resolve a user-supplied name and return its exact registry row."""
    documents = await asyncio.to_thread(kb.documents, user_id)
    exact_name = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
    if exact_name is None:
        raise _unknown_document(doc_name, documents)

    document = next(
        (d for d in documents if d["filename"] == exact_name),
        None,
    )
    if document is None:
        raise _unknown_document(doc_name, documents)

    return document


def build_custom_tools(kb: KnowledgeBase) -> list:

    @tool
    async def kb_stats(runtime: ToolRuntime[Ctx], top_keywords: int = 5) -> str:
        """Show statistics about the calling user's knowledge base.
        Use this when the user asks how many documents, sections or chunks are stored,
        or asks for the most common topics and keywords.

        Args:
            top_keywords: Number of top keywords to display, from 1 to 20 (default 5).
        """
        started = time.perf_counter()
        status = "ok"

        try:
            if isinstance(top_keywords, bool) or not isinstance(top_keywords, int) or not 1 <= top_keywords <= 20:
                raise ToolException(
                    "ERROR[INVALID_INPUT]: top_keywords must be an integer from 1 to 20. "
                    "Next step: provide a value between 1 and 20."
                )

            user_id = runtime.context.user_id
            stats = await asyncio.to_thread(kb.stats, user_id)

            if stats["documents"] == 0:
                return "The knowledge base is empty."

            keywords = stats["keywords"][:top_keywords]
            keyword_text = ", ".join(keywords) if keywords else "(none)"

            return (
                f"{stats['documents']} documents, {stats['sections']} sections, "
                f"{stats['chunks']} chunks. Top keywords: {keyword_text}."
        )

        except Exception:
            status = "error"
            raise

        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            KBSTATS_CALLS.add(1, {"status": status})
            log.info(
                "kb_stats_done",
                tool="kb_stats",
                status=status,
                duration_ms=duration_ms,
            )

    @tool
    async def document_outline(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
        """Show the structure and section summaries of one document in the calling user's knowledge base.
        Use this when the user asks what a document contains, how it is organised,
        or which section covers a topic. After finding the relevant section, search it
        with search_knowledge_base.

        Args:
            doc_name: The file name of a document in the user's knowledge-base list.
        """
        started = time.perf_counter()
        status = "ok"
        section_count = 0

        try:
            user_id = runtime.context.user_id
            document = await _find_document(kb, user_id, doc_name)

            sections = await asyncio.to_thread(
                kb.outline, user_id, document["doc_id"]
            )
            section_count = len(sections)

            lines = [
                f"Outline of '{document['filename']}' ({section_count} sections):"
            ]

            for position, section in enumerate(sections, start=1):
                page_start = section["page_start"]
                page_end = section["page_end"]

                if page_start is None:
                    page_text = ""
                elif page_start == page_end:
                    page_text = f" (p.{page_start})"
                else:
                    page_text = f" (p.{page_start}-{page_end})"

                lines.append(
                    f"{position}. [L{section['level']}] {section['heading']}"
                    f"{page_text}: {section['summary']}"
                )

            return "\n".join(lines)

        except Exception:
            status = "error"
            raise

        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            OUTLINE_CALLS.add(1, {"status": status})
            log.info(
                "document_outline_done",
                tool="document_outline",
                status=status,
                duration_ms=duration_ms,
                sections=section_count,
            )

    @tool
    async def delete_document(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
        """Delete one document from the calling user's knowledge base.
        Use this when the user explicitly asks to remove a document.

        Args:
            doc_name: The file name of a document in the user's knowledge-base list.
        """
        user_id = runtime.context.user_id
        document = await _find_document(kb, user_id, doc_name)

        deleted = await asyncio.to_thread(
            kb.delete_document, user_id, document["doc_id"]
        )

        if not deleted:
            documents = await asyncio.to_thread(kb.documents, user_id)
            raise _unknown_document(doc_name, documents)

        remaining = await asyncio.to_thread(kb.documents, user_id)

        return (
            f"Deleted '{document['filename']}'. "
            f"{len(remaining)} documents left."
        )

    @tool
    async def update_document(
        doc_name: str,
        source: str,
        runtime: ToolRuntime[Ctx],
    ) -> str:
        """Replace an existing document in the calling user's knowledge base with a newly uploaded version.
        Use this when the user explicitly asks to update or replace a document.
        The new upload is ingested under the existing document's file name before
        the old version is removed.

        Args:
            doc_name: The existing file name in the user's knowledge-base list.
            source: The file_id of the newly uploaded replacement file.
        """
        user_id = runtime.context.user_id
        old_document = await _find_document(kb, user_id, doc_name)

        try:
            path, _ = await asyncio.to_thread(
                files.resolve, user_id, source.strip()
            )

            result, existing = await ingest_file(
                kb,
                user_id,
                path,
                old_document["filename"],
                runtime.config,
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

        if existing:
            return f"'{old_document['filename']}' is already up to date."

        # Only remove the old version after the new version has been ingested successfully.
        deleted = await asyncio.to_thread(
            kb.delete_document, user_id, old_document["doc_id"]
        )

        if not deleted:
            raise ToolException(
                "ERROR[UPDATE_FAILED]: The new version was ingested, but the old version "
                "could not be removed. Next step: ask an administrator to inspect the knowledge base."
            )

        return (
            f"Updated '{old_document['filename']}': "
            f"{len(result.sections)} sections, {result.n_chunks} chunks."
        )

    for custom_tool in (
        kb_stats,
        document_outline,
        delete_document,
        update_document,
    ):
        custom_tool.handle_tool_error = True

    return [
        kb_stats,
        document_outline,
        delete_document,
        update_document,
    ]