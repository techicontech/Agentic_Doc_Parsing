"""The LLM agents, as Google ADK LlmAgents over LiteLLM (spec §3, §14).

Only steps that need judgment live here. Everything deterministic — lexical search,
RRF fusion, ranking rules, the verification gate — stays a plain function elsewhere.

Contracts (each name below):
  convention_detector — Input: sampled headers/footers. Output: CitationConvention JSON.
    Model: settings.llm_model. Failure: heuristic_detect, then page-number fallback.
  page_router — Input: page text density + sample. Output: {route, reason}.
    Model: settings.llm_model. Failure: vision_ocr bucket.
  vision_ocr — Input: PNG. Output: markdown OCR. Model: settings.ocr_vision_model.
    Failure: caller retry; never invent text.
  query_router — Input: question. Output: {labels, reason}. Failure: all three paths.
  tree_navigator — Input: question + section catalog. Output: {doc_codes, procedure_nos, titles}.
    Failure: empty structural path.
  answer_synthesizer — Input: question + evidence. Output: grounded answer string.
    Failure: evidence-only dump (see nodes.synthesize_answer).

Every model string is overridable via config, not hardcoded in the agent files.
Every model call goes through LiteLLM. If ADK is unavailable or an agent run fails,
the same prompt is sent through `marine_docs.llm.chat_completion`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from marine_docs.config import get_settings
from marine_docs.llm import chat_completion, configure_litellm, llm_ready

logger = logging.getLogger(__name__)

PAGE_ROUTER = "page_router"
VISION_OCR = "vision_ocr"
QUERY_ROUTER = "query_router"
TREE_NAVIGATOR = "tree_navigator"
ANSWER_SYNTHESIZER = "answer_synthesizer"

CONVENTION_DETECTOR = "convention_detector"

APP_NAME = "marine_docs_intelligence"

INSTRUCTIONS: dict[str, str] = {
    CONVENTION_DETECTOR: """You infer how THIS technical manual wants pages cited.

You will see sampled headers and footers from one PDF (any manufacturer). Identify:
(a) a consistent citation code readers are told to quote, if any;
(b) a separate revision/edition indicator, and whether it is per-section or per-document;
(c) a plain printed page number, independent of (a).

Do not assume MAN B&W, Wartsila, or any other maker's format. If there is no
structured convention, say so.

Return ONLY JSON:
{"citation_field":"citation_key","pattern":"regex for the quoted code",
 "revision_field":"edition or revision or null",
 "revision_granularity":"per_section|per_document|none",
 "label_words":["section"],"quote_hint":"short template or null",
 "has_printed_page":true,"no_structured_convention":false,"notes":"short"}""",
    PAGE_ROUTER: """You route one PDF page of a marine engine manual to an extractor.
The heuristic router could not decide because the page mixes prose and drawings.

- "docling": the page is mostly readable prose/tables; the text layer is trustworthy.
- "vision": the page is mostly diagrams/plates, or its text layer is broken/garbled.

Return ONLY JSON: {"route": "docling", "reason": "short why"}""",
    VISION_OCR: """You are OCR and captioning for a marine technical manual page
(diagrams, plates, tables). Extract ALL readable text exactly. Preserve labels, part
numbers, procedure and plate codes, dimensions, and table structure in markdown.
For each drawing, list every callout number with its label. Never invent text.
Reply with markdown only.""",
    QUERY_ROUTER: """You are the query router for a technical-manual retrieval system.
Classify the question with one or more labels (multi-label, not exclusive):

- lexical: exact identifiers, procedure/plate codes, numbers, torques, clearances, spec lookups
- structural: procedures, how-to, multi-step work, keep/replace criteria in the chapter tree
- visual: diagrams, plates, panels, wiring/piping, "show me", figures

Keep/replace, wear-limit, and "is this still valid" questions need BOTH lexical and structural.
Callout numbers, plate codes, panel/drawing IDs, and "show me the diagram" need visual as well.

Return ONLY JSON: {"labels": ["lexical","structural"], "reason": "short why"}""",
    TREE_NAVIGATOR: """You navigate a technical maintenance manual the way an engineer uses
the table of contents: no embeddings, just the hierarchy.

Given the question and the section catalog, pick the 3-8 most relevant sections.
Use only real doc_codes, procedure numbers and titles from the catalog.
For a keep/replace, wear, clearance, or criteria question, pick BOTH the numeric
data sheet AND the checking/inspection procedure that share the same component
title. Do not prefer one code family over another.
If the question asks about hours, overhaul interval, or a maintenance schedule,
include the schedule section as well as the related checking procedure.
If the question names a plate, drawing, panel, or callout, include that plate
section as well as the procedure that uses it.
Pick sections whose component title shares the most words with the question.
Do not substitute a sibling component that only shares a similar number or a
shared table-row code.

Return ONLY JSON: {"doc_codes": ["..."], "procedure_nos": ["..."], "titles": ["..."]}""",
    ANSWER_SYNTHESIZER: """You are a technical manual assistant.
Answer ONLY from the provided evidence. If the evidence does not answer the question,
say so plainly instead of guessing.

Cite the manual's own reference using the citation_key on the evidence (plus
revision/edition when present). Printed and PDF page numbers are supplementary
context, never the primary citation. One manufacturer's form is "Procedure
902-1.3 Edition 0286"; another may be "Section 4.2.1 Rev C" — quote whatever
the evidence carries. Be exact with numbers, units and step order, and
never invent a value. Mention relevant diagram panels by drawing code and step number.

Reading rules that apply to any manual:
- If a value is stated as N above/below a named reference (adjustment sheet, recorded
  value, as-fitted, original, previously measured), N is a delta, not an absolute
  limit. Apply it to that reference when the reference is in evidence; otherwise
  report the delta as written.
- When both a numeric data sheet and a checking/inspection procedure for the same
  component are in evidence, apply the procedure's if/then (reuse vs replace vs
  inspect) first. A limits table alone is not the full decision.
- Checklist / checkbox items: only items the evidence marks as selected, ticked, or
  required apply. Unmarked or empty boxes are not requirements.
- If two numeric values for the same quantity disagree, report both with citations.
  Do not pick one silently.
- Arithmetic honesty: never say A exceeds B if A is less than B. Compare numbers
  before choosing a band. If a measured value is below an inspect/replace threshold,
  do not apply the exceeded-branch action. If a check uses a different measured
  quantity than the question supplies, do not treat the question's number as that
  check's input.
- Scene-setting (at sea, in port, dry dock, afloat) is context for which checks are
  valid, not an extra OEM condition token you must match in every sentence.
- If a criteria table maps a measurement band to a replacement type (standard /
  oversize / dummy / scrap), that mapping is the decision for that measurement.
  Do not headline reuse when the band says otherwise. An extra check that is
  required before reuse applies only when the band or procedure actually selects
  reuse — it does not override oversize/dummy/scrap.
- If the evidence equates two names (A = B, also referred to as), they are the
  same item. Use the procedure named for that item. Do not invert the equation.
- If evidence says a measurement is valid only afloat / not in dry dock, and the
  question is in dry dock, say the measurement is not valid here.
- A 50% reduction of original L is reached only when remaining L is at or below
  half the original. Remaining L above that value has not hit the wear limit.
- When comparing two components, quote every keep/replace rule that appears for
  each. A shared clearance-delta sentence does not make the logic identical if
  one component also has a unique wear rule (oil wedge, wear rate, etc.).
- A unit/cylinder number (Unit 5, cylinder 4) is a location on the engine, not a
  component type. Pick the component from the named parts and symptoms in the
  question, not from the unit number.
- An extra check that compares wear of two different components must use both
  components' wear. Do not rewrite it as a second reading of one dimension.
- If the question says a clearance is N above/below a named reference (Adjustment
  Sheet, as-fitted, recorded), compare N only to the procedure's delta threshold.
  Do not treat N as the absolute clearance against min/max data-sheet bands.
- Unmarked checklist lines (no tick/X) are not required. Do not copy a tick from
  a different component's checklist onto the component that was asked about.
  In a checklist table, a mark belongs to the item it sits beside; an X after an
  unmarked line belongs to the next row, not that unmarked line.
- Short if/then lists (time bands, OK / observe / overhaul) are the requirement
  when they match the question. Prefer them over unrelated overhaul-workshop notes
  in the same procedure.
- Prefer the component whose title shares the most words with the question. A
  data sheet that only reuses a table-row code on a different component is not
  the source. Quote the named plate/drawing; a sibling plate is not the diagram.
- A drawing caption that names Checking, Overhaul, Dismantling, or Mounting is
  the action of that panel even if a parent heading differs.
- Callout numbers come from the OCR/callout table in evidence, not from guessing.""",
}


class _Loop:
    """One background event loop for all agent runs.

    ADK is async; the ingest and chat paths are sync. Reusing a single loop keeps the
    LiteLLM HTTP clients alive between calls and avoids tearing down SSL transports
    on every question.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def run(self, coro):
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever,
                    name="adk-agent-loop",
                    daemon=True,
                ).start()
            loop = self._loop
        return asyncio.run_coroutine_threadsafe(coro, loop).result()


_LOOP = _Loop()
_AGENTS: dict[str, Any] = {}
_ADK_AVAILABLE: bool | None = None


def adk_available() -> bool:
    global _ADK_AVAILABLE
    if _ADK_AVAILABLE is None:
        try:
            import google.adk  # noqa: F401

            _ADK_AVAILABLE = True
        except Exception:
            logger.warning("google-adk not installed; agents fall back to direct LiteLLM calls")
            _ADK_AVAILABLE = False
    return _ADK_AVAILABLE


def get_agent(name: str):
    """Build (once) the LlmAgent for one pipeline step."""
    if name in _AGENTS:
        return _AGENTS[name]

    configure_litellm()

    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    settings = get_settings()
    model_id = settings.ocr_vision_model if name == VISION_OCR else settings.llm_model
    api_base = (settings.litellm_api_base or "").rstrip("/")
    agent = LlmAgent(
        name=name,
        model=LiteLlm(
            # The proxy is OpenAI-compatible, so LiteLLM needs the openai/ prefix.
            model=f"openai/{model_id}" if api_base else model_id,
            api_base=api_base or None,
            api_key=settings.litellm_api_key or settings.anthropic_api_key,
        ),
        instruction=INSTRUCTIONS[name],
        description=f"{name} agent for the marine manual pipeline",
    )
    _AGENTS[name] = agent
    return agent


def ask(
    name: str,
    prompt: str,
    *,
    image_png: bytes | None = None,
    max_tokens: int = 1200,
) -> str:
    """Run one agent turn and return its text. Falls back to a direct LiteLLM call."""
    if not llm_ready():
        raise RuntimeError("No LiteLLM credentials configured")

    if adk_available():
        try:
            return _LOOP.run(_run_agent(name, prompt, image_png))
        except Exception:
            logger.exception("ADK agent %s failed; using direct LiteLLM call", name)

    return _direct_call(name, prompt, image_png=image_png, max_tokens=max_tokens)


async def _run_agent(name: str, prompt: str, image_png: bytes | None) -> str:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    agent = get_agent(name)
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="engineer")
    parts = [types.Part(text=prompt)]
    if image_png:
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/png", data=image_png)))

    chunks: list[str] = []
    async for event in runner.run_async(
        user_id="engineer",
        session_id=session.id,
        new_message=types.Content(role="user", parts=parts),
    ):
        if event.content and event.content.parts:
            chunks.extend(p.text for p in event.content.parts if getattr(p, "text", None))
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError(f"agent {name} returned no text")
    return text


def _direct_call(
    name: str,
    prompt: str,
    *,
    image_png: bytes | None,
    max_tokens: int,
) -> str:
    import base64

    content: Any = prompt
    if image_png:
        data_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    settings = get_settings()
    return chat_completion(
        [
            {"role": "system", "content": INSTRUCTIONS[name]},
            {"role": "user", "content": content},
        ],
        model=settings.ocr_vision_model if name == VISION_OCR else None,
        temperature=0.0,
        max_tokens=max_tokens,
    )
