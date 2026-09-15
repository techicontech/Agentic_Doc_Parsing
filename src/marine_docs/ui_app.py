"""Basic Gradio chatbot UI over the marine docs knowledge model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from marine_docs.chat import answer_query
from marine_docs.config import get_settings
from marine_docs.device import resolve_torch_device
from marine_docs.llm import llm_ready
from marine_docs.retrieval import get_fleet_context


def _format_response(resp) -> str:
    lines = [resp.answer, "", "---", "### Citations"]
    if not resp.citations:
        lines.append("_None_")
    for c in resp.citations:
        section = " > ".join(c.get("section") or [])
        lines.append(
            f"- p.{c.get('page')} | `{c.get('doc_code') or '-'}` | {section} "
            f"| source={c.get('source')}"
        )
        if c.get("image_path"):
            lines.append(f"  - figure: `{c.get('figure')}` → `{c.get('image_path')}`")
    lines.extend(
        [
            "",
            "### Verification",
            f"```json\n{json.dumps(resp.verification, indent=2)}\n```",
        ]
    )
    return "\n".join(lines)


def chat_fn(message: str, history: list, equipment: str):
    if not message or not message.strip():
        return history, ""
    resp = answer_query(message.strip(), equipment_context=equipment or "S50MC-C")
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": _format_response(resp)},
    ]
    return history, ""


def build_ui():
    settings = get_settings()
    fleet = get_fleet_context()
    device = resolve_torch_device(settings.docling_device)
    status = (
        f"DB `{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}` · "
        f"Docling device `{device}` · "
        f"LLM `{'ready' if llm_ready() else 'not configured'}` · "
        f"Fleet `{(fleet or {}).get('model', 'none')}`"
    )

    with gr.Blocks(title="Marine Manual Assistant") as demo:
        gr.Markdown(
            f"""# Marine Manual Assistant
**MAN B&W technical docs** — evidence-grounded answers with citations / abstain.

`{status}`

Ingest Vol II first (`python scripts/run_ingest.py`), then ask questions here.
"""
        )
        equipment = gr.Textbox(value="S50MC-C", label="Equipment context", scale=1)
        chatbot = gr.Chatbot(label="Chat", height=480, type="messages")
        msg = gr.Textbox(
            label="Question",
            placeholder="e.g. What is the fuel valve tightening torque (D09-41)? Show the mounting diagram.",
        )
        with gr.Row():
            send = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear")

        gr.Examples(
            examples=[
                ["What is fuel valve tightening torque D09-41?"],
                ["How do I dismantle the cylinder cover (M90101)?"],
                ["Show me the Cylinder Cover Panel diagram P90151"],
                ["What is the torque for exhaust valve stud D01-01?"],
                ["What is the fuel valve torque for engine S70MC-C?"],
                ["What does Volume I say about scavenge air receiver operation?"],
            ],
            inputs=msg,
        )

        send.click(chat_fn, inputs=[msg, chatbot, equipment], outputs=[chatbot, msg])
        msg.submit(chat_fn, inputs=[msg, chatbot, equipment], outputs=[chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
