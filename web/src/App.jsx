import { useEffect, useRef, useState } from "react";

async function api(path, options) {
  const res = await fetch(path, options);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail = data?.detail;
    throw new Error(
      typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : text || `HTTP ${res.status}`
    );
  }
  return data;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [job, setJob] = useState(null);
  const [hasData, setHasData] = useState(false);
  const bottomRef = useRef(null);
  const fileRef = useRef(null);
  const pollRef = useRef(null);

  const ingesting = job?.status === "running" || uploading;
  const canChat = hasData && !ingesting;

  async function refreshStatus() {
    const j = await api("/api/ingest/status");
    setJob(j);
    setHasData(Boolean(j.has_data));
    return j;
  }

  useEffect(() => {
    refreshStatus().catch(() => {});
  }, []);

  useEffect(() => {
    if (job?.status !== "running") {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(() => {
      refreshStatus()
        .then((j) => {
          if (j.status === "completed" || j.status === "failed") {
            clearInterval(pollRef.current);
            setUploading(false);
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(pollRef.current);
  }, [job?.status]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy, job?.message]);

  async function onFileSelected(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please choose a PDF file.");
      return;
    }
    setError("");
    setUploading(true);
    setMessages([]);
    try {
      const body = new FormData();
      body.append("file", file);
      await api("/api/ingest/upload", { method: "POST", body });
      await refreshStatus();
    } catch (err) {
      setError(err.message || String(err));
      setUploading(false);
    }
  }

  async function send() {
    const message = input.trim();
    if (!message || busy || !canChat) return;
    setError("");
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setBusy(true);
    try {
      const data = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          abstained: data.abstained,
          citations: data.citations || [],
          diagrams: data.diagrams || [],
          retrieval_notes: data.retrieval_notes || {},
          verification: data.verification || {},
        },
      ]);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Marine Manual Chat</h1>
          <p className="sub">Upload a PDF, wait for ingest, then ask anything.</p>
        </div>
        <div className="header-actions">
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={onFileSelected}
          />
          <button
            type="button"
            className="primary"
            disabled={ingesting}
            onClick={() => fileRef.current?.click()}
          >
            {ingesting ? "Ingesting…" : "Upload PDF"}
          </button>
        </div>
      </header>

      {(ingesting || job?.pdf_name || job?.status === "failed" || job?.status === "completed") && (
        <div className={`banner ${job?.status === "failed" ? "fail" : ""}`}>
          {job?.pdf_name ? <strong>{job.pdf_name}</strong> : null}
          <span>
            {job?.status === "failed"
              ? job?.error || "Ingest failed"
              : job?.message || "Working…"}
          </span>
          {ingesting && typeof job?.percent === "number" && (
            <span className="pct">{job.percent}%</span>
          )}
          {ingesting && (
            <div className="bar">
              <div className="bar-fill" style={{ width: `${Math.min(100, job?.percent || 0)}%` }} />
            </div>
          )}
        </div>
      )}

      <main className="chat">
        {messages.length === 0 && !ingesting && (
          <div className="empty">
            {canChat
              ? "Ask about specs, procedures, diagrams, or troubleshooting."
              : "Click Upload PDF to start."}
          </div>
        )}

        {ingesting && (
          <div className="bubble assistant pending">
            Ingesting{job?.pdf_name ? ` “${job.pdf_name}”` : ""}…
            {typeof job?.percent === "number" ? ` ${job.percent}%` : ""}
            {job?.done != null && job?.total != null ? ` · ${job.done}/${job.total}` : ""}
          </div>
        )}

        {messages.map((m, idx) => (
          <div key={idx} className={`bubble ${m.role}${m.abstained ? " abstain" : ""}`}>
            <div className="role">{m.role === "user" ? "You" : "Assistant"}</div>
            <div className="content">{m.content}</div>

            {m.retrieval_notes?.paths_fired?.length > 0 && (
              <div className="paths">
                Paths: {(m.retrieval_notes.paths_fired || []).join(" · ")}
                {m.retrieval_notes.fusion_method ? ` · ${m.retrieval_notes.fusion_method}` : ""}
                {m.retrieval_notes.exact_identifier_hits
                  ? ` · ${m.retrieval_notes.exact_identifier_hits} exact-code hit(s)`
                  : ""}
                {m.retrieval_notes.panels_pulled
                  ? ` · ${m.retrieval_notes.panels_pulled} step panel(s)`
                  : ""}
                {m.retrieval_notes.cross_refs_resolved
                  ? ` · ${m.retrieval_notes.cross_refs_resolved} cross-ref hop(s)`
                  : ""}
              </div>
            )}

            {m.diagrams?.length > 0 && (
              <div className="diagrams">
                <div className="section-label">Diagrams</div>
                <div className="diagram-grid">
                  {m.diagrams.map((d) => (
                    <figure key={d.image_path || d.url}>
                      <a href={d.url} target="_blank" rel="noreferrer">
                        <img src={d.url} alt={d.label || `page ${d.page}`} loading="lazy" />
                      </a>
                      <figcaption>
                        {d.label || "panel"}
                        {d.step ? ` · step ${d.step}` : ""}
                        {d.ref ? ` · ${d.ref}` : ""}
                      </figcaption>
                    </figure>
                  ))}
                </div>
              </div>
            )}

            {m.citations?.length > 0 && (
              <div className="citations">
                <div className="section-label">Citations</div>
                <ul>
                  {m.citations.slice(0, 5).map((c, i) => (
                    <li key={i}>
                      <span className="cite-ref">{c.ref || `page ${c.page}`}</span>
                      {c.component ? ` · ${c.component}` : ""}
                      {c.action ? ` · ${c.action}` : ""}
                      <span className="cite-page">
                        {c.page_printed ? ` · printed p.${c.page_printed}` : ""}
                        {` · pdf p.${c.page}`}
                        {c.step ? ` · step ${c.step}` : ""}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}

        {busy && <div className="bubble assistant pending">Thinking…</div>}
        <div ref={bottomRef} />
      </main>

      {error && <div className="error">{error}</div>}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            canChat
              ? "Ask a question…"
              : ingesting
                ? "Wait for ingest to finish…"
                : "Upload a PDF first…"
          }
          rows={2}
          disabled={!canChat || busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button type="submit" disabled={!canChat || busy || !input.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
