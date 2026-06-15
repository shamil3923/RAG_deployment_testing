import { useEffect, useRef, useState } from "react";
import * as api from "./api";

export default function App() {
  const [messages, setMessages] = useState([]); // {role, content, sources?}
  const [input, setInput] = useState("");
  const [docText, setDocText] = useState("");
  const [chunks, setChunks] = useState(0);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const endRef = useRef(null);

  const refresh = () => api.health().then((h) => setChunks(h.chunks)).catch(() => {});
  useEffect(() => { refresh(); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function send() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await api.chat(q);
      setMessages((m) => [...m, { role: "assistant", content: res.answer, sources: res.sources }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  async function ingest() {
    if (!docText.trim()) return;
    setStatus("Ingesting…");
    try {
      const res = await api.ingestText(docText);
      setChunks(res.chunks);
      setDocText("");
      setStatus(`Added ${res.ingested_chunks} chunks.`);
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
  }

  async function onFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus(`Uploading ${file.name}…`);
    try {
      const res = await api.ingestFile(file);
      setChunks(res.chunks);
      setStatus(`Added ${res.ingested_chunks} chunks from ${file.name}.`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      e.target.value = "";
    }
  }

  async function clearAll() {
    await api.reset();
    setChunks(0);
    setMessages([]);
    setStatus("Knowledge base cleared.");
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>📚 Knowledge base</h2>
        <p className="muted">{chunks} chunks indexed</p>

        <textarea
          placeholder="Paste text to add to the knowledge base…"
          value={docText}
          onChange={(e) => setDocText(e.target.value)}
          rows={8}
        />
        <button onClick={ingest} disabled={!docText.trim()}>Add text</button>

        <label className="filebtn">
          Upload pdf / docx / image / txt
          <input
            type="file"
            accept=".txt,.md,.pdf,.docx,.png,.jpg,.jpeg,.webp,.gif,.bmp"
            onChange={onFile}
            hidden
          />
        </label>

        <button className="ghost" onClick={clearAll}>Clear all</button>
        {status && <p className="status">{status}</p>}
      </aside>

      <main className="chat">
        <header>
          <h1>Q&A RAG Assistant</h1>
          <span className="muted">Qwen · NVIDIA API · Chroma</span>
        </header>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              Add some documents on the left, then ask a question.
              <br />Try: <em>“What are the technical skills candidates have?”</em>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">{m.content}</div>
              {m.sources?.length > 0 && (
                <details className="sources">
                  <summary>{m.sources.length} sources</summary>
                  {m.sources.map((s) => (
                    <div key={s.n} className="source">
                      <strong>[{s.n}] {s.source}</strong> · score {s.score}
                      <p>{s.preview}…</p>
                    </div>
                  ))}
                </details>
              )}
            </div>
          ))}
          {loading && <div className="msg assistant"><div className="bubble typing">Thinking…</div></div>}
          <div ref={endRef} />
        </div>

        <div className="composer">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Ask a question about your documents…"
          />
          <button onClick={send} disabled={loading || !input.trim()}>Send</button>
        </div>
      </main>
    </div>
  );
}
