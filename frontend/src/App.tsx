import { useEffect, useMemo, useState } from "react";
import { analyzeConversation, transcribeImage } from "./api";
import { samples } from "./samples";
import type { AnalyzeResponse, Message, Speaker } from "./types";

type Page =
  | "overview"
  | "analyze"
  | "processing"
  | "context"
  | "results"
  | "classification"
  | "research"
  | "dictionary";

const NAV_ITEMS: Array<{ id: Page; label: string; icon: string }> = [
  { id: "overview", label: "Overview", icon: "◈" },
  { id: "analyze", label: "Analyze Conversation", icon: "▤" },
  { id: "processing", label: "Text Processing", icon: "⌘" },
  { id: "context", label: "Contextual Analysis", icon: "◎" },
  { id: "results", label: "Detection Results", icon: "▥" },
  { id: "classification", label: "Classification Details", icon: "⬡" },
  { id: "research", label: "Research Evaluation", icon: "▧" },
  { id: "dictionary", label: "Dictionary", icon: "▣" },
];

const CATEGORY_DETAILS: Record<string, string> = {
  "Physical Threat": "Threats of bodily harm, injury, or violence.",
  "Psychological / Emotional Threat": "Intimidation or conduct that may cause fear or distress.",
  "Unwanted Sexual Remarks/Comments": "Unsolicited sexual statements, requests, or comments.",
  "Misogynistic Remarks/Comments": "Expressions that demean or express hatred toward women.",
  "Sexist Remarks/Comments": "Stereotyping or discrimination based on sex or gender.",
  "Homophobic Remarks/Comments": "Remarks targeting sexual orientation or same-sex attraction.",
  "Transphobic Remarks/Comments": "Remarks targeting gender identity or expression.",
  Cyberstalking: "Repeated unwanted digital pursuit, tracking, or monitoring.",
  "Incessant Messaging": "Persistent unwanted messages, particularly after refusal.",
  "Unauthorized Sharing of Personal Information": "Sharing private information without consent.",
};

const DICTIONARY_ENTRIES = [
  { term: "grabe", meaning: "Filipino expression of intensity or disbelief", category: "Expression", normalized: "grabe" },
  { term: "malandi", meaning: "Flirtatious or sexually suggestive", category: "Tagalog", normalized: "flirtatious" },
  { term: "bastos", meaning: "Rude or offensive", category: "Tagalog", normalized: "rude" },
  { term: "lol", meaning: "Laughing out loud", category: "Slang", normalized: "laughing" },
  { term: "idk", meaning: "I do not know", category: "Abbreviation", normalized: "i do not know" },
  { term: "😠", meaning: "Anger; meaning depends on conversational context", category: "Emoji", normalized: "angry" },
  { term: "🥵", meaning: "Overheated or sexually suggestive", category: "Emoji", normalized: "hot" },
  { term: "😭", meaning: "Crying or feeling overwhelmed", category: "Emoji", normalized: "crying" },
];

const PAGE_INFO: Record<Page, { title: string; subtitle: string }> = {
  overview: {
    title: "Overview",
    subtitle: "A contextual lens for understanding multilingual online harassment.",
  },
  analyze: {
    title: "Analyze Conversation",
    subtitle: "Submit a conversation for contextual and rule-guided analysis.",
  },
  processing: {
    title: "Text Processing",
    subtitle: "View how multilingual text is prepared before classification.",
  },
  context: {
    title: "Contextual Analysis",
    subtitle: "Review the contextual model evidence and conversation-level result.",
  },
  results: {
    title: "Detection Results",
    subtitle: "Inspect detected indicators, category scores, and the rule-based outcome.",
  },
  classification: {
    title: "Classification Details",
    subtitle: "Explore category scoring and the evidence behind the prediction.",
  },
  research: {
    title: "Research Evaluation",
    subtitle: "Review model evaluation status and research safeguards.",
  },
  dictionary: {
    title: "Dictionary",
    subtitle: "Browse examples of informal language and emoji normalization.",
  },
};

function pageFromHash(): Page {
  const hash = window.location.hash.slice(1) as Page;
  return NAV_ITEMS.some((item) => item.id === hash) ? hash : "overview";
}

function parseTranscript(text: string): Message[] {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const speakers = new Map<string, Speaker>();
  return lines.map((line, index): Message => {
    const match = line.match(/^([^:]{1,40}):\s*(.*)$/);
    if (!match) {
      return { speaker: index % 2 === 0 ? "User A" : "User B", text: line };
    }
    const name = match[1].trim();
    const key = name.toLowerCase();
    let assigned: Speaker | undefined = speakers.get(key);
    if (!assigned) {
      const isSecond =
        /\b(user\s*b|person\s*b|receiver|recipient|other)\b/i.test(name) ||
        (speakers.size > 0 && !/\b(user\s*a|person\s*a|sender|you)\b/i.test(name));
      assigned = isSecond ? "User B" : "User A";
      speakers.set(key, assigned);
    }
    return { speaker: assigned, text: match[2].trim() };
  }).filter((message) => message.text.length > 0);
}

export default function App() {
  const [page, setPage] = useState<Page>(() => pageFromHash());
  const [messages, setMessages] = useState<Message[]>([]);
  const [speaker, setSpeaker] = useState<Speaker>("User A");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return localStorage.getItem("haraya:theme") === "light" ? "light" : "dark";
  });
  const [dictionarySearch, setDictionarySearch] = useState("");
  const [dictionaryCategory, setDictionaryCategory] = useState("All");

  useEffect(() => {
    const syncPage = () => setPage(pageFromHash());
    window.addEventListener("hashchange", syncPage);
    return () => window.removeEventListener("hashchange", syncPage);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("haraya:theme", theme);
  }, [theme]);

  const detectedIndicators = useMemo(
    () => result?.legal_indicators.filter((indicator) => indicator.present) ?? [],
    [result],
  );
  const dictionaryEntries = useMemo(() => {
    const query = dictionarySearch.trim().toLowerCase();
    return DICTIONARY_ENTRIES.filter((entry) => {
      const matchesCategory = dictionaryCategory === "All" || entry.category === dictionaryCategory;
      const matchesQuery =
        !query ||
        `${entry.term} ${entry.meaning} ${entry.normalized}`.toLowerCase().includes(query);
      return matchesCategory && matchesQuery;
    });
  }, [dictionaryCategory, dictionarySearch]);

  function navigate(nextPage: Page) {
    if (page === nextPage) return;
    window.location.hash = nextPage;
    setPage(nextPage);
  }

  function startNewAnalysis() {
    setMessages([]);
    setResult(null);
    setError(null);
    navigate("analyze");
  }

  async function runAnalysis(input = messages, source = "Manual Input") {
    if (!input.length) {
      setError("Add at least one message before running the analysis.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const response = await analyzeConversation(input, source);
      setMessages(input);
      setResult(response);
      navigate("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  function addMessage() {
    const text = draft.trim();
    if (!text) return;
    setMessages((current) => [...current, { speaker, text }]);
    setDraft("");
    setError(null);
  }

  async function onImage(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const dataUrl = await fileToDataUrl(file);
      const ocr = await transcribeImage(dataUrl);
      if (ocr.error) {
        setError(ocr.error);
        return;
      }
      const transcribed: Message[] = ocr.bubbles.map((bubble) => ({
        speaker: bubble.speaker,
        text: bubble.text,
      }));
      if (!transcribed.length) {
        setError("No readable chat bubbles were found in the image.");
        return;
      }
      setMessages(transcribed);
      const response = await analyzeConversation(transcribed, "Image-to-Text");
      setResult(response);
      navigate("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image transcription failed.");
    } finally {
      setBusy(false);
    }
  }

  async function onTranscript(file: File | undefined) {
    if (!file) return;
    try {
      const transcript = parseTranscript(await file.text());
      if (!transcript.length) {
        setError("The transcript file did not contain any readable messages.");
        return;
      }
      setMessages(transcript);
      setResult(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transcript could not be read.");
    }
  }

  const info = PAGE_INFO[page];

  return (
    <div className={`haraya-shell theme-${theme}`}>
      <aside className="sidebar">
        <a className="brand-lockup" href="#overview" onClick={() => navigate("overview")}>
          <span className="brand-mark" aria-hidden="true">✿</span>
          <span>
            <strong>HARAYA</strong>
            <small>CONTEXTUAL CLASSIFICATION</small>
          </span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => (
            <button
              type="button"
              key={item.id}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => navigate(item.id)}
              aria-current={page === item.id ? "page" : undefined}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              <span>{item.label}</span>
              {page === item.id ? <span className="nav-current" /> : null}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="prototype-card">
            <span className="prototype-pip" />
            <div><strong>Research prototype</strong><small>Not a legal finding</small></div>
          </div>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}
          >
            <span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>
            Switch to {theme === "dark" ? "light" : "dark"} mode
          </button>
        </div>
      </aside>

      <div className="main-column">
        <header className="page-header">
          <div className="header-art" aria-hidden="true"><span /><span /></div>
          <div className="header-inner">
            <div>
              <div className="eyebrow">HARAYA 2.0 <span>/</span> {info.title.toUpperCase()}</div>
              <h1>{info.title}</h1>
              <p>{info.subtitle}</p>
            </div>
            <button type="button" className="header-action" onClick={startNewAnalysis}>
              <span aria-hidden="true">＋</span> New analysis
            </button>
          </div>
        </header>

        <main className="page-content">
          {page === "overview" && (
            <OverviewPage onNavigate={navigate} onNewAnalysis={startNewAnalysis} />
          )}
          {page === "analyze" && (
            <AnalyzePage
              messages={messages}
              setMessages={setMessages}
              speaker={speaker}
              setSpeaker={setSpeaker}
              draft={draft}
              setDraft={setDraft}
              busy={busy}
              error={error}
              setError={setError}
              result={result}
              onAddMessage={addMessage}
              onAnalyze={() => void runAnalysis()}
              onImage={(file) => void onImage(file)}
              onTranscript={(file) => void onTranscript(file)}
            />
          )}
          {page === "processing" && <ProcessingPage result={result} onAnalyze={() => navigate("analyze")} />}
          {page === "context" && <ContextPage result={result} onAnalyze={() => navigate("analyze")} />}
          {page === "results" && (
            <ResultsPage
              result={result}
              messages={messages}
              detectedIndicators={detectedIndicators}
              onAnalyze={() => navigate("analyze")}
              onDetails={() => navigate("classification")}
            />
          )}
          {page === "classification" && <ClassificationPage result={result} onAnalyze={() => navigate("analyze")} />}
          {page === "research" && <ResearchPage />}
          {page === "dictionary" && (
            <DictionaryPage
              entries={dictionaryEntries}
              search={dictionarySearch}
              setSearch={setDictionarySearch}
              category={dictionaryCategory}
              setCategory={setDictionaryCategory}
            />
          )}
        </main>
        <footer className="page-footer">
          <span>HARAYA 2.0 · Research prototype</span>
          <span>Safe Spaces Act mapping · Outputs are not legal advice</span>
        </footer>
      </div>
    </div>
  );
}

function OverviewPage({
  onNavigate,
  onNewAnalysis,
}: {
  onNavigate: (page: Page) => void;
  onNewAnalysis: () => void;
}) {
  const categories = Object.entries(CATEGORY_DETAILS);
  return (
    <div className="page-stack">
      <section className="hero-panel">
        <div className="hero-art" aria-hidden="true">
          <span className="hero-orbit orbit-one" /><span className="hero-orbit orbit-two" />
          <span className="hero-core">✿</span><span className="hero-node node-one" />
          <span className="hero-node node-two" /><span className="hero-node node-three" />
        </div>
        <div className="hero-copy">
          <span className="pill"><span className="live-dot" /> CONTEXTUAL TEXT CLASSIFICATION FRAMEWORK</span>
          <h2>Recognizing harm.<br />Building safer spaces.</h2>
          <p>
            HARAYA examines conversations, language, and message patterns to surface signals
            associated with gender-based online harassment. Each result includes the indicators
            and rule scores that informed it.
          </p>
          <div className="button-row">
            <button type="button" className="button-primary" onClick={onNewAnalysis}>Analyze a conversation <span>↗</span></button>
            <button type="button" className="button-secondary" onClick={() => onNavigate("results")}>View latest result</button>
          </div>
        </div>
        <div className="hero-footnote"><span>✳</span> Human-centered research · Explainable outputs · Multilingual context</div>
      </section>

      <section className="feature-grid">
        {[
          ["01", "Conversation first", "Reads a sequence of messages instead of treating one phrase in isolation.", "▤"],
          ["02", "Multilingual by design", "Prepares English, Filipino, Taglish, informal language, and emoji.", "◎"],
          ["03", "Grounded in rules", "Maps detected signals to draft operational categories under RA 11313.", "⬡"],
        ].map(([number, title, copy, icon]) => (
          <article className="feature-card" key={title}>
            <div className="feature-top"><span>{number}</span><b aria-hidden="true">{icon}</b></div>
            <h3>{title}</h3><p>{copy}</p>
          </article>
        ))}
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><span className="section-kicker">HARAYA TAXONOMY</span><h2>Possible harassment categories</h2></div>
          <span className="count-badge">10 signal classes</span>
        </div>
        <div className="category-grid">
          {categories.map(([title, description], index) => (
            <article className="category-card" key={title}>
              <span className="category-number">{String(index + 1).padStart(2, "0")}</span>
              <span className="category-arrow">↗</span>
              <h3>{title}</h3><p>{description}</p>
            </article>
          ))}
        </div>
      </section>
      <div className="notice">
        <span className="notice-icon">i</span>
        <p><strong>Research prototype:</strong> Scores are based on draft operational rules and are not legal advice or a legal finding.</p>
      </div>
    </div>
  );
}

function AnalyzePage({
  messages,
  setMessages,
  speaker,
  setSpeaker,
  draft,
  setDraft,
  busy,
  error,
  setError,
  result,
  onAddMessage,
  onAnalyze,
  onImage,
  onTranscript,
}: {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  speaker: Speaker;
  setSpeaker: (speaker: Speaker) => void;
  draft: string;
  setDraft: (draft: string) => void;
  busy: boolean;
  error: string | null;
  setError: (error: string | null) => void;
  result: AnalyzeResponse | null;
  onAddMessage: () => void;
  onAnalyze: () => void;
  onImage: (file: File | undefined) => void;
  onTranscript: (file: File | undefined) => void;
}) {
  return (
    <div className="analysis-layout">
      <section className="panel input-panel">
        <PanelHeading marker="01" title="Conversation input" detail={`${messages.length} messages`} />
        <div className="conversation-board">
          {messages.length ? messages.map((message, index) => (
            <div className={`chat-row ${message.speaker === "User B" ? "from-b" : ""}`} key={`${index}-${message.speaker}`}>
              <span className="chat-avatar">{message.speaker === "User B" ? "❀" : "✿"}</span>
              <div className="chat-message">
                <small>{message.speaker}<span>MESSAGE {String(index + 1).padStart(2, "0")}</span></small>
                <p>{message.text}</p>
              </div>
            </div>
          )) : (
            <div className="empty-state">
              <span className="empty-mark">▤</span>
              <strong>Your conversation will appear here</strong>
              <p>Add messages below, upload a transcript, or use a sample to get started.</p>
            </div>
          )}
        </div>

        <div className="message-composer">
          <label className="field-label" htmlFor="speaker">SPEAKER</label>
          <select id="speaker" value={speaker} onChange={(event) => setSpeaker(event.target.value as Speaker)}>
            <option>User A</option><option>User B</option>
          </select>
          <label className="field-label" htmlFor="message-draft">MESSAGE TEXT</label>
          <textarea
            id="message-draft"
            value={draft}
            placeholder="Type in English, Filipino, or Taglish…"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                onAddMessage();
              }
            }}
          />
          <div className="composer-hint"><kbd>Enter</kbd> to add message <span>·</span> <kbd>Shift + Enter</kbd> for a new line</div>
        </div>

        <div className="input-tools">
          <label className="upload-button">
            <span>▧</span> Screenshot
            <input type="file" accept="image/*" hidden onChange={(event) => {
              onImage(event.target.files?.[0]);
              event.currentTarget.value = "";
            }} />
          </label>
          <label className="upload-button">
            <span>↥</span> Transcript
            <input type="file" accept=".txt,text/plain" hidden onChange={(event) => {
              onTranscript(event.target.files?.[0]);
              event.currentTarget.value = "";
            }} />
          </label>
          <select
            aria-label="Load a sample conversation"
            className="sample-select"
            defaultValue=""
            onChange={(event) => {
              const sample = samples.find((item) => item.name === event.target.value);
              if (sample) {
                setMessages(sample.messages);
                setError(null);
              }
              event.currentTarget.value = "";
            }}
          >
            <option value="" disabled>Load sample…</option>
            {samples.map((sample) => <option key={sample.name} value={sample.name}>{sample.name}</option>)}
          </select>
          <button type="button" className="text-button" onClick={() => {
            setMessages([]);
            setError(null);
          }} disabled={!messages.length}>Clear all</button>
        </div>
        {error ? <p className="inline-error" role="alert">{error}</p> : null}
        <div className="analyze-action">
          <span>Messages are analyzed as a conversation, not individually.</span>
          <button type="button" className="button-primary" onClick={onAnalyze} disabled={busy || !messages.length}>
            {busy ? <><span className="spinner" /> Analyzing…</> : <>Run analysis <span>↗</span></>}
          </button>
        </div>
      </section>

      <section className="panel workflow-panel">
        <PanelHeading marker="02" title="Analysis workflow" detail="HARAYA 2.0" />
        <div className="workflow-list">
          {[
            ["01", "Text preprocessing", "Clean, normalize, and identify language"],
            ["02", "mBERT context analysis", "Contextual language evidence"],
            ["03", "Legal indicator detection", "Check conversation-level signals"],
            ["04", "Rule-based scoring", "Compare evidence to draft thresholds"],
            ["05", "Classification + explanation", "Return prediction with supporting evidence"],
          ].map(([number, title, description], index) => (
            <div className="workflow-step" key={number}>
              <span className={`step-marker ${result ? "complete" : ""}`}>{result ? "✓" : number}</span>
              <div><strong>{title}</strong><small>{description}</small></div>
              {index < 4 ? <span className="step-connector" /> : null}
            </div>
          ))}
        </div>
        <div className="workflow-note">
          <span>✳</span>
          <p>The rule-based legal scoring engine makes the final decision. mBERT is contextual evidence only.</p>
        </div>
        {result ? (
          <div className="last-result">
            <span className="section-kicker">LATEST PREDICTION</span>
            <strong>{result.prediction}</strong>
            <span>{Math.round(result.confidence * 100)}% confidence · {result.language}</span>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function ProcessingPage({ result, onAnalyze }: { result: AnalyzeResponse | null; onAnalyze: () => void }) {
  if (!result) return <NoResultPage onAnalyze={onAnalyze} title="No processed conversation yet" />;
  return (
    <div className="page-stack">
      <section className="panel">
        <PanelHeading marker="01" title="Original conversation" detail={`${result.processed_messages.length} messages`} />
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>#</th><th>Speaker</th><th>Original</th><th>Cleaned</th><th>Normalized</th></tr></thead>
            <tbody>{result.processed_messages.map((message, index) => (
              <tr key={`${message.speaker}-${index}`}>
                <td className="mono">{String(index + 1).padStart(3, "0")}</td>
                <td>{message.speaker}</td><td>{message.original}</td><td>{message.cleaned}</td><td>{message.normalized}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
      <div className="metric-grid three">
        <MetricCard label="LANGUAGE" value={result.language} detail="Detected across conversation" />
        <MetricCard label="MESSAGES" value={String(result.processed_messages.length)} detail="Conversation units processed" />
        <MetricCard label="TOKENS" value={String(result.processed_messages.reduce((sum, message) => sum + message.tokens.length, 0))} detail="Prepared for contextual analysis" />
      </div>
      <section className="panel">
        <PanelHeading marker="02" title="Tokenization and normalization" detail="Per-message view" />
        <div className="processed-list">
          {result.processed_messages.map((message, index) => (
            <article className="processed-item" key={`processed-${index}`}>
              <div className="processed-title"><strong>Message {String(index + 1).padStart(2, "0")}</strong><span>{message.language} · {message.speaker}</span></div>
              <p>{message.normalized || "—"}</p>
              <div className="token-row">{message.tokens.map((token, tokenIndex) => <code key={`${token}-${tokenIndex}`}>{token}</code>)}</div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function ContextPage({ result, onAnalyze }: { result: AnalyzeResponse | null; onAnalyze: () => void }) {
  if (!result) return <NoResultPage onAnalyze={onAnalyze} title="No contextual analysis available" />;
  const distribution = result.mbert.distribution;
  return (
    <div className="page-stack">
      <section className="context-hero panel">
        <div>
          <span className="section-kicker">CONVERSATION-LEVEL CLASSIFICATION</span>
          <h2>{result.prediction}</h2>
          <p>{result.legal_explanation}</p>
        </div>
        <div className="confidence-ring" style={{ "--confidence": `${Math.round(result.confidence * 100)}%` } as React.CSSProperties}>
          <strong>{Math.round(result.confidence * 100)}<small>%</small></strong><span>CONFIDENCE</span>
        </div>
      </section>
      <div className="metric-grid three">
        <MetricCard label="LANGUAGE" value={result.language} detail="Detected language" />
        <MetricCard label="CONTEXT MODEL" value={result.mbert.encoder_loaded ? "mBERT active" : "Unavailable"} detail={result.mbert.model_id} />
        <MetricCard label="CLASSIFIER" value={result.mbert.classifier_loaded ? "Loaded" : "Not configured"} detail="Contextual evidence only" />
      </div>
      <section className="panel">
        <PanelHeading marker="01" title="Contextual model distribution" detail="Optional classifier evidence" />
        {distribution.length ? (
          <div className="bar-list">{distribution.map((item) => (
            <div className="bar-item" key={item.label}>
              <div><span>{item.label}</span><strong>{Math.round(item.score * 100)}%</strong></div>
              <div className="bar-track"><span style={{ width: `${Math.max(0, Math.min(100, item.score * 100))}%` }} /></div>
            </div>
          ))}</div>
        ) : (
          <div className="empty-inline">
            <strong>No fine-tuned classifier distribution is configured.</strong>
            <p>The encoder and rule engine can still provide the current result. Configure a compatible classifier to show its label distribution here.</p>
          </div>
        )}
      </section>
      <section className="panel">
        <PanelHeading marker="02" title="Conversation document" detail="Normalized text supplied for analysis" />
        <pre className="conversation-document">{result.conversation_text}</pre>
      </section>
    </div>
  );
}

function ResultsPage({
  result,
  messages,
  detectedIndicators,
  onAnalyze,
  onDetails,
}: {
  result: AnalyzeResponse | null;
  messages: Message[];
  detectedIndicators: AnalyzeResponse["legal_indicators"];
  onAnalyze: () => void;
  onDetails: () => void;
}) {
  if (!result) return <NoResultPage onAnalyze={onAnalyze} title="No conversation analyzed yet" />;
  return (
    <div className="page-stack">
      <section className="result-banner">
        <div className="result-icon">{result.detected ? "!" : "✓"}</div>
        <div className="result-copy">
          <span className="section-kicker">HARAYA PREDICTION</span>
          <h2>{result.prediction}</h2>
          <p>{result.legal_explanation}</p>
        </div>
        <div className="result-confidence"><strong>{Math.round(result.confidence * 100)}<small>%</small></strong><span>CONFIDENCE</span></div>
      </section>
      <div className="metric-grid three">
        <MetricCard label="RULE SCORE" value={`${result.rule_score} / ${result.rule_max_score || "—"}`} detail="Winning category score" />
        <MetricCard label="MESSAGES" value={String(messages.length)} detail={`Source: ${result.source}`} />
        <MetricCard label="INDICATORS FOUND" value={String(detectedIndicators.length)} detail="Conversation-level evidence" />
      </div>
      <section className="panel">
        <div className="panel-heading-row">
          <PanelHeading marker="01" title="Legal indicators" detail={`${detectedIndicators.length} detected`} />
          <button type="button" className="text-button" onClick={onDetails}>View scoring details ↗</button>
        </div>
        <div className="indicator-grid">
          {result.legal_indicators.map((item) => (
            <article className={`indicator-card ${item.present ? "is-present" : ""}`} key={item.id}>
              <span className="indicator-check">{item.present ? "✓" : "○"}</span>
              <div><strong>{item.label}</strong><p>{item.present ? item.evidence[0] || "Indicator detected in conversation context." : "No supporting evidence detected."}</p></div>
              <span className={`status-chip ${item.present ? "found" : ""}`}>{item.present ? "FOUND" : "NOT FOUND"}</span>
            </article>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelHeading marker="02" title="Category score summary" detail="Rule engine output" />
        <div className="category-score-list">
          {result.category_scores.map((category) => (
            <div className="category-score-row" key={category.label}>
              <div><strong>{category.label}</strong><span className={`decision-chip ${category.decision.toLowerCase().replaceAll(" ", "-")}`}>{category.decision}</span></div>
              <div className="category-score-value">{category.score}<small> / {category.max_score}</small></div>
              <div className="bar-track"><span style={{ width: `${category.max_score ? Math.min(100, category.score / category.max_score * 100) : 0}%` }} /></div>
            </div>
          ))}
        </div>
      </section>
      <section className="panel conversation-result">
        <PanelHeading marker="03" title="Analyzed conversation" detail={`${messages.length} messages`} />
        {messages.map((message, index) => (
          <div className={`chat-row ${message.speaker === "User B" ? "from-b" : ""}`} key={`${index}-${message.speaker}`}>
            <span className="chat-avatar">{message.speaker === "User B" ? "❀" : "✿"}</span>
            <div className="chat-message"><small>{message.speaker}<span>MESSAGE {String(index + 1).padStart(2, "0")}</span></small><p>{message.text}</p></div>
          </div>
        ))}
      </section>
      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  );
}

function ClassificationPage({ result, onAnalyze }: { result: AnalyzeResponse | null; onAnalyze: () => void }) {
  if (!result) return <NoResultPage onAnalyze={onAnalyze} title="No classification details yet" />;
  return (
    <div className="page-stack">
      <section className="panel">
        <PanelHeading marker="01" title="How the result was determined" detail="Rule-based legal scoring" />
        <div className="explain-summary">
          <span className="result-icon">{result.detected ? "!" : "✓"}</span>
          <div><span className="section-kicker">PREDICTED CATEGORY</span><h2>{result.prediction}</h2><p>{result.legal_explanation}</p></div>
        </div>
        <div className="rule-callout"><strong>Decision source</strong><p>mBERT provides contextual evidence only. The rule-based engine combines detected indicators and compares their weighted scores with the configured draft thresholds.</p></div>
      </section>
      <section className="panel">
        <PanelHeading marker="02" title="Category scoring breakdown" detail="Score compared with threshold" />
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Category</th><th>Score</th><th>Decision</th><th>Matched indicators</th><th>Legal reference</th></tr></thead>
            <tbody>{result.category_scores.map((category) => (
              <tr key={category.label}>
                <td><strong>{category.label}</strong></td><td className="mono">{category.score} / {category.max_score}</td>
                <td><span className={`decision-chip ${category.decision.toLowerCase().replaceAll(" ", "-")}`}>{category.decision}</span></td>
                <td>{category.matched_indicators.join(", ") || "—"}</td><td>{category.legal_reference}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
      <section className="panel">
        <PanelHeading marker="03" title="Indicator evidence" detail="Detected from this conversation" />
        <div className="indicator-grid">
          {result.legal_indicators.filter((item) => item.present).map((item) => (
            <article className="indicator-card is-present" key={item.id}>
              <span className="indicator-check">✓</span><div><strong>{item.label}</strong>
                {item.evidence.map((evidence, index) => <p key={`${item.id}-${index}`}>{evidence}</p>)}
              </div><span className="status-chip found">FOUND</span>
            </article>
          ))}
          {!result.legal_indicators.some((item) => item.present) ? <div className="empty-inline"><strong>No positive indicators detected.</strong><p>Review all category scores above for the explanation.</p></div> : null}
        </div>
      </section>
    </div>
  );
}

function ResearchPage() {
  return (
    <div className="page-stack">
      <section className="research-hero panel">
        <span className="research-icon">▧</span>
        <div><span className="section-kicker">EVALUATION WORKSPACE</span><h2>Research metrics are not loaded in the browser</h2>
          <p>The current backend provides conversation analysis and OCR endpoints. Dataset training and held-out evaluation run as a backend command; this page does not report live model metrics.</p>
        </div>
      </section>
      <div className="metric-grid three">
        <MetricCard label="EVALUATION STATUS" value="Not loaded" detail="No evaluation report API is configured" />
        <MetricCard label="CURRENT DATASET" value="110 rows" detail="Synthetic pilot · expert review required" />
        <MetricCard label="REPORT LOCATION" value="evaluation.json" detail="Generated by the backend training command" />
      </div>
      <section className="panel">
        <PanelHeading marker="01" title="Run the dataset evaluation" detail="From the backend folder" />
        <p className="panel-copy">The current CSV is a synthetic pilot dataset marked for expert review. Any results are an implementation check, not evidence of real-world performance.</p>
        <pre className="command-block">.\.venv\Scripts\python.exe -m app.train --dataset .\csv\HARAYA_2_0_pilot_110.csv --output-dir .\models\haraya-classifier</pre>
        <div className="notice"><span className="notice-icon">!</span><p>Use independently collected and expert-reviewed conversations for a meaningful research evaluation. Keep the test split separate from model and rule tuning.</p></div>
      </section>
      <section className="panel">
        <PanelHeading marker="02" title="Metrics to report" detail="From a held-out conversation split" />
        <div className="metric-explain-grid">
          {[
            ["Accuracy", "Share of held-out conversations assigned the correct label."],
            ["Macro F1", "Unweighted average F1 across categories, useful when categories vary in size."],
            ["Per-class precision / recall", "Shows false positives and missed examples for each category."],
          ].map(([title, copy]) => <article className="metric-explain" key={title}><strong>{title}</strong><p>{copy}</p></article>)}
        </div>
      </section>
    </div>
  );
}

function DictionaryPage({
  entries,
  search,
  setSearch,
  category,
  setCategory,
}: {
  entries: typeof DICTIONARY_ENTRIES;
  search: string;
  setSearch: (value: string) => void;
  category: string;
  setCategory: (value: string) => void;
}) {
  const categories = ["All", ...new Set(DICTIONARY_ENTRIES.map((entry) => entry.category))];
  return (
    <div className="page-stack">
      <section className="metric-grid four">
        <MetricCard label="REFERENCE EXAMPLES" value={String(DICTIONARY_ENTRIES.length)} detail="Illustrative entries" />
        {categories.slice(1).map((item) => <MetricCard key={item} label={item.toUpperCase()} value={String(DICTIONARY_ENTRIES.filter((entry) => entry.category === item).length)} detail="Examples shown" />)}
      </section>
      <section className="panel">
        <PanelHeading marker="01" title="Normalization reference" detail="Search the illustrative examples" />
        <p className="panel-copy">These examples illustrate common normalization cues. The backend lexicon is authoritative and may contain additional entries.</p>
        <div className="dictionary-filters">
          <label className="search-field"><span aria-hidden="true">⌕</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search term, meaning, or normalized form" /></label>
          <label className="category-filter">Category
            <select value={category} onChange={(event) => setCategory(event.target.value)}>{categories.map((item) => <option key={item}>{item}</option>)}</select>
          </label>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Term / Symbol</th><th>Meaning</th><th>Category</th><th>Normalized form</th></tr></thead>
            <tbody>{entries.map((entry) => (
              <tr key={`${entry.category}-${entry.term}`}><td><strong>{entry.term}</strong></td><td>{entry.meaning}</td><td><span className="status-chip found">{entry.category}</span></td><td className="mono">{entry.normalized}</td></tr>
            ))}</tbody>
          </table>
          {!entries.length ? <div className="empty-inline"><strong>No matching entries</strong><p>Try a different term or clear the category filter.</p></div> : null}
        </div>
      </section>
    </div>
  );
}

function NoResultPage({ title, onAnalyze }: { title: string; onAnalyze: () => void }) {
  return (
    <section className="panel no-result">
      <span className="empty-mark">◎</span><h2>{title}</h2>
      <p>Submit a conversation first to populate this analysis stage with real backend output.</p>
      <button type="button" className="button-primary" onClick={onAnalyze}>Go to conversation input <span>↗</span></button>
    </section>
  );
}

function PanelHeading({ marker, title, detail }: { marker: string; title: string; detail?: string }) {
  return (
    <div className="panel-heading">
      <span className="panel-marker">{marker}</span><h2>{title}</h2>
      {detail ? <span className="panel-detail">{detail}</span> : null}
    </div>
  );
}

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <article className="metric-card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>;
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("The image could not be read."));
    reader.readAsDataURL(file);
  });
}
