/* FlashServe-Lite — React UI with Continuous Batching, Speculative Decoding & Quantization */
const { useState, useEffect, useRef, useCallback } = React;

// ─── Header with Navigation Tabs ──────────────────────────────────
function Header({ online, modelName, activeTab, onSelectTab }) {
  return (
    <header className="header">
      <div className="header-left">
        <span className="logo">⚡ FlashServe-Lite</span>
        <span className="logo-sub">Mini LLM Inference Gateway</span>
      </div>

      <nav className="header-nav">
        <button
          className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => onSelectTab('chat')}
        >
          <span>⚡</span> Serving & Batching
        </button>
        <button
          className={`nav-tab ${activeTab === 'speculative' ? 'active' : ''}`}
          onClick={() => onSelectTab('speculative')}
        >
          <span>🔮</span> Speculative Decoding Lab
        </button>
        <button
          className={`nav-tab ${activeTab === 'quantization' ? 'active' : ''}`}
          onClick={() => onSelectTab('quantization')}
        >
          <span>⚖️</span> Quantization Profiler
        </button>
      </nav>

      <div className="header-right">
        <div className="status-badge">
          <span className={`status-dot ${online ? 'online' : 'offline'}`} />
          {online ? 'Connected' : 'Disconnected'}
        </div>
        {modelName && <span className="model-badge">{modelName.split('/').pop()}</span>}
      </div>
    </header>
  );
}

// ─── Sidebar Cards (Continuous Batching & Paged KV) ───────────────
function ModelInfo({ stats }) {
  const info = stats || {};
  const modelShort = info.model ? info.model.split('/').pop() : '—';
  return (
    <div className="sidebar-card">
      <div className="card-title">Model Architecture</div>
      <div className="stat-row"><span className="stat-label">Model</span><span className="stat-value" title={info.model || ''}>{modelShort}</span></div>
      <div className="stat-row"><span className="stat-label">Precision</span><span className="stat-value">{info.quantization ? info.quantization.toUpperCase() : (info.dtype || 'FP16')}</span></div>
      <div className="stat-row"><span className="stat-label">Device</span><span className="stat-value">{info.device || 'CUDA'}</span></div>
    </div>
  );
}

function EngineStats({ stats }) {
  const s = stats || {};
  return (
    <div className="sidebar-card">
      <div className="card-title">Continuous Batching</div>
      <div className="stat-row"><span className="stat-label">Running Requests</span><span className="stat-value">{s.running_sequences ?? 0}</span></div>
      <div className="stat-row"><span className="stat-label">Waiting in Queue</span><span className="stat-value">{s.waiting_requests ?? s.waiting_sequences ?? 0}</span></div>
      <div className="stat-row"><span className="stat-label">Total Completed</span><span className="stat-value">{s.total_completed ?? s.completed_sequences ?? 0}</span></div>
    </div>
  );
}

function KVCacheMonitor({ stats }) {
  const kv = stats?.kv_cache || {};
  const util = kv.utilization_pct != null ? kv.utilization_pct : (kv.utilization != null ? kv.utilization * 100 : 0);
  const pctClass = util < 60 ? 'progress-green' : util < 85 ? 'progress-yellow' : 'progress-red';
  const memUsed = kv.used_mb != null ? kv.used_mb : 0;
  const memTotal = kv.pool_size_mb != null ? kv.pool_size_mb : 0;
  return (
    <div className="sidebar-card">
      <div className="card-title">Paged KV Cache</div>
      <div className="stat-row"><span className="stat-label">Utilization</span><span className="stat-value">{util.toFixed(1)}%</span></div>
      <div className="progress-bar-track"><div className={`progress-bar-fill ${pctClass}`} style={{ width: `${Math.min(util, 100)}%` }} /></div>
      <div className="stat-row" style={{ marginTop: 8 }}>
        <span className="stat-label">Allocated Blocks</span>
        <span className="stat-value">{kv.used_blocks ?? 0} / {kv.total_blocks ?? 0}</span>
      </div>
      <div className="stat-row"><span className="stat-label">Pool VRAM</span><span className="stat-value">{memUsed.toFixed(0)} / {memTotal.toFixed(0)} MB</span></div>
    </div>
  );
}

function PerformanceMetrics({ stats, liveTokensPerSec }) {
  return (
    <div className="sidebar-card">
      <div className="card-title">Throughput Telemetry</div>
      <div className="stat-row"><span className="stat-label">Tokens/sec</span><span className="stat-value" style={{ color: liveTokensPerSec > 0 ? '#4ade80' : undefined }}>{liveTokensPerSec > 0 ? liveTokensPerSec.toFixed(1) : '—'}</span></div>
      <div className="stat-row"><span className="stat-label">Active Sequences</span><span className="stat-value">{stats?.running_sequences ?? 0}</span></div>
    </div>
  );
}

function Sidebar({ stats, liveTokensPerSec }) {
  return (
    <aside className="sidebar">
      <ModelInfo stats={stats} />
      <EngineStats stats={stats} />
      <KVCacheMonitor stats={stats} />
      <PerformanceMetrics stats={stats} liveTokensPerSec={liveTokensPerSec} />
    </aside>
  );
}

// ─── Chat View Components ─────────────────────────────────────────
function GenerationCard({ gen }) {
  return (
    <div className="generation-card">
      <div className="gen-prompt-section">
        <div className="gen-prompt-label">Prompt</div>
        <div>{gen.prompt}</div>
      </div>
      {gen.error ? (
        <div className="gen-error">⚠ {gen.error}</div>
      ) : (
        <div className="gen-output-section">
          <span className="gen-output-text">{gen.output}</span>
          {gen.streaming && <span className="typing-cursor" />}
        </div>
      )}
      {!gen.streaming && !gen.error && gen.totalTokens > 0 && (
        <div className="gen-meta">
          <span>{gen.totalTokens} tokens</span>
          {gen.tokensPerSec > 0 && <span>{gen.tokensPerSec.toFixed(1)} tok/s</span>}
          {gen.elapsed > 0 && <span>{gen.elapsed.toFixed(2)}s</span>}
        </div>
      )}
    </div>
  );
}

function OutputDisplay({ generations }) {
  const containerRef = useRef(null);
  const autoScrollRef = useRef(true);
  const prevLenRef = useRef(generations.length);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const isUp = distanceFromBottom > 80;
    autoScrollRef.current = !isUp;
    setShowScrollBtn(isUp && generations.length > 0);
  };

  const scrollToBottom = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({ top: containerRef.current.scrollHeight, behavior: 'smooth' });
      autoScrollRef.current = true;
      setShowScrollBtn(false);
    }
  };

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (generations.length > prevLenRef.current) {
      autoScrollRef.current = true;
      setShowScrollBtn(false);
      el.scrollTop = el.scrollHeight;
    } else if (autoScrollRef.current) {
      el.scrollTop = el.scrollHeight;
    }
    prevLenRef.current = generations.length;
  }, [generations]);

  if (generations.length === 0) {
    return (
      <div className="output-display" ref={containerRef}>
        <div className="empty-state">
          <div className="empty-state-icon">⚡</div>
          <div className="empty-state-text">FlashServe-Lite Engine Ready</div>
          <div className="empty-state-hint">Type a prompt below to test Continuous Batching & Paged KV Cache</div>
        </div>
      </div>
    );
  }

  return (
    <React.Fragment>
      <div className="output-display" ref={containerRef} onScroll={handleScroll}>
        {generations.map((g) => <GenerationCard key={g.id} gen={g} />)}
      </div>
      {showScrollBtn && (
        <button className="scroll-bottom-btn" onClick={scrollToBottom}>
          ↓ Scroll to bottom
        </button>
      )}
    </React.Fragment>
  );
}

function PromptInput({ onSubmit, isGenerating }) {
  const [text, setText] = useState('');
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [stream, setStream] = useState(true);
  const taRef = useRef(null);

  const autoResize = () => {
    const ta = taRef.current;
    if (ta) { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'; }
  };

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed || isGenerating) return;
    onSubmit({ prompt: trimmed, max_tokens: maxTokens, temperature, top_p: topP, top_k: topK, stream });
    setText('');
    if (taRef.current) taRef.current.style.height = 'auto';
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); handleSubmit(); }
  };

  return (
    <div className="prompt-section">
      <div className="settings-row">
        <div className="setting-group">
          <label>Max Tokens</label>
          <input type="range" min="1" max="512" value={maxTokens} onChange={e => setMaxTokens(+e.target.value)} />
          <span className="setting-value">{maxTokens}</span>
        </div>
        <div className="setting-group">
          <label>Temp</label>
          <input type="range" min="0" max="2" step="0.05" value={temperature} onChange={e => setTemperature(+e.target.value)} />
          <span className="setting-value">{temperature.toFixed(2)}</span>
        </div>
        <div className="setting-group">
          <label>top_p</label>
          <input type="number" min="0" max="1" step="0.05" value={topP} onChange={e => setTopP(+e.target.value)} />
        </div>
        <div className="setting-group">
          <label>top_k</label>
          <input type="number" min="1" max="200" value={topK} onChange={e => setTopK(+e.target.value)} />
        </div>
        <label className="stream-toggle"><input type="checkbox" checked={stream} onChange={e => setStream(e.target.checked)} /> Stream</label>
      </div>
      <div className="prompt-row">
        <textarea
          ref={taRef}
          className="prompt-textarea"
          rows={1}
          placeholder="Enter prompt to test continuous batching..."
          value={text}
          onChange={e => { setText(e.target.value); autoResize(); }}
          onKeyDown={onKeyDown}
          disabled={isGenerating}
        />
        <button className="send-btn" onClick={handleSubmit} disabled={isGenerating || !text.trim()}>
          {isGenerating ? <span className="spinner" /> : 'Send'}
        </button>
      </div>
      <div className="shortcut-hint">Ctrl + Enter to send</div>
    </div>
  );
}

// ─── Speculative Decoding Lab View ────────────────────────────────
function SpeculativeLab() {
  const [targetModel, setTargetModel] = useState('Qwen/Qwen2.5-1.5B-Instruct');
  const [draftModel, setDraftModel] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [k, setK] = useState(4);
  const [maxTokens, setMaxTokens] = useState(24);
  const [prompt, setPrompt] = useState('Artificial intelligence and machine learning are');
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState([]);
  const [summary, setSummary] = useState(null);
  const [fullOutput, setFullOutput] = useState('');
  const [error, setError] = useState(null);

  const presets = [
    'Artificial intelligence and machine learning are',
    'The primary advantage of speculative decoding is that',
    'Continuous batching maximizes GPU utilization by',
  ];

  const runSpeculative = async () => {
    if (running || !prompt.trim()) return;
    setRunning(true);
    setSteps([]);
    setSummary(null);
    setFullOutput('');
    setError(null);

    try {
      const resp = await fetch('/api/speculative', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          k: k,
          max_tokens: maxTokens,
          target_model: targetModel,
          draft_model: draftModel,
        }),
      });

      if (!resp.ok) throw new Error(`Server returned ${resp.status}: ${await resp.text()}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const dataStr = line.slice(6).trim();
          if (dataStr === '[DONE]') continue;
          try {
            const data = JSON.parse(dataStr);
            if (data.type === 'step') {
              setSteps(prev => [...prev, data]);
              setFullOutput(prev => prev + data.emitted_text);
            } else if (data.type === 'summary') {
              setSummary(data);
            }
          } catch {}
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="lab-container">
      <div className="lab-header">
        <div className="lab-title-group">
          <h2><span>🔮</span> Speculative Decoding Interactive Lab</h2>
          <div className="lab-desc">
            Test the Draft-and-Verify acceleration cycle in real-time. The lightweight draft model proposes K candidate tokens, and the target model verifies all K in a single forward pass.
          </div>
        </div>
      </div>

      <div className="spec-grid">
        {/* Controls Card */}
        <div className="spec-controls-card">
          <div className="control-field">
            <label className="control-label">Target (Verification) Model</label>
            <select
              className="control-select"
              value={targetModel}
              onChange={e => setTargetModel(e.target.value)}
              disabled={running}
            >
              <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen2.5-1.5B-Instruct (Target)</option>
              <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5-0.5B-Instruct (Fast Target)</option>
            </select>
          </div>

          <div className="control-field">
            <label className="control-label">Draft (Proposer) Model</label>
            <select
              className="control-select"
              value={draftModel}
              onChange={e => setDraftModel(e.target.value)}
              disabled={running}
            >
              <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5-0.5B-Instruct (Draft)</option>
            </select>
          </div>

          <div className="control-field">
            <label className="control-label">
              <span>Lookahead Tokens (K)</span>
              <span>{k}</span>
            </label>
            <input
              type="range"
              min="1"
              max="6"
              value={k}
              onChange={e => setK(+e.target.value)}
              disabled={running}
            />
          </div>

          <div className="control-field">
            <label className="control-label">
              <span>Max Tokens to Generate</span>
              <span>{maxTokens}</span>
            </label>
            <input
              type="range"
              min="8"
              max="64"
              step="4"
              value={maxTokens}
              onChange={e => setMaxTokens(+e.target.value)}
              disabled={running}
            />
          </div>

          <div className="control-field">
            <label className="control-label">Prompt</label>
            <textarea
              className="control-input"
              rows={3}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              disabled={running}
            />
            <div className="preset-pills">
              {presets.map((p, i) => (
                <button
                  key={i}
                  className="preset-pill"
                  onClick={() => setPrompt(p)}
                  disabled={running}
                >
                  {p.slice(0, 28)}…
                </button>
              ))}
            </div>
          </div>

          <button
            className="run-spec-btn"
            onClick={runSpeculative}
            disabled={running || !prompt.trim()}
          >
            {running ? <span className="spinner" /> : '⚡ Run Speculative Decoding'}
          </button>

          {error && <div className="gen-error">⚠ {error}</div>}
        </div>

        {/* Results Area */}
        <div className="spec-results-area">
          {/* Telemetry Banner */}
          <div className="spec-stats-banner">
            <div className="spec-stat-box">
              <span className="spec-stat-sub">Effective Speedup</span>
              <span className="spec-stat-val highlight">
                {summary ? `${summary.speedup}x` : (steps.length > 0 ? `${steps[steps.length - 1].speedup}x` : '—')}
              </span>
              <span className="spec-stat-sub">Tokens / Target Call</span>
            </div>
            <div className="spec-stat-box">
              <span className="spec-stat-sub">Draft Acceptance</span>
              <span className="spec-stat-val">
                {summary ? `${summary.acceptance_rate}%` : '—'}
              </span>
              <span className="spec-stat-sub">Verification Rate</span>
            </div>
            <div className="spec-stat-box">
              <span className="spec-stat-sub">Target GPU Calls</span>
              <span className="spec-stat-val">
                {summary ? summary.target_calls : steps.length}
              </span>
              <span className="spec-stat-sub">Forward Passes</span>
            </div>
            <div className="spec-stat-box">
              <span className="spec-stat-sub">Total Generated</span>
              <span className="spec-stat-val">
                {summary ? summary.total_tokens : (steps.length > 0 ? steps[steps.length - 1].running_tokens : 0)}
              </span>
              <span className="spec-stat-sub">Tokens</span>
            </div>
          </div>

          {/* Full Emitted Text */}
          {fullOutput && (
            <div className="generation-card" style={{ marginBottom: 0 }}>
              <div className="gen-prompt-label">Live Verified Output</div>
              <div className="gen-output-text" style={{ marginTop: 6 }}>
                <strong>{prompt}</strong>
                <span style={{ color: '#a78bfa' }}>{fullOutput}</span>
                {running && <span className="typing-cursor" />}
              </div>
            </div>
          )}

          {/* Step Timeline */}
          <div className="steps-timeline">
            {steps.map((st, i) => (
              <div key={i} className="spec-step-card">
                <div className="spec-step-header">
                  <span className="step-call-badge">
                    Step #{st.step} • Target Model Call #{st.target_calls}
                  </span>
                  <span className={`step-acc-badge ${st.accepted >= 3 ? 'high' : 'med'}`}>
                    ✓ {st.accepted}/{st.drafted} Accepted
                  </span>
                </div>

                <div className="step-tokens-row">
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Draft Verified:</span>
                  {st.tokens.map((tk, tIdx) => (
                    <span key={tIdx} className="token-chip accepted">
                      ✓ {JSON.stringify(tk)}
                    </span>
                  ))}
                </div>

                <div className="step-emitted-text">
                  Emitted chunk: <strong>{JSON.stringify(st.emitted_text)}</strong> • Running Total: {st.running_tokens} tokens ({st.speedup}x speedup)
                </div>
              </div>
            ))}

            {steps.length === 0 && !running && (
              <div className="empty-state" style={{ minHeight: 200 }}>
                <div className="empty-state-icon">🔮</div>
                <div className="empty-state-text">Ready to Test Speculative Decoding</div>
                <div className="empty-state-hint">
                  Click "Run Speculative Decoding" to observe candidate drafting and parallel verification live.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Quantization Profiler View ───────────────────────────────────
function QuantizationProfiler() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [layerDim, setLayerDim] = useState(1536);
  const [error, setError] = useState(null);

  const fetchBenchmark = async (dim) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/api/quantization/benchmark?layer_dim=${dim || layerDim}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBenchmark(1536);
  }, []);

  const bench = data?.live_benchmark;

  return (
    <div className="lab-container">
      <div className="lab-header">
        <div className="lab-title-group">
          <h2><span>⚖️</span> Quantization Profiler & Memory Matrix</h2>
          <div className="lab-desc">
            Compare model memory footprints, weight compression ratios, and reconstruction fidelity across FP16 (half), INT8, and INT4 precision modes.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <select
            className="control-select"
            value={layerDim}
            onChange={e => {
              const d = +e.target.value;
              setLayerDim(d);
              fetchBenchmark(d);
            }}
            disabled={loading}
          >
            <option value="896">896×896 (Qwen 0.5B Layer)</option>
            <option value="1536">1536×1536 (Qwen 1.5B Layer)</option>
            <option value="2048">2048×2048 (Wide Projection)</option>
          </select>
          <button
            className="run-spec-btn"
            style={{ height: 36, padding: '0 16px', fontSize: 13 }}
            onClick={() => fetchBenchmark(layerDim)}
            disabled={loading}
          >
            {loading ? <span className="spinner" /> : '⚡ Re-run Benchmark'}
          </button>
        </div>
      </div>

      {error && <div className="gen-error">⚠ {error}</div>}

      {/* Live Benchmark Cards */}
      {bench && (
        <div className="quant-cards-grid">
          {/* FP16 */}
          <div className="quant-card">
            <div className="quant-card-header">
              <span className="quant-card-title">FP16 Half Precision</span>
              <span className="quant-badge fp16">16-bit Float</span>
            </div>
            <div>
              <div className="quant-size-val">{bench.fp16.size_mb} MB</div>
              <div className="quant-size-sub">{bench.fp16.bytes.toLocaleString()} bytes</div>
            </div>
            <div className="quant-metric-row">
              <span>Memory Footprint</span>
              <span>100% (Baseline)</span>
            </div>
            <div className="quant-metric-row">
              <span>Precision Loss</span>
              <span>0.00000 (Exact)</span>
            </div>
            <div className="quant-metric-row">
              <span>Cosine Similarity</span>
              <span>1.00000</span>
            </div>
            <div className="quant-metric-row">
              <span>Hardware Acceleration</span>
              <span>Native Tensor Cores</span>
            </div>
          </div>

          {/* INT8 */}
          <div className="quant-card" style={{ borderLeft: '3px solid #10b981' }}>
            <div className="quant-card-header">
              <span className="quant-card-title">INT8 Quantization</span>
              <span className="quant-badge int8">8-bit Integer</span>
            </div>
            <div>
              <div className="quant-size-val" style={{ color: '#34d399' }}>{bench.int8.size_mb} MB</div>
              <div className="quant-size-sub">{bench.int8.bytes.toLocaleString()} bytes (50% saved)</div>
            </div>
            <div className="quant-metric-row">
              <span>Memory Reduction</span>
              <span style={{ color: '#34d399' }}>-50.0% VRAM</span>
            </div>
            <div className="quant-metric-row">
              <span>Reconstruction MSE</span>
              <span>{bench.int8.mse}</span>
            </div>
            <div className="quant-metric-row">
              <span>Cosine Similarity</span>
              <span>{bench.int8.cosine_similarity} (&gt;99.99%)</span>
            </div>
            <div className="quant-metric-row">
              <span>Dynamic Scale</span>
              <span>{bench.int8.scale}</span>
            </div>
          </div>

          {/* INT4 */}
          <div className="quant-card" style={{ borderLeft: '3px solid #c084fc' }}>
            <div className="quant-card-header">
              <span className="quant-card-title">INT4 (NF4 / FP4)</span>
              <span className="quant-badge int4">4-bit Packed</span>
            </div>
            <div>
              <div className="quant-size-val" style={{ color: '#c084fc' }}>{bench.int4.size_mb} MB</div>
              <div className="quant-size-sub">{bench.int4.bytes.toLocaleString()} bytes (75% saved)</div>
            </div>
            <div className="quant-metric-row">
              <span>Memory Reduction</span>
              <span style={{ color: '#c084fc' }}>-75.0% VRAM</span>
            </div>
            <div className="quant-metric-row">
              <span>Reconstruction MSE</span>
              <span>{bench.int4.mse}</span>
            </div>
            <div className="quant-metric-row">
              <span>Cosine Similarity</span>
              <span>{bench.int4.cosine_similarity}</span>
            </div>
            <div className="quant-metric-row">
              <span>Dynamic Scale</span>
              <span>{bench.int4.scale}</span>
            </div>
          </div>
        </div>
      )}

      {/* Model Architecture Matrix Table */}
      {data?.matrix && (
        <div className="quant-table-card">
          <div className="card-title">
            Target & Draft Model VRAM Sizing Matrix (RTX 3050 6GB Laptop GPU Budget: {data.total_vram_gb} GB)
          </div>
          <table className="matrix-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Precision Mode</th>
                <th>Weights VRAM</th>
                <th>Total RAM with KV</th>
                <th>VRAM Savings</th>
                <th>Expected Speed</th>
                <th>RTX 3050 Fit</th>
              </tr>
            </thead>
            <tbody>
              {data.matrix.map((row, idx) => (
                <tr key={idx}>
                  <td><strong>{row.model}</strong></td>
                  <td>{row.precision}</td>
                  <td>{row.weights_mb} MB</td>
                  <td>{row.total_ram_mb} MB</td>
                  <td><span style={{ color: row.memory_saved.includes('Baseline') ? 'var(--text-muted)' : '#4ade80' }}>{row.memory_saved}</span></td>
                  <td>~{row.tok_per_sec} tok/s</td>
                  <td><span className="fit-chip ok">✓ {row.fit_status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Docker bitsandbytes Integration Note */}
      <div className="quant-table-card">
        <div className="card-title">Hardware Kernel Acceleration & Containerization</div>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          In FlashServe-Lite, INT8 and INT4 quantization are implemented via PyTorch and HuggingFace <code>bitsandbytes</code> in <code>flashserve/quantization.py</code>.
          On Windows environments with Application Control policies, standard FP16 tensor-core acceleration runs natively on CUDA. For containerized INT4 hardware kernels:
        </p>
        <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: 12, borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 12, color: '#38bdf8' }}>
          docker run --gpus all -p 8000:8000 flashserve-lite --quantization int4
        </div>
      </div>
    </div>
  );
}

// ─── Main App Layout ──────────────────────────────────────────────
function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [online, setOnline] = useState(false);
  const [stats, setStats] = useState(null);
  const [generations, setGenerations] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [liveTokensPerSec, setLiveTokensPerSec] = useState(0);

  // Health check
  useEffect(() => {
    const check = () => fetch('/v1/health').then(r => setOnline(r.ok)).catch(() => setOnline(false));
    check();
    const id = setInterval(check, 4000);
    return () => clearInterval(id);
  }, []);

  // Stats polling
  useEffect(() => {
    const poll = () => fetch('/v1/stats').then(r => r.json()).then(setStats).catch(() => {});
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  const handleSubmit = useCallback(async (params) => {
    const genId = 'gen-' + Date.now() + '-' + Math.floor(Math.random() * 10000);
    const newGen = { id: genId, prompt: params.prompt, output: '', streaming: true, error: null, totalTokens: 0, tokensPerSec: 0, elapsed: 0 };
    setGenerations(prev => [...prev, newGen]);
    setIsGenerating(true);
    setLiveTokensPerSec(0);
    const startTime = performance.now();
    let tokenCount = 0;

    const update = (fn) => setGenerations(prev => prev.map(g => g.id === genId ? fn(g) : g));

    try {
      if (params.stream) {
        const resp = await fetch('/v1/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: params.prompt, max_tokens: params.max_tokens, temperature: params.temperature, top_p: params.top_p, top_k: params.top_k, stream: true }),
        });
        if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6).trim();
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              const token = parsed.choices?.[0]?.text ?? parsed.token ?? '';
              if (token) {
                tokenCount++;
                const elapsed = (performance.now() - startTime) / 1000;
                const tps = tokenCount / elapsed;
                setLiveTokensPerSec(tps);
                update(g => ({ ...g, output: g.output + token, totalTokens: tokenCount, tokensPerSec: tps, elapsed }));
              }
            } catch {}
          }
        }
        const elapsed = (performance.now() - startTime) / 1000;
        update(g => ({ ...g, streaming: false, elapsed, tokensPerSec: tokenCount / elapsed }));
      } else {
        const resp = await fetch('/v1/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: params.prompt, max_tokens: params.max_tokens, temperature: params.temperature, top_p: params.top_p, top_k: params.top_k, stream: false }),
        });
        if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
        const json = await resp.json();
        const text = json.choices?.[0]?.text ?? '';
        const elapsed = (performance.now() - startTime) / 1000;
        tokenCount = json.usage?.completion_tokens ?? text.split(/\s+/).length;
        update(g => ({ ...g, output: text, streaming: false, totalTokens: tokenCount, tokensPerSec: tokenCount / elapsed, elapsed }));
      }
    } catch (err) {
      update(g => ({ ...g, streaming: false, error: err.message }));
    } finally {
      setIsGenerating(false);
      setLiveTokensPerSec(0);
    }
  }, []);

  const modelName = stats?.model || null;

  return (
    <div className="app-layout">
      <Header
        online={online}
        modelName={modelName}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />

      {activeTab === 'chat' && (
        <div className="main-layout">
          <Sidebar stats={stats} liveTokensPerSec={liveTokensPerSec} />
          <div className="chat-area">
            <OutputDisplay generations={generations} />
            <PromptInput onSubmit={handleSubmit} isGenerating={isGenerating} />
          </div>
        </div>
      )}

      {activeTab === 'speculative' && (
        <SpeculativeLab />
      )}

      {activeTab === 'quantization' && (
        <QuantizationProfiler />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
