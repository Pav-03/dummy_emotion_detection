import { useState, useEffect, useRef } from 'react'
 
const API_URL = '/api'
 
export default function App() {
  const [token, setToken] = useState(null)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loginError, setLoginError] = useState('')
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [health, setHealth] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const textareaRef = useRef(null)
 
  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])
 
  useEffect(() => {
    if (token) {
      fetchModelInfo()
    }
  }, [token])
 
  async function checkHealth() {
    try {
      const res = await fetch(`${API_URL}/health`)
      const data = await res.json()
      setHealth(data)
    } catch {
      setHealth({ status: 'unreachable' })
    }
  }
 
  async function fetchModelInfo() {
    try {
      const res = await fetch(`${API_URL}/model-info`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        setModelInfo(data)
      }
    } catch {
      /* ignore */
    }
  }
 
  async function handleLogin(e) {
    e.preventDefault()
    setLoginError('')
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      if (!res.ok) {
        setLoginError('Invalid username or password')
        return
      }
      const data = await res.json()
      setToken(data.access_token)
    } catch {
      setLoginError('Cannot connect to API')
    }
  }
 
  async function handlePredict(e) {
    e.preventDefault()
    if (!text.trim() || loading) return
    setLoading(true)
    const start = performance.now()
 
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ text }),
      })
 
      if (res.status === 401) {
        setToken(null)
        setLoginError('Session expired. Please login again.')
        return
      }
 
      const data = await res.json()
      const latency = Math.round(performance.now() - start)
 
      const prediction = {
        text: text,
        emotion: data.emotion,
        confidence: data.confidence,
        model_version: data.model_version,
        latency: latency,
        timestamp: new Date().toLocaleTimeString(),
      }
 
      setResult(prediction)
      setHistory((prev) => [prediction, ...prev].slice(0, 20))
      setText('')
      if (textareaRef.current) textareaRef.current.focus()
    } catch {
      setResult({ error: 'Failed to get prediction' })
    } finally {
      setLoading(false)
    }
  }
 
  async function handleBatchPredict() {
    const texts = text
      .split('\n')
      .map((t) => t.trim())
      .filter((t) => t.length > 0)
 
    if (texts.length < 2 || loading) return
    setLoading(true)
 
    try {
      const res = await fetch(`${API_URL}/predict/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ texts }),
      })
 
      if (res.status === 401) {
        setToken(null)
        setLoginError('Session expired. Please login again.')
        return
      }
 
      const data = await res.json()
      const batchResults = data.predictions.map((p, i) => ({
        text: texts[i],
        emotion: p.emotion,
        confidence: p.confidence,
        model_version: p.model_version,
        latency: 0,
        timestamp: new Date().toLocaleTimeString(),
      }))
 
      setResult({ batch: true, predictions: batchResults, total: data.total })
      setHistory((prev) => [...batchResults, ...prev].slice(0, 20))
      setText('')
    } catch {
      setResult({ error: 'Batch prediction failed' })
    } finally {
      setLoading(false)
    }
  }
 
  function handleLogout() {
    setToken(null)
    setResult(null)
    setHistory([])
    setModelInfo(null)
    setUsername('')
    setPassword('')
  }
 
  const isHealthy = health?.status === 'healthy'
  const lineCount = text.split('\n').filter((t) => t.trim()).length
 
  if (!token) {
    return (
      <div className="app">
        <div className="login-container">
          <div className="login-card">
            <div className="login-header">
              <div className="logo-mark">E</div>
              <h1>Emotion detection</h1>
              <p className="subtitle">ML-powered sentiment analysis</p>
            </div>
 
            <form onSubmit={handleLogin}>
              <div className="input-group">
                <label>Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter username"
                  autoFocus
                />
              </div>
              <div className="input-group">
                <label>Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                />
              </div>
              {loginError && <p className="error-text">{loginError}</p>}
              <button type="submit" className="btn-primary">
                Sign in
              </button>
              <button
                type="button"
                className="btn-secondary"
                style={{ width: '100%', marginTop: '10px' }}
                onClick={() => {
                  setUsername('pavan')
                  setPassword('secure123')
                }}
              >
                Fill demo credentials
              </button>
            </form>
 
            <div className="login-footer">
              <span className={`health-dot ${isHealthy ? 'healthy' : 'unhealthy'}`} />
              <span className="health-text">
                API {isHealthy ? 'connected' : 'unreachable'}
              </span>
            </div>
          </div>
        </div>
      </div>
    )
  }
 
  return (
    <div className="app">
      <div className="dashboard">
        <header className="header">
          <div className="header-left">
            <div className="logo-mark small">E</div>
            <div>
              <h1>Emotion detection</h1>
              <p className="header-meta">
                {modelInfo
                  ? `${modelInfo.model_type} · ${modelInfo.features} · ${modelInfo.version}`
                  : 'Loading model info...'}
              </p>
            </div>
          </div>
          <div className="header-right">
            <div className={`health-badge ${isHealthy ? 'healthy' : 'unhealthy'}`}>
              <span className={`health-dot ${isHealthy ? 'healthy' : 'unhealthy'}`} />
              {isHealthy ? 'Healthy' : 'Unhealthy'}
            </div>
            <button className="btn-ghost" onClick={handleLogout}>
              Sign out
            </button>
          </div>
        </header>
 
        <main className="main">
          <div className="predict-section">
            <form onSubmit={handlePredict}>
              <div className="textarea-wrapper">
                <textarea
                  ref={textareaRef}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Try: feeling great today, i hate everything, so happy right now&#10;&#10;For batch mode, put each text on a new line."
                  rows={3}
                  autoFocus
                />
                <div className="textarea-footer">
                  <span className="char-count">
                    {text.length > 0 && `${text.length} chars`}
                    {lineCount > 1 && ` · ${lineCount} lines`}
                  </span>
                  <div className="btn-group">
                    {lineCount > 1 && (
                      <button
                        type="button"
                        className="btn-secondary"
                        onClick={handleBatchPredict}
                        disabled={loading}
                      >
                        Batch analyze ({lineCount})
                      </button>
                    )}
                    <button
                      type="submit"
                      className="btn-primary"
                      disabled={loading || !text.trim()}
                    >
                      {loading ? 'Analyzing...' : 'Analyze'}
                    </button>
                  </div>
                </div>
              </div>
            </form>
 
            {result && !result.error && !result.batch && (
              <div className="result-card" key={result.timestamp}>
                <div className="result-main">
                  <div
                    className={`emotion-badge ${result.emotion === 'positive' ? 'positive' : 'negative'}`}
                  >
                    <span className="emotion-icon">
                      {result.emotion === 'positive' ? '+' : '-'}
                    </span>
                    <div>
                      <span className="emotion-label">{result.emotion}</span>
                      <span className="emotion-confidence">
                        {(result.confidence * 100).toFixed(1)}% confidence
                      </span>
                    </div>
                  </div>
                  <p className="result-text">"{result.text}"</p>
                </div>
                <div className="result-meta">
                  <span>{result.latency}ms</span>
                  <span>{result.model_version}</span>
                </div>
              </div>
            )}
 
            {result && result.batch && (
              <div className="result-card batch-result" key="batch">
                <p className="batch-header">
                  Batch result — {result.total} predictions
                </p>
                {result.predictions.map((p, i) => (
                  <div key={i} className="batch-row">
                    <span
                      className={`batch-tag ${p.emotion === 'positive' ? 'positive' : 'negative'}`}
                    >
                      {p.emotion === 'positive' ? 'POS' : 'NEG'}
                    </span>
                    <span className="batch-text">{p.text}</span>
                    <span className="batch-confidence">
                      {(p.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
 
            {result && result.error && (
              <div className="error-card">{result.error}</div>
            )}
          </div>
 
          <div className="history-section">
            <h2>Recent predictions</h2>
            {history.length === 0 ? (
              <p className="empty-state">
                No predictions yet. Type something above to get started.
              </p>
            ) : (
              <div className="history-list">
                {history.map((item, i) => (
                  <div key={i} className="history-row">
                    <span
                      className={`batch-tag ${item.emotion === 'positive' ? 'positive' : 'negative'}`}
                    >
                      {item.emotion === 'positive' ? 'POS' : 'NEG'}
                    </span>
                    <span className="history-text">{item.text}</span>
                    <span className="history-conf">
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                    <span className="history-time">{item.timestamp}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </main>
 
        <footer className="footer">
          <span>Built by Pavan Modi</span>
          <span className="footer-sep">·</span>
          <span>XGBoost + BoW</span>
          <span className="footer-sep">·</span>
          <span>FastAPI + Docker + AWS</span>
          <span className="footer-sep">·</span>
          <a
            href="https://github.com/Pav-03/dummy_emotion_detection"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
        </footer>
      </div>
    </div>
  )
}
 