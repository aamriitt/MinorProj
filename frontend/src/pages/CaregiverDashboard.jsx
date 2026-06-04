import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUser } from '../context/UserContext.jsx'
import { useSocket } from '../context/SocketContext.jsx'
import { getGreeting, timeNow, ALERT_ICONS, ALERT_COLORS, ALERT_BGS, RISK_CONFIG, PROB_COLORS, CLASS_LABELS, VIDEO_URL, API_BASE } from '../utils/helpers.js'
import './CaregiverDashboard.css'

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }

export default function CaregiverDashboard() {
  const { user, logout: doLogout } = useUser()
  const navigate = useNavigate()
  const { connected, alerts, status, riskUpdate, emit, onEvent, clearAlerts, testAlert, simulateRisk, addToast } = useSocket()

  const [messages, setMessages] = useState([])
  const [msgInput, setMsgInput] = useState('')
  const [motionData, setMotionData] = useState([30,50,20,60,45,70,55,80,40,60,75,50,65,45,70,55,80,60,50,0])
  const [inactData, setInactData] = useState([5,10,8,15,6,20,10,12,8,25,10,15,8,20,12,10,15,8,20,0])
  const [fallCount, setFallCount] = useState(0)
  const [emergencies, setEmergencies] = useState([])
  const [clock, setClock] = useState('')
  const msgEndRef = useRef(null)

  const cn = user?.caregiverName || 'Caregiver'
  const pn = user?.patientName || 'Patient'
  const pa = user?.patientAge || '—'

  useEffect(() => { if (!user) navigate('/') }, [user, navigate])
  useEffect(() => {
    const t = setInterval(() => setClock(new Date().toLocaleTimeString()), 1000)
    return () => clearInterval(t)
  }, [])

  // Listen for messages
  useEffect(() => {
    const unsub1 = onEvent('caregiver_message', m => {})
    const unsub2 = onEvent('patient_message', m => {
      setMessages(prev => [...prev, { ...m, isMe: false }])
      addToast(`${m.sender}: "${m.message}"`, 'info')
    })
    return () => { unsub1(); unsub2() }
  }, [onEvent, addToast])

  // Track alerts for emergencies/falls
  useEffect(() => {
    if (alerts.length > 0) {
      const latest = alerts[0]
      if (latest && (latest.type === 'fall' || latest.type === 'sos')) {
        setFallCount(prev => prev + 1)
        setEmergencies(prev => [latest, ...prev].slice(0, 6))
        addToast(latest.message, 'critical')
      } else if (latest && latest.type === 'inactivity') {
        setEmergencies(prev => [latest, ...prev].slice(0, 6))
        addToast(latest.message, 'warning')
      }
    }
  }, [alerts.length])

  // Update charts on status change
  useEffect(() => {
    setMotionData(prev => {
      const next = [...prev, status.motion ? Math.floor(Math.random()*50)+50 : Math.floor(Math.random()*10)]
      return next.slice(-20)
    })
    setInactData(prev => {
      const next = [...prev, Math.min(Math.floor(status.inactive_duration || 0), 100)]
      return next.slice(-20)
    })
  }, [status])

  useEffect(() => { msgEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  function sendMsg(text) {
    const msg = (text || msgInput).trim()
    if (!msg) return
    const payload = { message: msg, sender: cn, timestamp: timeNow() }
    emit('caregiver_message', payload)
    setMessages(prev => [...prev, { ...payload, isMe: true }])
    setMsgInput('')
  }

  function handleSimulate(scenario) {
    simulateRisk(scenario).then(() => {
      const labels = { heart_attack:'Heart Attack Risk simulated', panic:'Panic Attack simulated', normal:'Normal state restored', false_alarm:'False Alarm simulated' }
      const types = { heart_attack:'critical', panic:'warning', normal:'info', false_alarm:'info' }
      addToast(labels[scenario]||'Simulation triggered', types[scenario]||'info')
    })
  }

  function handleLogout() { if (confirm('Log out?')) { doLogout(); navigate('/') } }

  const dur = Math.floor(status.inactive_duration || 0)
  const risk = riskUpdate || { label:0, class_name:'Normal', confidence:0, severity:'none', probs:{}, overridden:false, timestamp:'' }
  const rCfg = RISK_CONFIG[risk.label] || RISK_CONFIG[0]

  function drawBars(data, light, dark) {
    const max = Math.max(...data, 1)
    return data.map((v, i) => (
      <div key={i} className="chart-bar" style={{
        flex:1, borderRadius:'2px 2px 0 0', minHeight:'3px',
        height: `${Math.max(Math.round(v/max*100),4)}%`,
        background: i === data.length-1 ? dark : light,
      }} title={v} />
    ))
  }

  if (!user) return null

  return (
    <div className="cg-page">
      {/* HEADER */}
      <header className="cg-header glass">
        <div className="cg-header-left">
          <div className="cg-header-logo"><span className="material-symbols-outlined icon-filled" style={{color:'white'}}>shield</span></div>
          <div><h1 className="cg-header-brand">CareWatch</h1><p className="cg-header-sub">Caregiver Portal</p></div>
        </div>
        <nav className="cg-header-nav">
          <a href="/caregiver" className="cg-nav-active">Dashboard</a>
          <a href="/patient" className="cg-nav-link">Patient View</a>
        </nav>
        <div className="cg-header-right">
          <div className="cg-header-user">
            <div className="cg-avatar">{cn.charAt(0).toUpperCase()}</div>
            <div className="cg-user-info"><p className="cg-user-name">{cn}</p><p className="cg-user-role">Caregiver</p></div>
          </div>
          <button onClick={handleLogout} className="cg-logout-btn" title="Logout">
            <span className="material-symbols-outlined">logout</span>
          </button>
        </div>
      </header>

      {/* SIDEBAR */}
      <aside className="cg-sidebar">
        <div className="cg-sidebar-inner">
          <div className="cg-sidebar-top">
            <h2>Care Portal</h2>
            <p>Monitoring <strong>{pn}</strong></p>
            <div className="cg-sidebar-patient-card">
              <div><p className="label">Patient</p><p className="value">{pn}</p></div>
              <div style={{textAlign:'right'}}><p className="label">Age</p><p className="value">{pa} yrs</p></div>
            </div>
          </div>
          <div className="cg-sidebar-nav">
            <button className="cg-snav-active"><span className="material-symbols-outlined icon-filled">grid_view</span>Dashboard</button>
            <button className="cg-snav-item"><span className="material-symbols-outlined">notifications_active</span>Alerts<span className="cg-snav-badge">{alerts.length}</span></button>
            <button className="cg-snav-item"><span className="material-symbols-outlined">forum</span>Messages</button>
            <button className="cg-snav-item"><span className="material-symbols-outlined">analytics</span>Analytics</button>
          </div>
          <div className="cg-sidebar-bottom">
            <div className="cg-sidebar-status-card">
              <p className="label">System Status</p>
              <div className="cg-status-row">
                <span className="cg-dot" style={{background: connected ? '#22c55e' : '#a83836'}} />
                <span style={{fontSize:'0.75rem',fontWeight:700,color: connected ? '#15803d' : '#a83836'}}>{connected ? 'System Online' : 'Disconnected'}</span>
              </div>
            </div>
            <div className="cg-sidebar-status-card">
              <p className="label">Last Activity</p>
              <p className="value">{status.motion ? 'Just now' : `${dur}s ago`}</p>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN */}
      <main className="cg-main">
        <section className="cg-greeting fade-up">
          <h1>{getGreeting()}, <span>{cn}</span>.</h1>
          <p>Monitoring <strong style={{color:'var(--primary)'}}>{pn}</strong> • <span style={{color:'var(--primary)',fontWeight:700}}>{status.fall ? '⚠ Fall detected!' : status.motion ? 'Active & moving' : 'All systems active'}</span></p>
        </section>

        <div className="cg-grid">
          {/* LEFT COLUMN */}
          <div className="cg-left">
            {/* LIVE CAMERA */}
            <div className="card">
              <div className="card-header">
                <h2><span className="material-symbols-outlined icon-filled" style={{color:'var(--primary)'}}>videocam</span>Live Camera Feed — <span style={{color:'var(--primary)'}}>{pn}'s Room</span></h2>
                <div style={{display:'flex',alignItems:'center',gap:'0.5rem'}}>
                  <span className="cg-dot pulse" style={{background:'var(--error)'}} />
                  <span style={{fontSize:'0.6875rem',fontWeight:900,color:'var(--error)',textTransform:'uppercase',letterSpacing:'0.08em'}}>LIVE</span>
                </div>
              </div>
              <div className="cg-video-wrap">
                <img src={VIDEO_URL} alt="Live feed" className="cg-video-img" />
                <div className="cg-video-brackets">
                  <div className="bracket tl"/><div className="bracket tr"/><div className="bracket bl"/><div className="bracket br"/>
                </div>
                {status.fall && <div className="cg-fall-overlay"><div className="cg-fall-badge">⚠ FALL DETECTED</div></div>}
                <div className="cg-video-bottom">
                  <div style={{display:'flex',alignItems:'center',gap:'0.5rem'}}>
                    <span className="cg-dot pulse" style={{background:'#4ade80'}} />
                    <span style={{color:'white',fontSize:'0.75rem',fontWeight:700}}>{status.fall ? '⚠ FALL DETECTED' : status.motion ? '● Moving' : `● Still — ${dur}s`}</span>
                  </div>
                  <span style={{color:'rgba(255,255,255,0.6)',fontSize:'0.75rem'}}>{clock}</span>
                </div>
              </div>
              <div className="cg-video-stats">
                <div><p className="label">Motion</p><p className="value" style={{color: status.motion ? '#15803d' : '#5f5f5c'}}>{status.motion ? '● Active' : '◯ Still'}</p></div>
                <div><p className="label">Inactive</p><p className="value" style={{color: dur > 55 ? '#b45309' : 'var(--primary)'}}>{dur}s</p></div>
                <div><p className="label">Camera</p><p className="value" style={{color: status.camera_ok ? 'var(--primary)' : 'var(--error)'}}>{status.camera_ok ? 'Active ✓' : 'No Camera'}</p></div>
              </div>
            </div>

            {/* STAT CARDS */}
            <div className="cg-stat-grid">
              <div className="cg-stat-card"><span className="material-symbols-outlined icon-filled" style={{color:'var(--primary)'}}>person</span><p className="label">Patient</p><p className="value">{pn}</p></div>
              <div className="cg-stat-card"><span className="material-symbols-outlined icon-filled" style={{color:'var(--secondary)'}}>cake</span><p className="label">Age</p><p className="value">{pa} yrs</p></div>
              <div className="cg-stat-card"><span className="material-symbols-outlined icon-filled" style={{color:'var(--tertiary)'}}>timer</span><p className="label">Inactive</p><p className="value" style={{color: dur>55?'#b45309':'var(--primary)'}}>{dur}s</p></div>
              <div className="cg-stat-card cg-stat-alert"><span className="material-symbols-outlined icon-filled" style={{color:'var(--error)'}}>warning</span><p className="label" style={{color:'rgba(168,56,54,0.6)'}}>Alerts</p><p className="value" style={{color:'var(--error)'}}>{alerts.length}</p></div>
            </div>

            {/* ALERT FEED */}
            <div className="card">
              <div className="card-header">
                <h2><span className="material-symbols-outlined icon-filled" style={{color:'var(--error)'}}>notifications_active</span>Live Alert Feed</h2>
                <div style={{display:'flex',gap:'0.5rem'}}>
                  <button className="btn btn-ghost" onClick={clearAlerts}>Clear</button>
                  <button className="btn" style={{background:'rgba(145,234,253,0.4)',color:'var(--primary)'}} onClick={testAlert}>Test Alert</button>
                </div>
              </div>
              <div className="cg-alert-feed">
                {alerts.length === 0 ? (
                  <div className="cg-empty"><span className="material-symbols-outlined" style={{fontSize:'2rem',opacity:0.2}}>notifications</span><p>No alerts — monitoring active</p></div>
                ) : alerts.slice(0,25).map((a, i) => {
                  const isCrit = a.type==='fall'||a.type==='sos'
                  return (
                    <div key={i} className={`cg-alert-item slide-in ${isCrit?'cg-alert-critical':''}`}>
                      <div className="cg-alert-icon" style={{background: ALERT_BGS[a.type]?.bg || ALERT_BGS.general.bg, borderColor: ALERT_BGS[a.type]?.border || ALERT_BGS.general.border}}>
                        <span className="material-symbols-outlined icon-filled" style={{color: ALERT_COLORS[a.type]||'#5f5f5c',fontSize:'1rem'}}>{ALERT_ICONS[a.type]||'info'}</span>
                      </div>
                      <div className="cg-alert-body">
                        <div style={{display:'flex',alignItems:'center',gap:'0.5rem',marginBottom:'0.125rem'}}>
                          <span className="cg-alert-type" style={{color: isCrit?'var(--error)':'var(--on-surface-variant)'}}>{a.type}</span>
                          {isCrit && <span className="cg-alert-crit-badge">CRITICAL</span>}
                        </div>
                        <p className="cg-alert-msg">{a.message}</p>
                        <p className="cg-alert-time">{a.timestamp}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* ANALYTICS */}
            <div className="card">
              <div className="card-header"><h2><span className="material-symbols-outlined icon-filled" style={{color:'var(--primary)'}}>analytics</span>Activity Analytics</h2></div>
              <div className="cg-analytics">
                <div>
                  <p className="label">Motion Activity</p>
                  <div className="cg-chart">{drawBars(motionData, '#91eafd', '#006978')}</div>
                  <div className="cg-chart-labels"><span>Older</span><span>Now</span></div>
                </div>
                <div>
                  <p className="label">Inactivity Duration</p>
                  <div className="cg-chart">{drawBars(inactData, '#fde68a', '#b45309')}</div>
                  <div className="cg-chart-labels"><span>Older</span><span>Now</span></div>
                </div>
                <div>
                  <p className="label">Fall Incidents</p>
                  <div className="cg-fall-stats">
                    <div className="cg-fall-row"><span>Today</span><span className="cg-fall-count">{fallCount}</span></div>
                    <div className="cg-fall-bar-bg"><div className="cg-fall-bar-fill" style={{width:`${Math.min(fallCount*25,100)}%`}} /></div>
                    <div className={`cg-fall-notice ${fallCount>0?'cg-fall-notice-bad':''}`}>
                      <span className="material-symbols-outlined icon-filled" style={{color: fallCount>0?'#a83836':'#16a34a',fontSize:'1rem'}}>{fallCount>0?'warning':'check_circle'}</span>
                      <span>{fallCount>0?`${fallCount} fall(s) detected today`:'No falls detected'}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN */}
          <div className="cg-right">
            {/* ML HEALTH RISK */}
            <div className="card" style={{borderColor: risk.label===1?'#fca5a5':risk.label===2?'#fed7aa':'', boxShadow: risk.label===1?'0 0 0 3px rgba(168,56,54,.08)':risk.label===2?'0 0 0 3px rgba(249,115,22,.08)':''}}>
              <div className="card-header">
                <h2><span className="material-symbols-outlined icon-filled" style={{color:'var(--secondary)'}}>biotech</span>AI Health Risk Prediction</h2>
                <div style={{display:'flex',alignItems:'center',gap:'0.5rem'}}>
                  <span className="cg-dot pulse" style={{background: (risk.label===1||risk.label===2)?'#ef4444':'#22c55e'}} />
                  <span style={{fontSize:'0.625rem',fontWeight:700,textTransform:'uppercase',letterSpacing:'0.08em',color: (risk.label===1||risk.label===2)?'#b91c1c':'#15803d'}}>{(risk.label===1||risk.label===2)?'RISK DETECTED':'ML Active'}</span>
                </div>
              </div>
              <div className="cg-risk-body">
                <div className="cg-risk-main">
                  <div className="cg-risk-icon-wrap" style={{background:rCfg.bg,border:`3px solid ${rCfg.border}`}}>
                    <span className="material-symbols-outlined icon-filled" style={{fontSize:'1.875rem',color:rCfg.iconColor}}>{rCfg.icon}</span>
                  </div>
                  <div style={{flex:1}}>
                    <p className="label">Current Status</p>
                    <p className="cg-risk-class">{risk.class_name||'Normal'}</p>
                    <p style={{fontSize:'0.75rem',color:'var(--on-surface-variant)',marginTop:'0.125rem'}}>Confidence: <strong style={{color:'var(--primary)'}}>{risk.confidence||0}%</strong></p>
                  </div>
                  <div style={{textAlign:'right',flexShrink:0}}>
                    <div className="cg-risk-badge" style={{background:rCfg.badgeBg,color:rCfg.badgeColor}}>{rCfg.label}</div>
                    <p style={{fontSize:'0.625rem',color:'var(--on-surface-variant)',marginTop:'0.25rem'}}>{risk.timestamp||'—'}</p>
                  </div>
                </div>
                {/* Probability bars */}
                <div className="cg-risk-probs">
                  {CLASS_LABELS.map((name, i) => {
                    const pct = risk.probs?.[name] || 0
                    return (
                      <div key={name}>
                        <div className="cg-prob-header"><span>{name}</span><span style={{color:PROB_COLORS[i]}}>{pct}%</span></div>
                        <div className="cg-prob-bar-bg"><div className="cg-prob-bar-fill" style={{width:`${pct}%`,background:PROB_COLORS[i]}} /></div>
                      </div>
                    )
                  })}
                </div>
                {risk.overridden && <div className="cg-false-alarm"><span className="material-symbols-outlined icon-filled" style={{color:'#2563eb',fontSize:'1rem'}}>info</span><p>Low confidence — reclassified as False Alarm by filter</p></div>}
                <div>
                  <p className="label" style={{marginBottom:'0.5rem'}}>Simulate Scenario (Demo)</p>
                  <div className="cg-sim-grid">
                    <button className="cg-sim-btn cg-sim-normal" onClick={() => handleSimulate('normal')}>✓ Normal</button>
                    <button className="cg-sim-btn cg-sim-heart" onClick={() => handleSimulate('heart_attack')}>❤ Heart Risk</button>
                    <button className="cg-sim-btn cg-sim-panic" onClick={() => handleSimulate('panic')}>⚡ Panic Attack</button>
                    <button className="cg-sim-btn cg-sim-false" onClick={() => handleSimulate('false_alarm')}>✗ False Alarm</button>
                  </div>
                </div>
              </div>
            </div>

            {/* EMERGENCY TRIGGERS */}
            <div className="cg-emergency-card">
              <div className="cg-emergency-header">
                <span className="material-symbols-outlined icon-filled" style={{color:'var(--error)'}}>emergency</span>
                <h2>Emergency Triggers</h2>
              </div>
              <div className="cg-emergency-list">
                {emergencies.length === 0 ? (
                  <div className="cg-empty"><span className="material-symbols-outlined" style={{fontSize:'2rem',color:'rgba(168,56,54,0.2)'}}>shield</span><p>No triggers received</p></div>
                ) : emergencies.map((a, i) => {
                  const sev = (a.type==='fall'||a.type==='sos') ? 'CRITICAL' : 'WARNING'
                  return (
                    <div key={i} className="cg-emergency-item slide-in">
                      <span className="material-symbols-outlined icon-filled" style={{color:'var(--error)',fontSize:'1.5rem'}}>{ALERT_ICONS[a.type]||'warning'}</span>
                      <div>
                        <div style={{display:'flex',alignItems:'center',gap:'0.5rem',marginBottom:'0.25rem'}}>
                          <span className={`cg-sev-badge ${sev==='CRITICAL'?'cg-sev-crit':'cg-sev-warn'}`}>{sev}</span>
                          <span style={{fontSize:'0.625rem',color:'var(--on-surface-variant)'}}>{a.timestamp}</span>
                        </div>
                        <p style={{fontSize:'0.875rem',fontWeight:600}}>{a.message}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* MESSAGES */}
            <div className="card">
              <div className="card-header"><h2><span className="material-symbols-outlined icon-filled" style={{color:'var(--primary)'}}>forum</span>Send Message to <span style={{color:'var(--primary)',marginLeft:'0.25rem'}}>{pn}</span></h2></div>
              <div className="cg-msg-thread">
                {messages.length === 0 ? <div className="cg-msg-placeholder">Messages will appear here</div> : messages.map((m, i) => (
                  <div key={i} className={`cg-msg-bubble ${m.isMe ? 'cg-msg-me' : 'cg-msg-them'} slide-in`}>
                    <div className={m.isMe ? 'cg-msg-me-inner' : 'cg-msg-them-inner'}>
                      <p className="cg-msg-text">{m.message}</p>
                      <p className="cg-msg-meta">{m.sender} • {m.timestamp}</p>
                    </div>
                  </div>
                ))}
                <div ref={msgEndRef} />
              </div>
              <div className="cg-quick-msgs">
                <p className="label">Quick Messages</p>
                <div className="cg-quick-btns">
                  <button onClick={() => sendMsg('Are you okay? 💙')}>Are you okay?</button>
                  <button onClick={() => sendMsg('💊 Time to take your medicine!')}>Take medicine</button>
                  <button onClick={() => sendMsg('📞 Please call me!')}>Call me</button>
                  <button onClick={() => sendMsg('💧 Please drink some water!')}>Drink water</button>
                </div>
              </div>
              <div className="cg-msg-input-area">
                <input value={msgInput} onChange={e => setMsgInput(e.target.value)} onKeyDown={e => e.key==='Enter' && sendMsg()} placeholder="Type a message…" className="cg-msg-input" />
                <button className="cg-msg-send" onClick={() => sendMsg()}><span className="material-symbols-outlined">send</span></button>
              </div>
            </div>

            {/* ALERT HISTORY */}
            <div className="card">
              <div className="card-header"><h2><span className="material-symbols-outlined" style={{fontSize:'1.125rem'}}>history</span>Alert History</h2></div>
              <div className="cg-history">
                {alerts.length === 0 ? <div className="cg-empty"><p>No history yet</p></div> : alerts.slice(0,20).map((a, i) => (
                  <div key={i} className="cg-history-item">
                    <span className="material-symbols-outlined icon-filled" style={{color: ALERT_COLORS[a.type]||'#5f5f5c',fontSize:'1rem'}}>{ALERT_ICONS[a.type]||'info'}</span>
                    <div style={{flex:1,minWidth:0}}><p className="cg-history-msg">{a.message}</p><p className="cg-history-time">{a.timestamp}</p></div>
                    <span className="cg-history-badge" style={{background: ALERT_BGS[a.type]?.bg, borderColor: ALERT_BGS[a.type]?.border}}>{a.type}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* MOBILE NAV */}
      <nav className="cg-mobile-nav glass">
        <a href="/caregiver" className="cg-mob-active"><span className="material-symbols-outlined">home</span><span>Home</span></a>
        <a href="#"><span className="material-symbols-outlined">notifications_active</span><span>Alerts</span></a>
        <a href="#"><span className="material-symbols-outlined">forum</span><span>Messages</span></a>
        <a href="/patient"><span className="material-symbols-outlined">elderly</span><span>Patient</span></a>
      </nav>
    </div>
  )
}
