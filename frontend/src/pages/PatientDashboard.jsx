import { useState, useEffect, useRef } from 'react'
import { useUser } from '../context/UserContext.jsx'
import { useSocket } from '../context/SocketContext.jsx'
import { getGreeting, timeNow, API_BASE } from '../utils/helpers.js'
import './PatientDashboard.css'

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') }

const HINDI_R = {
  'मदद चाहिए': { hi:'देखभालकर्ता को तुरंत सूचित किया जा रहा है।', en:'Alerting your caregiver immediately.', alert:true },
  'पानी चाहिए': { hi:'ठीक है, पानी लाया जाएगा।', en:'Water will be arranged for you.', alert:false },
  'दवाई': { hi:'दवाई याद दिलाई गई।', en:'Medicine reminder noted.', alert:true },
  'ठीक हूँ': { hi:'बहुत अच्छा! हम आपकी निगरानी कर रहे हैं।', en:'Glad to hear that!', alert:false },
  'default': { hi:'आपकी बात सुनी गई। मैं यहाँ हूँ।', en:"I heard you. I'm here to help.", alert:false },
}

const RISK_BANNER_CFG = {
  'Heart Attack Risk': { bg:'#fee2e2',border:'#fca5a5',iconBg:'#ef4444',icon:'emergency',badgeBg:'#fee2e2',badgeColor:'#a83836',badge:'CRITICAL',barColor:'#ef4444',title:'Heart Risk Alert',en:'Abnormal movement patterns detected. Please stay calm and alert caregiver.' },
  'Panic Attack': { bg:'#fff7ed',border:'#fed7aa',iconBg:'#f97316',icon:'psychology',badgeBg:'#fff7ed',badgeColor:'#c2410c',badge:'HIGH',barColor:'#f97316',title:'Panic Alert',en:'Take slow deep breaths. You are safe. Caregiver has been notified.' },
}

function speak(text) {
  if (!text?.trim() || !('speechSynthesis' in window)) return
  const isHindi = /[\u0900-\u097F]/.test(text)
  window.speechSynthesis.cancel()
  const u = new SpeechSynthesisUtterance(text)
  u.lang = isHindi ? 'hi-IN' : 'en-IN'
  u.rate = isHindi ? 0.88 : 0.92
  u.pitch = 1.05
  const voices = window.speechSynthesis.getVoices()
  if (voices.length) {
    const pref = voices.find(v => v.lang === u.lang) || voices.find(v => v.lang.startsWith(isHindi?'hi':'en')) || voices[0]
    if (pref) u.voice = pref
  }
  window.speechSynthesis.speak(u)
}

export default function PatientDashboard() {
  const { user } = useUser()
  const { emit, onEvent, testAlert } = useSocket()
  const pn = user?.patientName || 'Friend'
  const cn = user?.caregiverName || 'Caregiver'

  const [chatMsgs, setChatMsgs] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [showSOS, setShowSOS] = useState(false)
  const [riskBanner, setRiskBanner] = useState(null)
  const [cgMsg, setCgMsg] = useState(null)
  const chatEndRef = useRef(null)

  const [reminders, setReminders] = useState([
    { icon:'pill', label:'Morning Vitamins', hi:'सुबह की दवाई', sub:'After breakfast • 8:30 AM', theme:'secondary', done:false, caregiverMsg:`✅ ${pn} has taken their morning vitamins.` },
    { icon:'water_drop', label:'Hydration Break', hi:'पानी पिएं', sub:'Drink 250ml water', theme:'primary', done:false, caregiverMsg:`💧 ${pn} has had their water / hydration break.` },
    { icon:'directions_walk', label:'Gentle Stroll', hi:'सुबह की सैर', sub:'Garden area • 10:00 AM', theme:'tertiary', done:false, caregiverMsg:`🚶 ${pn} has completed their gentle stroll / exercise.` },
    { icon:'video_call', label:`Call with ${cn}`, hi:`${cn} से बात`, sub:'Family check-in • 6:00 PM', theme:'neutral', done:false, caregiverMsg:`📞 ${pn} has completed their family check-in call.` },
  ])

  useEffect(() => { addAI(`नमस्ते ${pn} जी! मैं आपकी देखभाल के लिए यहाँ हूँ।`, `Hello ${pn}! I am here to take care of you today.`) }, [])
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior:'smooth' }) }, [chatMsgs])

  useEffect(() => {
    const unsub1 = onEvent('caregiver_message', m => {
      setCgMsg(m)
      addAI(`${cn} का संदेश: "${m.message}"`, `Message from ${m.sender||cn}: "${m.message}"`)
      speak(`${cn} कह रहे हैं: ${m.message}`)
      setTimeout(() => setCgMsg(null), 10000)
    })
    const unsub2 = onEvent('patient_risk_notification', r => {
      const cfg = RISK_BANNER_CFG[r.class_name]
      if (cfg) {
        setRiskBanner({ ...r, cfg })
        addAI(r.hindi_msg||'', cfg.en)
        speak(r.hindi_msg||'')
        if (r.severity !== 'critical') setTimeout(() => setRiskBanner(null), 30000)
      }
    })
    return () => { unsub1(); unsub2() }
  }, [onEvent, cn])

  function addUser(msg) { setChatMsgs(prev => [...prev, { type:'user', text:msg, time:timeNow() }]) }
  function addAI(hi, en) { setChatMsgs(prev => [...prev, { type:'ai', hi, en, time:timeNow() }]) }

  function doCmd(cmd) {
    addUser(cmd)
    const r = HINDI_R[cmd] || HINDI_R['default']
    setTimeout(() => { addAI(r.hi, r.en); speak(r.hi); if (r.alert) testAlert() }, 400)
  }

  function sendChat() {
    const msg = chatInput.trim(); if (!msg) return
    setChatInput('')
    addUser(msg)
    emit('patient_message', { message:msg, sender:pn, timestamp:timeNow(), fromPatient:true })
    let resp = HINDI_R['default']
    for (const [k,v] of Object.entries(HINDI_R)) { if (k!=='default' && msg.includes(k)) { resp=v; break } }
    setTimeout(() => { addAI(resp.hi, resp.en); speak(resp.hi); if (resp.alert) testAlert() }, 600)
  }

  function triggerSOS() {
    setShowSOS(true)
    testAlert()
    emit('sos_trigger', { patient:pn, timestamp:new Date().toLocaleString() })
    addAI(`🚨 SOS भेजा गया! ${cn} को सूचित किया जा रहा है।`, `SOS sent! ${cn} has been alerted.`)
  }

  function toggleReminder(i) {
    setReminders(prev => {
      const next = [...prev]; next[i] = { ...next[i], done: !next[i].done }
      if (next[i].done) {
        addAI(next[i].hi + ' — हो गया! ✓', next[i].label + ' — Done! Great job.')
        speak(next[i].hi + ' — हो गया! बहुत अच्छे।')
        emit('patient_message', { message:next[i].caregiverMsg, sender:pn, timestamp:timeNow(), fromPatient:true, isReminder:true })
        testAlert()
      }
      return next
    })
  }

  function quickAction(type) {
    const Q = {
      call: { hi:`${cn} को कॉल किया जा रहा है।`, en:`Calling ${cn} for you.`, alert:true },
      help: { hi:'मदद के लिए सूचना भेजी गई है। वे जल्द आएंगे।', en:"Help has been requested.", alert:true },
      fine: { hi:'बहुत अच्छा! आप ठीक हैं, यह जानकर अच्छा लगा।', en:"Wonderful! Glad you're doing well.", alert:false },
    }
    const r = Q[type]
    addUser(type==='call'?'Call Caregiver':type==='help'?'Need Help':"I'm Fine")
    addAI(r.hi, r.en); speak(r.hi)
    if (r.alert) testAlert()
  }

  const tasksLeft = reminders.filter(r => !r.done).length

  return (
    <div className="pt-page">
      {/* SOS Overlay */}
      {showSOS && (
        <div className="pt-sos-overlay">
          <div className="pt-sos-ring pulse-ring"><span className="material-symbols-outlined icon-filled" style={{color:'white',fontSize:'3.5rem'}}>emergency</span></div>
          <div style={{textAlign:'center',padding:'0 2rem'}}>
            <p className="pt-sos-hi deva">SOS भेज दिया!</p>
            <p className="pt-sos-en">SOS Sent!</p>
            <p className="pt-sos-sub deva">{cn} को सूचित किया जा रहा है…</p>
            <p style={{fontSize:'1.125rem',color:'rgba(255,255,255,0.7)'}}>Your caregiver has been alerted.</p>
          </div>
          <div className="pt-sos-connecting"><span className="cg-dot pulse" style={{background:'white',width:'0.75rem',height:'0.75rem'}} /><span>Connecting to caregiver…</span></div>
          <button className="pt-sos-cancel deva" onClick={() => { setShowSOS(false); addAI('ठीक है, SOS रद्द कर दिया गया।', 'SOS cancelled. Glad you\'re okay!') }}>मैं ठीक हूँ — I'm OK / Cancel</button>
        </div>
      )}

      {/* Header */}
      <header className="pt-header glass">
        <div className="pt-header-left">
          <div className="pt-header-logo"><span className="material-symbols-outlined icon-filled" style={{color:'white',fontSize:'1.5rem'}}>shield</span></div>
          <div><h1 className="pt-header-brand">CareWatch</h1><p className="deva" style={{fontSize:'0.75rem',color:'var(--on-surface-variant)'}}>नमस्ते, {pn} जी!</p></div>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:'0.75rem'}}>
          <div className="pt-monitoring-badge"><span className="cg-dot pulse" style={{background:'#22c55e'}} /><span>Being Monitored</span></div>
          <div className="pt-avatar">{pn.charAt(0).toUpperCase()}</div>
        </div>
      </header>

      {/* Main */}
      <main className="pt-main">
        <section className="pt-greeting fade-up">
          <h1>{getGreeting()}, <span style={{color:'var(--primary)'}}>{pn}</span>.</h1>
          <p className="deva">{pn} जी, आज का दिन अच्छा रहे।</p>
        </section>

        {/* Monitoring Notice */}
        <div className="pt-monitor-notice fade-up">
          <div className="pt-notice-icon"><span className="material-symbols-outlined icon-filled" style={{color:'white',fontSize:'1.5rem'}}>videocam</span></div>
          <div><p style={{fontWeight:700,color:'var(--on-primary-container)'}}>You are being monitored for your safety.</p><p className="deva" style={{fontSize:'0.875rem',color:'rgba(0,87,99,0.7)',marginTop:'0.125rem'}}>आपकी सुरक्षा के लिए आपकी निगरानी की जा रही है।</p></div>
          <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:'0.5rem',flexShrink:0}}><span className="cg-dot pulse" style={{background:'#22c55e'}} /><span style={{fontSize:'0.75rem',fontWeight:700,color:'#15803d'}}>ACTIVE</span></div>
        </div>

        {/* Risk Banner */}
        {riskBanner && (
          <div className="pt-risk-banner fade-up" style={{background:riskBanner.cfg.bg,borderColor:riskBanner.cfg.border}}>
            <div style={{display:'flex',alignItems:'flex-start',gap:'1rem'}}>
              <div className="pt-risk-icon" style={{background:riskBanner.cfg.iconBg}}><span className="material-symbols-outlined icon-filled" style={{color:'white',fontSize:'1.5rem'}}>{riskBanner.cfg.icon}</span></div>
              <div style={{flex:1}}>
                <div style={{display:'flex',alignItems:'center',gap:'0.5rem',marginBottom:'0.25rem'}}>
                  <p style={{fontWeight:900,color:riskBanner.cfg.badgeColor}}>{riskBanner.cfg.title}</p>
                  <span style={{fontSize:'0.625rem',fontWeight:900,padding:'0.125rem 0.5rem',borderRadius:'var(--radius-full)',background:riskBanner.cfg.badgeBg,color:riskBanner.cfg.badgeColor}}>{riskBanner.cfg.badge}</span>
                </div>
                <p className="deva" style={{fontWeight:600,color:riskBanner.cfg.badgeColor}}>{riskBanner.hindi_msg}</p>
                <p style={{fontSize:'0.875rem',opacity:0.7,marginTop:'0.25rem'}}>{riskBanner.cfg.en}</p>
                <div style={{display:'flex',alignItems:'center',gap:'0.5rem',marginTop:'0.5rem'}}>
                  <span className="label">AI Confidence:</span>
                  <div style={{flex:1,height:'0.375rem',background:'rgba(0,0,0,0.1)',borderRadius:'var(--radius-full)',overflow:'hidden'}}><div style={{height:'100%',width:`${riskBanner.confidence||0}%`,background:riskBanner.cfg.barColor,borderRadius:'var(--radius-full)',transition:'width 0.7s'}} /></div>
                  <span style={{fontSize:'0.625rem',fontWeight:700,color:riskBanner.cfg.badgeColor}}>{riskBanner.confidence||0}%</span>
                </div>
              </div>
              <button onClick={() => setRiskBanner(null)} style={{background:'none',border:'none',cursor:'pointer',opacity:0.4,color:'currentcolor'}}><span className="material-symbols-outlined" style={{fontSize:'1.125rem'}}>close</span></button>
            </div>
          </div>
        )}

        <div className="pt-grid">
          {/* LEFT */}
          <div className="pt-left">
            {/* AI Chat */}
            <div className="card pt-chat-card">
              <div className="card-header">
                <div style={{display:'flex',alignItems:'center',gap:'0.75rem'}}>
                  <div style={{width:'2.75rem',height:'2.75rem',background:'var(--primary)',borderRadius:'50%',display:'flex',alignItems:'center',justifyContent:'center',boxShadow:'var(--shadow-sm)'}}><span className="material-symbols-outlined icon-filled" style={{color:'white'}}>smart_toy</span></div>
                  <div><h3 style={{fontWeight:700,fontSize:'1rem'}}>Serene AI — सेरेन साथी</h3><div style={{display:'flex',alignItems:'center',gap:'0.375rem'}}><span className="cg-dot" style={{background:'#22c55e'}} /><span style={{fontSize:'0.75rem',fontWeight:700,color:'#15803d',textTransform:'uppercase',letterSpacing:'0.08em'}}>Active & Ready</span></div></div>
                </div>
              </div>
              <div className="pt-chat-thread" id="chat-thread">
                {chatMsgs.map((m, i) => m.type === 'user' ? (
                  <div key={i} className="pt-chat-user chat-out">
                    <div className="pt-chat-user-bubble">
                      <p className="deva">{m.text}</p>
                      <p className="pt-chat-meta">{pn} • {m.time}</p>
                    </div>
                  </div>
                ) : (
                  <div key={i} className="pt-chat-ai chat-in">
                    <div className="pt-chat-ai-avatar"><span className="material-symbols-outlined icon-filled" style={{color:'var(--primary)'}}>smart_toy</span></div>
                    <div className="pt-chat-ai-bubble">
                      <p className="deva">{m.hi}</p>
                      {m.en && <p style={{fontSize:'0.875rem',color:'rgba(0,87,99,0.7)',marginTop:'0.25rem'}}>{m.en}</p>}
                      <p className="pt-chat-meta">Serene • {m.time}</p>
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
              {cgMsg && (
                <div className="pt-cg-banner">
                  <span className="material-symbols-outlined icon-filled" style={{color:'var(--secondary)'}}>mark_chat_read</span>
                  <div><p className="label">Message from {cgMsg.sender||cn}</p><p style={{fontSize:'0.875rem',fontWeight:600,color:'var(--on-secondary-container)'}}>{cgMsg.message}</p></div>
                </div>
              )}
              <div className="pt-chat-input-area">
                <div style={{display:'flex',gap:'0.75rem',alignItems:'center'}}>
                  <input value={chatInput} onChange={e => setChatInput(e.target.value)} onKeyDown={e => e.key==='Enter' && sendChat()} placeholder="Type or speak… / बोलें या लिखें…" className="pt-chat-input" />
                  <button className="pt-chat-send big-btn" onClick={sendChat}><span className="material-symbols-outlined">send</span></button>
                </div>
                <div className="pt-quick-chips">
                  <button onClick={() => doCmd('मदद चाहिए')} className="pt-chip pt-chip-sos deva">🆘 मदद चाहिए</button>
                  <button onClick={() => doCmd('पानी चाहिए')} className="pt-chip pt-chip-water deva">💧 पानी चाहिए</button>
                  <button onClick={() => doCmd('दवाई')} className="pt-chip pt-chip-med deva">💊 दवाई</button>
                  <button onClick={() => doCmd('ठीक हूँ')} className="pt-chip pt-chip-ok deva">✅ ठीक हूँ</button>
                </div>
              </div>
            </div>

            {/* Reminders */}
            <div className="pt-reminders">
              <div className="pt-reminders-header">
                <h2><span className="material-symbols-outlined icon-filled" style={{color:'var(--primary)'}}>event_note</span>Today's Schedule</h2>
                <span className="pt-tasks-badge">{tasksLeft} Task{tasksLeft!==1?'s':''}</span>
              </div>
              <div className="pt-reminders-grid">
                {reminders.map((r, i) => (
                  <div key={i} className={`reminder-card pt-reminder-${r.theme} ${r.done?'pt-reminder-done':''}`} onClick={() => toggleReminder(i)}>
                    <div className="pt-reminder-icon"><span className="material-symbols-outlined icon-filled" style={{fontSize:'1.5rem'}}>{r.icon}</span></div>
                    <div style={{flex:1}}><h3 style={{fontWeight:700}}>{r.label}</h3><p className="deva" style={{fontSize:'0.875rem',opacity:0.7}}>{r.hi}</p><p style={{fontSize:'0.75rem',opacity:0.6,marginTop:'0.125rem'}}>{r.sub}</p></div>
                    <div className={`pt-reminder-check ${r.done?'pt-reminder-checked':''}`}><span className="material-symbols-outlined" style={{fontSize:'1.125rem'}}>{r.done?'check':'circle'}</span></div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* RIGHT */}
          <div className="pt-right">
            {/* SOS */}
            <div className="pt-sos-card">
              <h2 className="deva" style={{fontSize:'1.25rem',fontWeight:700,color:'var(--on-error-container)'}}>तत्काल सहायता?</h2>
              <h2 style={{fontSize:'1rem',fontWeight:700,color:'var(--on-error-container)',marginBottom:'1.25rem'}}>Need Immediate Help?</h2>
              <div className="pt-sos-btn-wrap">
                <div className="pt-sos-ring-outer pulse-ring" />
                <div className="pt-sos-ring-inner pulse-ring" style={{animationDelay:'0.4s'}} />
                <button className="sos-btn pt-sos-button" onClick={triggerSOS}>
                  <span className="material-symbols-outlined icon-filled" style={{fontSize:'3rem'}}>sos</span>
                  <span style={{fontSize:'0.875rem',fontWeight:900}}>HELP!</span>
                </button>
              </div>
              <p className="deva" style={{fontSize:'0.875rem',fontWeight:700,color:'rgba(110,10,18,0.6)'}}>आपका देखभालकर्ता तुरंत सूचित होगा</p>
              <p style={{fontSize:'0.75rem',color:'rgba(110,10,18,0.5)',marginTop:'0.25rem'}}>Caregiver alerted immediately</p>
            </div>

            {/* Quick Help */}
            <div className="card">
              <div className="card-header"><h2 className="deva" style={{fontSize:'1rem'}}>त्वरित सहायता • Quick Help</h2></div>
              <div className="pt-quick-help">
                <button className="big-btn pt-qh-btn pt-qh-call" onClick={() => quickAction('call')}>
                  <div className="pt-qh-icon" style={{background:'var(--primary)'}}><span className="material-symbols-outlined icon-filled" style={{color:'white',fontSize:'1.5rem'}}>call</span></div>
                  <div><p style={{fontWeight:900}}>Call Caregiver</p><p className="deva" style={{fontSize:'0.875rem',opacity:0.7}}>देखभालकर्ता को कॉल करें</p></div>
                </button>
                <button className="big-btn pt-qh-btn pt-qh-help" onClick={() => quickAction('help')}>
                  <div className="pt-qh-icon" style={{background:'var(--secondary)'}}><span className="material-symbols-outlined icon-filled" style={{color:'white',fontSize:'1.5rem'}}>waving_hand</span></div>
                  <div><p style={{fontWeight:900}}>Need Help</p><p className="deva" style={{fontSize:'0.875rem',opacity:0.7}}>मुझे मदद चाहिए</p></div>
                </button>
                <button className="big-btn pt-qh-btn pt-qh-fine" onClick={() => quickAction('fine')}>
                  <div className="pt-qh-icon" style={{background:'#22c55e'}}><span className="material-symbols-outlined icon-filled" style={{color:'white',fontSize:'1.5rem'}}>thumb_up</span></div>
                  <div><p style={{fontWeight:900}}>I'm Fine ✓</p><p className="deva" style={{fontSize:'0.875rem',color:'#15803d'}}>मैं ठीक हूँ</p></div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Mobile Nav */}
      <nav className="pt-mobile-nav glass">
        <a href="/patient" className="pt-mob-active"><span className="material-symbols-outlined">home</span><span>Home</span></a>
        <a href="#"><span className="material-symbols-outlined">smart_toy</span><span>AI Chat</span></a>
        <button onClick={triggerSOS} className="pt-mob-sos"><span className="material-symbols-outlined icon-filled">sos</span><span>SOS</span></button>
        <a href="#"><span className="material-symbols-outlined">event_note</span><span>Schedule</span></a>
      </nav>
    </div>
  )
}
