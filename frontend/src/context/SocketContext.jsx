import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'
import { io } from 'socket.io-client'

const SocketContext = createContext(null)

// Backend URL — in dev mode (Vite proxy) this goes through /socket.io
// In production build served by Flask, it uses the same origin
const BACKEND_URL = import.meta.env.DEV
  ? 'http://127.0.0.1:5000'
  : window.location.origin

export function SocketProvider({ children }) {
  const socketRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const [alerts, setAlerts] = useState([])
  const [status, setStatus] = useState({
    monitoring: false,
    camera_ok: false,
    inactive_duration: 0,
    motion: false,
    fall: false,
    health_risk_label: 0,
    health_risk_class: 'Normal',
    health_risk_confidence: 0,
    health_risk_severity: 'none',
    health_risk_probs: {},
  })
  const [riskUpdate, setRiskUpdate] = useState(null)
  const [patientRisk, setPatientRisk] = useState(null)
  const [toasts, setToasts] = useState([])
  const toastIdRef = useRef(0)

  // Message callbacks — components register listeners
  const listenersRef = useRef({
    caregiver_message: [],
    patient_message: [],
    risk_update: [],
    patient_risk_notification: [],
  })

  const addToast = useCallback((msg, type = 'info') => {
    const id = ++toastIdRef.current
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 6000)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  useEffect(() => {
    const socket = io(BACKEND_URL, {
      transports: ['websocket', 'polling'],
    })
    socketRef.current = socket

    socket.on('connect', () => setConnected(true))
    socket.on('disconnect', () => setConnected(false))

    socket.on('alert_history', (data) => {
      setAlerts([...data].reverse())
    })

    socket.on('new_alert', (alert) => {
      setAlerts(prev => [alert, ...prev])
    })

    socket.on('alerts_cleared', () => {
      setAlerts([])
    })

    socket.on('status_update', (s) => {
      setStatus(s)
    })

    socket.on('risk_update', (r) => {
      setRiskUpdate(r)
      listenersRef.current.risk_update.forEach(fn => fn(r))
    })

    socket.on('patient_risk_notification', (r) => {
      setPatientRisk(r)
      listenersRef.current.patient_risk_notification.forEach(fn => fn(r))
    })

    socket.on('caregiver_message', (m) => {
      listenersRef.current.caregiver_message.forEach(fn => fn(m))
    })

    socket.on('patient_message', (m) => {
      listenersRef.current.patient_message.forEach(fn => fn(m))
    })

    return () => { socket.disconnect() }
  }, [])

  const emit = useCallback((event, data) => {
    if (socketRef.current) socketRef.current.emit(event, data)
  }, [])

  const onEvent = useCallback((event, fn) => {
    if (listenersRef.current[event]) {
      listenersRef.current[event].push(fn)
    }
    return () => {
      if (listenersRef.current[event]) {
        listenersRef.current[event] = listenersRef.current[event].filter(f => f !== fn)
      }
    }
  }, [])

  const clearAlerts = useCallback(async () => {
    await fetch('/clear', { method: 'POST' })
  }, [])

  const testAlert = useCallback(async () => {
    await fetch('/test_alert', { method: 'POST' })
  }, [])

  const simulateRisk = useCallback(async (scenario) => {
    const res = await fetch('/simulate_risk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario }),
    })
    return res.json()
  }, [])

  const sendMessage = useCallback(async (message, sender) => {
    await fetch('/send_message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, sender }),
    })
  }, [])

  return (
    <SocketContext.Provider value={{
      connected, alerts, status, riskUpdate, patientRisk,
      emit, onEvent, clearAlerts, testAlert, simulateRisk, sendMessage,
      toasts, addToast, removeToast,
    }}>
      {children}
    </SocketContext.Provider>
  )
}

export function useSocket() {
  const ctx = useContext(SocketContext)
  if (!ctx) throw new Error('useSocket must be used within SocketProvider')
  return ctx
}
