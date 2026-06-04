/** Escape HTML special characters */
export function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Get time-of-day greeting */
export function getGreeting() {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

/** Format time as HH:MM */
export function timeNow() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** Alert icon/color config */
export const ALERT_ICONS = {
  fall: 'emergency', inactivity: 'timer', voice: 'record_voice_over',
  sos: 'sos', reminder: 'alarm', test: 'science',
  general: 'info', heart_attack_risk: 'emergency', panic_attack_risk: 'psychology',
}

export const ALERT_COLORS = {
  fall: '#a83836', inactivity: '#d97706', voice: '#605890',
  sos: '#a83836', test: '#006978', general: '#5f5f5c',
  heart_attack_risk: '#a83836', panic_attack_risk: '#c2410c',
}

export const ALERT_BGS = {
  fall: { bg: 'rgba(168,56,54,0.1)', border: 'rgba(168,56,54,0.3)' },
  inactivity: { bg: '#fffbeb', border: '#fde68a' },
  voice: { bg: 'rgba(229,222,255,0.3)', border: 'rgba(96,88,144,0.2)' },
  sos: { bg: 'rgba(168,56,54,0.1)', border: 'rgba(168,56,54,0.3)' },
  test: { bg: 'rgba(145,234,253,0.2)', border: 'rgba(0,105,120,0.2)' },
  general: { bg: 'var(--surface-container)', border: 'rgba(179,178,174,0.2)' },
}

/** Risk panel configuration */
export const RISK_CONFIG = {
  0: { icon: 'favorite', bg: '#dcfce7', border: '#bbf7d0', iconColor: '#15803d', badgeBg: '#dcfce7', badgeColor: '#15803d', label: 'None' },
  1: { icon: 'emergency', bg: '#fee2e2', border: '#fca5a5', iconColor: '#a83836', badgeBg: '#fee2e2', badgeColor: '#a83836', label: 'CRITICAL' },
  2: { icon: 'psychology', bg: '#fff7ed', border: '#fed7aa', iconColor: '#c2410c', badgeBg: '#fff7ed', badgeColor: '#c2410c', label: 'HIGH' },
  3: { icon: 'info', bg: '#eff6ff', border: '#bfdbfe', iconColor: '#1d4ed8', badgeBg: '#eff6ff', badgeColor: '#1d4ed8', label: 'Low' },
}

export const PROB_COLORS = ['#22c55e', '#ef4444', '#f97316', '#3b82f6']
export const CLASS_LABELS = ['Normal', 'Heart Attack Risk', 'Panic Attack', 'False Alarm']

/** Backend API base — in dev, Vite proxy; in production, same origin */
export const API_BASE = import.meta.env.DEV ? 'http://127.0.0.1:5000' : ''

/** Video stream URL */
export const VIDEO_URL = `${API_BASE}/video`
