import { useSocket } from '../context/SocketContext.jsx'
import './ToastContainer.css'

const TOAST_CONFIG = {
  critical: { bg: 'toast-critical', icon: 'emergency', label: 'CRITICAL' },
  warning:  { bg: 'toast-warning',  icon: 'warning',   label: 'WARNING' },
  info:     { bg: 'toast-info',     icon: 'info',      label: 'INFO' },
}

export default function ToastContainer() {
  const { toasts, removeToast } = useSocket()

  return (
    <div className="toast-container">
      {toasts.map(t => {
        const cfg = TOAST_CONFIG[t.type] || TOAST_CONFIG.info
        return (
          <div key={t.id} className={`toast-item ${cfg.bg} toast-in`}>
            <span className="material-symbols-outlined icon-filled toast-icon">{cfg.icon}</span>
            <div className="toast-body">
              <p className="toast-label">{cfg.label}</p>
              <p className="toast-msg">{t.msg}</p>
            </div>
            <button className="toast-close" onClick={() => removeToast(t.id)}>
              <span className="material-symbols-outlined" style={{ fontSize: '1rem' }}>close</span>
            </button>
          </div>
        )
      })}
    </div>
  )
}
