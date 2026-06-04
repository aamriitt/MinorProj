import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUser } from '../context/UserContext.jsx'
import './LoginPage.css'

export default function LoginPage() {
  const { user, login } = useUser()
  const navigate = useNavigate()

  const [caregiverName, setCaregiverName] = useState('')
  const [patientName, setPatientName] = useState('')
  const [patientAge, setPatientAge] = useState('')
  const [relationship, setRelationship] = useState('Child')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // If already logged in, allow re-login (don't auto-redirect)

  function handleSubmit(e) {
    e.preventDefault()
    setError('')

    if (!caregiverName.trim() || !patientName.trim() || !patientAge.trim()) {
      setError('Please fill in all required fields.')
      return
    }
    const age = parseInt(patientAge)
    if (age < 1 || age > 120) {
      setError('Please enter a valid age.')
      return
    }

    setLoading(true)
    login({ caregiverName: caregiverName.trim(), patientName: patientName.trim(), patientAge: age, relationship })
    setTimeout(() => navigate('/caregiver'), 800)
  }

  return (
    <div className="login-page">
      {/* Background decoration */}
      <div className="login-bg-decor">
        <div className="login-circle login-circle-1" />
        <div className="login-circle login-circle-2" />
        <div className="login-circle login-circle-3" />
      </div>

      {/* Card */}
      <div className="login-card fade-in">
        <div className="login-card-inner">
          {/* Logo */}
          <div className="login-logo-section">
            <div className="login-logo-icon">
              <span className="material-symbols-outlined icon-filled" style={{ color: 'white', fontSize: '1.875rem' }}>spa</span>
            </div>
            <h1 className="login-brand">CareWatch</h1>
            <p className="login-tagline">Elderly Care Monitoring System</p>
          </div>

          {/* Welcome */}
          <div className="login-welcome">
            <h2>Welcome 👋</h2>
            <p>Enter your details to start monitoring</p>
          </div>

          {/* Error */}
          {error && (
            <div className="login-error">
              <span className="material-symbols-outlined" style={{ fontSize: '1.125rem' }}>error</span>
              <p>{error}</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="login-form">
            <div className="login-field">
              <label>Caregiver Name <span className="required">*</span></label>
              <div className="login-input-wrap">
                <span className="material-symbols-outlined icon-filled login-input-icon">person</span>
                <input
                  type="text"
                  placeholder="e.g. Amrit, Sarah…"
                  value={caregiverName}
                  onChange={e => setCaregiverName(e.target.value)}
                  className="input-field"
                />
              </div>
            </div>

            <div className="login-field">
              <label>Patient Name <span className="required">*</span></label>
              <div className="login-input-wrap">
                <span className="material-symbols-outlined icon-filled login-input-icon">elderly</span>
                <input
                  type="text"
                  placeholder="e.g. Isha, Arthur…"
                  value={patientName}
                  onChange={e => setPatientName(e.target.value)}
                  className="input-field"
                />
              </div>
            </div>

            <div className="login-field">
              <label>Patient Age <span className="required">*</span></label>
              <div className="login-input-wrap">
                <span className="material-symbols-outlined icon-filled login-input-icon">cake</span>
                <input
                  type="number"
                  min="1"
                  max="120"
                  placeholder="e.g. 80"
                  value={patientAge}
                  onChange={e => setPatientAge(e.target.value)}
                  className="input-field"
                />
              </div>
            </div>

            <div className="login-field">
              <label>Relationship <span style={{ opacity: 0.5 }}>(optional)</span></label>
              <div className="login-input-wrap">
                <span className="material-symbols-outlined icon-filled login-input-icon">family_restroom</span>
                <select
                  value={relationship}
                  onChange={e => setRelationship(e.target.value)}
                  className="input-field login-select"
                >
                  <option value="Child">Child (Son/Daughter)</option>
                  <option value="Spouse">Spouse</option>
                  <option value="Sibling">Sibling</option>
                  <option value="Professional Caregiver">Professional Caregiver</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>

            <button type="submit" className="login-submit" disabled={loading}>
              {loading ? (
                <>
                  <span className="material-symbols-outlined spinning">refresh</span>
                  Setting up…
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined icon-filled">login</span>
                  Start Monitoring
                </>
              )}
            </button>
          </form>

          <p className="login-footer">Your data is stored locally and never shared.</p>
        </div>
      </div>
    </div>
  )
}
