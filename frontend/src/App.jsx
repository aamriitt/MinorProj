import { Routes, Route, Navigate } from 'react-router-dom'
import LoginPage from './pages/LoginPage.jsx'
import CaregiverDashboard from './pages/CaregiverDashboard.jsx'
import PatientDashboard from './pages/PatientDashboard.jsx'
import { UserProvider } from './context/UserContext.jsx'
import { SocketProvider } from './context/SocketContext.jsx'
import ToastContainer from './components/ToastContainer.jsx'

export default function App() {
  return (
    <UserProvider>
      <SocketProvider>
        <ToastContainer />
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/caregiver" element={<CaregiverDashboard />} />
          <Route path="/patient" element={<PatientDashboard />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </SocketProvider>
    </UserProvider>
  )
}
