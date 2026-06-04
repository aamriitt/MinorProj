import { createContext, useContext, useState, useEffect } from 'react'

const UserContext = createContext(null)

const STORAGE_KEY = 'serenecare_user'

export function UserProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      return raw ? JSON.parse(raw) : null
    } catch { return null }
  })

  function login(data) {
    const userData = { ...data, loginTime: new Date().toLocaleString() }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(userData))
    setUser(userData)
  }

  function logout() {
    localStorage.removeItem(STORAGE_KEY)
    setUser(null)
  }

  return (
    <UserContext.Provider value={{ user, login, logout }}>
      {children}
    </UserContext.Provider>
  )
}

export function useUser() {
  const ctx = useContext(UserContext)
  if (!ctx) throw new Error('useUser must be used within UserProvider')
  return ctx
}
