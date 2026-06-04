import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/video': 'http://127.0.0.1:5000',
      '/alerts': 'http://127.0.0.1:5000',
      '/status': 'http://127.0.0.1:5000',
      '/risk': 'http://127.0.0.1:5000',
      '/voice': 'http://127.0.0.1:5000',
      '/clear': 'http://127.0.0.1:5000',
      '/test_alert': 'http://127.0.0.1:5000',
      '/send_message': 'http://127.0.0.1:5000',
      '/simulate_risk': 'http://127.0.0.1:5000',
      '/set_patient': 'http://127.0.0.1:5000',
      '/api': 'http://127.0.0.1:5000',
      '/socket.io': {
        target: 'http://127.0.0.1:5000',
        ws: true,
      },
    },
  },
  build: {
    outDir: '../static/react',
    emptyOutDir: true,
  },
})
