import { useState, useEffect } from 'react'
import './App.css'
import Header from './components/Header'
import Stats from './components/Stats'
import UploadSection from './components/UploadSection'
import QuerySection from './components/QuerySection'
import ChatSection from './components/ChatSection'
import AlertContainer from './components/AlertContainer'
import axios from 'axios'

const API_BASE = 'http://localhost:8000/api'

function App() {
  const [stats, setStats] = useState({
    documents_uploaded: 0,
    total_chunks: 0,
    index_dimension: 384
  })
  const [documents, setDocuments] = useState({})
  const [alert, setAlert] = useState(null)

  useEffect(() => {
    loadStats()
    loadDocuments()
  }, [])

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stats`)
      if (response.data.success) {
        setStats(response.data.stats)
      }
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const loadDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE}/documents`)
      if (response.data.success) {
        setDocuments(response.data.documents)
      }
    } catch (error) {
      console.error('Failed to load documents:', error)
    }
  }

  const showAlert = (message, type) => {
    setAlert({ message, type })
    setTimeout(() => setAlert(null), 5000)
  }

  const handleUploadSuccess = () => {
    loadStats()
    loadDocuments()
    showAlert('Document uploaded and processing started!', 'success')
  }

  return (
    <div className="app">
      <Header />
      <div className="container">
        <Stats stats={stats} />
        <AlertContainer alert={alert} />
        
        <div className="main-grid">
          <UploadSection 
            documents={documents}
            onUploadSuccess={handleUploadSuccess}
            showAlert={showAlert}
          />
          <QuerySection showAlert={showAlert} />
        </div>

        {/* Chat Section - Full Width */}
        <div className="chat-container">
          <ChatSection />
        </div>
      </div>
    </div>
  )
}

export default App
