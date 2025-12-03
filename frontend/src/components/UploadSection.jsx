import React, { useState, useRef } from 'react'
import axios from 'axios'
import './UploadSection.css'

const API_BASE = 'http://localhost:8000/api'

const UploadSection = ({ documents, onUploadSuccess, showAlert }) => {
  const [uploading, setUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileSelect = (file) => {
    if (!file) return

    const allowed = ['pdf', 'docx', 'txt']
    const ext = file.name.split('.').pop().toLowerCase()
    
    if (!allowed.includes(ext)) {
      showAlert(`Invalid file type. Allowed: ${allowed.join(', ')}`, 'error')
      return
    }

    if (file.size > 16 * 1024 * 1024) {
      showAlert('File too large. Max size: 16MB', 'error')
      return
    }

    uploadFile(file)
  }

  const uploadFile = async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    setUploading(true)

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      if (response.data.success) {
        onUploadSuccess()
      }
    } catch (error) {
      showAlert(`Upload failed: ${error.response?.data?.detail || error.message}`, 'error')
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => {
    setDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleInputChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const documentsList = Object.values(documents)

  return (
    <div className="card">
      <div className="card-title">Upload Document</div>
      
      <div 
        className={`upload-area ${dragOver ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="upload-icon">📄</div>
        <h3>Drop your document here</h3>
        <p>or click to browse</p>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-light)' }}>
          Supports PDF, DOCX, TXT (Max 16MB)
        </p>
        <input 
          type="file" 
          ref={fileInputRef}
          className="file-input" 
          accept=".pdf,.docx,.txt"
          onChange={handleInputChange}
          style={{ display: 'none' }}
        />
        <button 
          className="btn"
          disabled={uploading}
          onClick={(e) => {
            e.stopPropagation()
            fileInputRef.current?.click()
          }}
        >
          {uploading ? 'Uploading...' : 'Select File'}
        </button>
      </div>

      {uploading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div className="progress-fill"></div>
          </div>
          <div className="progress-text">Processing document...</div>
        </div>
      )}

      <div style={{ marginTop: '2rem' }}>
        <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: 'var(--text-dark)' }}>
          Recent Documents
        </h3>
        {documentsList.length === 0 ? (
          <p style={{ color: 'var(--text-light)', textAlign: 'center' }}>
            No documents yet
          </p>
        ) : (
          <div className="documents-list">
            {documentsList.map((doc, index) => (
              <div key={index} className="document-item">
                <div className="document-info">
                  <h4>📄 {doc.filename}</h4>
                  <p>{doc.upload_time} • {doc.chunks_count} chunks</p>
                </div>
                <span className="badge">{doc.chunks_count}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default UploadSection
