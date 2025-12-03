import React, { useState } from 'react'
import axios from 'axios'
import './QuerySection.css'

const API_BASE = 'http://localhost:8000/api'

const QuerySection = ({ showAlert }) => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [searching, setSearching] = useState(false)

  const handleSearch = async () => {
    if (!query.trim()) {
      showAlert('Please enter a query', 'error')
      return
    }

    setSearching(true)

    try {
      const response = await axios.post(`${API_BASE}/query`, {
        query,
        top_k: 5
      })

      if (response.data.success) {
        setResults(response.data.results)
        showAlert(`Found ${response.data.results_count} relevant chunks`, 'success')
      }
    } catch (error) {
      const message = error.response?.data?.detail || error.message
      showAlert(`Search failed: ${message}`, 'error')
      setResults([])
    } finally {
      setSearching(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  return (
    <div className="card">
      <div className="card-title">Query Documents</div>
      
      <div className="search-container">
        <div className="search-input-group">
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="search-input" 
            placeholder="Ask a question about your documents..."
          />
          <button 
            className="btn btn-success" 
            onClick={handleSearch}
            disabled={searching}
          >
            {searching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      <div className="results-container">
        {results.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🔎</div>
            <p>Upload documents and ask questions to see results</p>
          </div>
        ) : (
          results.map((result, index) => (
            <div key={index} className="result-item">
              <div className="result-header">
                <strong>Result {index + 1}</strong>
                <span className="result-score">
                  Score: {result.combined_score.toFixed(3)}
                </span>
              </div>
              <div className="result-text">{result.text}</div>
              <div className="result-meta">
                <span>📄 {result.metadata.source}</span>
                <span>📍 Paragraph {result.metadata.paragraph_index}</span>
                <span>🔢 {result.metadata.token_count} tokens</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default QuerySection
