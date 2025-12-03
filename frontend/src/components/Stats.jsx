import React from 'react'
import './Stats.css'

const Stats = ({ stats }) => {
  return (
    <div className="stats-grid">
      <div className="stat-card">
        <div className="stat-value">{stats.documents_uploaded}</div>
        <div className="stat-label">Documents</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{stats.total_chunks}</div>
        <div className="stat-label">Total Chunks</div>
      </div>
      <div className="stat-card">
        <div className="stat-value">{stats.index_dimension}</div>
        <div className="stat-label">Embedding Dim</div>
      </div>
    </div>
  )
}

export default Stats
