import React, { useEffect } from 'react'
import './AlertContainer.css'

const AlertContainer = ({ alert }) => {
  if (!alert) return null

  return (
    <div className={`alert alert-${alert.type} show`}>
      {alert.message}
    </div>
  )
}

export default AlertContainer
