// src/components/Dashboard.js
import React, { useState, useEffect } from 'react';
import Prediction from './Prediction';

const Dashboard = () => {
  const [predictions, setPredictions] = useState([]);
  const [filteredType, setFilteredType] = useState('All');

  // Dummy data to simulate fetched predictions
  const dummyData = [
    {
      url: "http://phishy-example.com",
      prediction: "Legitimate",
      probability: 0.03,
    },
    {
      url: "http://login-alert-security-update.com",
      prediction: "Phishing",
      probability: 0.92,
    },
    {
      url: "https://normal-site.org",
      prediction: "Legitimate",
      probability: 0.11,
    },
    {
      url: "http://security-check-verification.com",
      prediction: "Phishing",
      probability: 0.89,
    },
  ];

  const fetchPredictions = () => {
    setPredictions(dummyData);
  };

  const clearPredictions = () => {
    setPredictions([]);
  };

  const handleFilterChange = (e) => {
    setFilteredType(e.target.value);
  };

  const filteredPredictions = predictions.filter((item) =>
    filteredType === 'All' ? true : item.prediction === filteredType
  );

  useEffect(() => {
    fetchPredictions();
  }, []);

  return (
    <div className="dashboard">
      <h1>Prediction Results</h1>

      <div style={{ marginBottom: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
        <button onClick={fetchPredictions}>🔄 Refresh</button>
        <button onClick={clearPredictions}>🗑️ Clear All</button>

        <label>
          Filter:
          <select value={filteredType} onChange={handleFilterChange} style={{ marginLeft: '0.5rem' }}>
            <option value="All">All</option>
            <option value="Phishing">Phishing</option>
            <option value="Legitimate">Legitimate</option>
          </select>
        </label>
      </div>

      <div className="predictions">
        {filteredPredictions.length > 0 ? (
          filteredPredictions.map((result, index) => (
            <Prediction key={index} data={result} />
          ))
        ) : (
          <p>No predictions available for selected filter.</p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
// Dashboard.js