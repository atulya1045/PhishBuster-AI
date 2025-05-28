import React from 'react';
import './Prediction.css';

function PredictionResult({ data }) {
  const { url, prediction, probability } = data;

  const getLabelColor = () => {
    switch (prediction.toLowerCase()) {
      case 'phishing':
      case 'spam':
        return 'label-danger';
      case 'legitimate':
      case 'ham':
        return 'label-safe';
      default:
        return '';
    }
  };

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <h3>{url}</h3>
        <span className={`prediction-label ${getLabelColor()}`}>{prediction}</span>
      </div>

      <div className="prediction-bar">
        <div
          className="prediction-fill"
          style={{
            width: `${Math.round(probability * 100)}%`,
            backgroundColor: probability > 0.85 ? '#28a745' : '#ffc107',
          }}
        >
          {Math.round(probability * 100)}%
        </div>
      </div>
    </div>
  );
}

export default PredictionResult;
// PredictionResult.js