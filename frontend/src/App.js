// src/App.js
import React, { useState } from 'react';
import PredictionResult from './components/PredictionResult';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [results, setResults] = useState([]);
  const [activeTab, setActiveTab] = useState('url');

  const handleSubmit = async () => {
    if (!input) return;

    try {
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input })
      });

      const data = await res.json();
      setResults(prev => [...prev, data]);
      setInput('');
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

  const filteredResults = results.filter(r => r.type === activeTab);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>PhishBuster AI Dashboard</h1>
        <p>Monitor phishing URLs and email spam in real-time</p>

        <div className="tabs">
          <button
            className={activeTab === 'url' ? 'active' : ''}
            onClick={() => setActiveTab('url')}
          >
            URLs
          </button>
          <button
            className={activeTab === 'email' ? 'active' : ''}
            onClick={() => setActiveTab('email')}
          >
            Emails
          </button>
        </div>

        <input
          className="search-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={`Enter ${activeTab === 'url' ? 'a URL' : 'email content'} to test...`}
        />
        <button onClick={handleSubmit}>Check</button>
      </header>

      <main className="app-main">
        {filteredResults.map((item, index) => (
          <PredictionResult key={index} data={item} />
        ))}
      </main>

      <footer className="app-footer">
        &copy; {new Date().getFullYear()} PhishBuster AI
      </footer>
    </div>
  );
}

export default App;
// App.js