import React from 'react';
import './Pages.css';

function Home() {
  return (
    <div className="page">
      <h1>Welcome to MyApp</h1>
      <p>This is the home page of our amazing application.</p>
      <div className="feature-grid">
        <div className="feature-card">
          <h3>Feature 1</h3>
          <p>Discover amazing features that will enhance your experience.</p>
        </div>
        <div className="feature-card">
          <h3>Feature 2</h3>
          <p>Explore our innovative solutions designed for you.</p>
        </div>
        <div className="feature-card">
          <h3>Feature 3</h3>
          <p>Experience the future of technology with our cutting-edge tools.</p>
        </div>
      </div>
    </div>
  );
}

export default Home; 