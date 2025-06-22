import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './TopBar.css';

function TopBar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="topbar">
      <div className="topbar-container">
        <div className="topbar-brand">
          <h2>Chat Is This Real?</h2>
          <h5>By Connect</h5>
        </div>
        
        <div className={`topbar-menu ${isMenuOpen ? 'active' : ''}`}>
          <Link to="/" className="topbar-link">Home</Link>
          <Link to="/about" className="topbar-link">About</Link>
          <Link to="/contact" className="topbar-link">Report a Problem</Link>
        </div>
        
        <div className="topbar-toggle" onClick={toggleMenu}>
          <span className="hamburger"></span>
          <span className="hamburger"></span>
          <span className="hamburger"></span>
        </div>
      </div>
    </nav>
  );
}

export default TopBar; 