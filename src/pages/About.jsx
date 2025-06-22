import React from 'react';
import './Pages.css';

function About() {
  return (
    <div className="page">
      <h1>About Us</h1>
      <p>Learn more about our company and mission.</p>
      
      <div className="about-content">
        <div className="about-section">
          <h2>Our Story</h2>
          <p>Founded with a vision to revolutionize the industry, we've been at the forefront of innovation for over a decade. Our team of experts is dedicated to delivering exceptional solutions that exceed expectations.</p>
        </div>
        
        <div className="about-section">
          <h2>Our Mission</h2>
          <p>To provide cutting-edge technology solutions that empower businesses and individuals to achieve their goals efficiently and effectively.</p>
        </div>
        
        <div className="about-section">
          <h2>Our Values</h2>
          <ul>
            <li>Innovation and Excellence</li>
            <li>Customer Satisfaction</li>
            <li>Integrity and Trust</li>
            <li>Continuous Improvement</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default About; 