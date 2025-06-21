import React from 'react';
import './Pages.css';

function Services() {
  return (
    <div className="page">
      <h1>Our Services</h1>
      <p>Discover the comprehensive range of services we offer to meet your needs.</p>
      
      <div className="services-grid">
        <div className="service-card">
          <h3>Web Development</h3>
          <p>Custom websites and web applications built with the latest technologies and best practices.</p>
          <ul>
            <li>Responsive Design</li>
            <li>E-commerce Solutions</li>
            <li>Content Management Systems</li>
          </ul>
        </div>
        
        <div className="service-card">
          <h3>Mobile Development</h3>
          <p>Native and cross-platform mobile applications for iOS and Android platforms.</p>
          <ul>
            <li>iOS Development</li>
            <li>Android Development</li>
            <li>React Native</li>
          </ul>
        </div>
        
        <div className="service-card">
          <h3>Consulting</h3>
          <p>Expert guidance and strategic planning to help your business grow and succeed.</p>
          <ul>
            <li>Technology Strategy</li>
            <li>Digital Transformation</li>
            <li>Performance Optimization</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Services; 