import React from 'react';
import './Pages.css';

function Contact() {
  return (
    <div className="page">
      <h1>Report a Problem</h1>
      <p>Report any issues you find so we can fix them.</p>
      
      <div className="contact-content">
        <div className="contact-info">
          <h2>Support Information</h2>
          <div className="contact-item">
            <h4>Email Support</h4>
            <p>support@myapp.com</p>
          </div>
          <div className="contact-item">
            <h4>Response Time</h4>
            <p>We typically respond within 24 hours</p>
          </div>
          <div className="contact-item">
            <h4>Before Reporting</h4>
            <p>Please check our FAQ section or try refreshing the page first.</p>
          </div>
        </div>
        
        <div className="contact-form">
          <h2>Report Your Issue</h2>
          <form>
            <div className="form-group">
              <label htmlFor="name">Your Name</label>
              <input type="text" id="name" name="name" placeholder="Enter your name" />
            </div>
            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <input type="email" id="email" name="email" placeholder="Enter your email" />
            </div>
            <div className="form-group">
              <label htmlFor="problem-type">Problem Type</label>
              <select id="problem-type" name="problem-type">
                <option value="">Select problem type</option>
                <option value="bug">Bug/Technical Issue</option>
                <option value="feature">Feature Request</option>
                <option value="performance">Performance Issue</option>
                <option value="ui">User Interface Problem</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="message">Problem Description</label>
              <textarea id="message" name="message" rows="5" placeholder="Please describe the problem you're experiencing in detail..."></textarea>
            </div>
            <button type="submit" className="submit-btn">Submit Report</button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Contact; 