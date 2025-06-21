import React, { useState } from 'react';
import './Pages.css';

function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    problemType: '',
    message: ''
  });
  const [showSuccess, setShowSuccess] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you would typically send the data to your backend
    console.log('Form submitted:', formData);
    
    // Show success message
    setShowSuccess(true);
    
    // Reset form
    setFormData({
      name: '',
      email: '',
      problemType: '',
      message: ''
    });
    
    // Hide success message after 5 seconds
    setTimeout(() => {
      setShowSuccess(false);
    }, 5000);
  };

  return (
    <div className="page">
      <h1>Report a Problem</h1>
      <p>Report any issues you find so we can fix them.</p>
      
      {showSuccess && (
        <div className="success-notification">
          <div className="success-content">
            <span className="success-icon">✓</span>
            <span className="success-text">Submitted successfully!</span>
          </div>
        </div>
      )}
      
      <div className="contact-form-full">
        <h2>Report Your Issue</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Your Name</label>
            <input 
              type="text" 
              id="name" 
              name="name" 
              value={formData.name}
              onChange={handleInputChange}
              placeholder="Enter your name" 
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input 
              type="email" 
              id="email" 
              name="email" 
              value={formData.email}
              onChange={handleInputChange}
              placeholder="Enter your email" 
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="problem-type">Problem Type</label>
            <select 
              id="problem-type" 
              name="problemType"
              value={formData.problemType}
              onChange={handleInputChange}
              required
            >
              <option value="">Select problem type</option>
              <option value="website">Website Issue</option>
              <option value="program">Program Issue</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="message">Problem Description</label>
            <textarea 
              id="message" 
              name="message" 
              rows="5" 
              value={formData.message}
              onChange={handleInputChange}
              placeholder="What is the problem?" 
              required
            ></textarea>
          </div>
          <button type="submit" className="submit-btn">Submit Report</button>
        </form>
      </div>
    </div>
  );
}

export default Contact; 