import React from 'react';
import './Pages.css';

function About() {
  return (
    <div className="page">
      <h1>About Us</h1>
      <p>Learn more about our company and our mission.</p>
      
      <div className="about-content">
        <div className="about-section">
          <h2>Our Story</h2>
          <p>Founded with a vision to revolutionize the Artificial Intelligence industry, we are driven by a shared passion to reshape the future through bold ideas and innovative thinking. As a group of dedicated high school students, we believe that age is not a barrier to impact. Our team is committed to pushing the boundaries of what's possible in AI, combining creativity, curiosity, and technical skill to develop solutions that can solve real-world problems. We understand the transformative potential of AI and aim to contribute meaningfully to its evolution. Through continuous learning, collaboration, and experimentation, we are confident that our project will leave a lasting mark on the field—and inspire others to dream big, just as we have.</p>
        </div>
        
        <div className="about-section">
          <h2>Our Mission</h2>
          <p>To retain integrity in visual media by making AI-generated content detectable.</p>
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
