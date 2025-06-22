import React, { useState } from 'react';
import { API_BASE_URL } from '../config';
import './Pages.css';

function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadType, setUploadType] = useState('image');
  const [activeTab, setActiveTab] = useState('image');
  const [randomValue, setRandomValue] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setSelectedVideo(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setUploadType('image');
    }
  };

  const handleVideoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedVideo(file);
      setSelectedImage(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setUploadType('video');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (selectedImage || selectedVideo) {
      const file = selectedImage || selectedVideo;
      
      setIsLoading(true); // Start loading
      
      try {
        console.log('Starting upload...');
        console.log('File:', file);
        console.log('API URL:', `${API_BASE_URL}/upload`);
        
        // Create FormData to send file
        const formData = new FormData();
        formData.append('file', file);
        
        // Send file to Python backend using config URL
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData,
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        const result = await response.json();
        console.log('Response body:', result);
        
        if (response.ok) {
          console.log('File uploaded successfully:', result);
          setRandomValue(result.random_value);
          alert(`${uploadType === 'image' ? 'Image' : 'Video'} uploaded successfully!`);
        } else {
          console.error('Upload failed:', result.error);
          alert(`Upload failed: ${result.error}`);
        }
      } catch (error) {
        console.error('Error uploading file:', error);
        console.error('Error details:', error.message);
        alert('Error uploading file. Please try again.');
      } finally {
        setIsLoading(false); // Stop loading regardless of success/failure
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        setSelectedVideo(null);
        setUploadType('image');
        setActiveTab('image');
      } else if (file.type.startsWith('video/')) {
        setSelectedVideo(file);
        setSelectedImage(null);
        setUploadType('video');
        setActiveTab('video');
      }
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const resetUpload = () => {
    setSelectedImage(null);
    setSelectedVideo(null);
    setPreviewUrl(null);
    setUploadType(activeTab);
    setRandomValue(null);
    setIsLoading(false);
  };

  return (
    <div className="page">
      {!previewUrl ? (
        <>
          <h1>Upload Your Image/Video</h1>
          
          <div className="upload-tabs">
            <button 
              className={`tab-btn ${activeTab === 'image' ? 'active' : ''}`}
              onClick={() => setActiveTab('image')}
            >
              Upload Image
            </button>
            <button 
              className={`tab-btn ${activeTab === 'video' ? 'active' : ''}`}
              onClick={() => setActiveTab('video')}
            >
              Upload Video
            </button>
          </div>
          
          <div className="upload-container">
            <div className="upload-area">
              <div className="upload-content">
                <h3>Drag & Drop your {activeTab} here</h3>
                <p>or</p>
                <label htmlFor={`${activeTab}-upload`} className="upload-btn">
                  Choose {activeTab === 'image' ? 'Image' : 'Video'}
                  <input
                    id={`${activeTab}-upload`}
                    type="file"
                    accept={activeTab === 'image' ? 'image/*' : 'video/*'}
                    onChange={activeTab === 'image' ? handleImageUpload : handleVideoUpload}
                    style={{ display: 'none' }}
                  />
                </label>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="preview-page">
          <div className="preview-header">
            <h1>Your {uploadType === 'image' ? 'Image' : 'Video'}</h1>
            <div className="preview-buttons">
              <button 
                onClick={handleSubmit} 
                className="submit-btn"
                disabled={isLoading}
              >
                {isLoading ? 'Uploading...' : 'Upload to Server'}
              </button>
              <button 
                onClick={resetUpload} 
                className="change-btn"
                disabled={isLoading}
              >
                Upload New File
              </button>
            </div>
          </div>
          
          <div className="large-preview">
            {uploadType === 'image' ? (
              <img src={previewUrl} alt="Preview" className="large-image" />
            ) : (
              <video src={previewUrl} controls className="large-video" />
            )}
            {randomValue && (
              <div className="random-value-display">
                <h3>Sigma Meter: {randomValue.toFixed(4)}</h3>
              </div>
            )}
            {isLoading && (
              <div className="loading-overlay">
                <div className="loading-spinner"></div>
                <p>Processing your {uploadType}...</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default Home; 