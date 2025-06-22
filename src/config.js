// API Configuration
const config = {
  development: {
    apiUrl: 'http://localhost:5000'
  },
  production: {
    apiUrl: 'https://chatisthisreal.onrender.com'
  }
};

const environment = import.meta.env.MODE || 'development';
export const API_BASE_URL = config[environment].apiUrl; 