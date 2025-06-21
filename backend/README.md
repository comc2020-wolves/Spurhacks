# Python Backend Setup

This is a Flask backend that receives uploaded files from the React frontend.

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run the Flask server:**
   ```bash
   python app.py
   ```

3. **The server will start on:** `http://localhost:5000`

## API Endpoints

- **POST /upload** - Upload a file
- **GET /health** - Health check

## File Storage

Uploaded files are saved in the `uploads/` directory.

## Adding Your Python Processing Logic

In the `upload_file()` function in `app.py`, you can add your custom Python processing:

```python
# After saving the file, add your processing logic here
result = your_python_function(filepath)
```

## CORS Configuration

The backend is configured to accept requests from the React frontend running on `http://localhost:5173`. 