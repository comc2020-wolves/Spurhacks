from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from imageModel import get_random_float

app = Flask(__name__)
# CORS configuration - allow all origins to fix the CORS issue
CORS(app, origins="*")  # Allow all origins for now

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wmv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print(f"Upload directory created/verified: {os.path.abspath(UPLOAD_FOLDER)}")
except Exception as e:
    print(f"Error creating upload directory: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    print("=== UPLOAD REQUEST RECEIVED ===")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            print("ERROR: No file in request.files")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        print(f"File object: {file}")
        print(f"File filename: {file.filename}")
        
        # Check if file was selected
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            print(f"ERROR: File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save the file
        if file.filename is None:
            print("ERROR: Filename is None")
            return jsonify({'error': 'Invalid filename'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        absolute_filepath = os.path.abspath(filepath)
        print(f"Attempting to save file to: {filepath}")
        print(f"Absolute filepath: {absolute_filepath}")
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"WARNING: File already exists: {filepath}")
            # Optionally, you can rename the file to avoid conflicts
            base_name, extension = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filepath):
                new_filename = f"{base_name}_{counter}{extension}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                counter += 1
            print(f"Using new filename: {os.path.basename(filepath)}")
        
        file.save(filepath)
        print(f"SUCCESS: File saved to {filepath}")
        
        # Verify file was actually saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"VERIFICATION: File exists with size {file_size} bytes")
        else:
            print(f"ERROR: File was not actually saved to {filepath}")
        
        # List contents of uploads directory
        uploads_contents = os.listdir(app.config['UPLOAD_FOLDER'])
        print(f"Uploads directory contents: {uploads_contents}")
        
        # Here you can add your Python processing logic
        # For example:
        # result = process_file(filepath)
        
        # Example usage of the random float function
        random_value = get_random_float(filepath)
        # Delete the uploaded file after processing
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"SUCCESS: File deleted from {filepath}")
            else:
                print(f"WARNING: File not found for deletion: {filepath}")
        except Exception as e:
            print(f"ERROR: Failed to delete file {filepath}: {str(e)}")
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'filepath': filepath,
            'random_value': random_value
        }), 200
        
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000) 