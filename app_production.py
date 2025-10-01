import os
import uuid
import cv2
import time
import threading
import warnings
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from model.inference import load_model, run_prediction
from utils.image_processing import validate_image

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
cv2.setNumThreads(1)

warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'Optimized_OmniCrack30k.pth'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading PyTorch model...")
start_time = time.time()
model = load_model(MODEL_PATH)
load_time = time.time() - start_time
print(f"🚀 Flask app ready! Model loaded in {load_time:.2f}s")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Clean up files older than 1 hour to prevent disk space issues."""
    try:
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up old file: {filename}")
    except Exception as e:
        print(f"Cleanup failed: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            
            if not file or not allowed_file(file.filename):
                flash('Invalid file type. Please upload JPG, PNG, or JPEG images only.')
                return redirect(request.url)
            
            filename = secure_filename(file.filename)
            unique_id = uuid.uuid4().hex
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
        
            file.save(save_path)
            
            is_valid, message = validate_image(save_path)
            if not is_valid:
                flash(message)
                if os.path.exists(save_path):
                    os.remove(save_path)
                return redirect(request.url)

            cleanup_old_files()
            
            print(f"🔍 Starting analysis for {filename}...")
            prediction_start = time.time()
            
            try:
                results = run_prediction(model, save_path)
                prediction_time = time.time() - prediction_start
                print(f"Analysis completed in {prediction_time:.2f}s")
            except Exception as e:
                print(f"Prediction failed: {str(e)}")
                flash(f"Analysis failed: {str(e)}")
                if os.path.exists(save_path):
                    os.remove(save_path)
                return redirect(request.url)
            
            try:
                original_img_url = url_for('static', filename=f'uploads/{unique_id}_{filename}')

                mask_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_mask.png")
                Image.fromarray(results['mask'] * 255).save(mask_img_path)
                mask_img_url = url_for('static', filename=f'uploads/{unique_id}_mask.png')

                overlay_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_overlay.png")
                Image.fromarray(results['overlay']).save(overlay_img_path)
                overlay_img_url = url_for('static', filename=f'uploads/{unique_id}_overlay.png')

                quantified_viz_url = None
                if results['quantified_viz'] is not None:
                    quant_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_quantified.png")
                    Image.fromarray(cv2.cvtColor(results['quantified_viz'], cv2.COLOR_BGR2RGB)).save(quant_img_path)
                    quantified_viz_url = url_for('static', filename=f'uploads/{unique_id}_quantified.png')
                
                return render_template('results.html',
                                       original_image=original_img_url,
                                       mask_image=mask_img_url,
                                       overlay_image=overlay_img_url,
                                       quantified_image=quantified_viz_url,
                                       crack_data=results['analysis'])
            
            except Exception as e:
                print(f"Image processing failed: {str(e)}")
                flash(f"Image processing failed: {str(e)}")
                for file_path in [save_path, mask_img_path, overlay_img_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                return redirect(request.url)
                
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            flash(f"An unexpected error occurred: {str(e)}")
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
