import os
import uuid
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from model.inference import load_model, run_prediction
from utils.image_processing import validate_image

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'Optimized_OmniCrack30k.pth'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a_super_secret_key' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading PyTorch model...")
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_id = uuid.uuid4().hex
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
            file.save(save_path)

            is_valid, message = validate_image(save_path)
            if not is_valid:
                flash(message)
                os.remove(save_path) 
                return redirect(request.url)

            results = run_prediction(model, save_path)
            
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
            
            return render_template('result.html',
                                   original_image=original_img_url,
                                   mask_image=mask_img_url,
                                   overlay_image=overlay_img_url,
                                   quantified_image=quantified_viz_url,
                                   crack_data=results['analysis'])
    
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)