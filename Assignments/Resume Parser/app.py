from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import os
from extract import extract

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_uploaded_resume():
     file_list = os.listdir('static/files')
     for filename in file_list:
          os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/')
def index():
     remove_uploaded_resume()
     return render_template('homePage.html')

@app.route('/uploadResume', methods=["POST"])
def uploadResume():
     remove_uploaded_resume()
     if 'File' not in request.files:
          flash('No file uploaded')
          return redirect('/')

     file_upload = request.files['File']
     file_upload.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file_upload.filename)))
     return redirect('/parsedResume')

@app.route('/parsedResume', methods=["GET"])
def displayParsedResume():
     filename = os.listdir('static/files')
     if len(filename) == 0:
          return redirect('/')

     skills, education = extract('./static/files/' + filename[0])
     
     return render_template('output.html', skills=skills, education=education)

if __name__ == "__main__":
    app.run(debug=True)

