from flask import Flask, flash, request, Response, redirect, abort
from werkzeug import secure_filename
from features_new import compute_features
import os
import requests
import pickle

UPLOAD_FOLDER = '/home/jonatron/website/fma_model'
ALLOWED_EXTENSIONS = set(['mp3', 'wav'])

fma_pipe = pickle.load(open('/home/jonatron/website/fma_model/fma_pipe.sav', 'rb'))

app = Flask(__name__)
# config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            df = compute_features(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #df.to_csv('test.csv')
            values = df.values
            values = values.reshape(1, -1)
            pred = str(fma_pipe.predict(values))
            return pred
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''



if __name__ == '__main__':
    app.run()
