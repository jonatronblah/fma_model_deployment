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
    genres = ['Hip-Hop', 'Pop', 'Rock', 'Experimental', 'Folk', 'Jazz',
       'Electronic', 'Spoken', 'International', 'Soul-RnB', 'Blues',
       'Country', 'Classical', 'Old-Time / Historic', 'Instrumental',
       'Easy Listening']
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        genre = str(request.form["genre"])
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dfu = compute_features(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            values = dfu.values
            values = values.reshape(1, -1)
            pred = str(fma_pipe.predict(values)[0])
            dfu = pd.DataFrame(values)
            dfu['user_genre'] = genre
            with open('/home/jonatron/website/fma_model/new_data.csv', 'a') as f:
                dfu.to_csv(f, header=False, index=None)

            return pred
    return render_template('fma_upload.html', genres=genres)



if __name__ == '__main__':
    app.run()
