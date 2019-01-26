from predict_breeds import play_dog_breeds

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

dirname = "static"
UPLOAD_FOLDER = "static/"
app = Flask(__name__)#creating an instance of the Flask class

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")#default page
def home():
    return render_template("home.html")#render_template() looks for a template (HTML file) in the templates folder

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/uploadFile", methods=['GET', 'POST'])
def upload_file(filePath=None, filter_img_paths=None, figdata_pngs=None, isHumanOrDogs=None, pred_messages=None, numOfOutputs=None):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    if request.method == 'POST':
        file = request.files['upload_file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filePath)
            filter_img_paths, figdata_pngs, isHumanOrDogs, pred_messages = play_dog_breeds(filePath)
            numOfOutputs = len(filter_img_paths)

        return render_template("upload.html", filePath=filePath,
                            filter_img_paths=filter_img_paths,
                            figdata_pngs=figdata_pngs,
                            isHumanOrDogs=isHumanOrDogs,
                            pred_messages=pred_messages,
                            numOfOutputs=numOfOutputs)
    else:
        return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True) #allows possible Python errors to appear on the web page.
