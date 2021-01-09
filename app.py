from __future__ import division, print_function

# Import fast.ai Library
import os

from fastai1.fastai.tabular import models
from fastai1.fastai.vision import Path, ImageDataBunch, cnn_learner, get_transforms, imagenet_stats, open_image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# path = Path("C:/Users/Gaurav/PycharmProjects/Water-classifier-fastai-master/path/models")
# classes = ['NORMAL', 'PNEUMONIA', 'COVID19']
# learn = load_learner(path, 'export.pkl')

path = Path("path")
classes = ['NORMAL', 'PNEUMONIA', 'COVID19']
data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = cnn_learner(data2, models.resnet50)
learn.load("C:/Users/Gaurav/PycharmProjects/Water-classifier-fastai-master/path/models/stage-1")


def model_predict(img_path):
    """
       model_predict will return the preprocessed image
    """
   
    img = open_image(img_path)
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
