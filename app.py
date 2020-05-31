from flask import Flask,render_template,request
import pandas as  pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.models import load_model
import os
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = './static/upload_images/'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] =10*1024*1024

ALLOWED_EXTENSIONS =['png','jpg','jpeg']
def allowed_file(filename):
    return '.'in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
        
def init():
    global graph
    graph =tf.get_default_graph()

def read_image(filename):
    img =load_img(filename,grayscale =True,target_size =(28,28))
    img =img_to_array(img)
    img =img.reshape(1,28,28,1)
    img =img.astype('float32')
    img =img/255.0
    return img

@app.route('/',methods =['GET','POST'])
def home():
    return render_template('home.html')
@app.route("/predict",methods =['GET','POST'])
def predict():
    if request.method =='POST':
        file =request.files['file']
        # try:
        if file and allowed_file(file.filename):
            filename =file.filename
            
            # print("path:",file_path)
            print("Inside this!!!")
            print("cwd", os.getcwd())
            #file.save(file_path)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("Image Uploaded Successfully:")

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img =read_image(file_path)
            with graph.as_default():
                model1 =tf.keras.models.load_model('./clothing_classification_model.h5')
                class_prediction =model1.predict_classes(img)
                print(class_prediction)
            
            if class_prediction[0]==0:
                product ="T-shirt/top"
            elif class_prediction[0]==1:
                product ="Trouser"
            elif class_prediction[0]==2:
                product ="Pullover"
            elif class_prediction[0]==3:
                product ="Dress"
            elif class_prediction[0]==4:
                product ="Coat"
            elif class_prediction[0]==5:
                product ="Sandal/Shoes"
            elif class_prediction[0]==6:
                product ="Shirt/ Coat"
            elif class_prediction[0]==7:
                product ="Sneaker/Shoes"
            elif class_prediction[0]==8:
                product ="Bag"
            else:
                product="Ankle boot"
            return render_template('predict.html', product =product,user_image = f"static/upload_images/{filename}")
        # except Exception as p:
            # return "Unable to read the file.please insert correct file"
    return render_template('predict.html')


if __name__ ==  __name__ == "__main__":
    init()
    app.run(debug =True)
