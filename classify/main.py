import base64
import io
import keras
import numpy as np
from keras.backend import tensorflow_backend as backend
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from django.conf import settings
import tensorflow

def detect(upload_image):
    result_name = upload_image.name
    result_list = []
    result_img = ''
    # 設定からモデルファイルのパスを取得
    model_file_path = settings.MODEL_FILE_PATH
    # kerasでモデルを読み込む
    model = tensorflow.keras.models.load_model(model_file_path)
    # アップロードされた画像ファイルをメモリ上でOpenCVのimageに格納
    img = load_img(upload_image, target_size=(32, 32))
    #画像を配列に変換し0-1で正規化
    temp_img_array = img_to_array(img) #画像から配列に変換
    temp_img_array = temp_img_array.astype('float32')/255.0
    temp_img_array = temp_img_array.reshape((1,32,32,3)) #適切な形のテンソルに変換

    predicted = model.predict_classes(temp_img_array)
    d = {'0':'オードリー若林', '1':'いきものがかりボーカル', '2':'カワウソ'}
    name = d[str(predicted[0])]
    result_list.append(name)

    return (result_list, result_name, img)
