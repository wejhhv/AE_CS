
##単体ファイルでの画像の

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Counter
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt


#matplotlibの画像サイズ4,4なら400px,400px
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

#既存モデルの読み込み(resnet
pretrained_model = tf.keras.applications.ResNet152V2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.resnet.decode_predictions

#! mobieNetの場合は上記の既存モデルの読み込み以下をコメントアウトして、下記のコード
'''

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
'''


#画像を224x224にリサイズ
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = image/255
  image = tf.image.resize(image, (224, 224))
  
  #image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  
  return image

#mobile2用
def preprocess_2(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  
  return image


#! 上位３のラベルを返す
def get_imagenet_label(probs):
  #print(decode_predictions(probs, top=3)[0])
  return decode_predictions(probs, top=3)[0]



#!これで画像からラベル情報を取得する
def check_label(image_location):
    
    image_raw = tf.io.read_file(image_location)
    
    #引数のファイルがBMP,GIF,JPEG,PNGいずれかであるかを解析し、tf.uint8のTensorに変換
    image = tf.image.decode_image(image_raw)
    image = preprocess(image)
    
    '''画像保存
    tf.keras.preprocessing.image.save_img(
    image_location+"file.jpg", image[0])
    '''
    
    status_list = get_imagenet_label(pretrained_model.predict(image))
    
    #表示
    plt.figure()
    plt.imshow(image[0]) # To change [-1, 1] to [0,1]
    plt.axis('off')    
    plt.title('{} : {:.2f}% \n {} : {:.2f}% \n {} : {:.2f}% '.format(
          status_list[0][1], status_list[0][2]*100,status_list[1][1], status_list[1][2]*100,status_list[2][1], status_list[2][2]*100))
    #plt.savefig('figure.png') 
    plt.show()
    
    
#実行
check_label("./cat.jpg")