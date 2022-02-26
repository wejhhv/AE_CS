
#! https://www.tensorflow.org/tutorials/generative/adversarial_fgsm?hl=ja を編集

# -*- coding: utf-8 -*

#途中で付け加え
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Counter
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import re
import glob


#matplotlibの画像サイズ4,4なら400px,400px
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False


''' 画像をこんな感じの型で表示
# tf.Tensor(
# [[1. 3.]
#  [3. 7.]], shape=(2, 2), dtype=float32)
# 行列があって、2個、2個,タイプは小数
# '''


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

l = [0,0,0,0,0,0,0,0,0,0]

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


#上位３のラベルを返す
def get_imagenet_label(probs):
  #print(decode_predictions(probs, top=3)[0])
  return decode_predictions(probs, top=3)[0]


##これで画像からラベル情報を取得する
def check_label(image_location):
    
    image_raw = tf.io.read_file(image_location)
    
    #引数のファイルがBMP,GIF,JPEG,PNGいずれかであるかを解析し、tf.uint8のTensorに変換
    image = tf.image.decode_image(image_raw)
   
    image = preprocess(image)
    
    #画像保存
    tf.keras.preprocessing.image.save_img(
    image_location+"file.jpg", image[0])
    
    #表示
    plt.figure()
    plt.imshow(image[0]) # To change [-1, 1] to [0,1]
      
    plt.axis('off')

    status_list = get_imagenet_label(pretrained_model.predict(image))
    
    plt.title('{} : {:.2f}% \n {} : {:.2f}% \n {} : {:.2f}% '.format(
          status_list[0][1], status_list[0][2]*100,status_list[1][1], status_list[1][2]*100,status_list[2][1], status_list[2][2]*100))
    plt.savefig('figure.png') 
    plt.show()
    

#! 敵対性サンプルの作成
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad


#! 画像表示用の関数
def display_images(image,esp,result,label):
  status_list = get_imagenet_label(pretrained_model.predict(image))
  #plt.figure()
  
      
  if status_list[0][1]==label:
    print(esp,1,end=" ")
  
  else:
    print(esp,0,end=" ")
  
  
  #画像の保存
  tf.keras.preprocessing.image.save_img(
    result+"/"+str(esp)+status_list[0][1]+".jpg", image[0])
  
  

#! 引数は画像パスと、生成用画像のフォルダー先
def create_ad_image(image_location,result):
    #image_path = tf.keras.utils.get_file('image.jog', image_location)
    
    #ファイルの読み込みとデコード
    image_raw = tf.io.read_file(image_location)
    image = tf.image.decode_image(image_raw)
    
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)

    #得られた摂動の視覚化
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    
    
    #このリストで摂動の値を指定する
    epsilons = [0,0.04,0.08,0.12,0.16,0.2]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]
    

    label_status_list = get_imagenet_label(pretrained_model.predict(image))
    
    
    #摂動の数だけfor
    for i, eps in enumerate(epsilons):
        
        adv_x = image + eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        
        #resultはフォルダー名　status_listは　lは
        #display_images(adv_x, descriptions[i],eps,result,status_list[0][1])
        
        status_list = get_imagenet_label(pretrained_model.predict(adv_x))
        #plt.figure()
            
        #status_listはepsごとのクラス名、label_status_listは摂動0でのクラス名（ノイズが入っていない状態） 
        if status_list[0][1]==label_status_list[0][1]:
            l[epsilons.index(eps)]+=1
          
            
        tf.keras.preprocessing.image.save_img(
            result+"/"+str(eps)+status_list[0][1]+".jpg", adv_x[0])
        
        #! Python上で圧縮センシングするならここにコードを記載


#animalフォルダーの全画像を取得
file_list = glob.glob("./animal/*")

for filename in file_list:
    
  dir = re.sub(r"\D", "", filename)  

  if not os.path.exists(dir):#ディレクトリがなかったら
      os.mkdir(dir)#作成したいフォルダ名を作成
    
  create_ad_image(filename,dir)
  
  #正解数を表示
  #print(l)


'''
add_folder="aho"
if not os.path.exists(add_folder):#ディレクトリがなかったら
      os.mkdir(add_folder)
      
create_ad_image("cat.jpg",add_folder)
'''

#check_label("./dama.jpg")

#for filename in file_list:
    #check_label(filename)