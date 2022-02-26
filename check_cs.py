'''正解率を求める箇所

'''


# -*- coding: utf-8 -*

#途中で付け加えた
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
# 行列があって、２個、２個,タイプは小数
# '''

global ac_label_global
fp=0
fn=0

'''
#既存モデルの読み込み
pretrained_model = tf.keras.applications.ResNet152V2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.resnet.decode_predictions
'''

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


l = [0,0,0,0,0,0,0,0,0,0]
#file_list = glob.glob("./learning_cat/*")

###画像を小数化してサイズを調整。この段階で４次元になる
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

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  #print(decode_predictions(probs, top=3)[0])
  return decode_predictions(probs, top=3)[0]


##これで画像からラベル情報を取得する
def check_label(image_location):
    #image_path = tf.keras.utils.get_file('image3.jpg',"./cat/cat1.jpg")
    image_raw = tf.io.read_file(image_location)
    
    #引数のファイルがBMP,GIF,JPEG,PNGいずれかであるかを解析し、tf.uint8のTensorに変換
    image = tf.image.decode_image(image_raw)
   
    image = preprocess(image)
    
    ##画像保存
    tf.keras.preprocessing.image.save_img(
    image_location+"dsfs.jpg", image[0])
        
    plt.figure()
    plt.imshow(image[0]) # To change [-1, 1] to [0,1]
    
    #これで枠線を消せるので、画像保存してからtitleつけて処理しよう    
    plt.axis('off')

    status_list = get_imagenet_label(pretrained_model.predict(image))
    
    
    
    plt.title('{} : {:.2f}% \n {} : {:.2f}% \n {} : {:.2f}% '.format(
          status_list[0][1], status_list[0][2]*100,status_list[1][1], status_list[1][2]*100,status_list[2][1], status_list[2][2]*100))
    plt.savefig('figure.png') 
    plt.show()
    

#CSをチェックする時用
def check_label_cs(image_location):
    image_raw = tf.io.read_file(image_location)
    image = tf.image.decode_image(image_raw)
   
    image = preprocess(image)
    
    status_list = get_imagenet_label(pretrained_model.predict(image))
    
    return status_list[0][1]    

###敵対性サンプルの作成
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


#####画像表示用の関数
def display_images(image, description,esp,result,label):
  status_list = get_imagenet_label(pretrained_model.predict(image))
  #plt.figure()
  
      
  if status_list[0][1]==label:
    print(esp,1,end=" ")
  
  else:
    print(esp,0,end=" ")
  
  tf.keras.preprocessing.image.save_img(
    result+"/"+str(esp)+status_list[0][1]+".jpg", image[0])
  
  '''
  plt.imshow(image[0])
  plt.title('{} \n {} : {:.2f}% \n {} : {:.2f}% \n {} : {:.2f}% '.format(description,
          status_list[0][1], status_list[0][2]*100,status_list[1][1], status_list[1][2]*100,status_list[2][1], status_list[2][2]*100))
  #plt.savefig("dog"+str(count)+".jpg")
  plt.show()
  '''

#####実際
def create_ad_image(image_location,result):
    #image_path = tf.keras.utils.get_file('image.jog', image_location)
    
    #ファイルの読み込みとデコード
    image_raw = tf.io.read_file(image_location)
    image = tf.image.decode_image(image_raw)
    
    
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)

    #得られた摂動の視覚化
    # Get the input label of the image.
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)

    epsilons = [0,0.001,0.003,0.007,0.01,0.04,0.08,0.1,0.12,0.15]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]
    

    
    label_status_list = get_imagenet_label(pretrained_model.predict(image))
    #print(label_status_list[0][1])
    
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
        #ここでadv_xについて圧縮センシングを行う
        
        #print("来てる？")
        #plt.imshow(perturbations[0]*0.5+0.5); 
        # # To change [-1, 1] to [0,1]
        #adv_x.save("test"+count+".jpg")

#画像の書き出し
def write_jpg(data):
    data = tf.cast(data, tf.int32)
    print(data)
    sample = tf.image.encode_jpeg(data)
    
    filepath="FILENAME.jpg"
    with open(filepath, 'wb') as fd:
        fd.write(sample)


path_current="./"
files = os.listdir(path_current)
files_dirs = [f for f in files if os.path.isdir(os.path.join(path_current, f))]

#全フォルダーの表示
#print(files_dirs)

#圧縮センシング時の正解数
cs_ans_list=[0,0,0,0,0,0,0,0,0,0]

for dir in files_dirs:

    #正解ラベルの取得
    normal_jpg_files=glob.glob("./"+dir+"//*.jpg")
    ans_label=normal_jpg_files[9][10:-4]
    
    #CSファイルのみを取り出し
    cs_jpg_files = glob.glob("./"+dir+"//CS*.jpg")
   
    
    for k in range(len(cs_jpg_files)):
        
        check_label=check_label_cs(cs_jpg_files[k])
        
        if ans_label==check_label:
            cs_ans_list[k]+=1

    #print(cs_ans_list)
            
            

    
    
    