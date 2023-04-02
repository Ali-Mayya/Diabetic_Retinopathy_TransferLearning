import os
import matplotlib.pyplot as plt
import tensorflow as tf
pic_size=48
data_path="./Dataset/train_images/"

cpt=0
plt.figure(0,figsize=(15,15))
for eye_id in os.listdir(data_path):
  cpt=cpt+1
  plt.subplot(4,4,cpt)
  img=tf.keras.preprocessing.image.load_img( data_path+eye_id , target_size=(300,300))
  plt.imshow(img)
  if cpt==16:
    break
plt.tight_layout()
plt.show()
