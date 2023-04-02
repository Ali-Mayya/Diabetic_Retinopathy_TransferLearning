from keras.preprocessing.image import ImageDataGenerator
#

val_path="D:/eye to diabets/Diabetic Retinopathy Detection/dataset blindness/Test_images"
data_path="D:/eye to diabets/Diabetic Retinopathy Detection/dataset blindness/Retinal_Images"
batch_size=64

train_datagen=ImageDataGenerator(rescale=1.0/255.0 ,
                                 width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 rotation_range = 20,
                                 horizontal_flip = True,
                                 # target_size = (224, 224),
                                 # reshape = (224, 224, 3)
                                 )
validation_datagen=ImageDataGenerator(rescale = 1.0/255)

train_generator= train_datagen.flow_from_directory(data_path ,
                                                 target_size = (224,224),
                                                 color_mode = "rgb",
                                                 batch_size = batch_size,
                                                 class_mode = "categorical",
                                                 shuffle = True)

validation_generator=validation_datagen.flow_from_directory(val_path,
                                                            target_size = (224,224),
                                                            color_mode = "rgb",
                                                            batch_size = batch_size,
                                                            class_mode ="categorical",
                                                            shuffle = False)

train_steps = int(train_generator.n/batch_size)
val_steps=int(validation_generator.n/batch_size)

#
#