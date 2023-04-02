from Data_Generator import *
from Build_Model import Retinopathy_model
# from IPython.core import history
from keras import callbacks
epochs=20
''' 
# from keras.callbacks import ModelCheckpoint
# checkpoint=ModelCheckpoint("pretraind_retino.h5",monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list=[checkpoint]
'''

epochs=20

import scipy
history = Retinopathy_model.fit_generator (  generator = train_generator,
                                   steps_per_epoch=train_generator.n//train_generator.batch_size,
                                   epochs=epochs,
                                   validation_data = validation_generator,
                                   validation_steps = validation_generator.n//validation_generator.batch_size,
                                   # callbacks=callbacks_list
                                )
#
Retinopathy_model.save("RetinopathyModel_transfer.h5")