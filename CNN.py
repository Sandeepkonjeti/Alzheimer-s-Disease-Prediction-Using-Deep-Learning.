import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D,BatchNormalization,Dropout,Rescaling,Flatten,Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D,Activation
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(176,176,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4,activation = 'softmax'))


model.compile(optimizer='adam',
loss=tf.losses.CategoricalCrossentropy(),
metrics=[keras.metrics.AUC(name='auc'),'acc'])


model.summary()

train_data_dir=r"C:\Users\VIJAYSHANKAR\OneDrive\Desktop\alzheimer\Alzheimer_s Dataset\train"
test_data_dir=r"C:\Users\VIJAYSHANKAR\OneDrive\Desktop\alzheimer\Alzheimer_s Dataset\test"

datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)

training_set = datagen.flow_from_directory(train_data_dir,
                                           target_size=(176,176),
                                           batch_size=32,
                                           class_mode='categorical',
                                           subset='training',
                                           shuffle=True)

validation_set = datagen.flow_from_directory(train_data_dir,
                                            target_size=(176,176),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=False,
                                            subset="validation"
                                            )
datagen = ImageDataGenerator(rescale=1./255)

test_set = datagen.flow_from_directory(test_data_dir,
                                         shuffle=False,
                                         target_size=(176,176),
                                         class_mode='categorical',
                                         batch_size = 32)

history=model.fit(training_set,steps_per_epoch=len(training_set),epochs = 50 ,validation_data = validation_set,validation_steps=len(validation_set),verbose=2)

model.save(r"model\cnn5-3.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['loss'],label='training loss',color='green')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(r"model\cnn_loss5-3.png")
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['acc'],label='training accuracy',color='green')
plt.plot(history.history['val_acc'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r"model\cnn_acc5-3.png")
plt.show()


plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['auc'],label='training AUC',color='green')
plt.plot(history.history['val_auc'],label='validation AUC')
plt.xlabel('# epochs')
plt.ylabel('AUC')
plt.legend()
plt.savefig(r"model\cnn_AUC5-3.png")
plt.show()

score=model.evaluate(test_set,verbose=1)

print(score)

# loss,auc,acc [0.06806784123182297, 0.9973264932632446, 0.9820312261581421]
