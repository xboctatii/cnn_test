from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, Conv2DTranspose
from utilz import img_generator, make_file_list
from PIL import Image
import numpy as np

def get_model(img_size = 64, start_filters = 64):
    filt = start_filters
    
    inputs = Input(shape=(img_size, img_size, 3), name='input') #32x32x3
    
    enc_1 = Conv2D(filt, kernel_size = (3,3), strides=(2,2), padding='same')(inputs) #16x16
    enorm_1 = BatchNormalization()(enc_1)
    eact_1 = Activation('relu')(enorm_1)
        
    dec_1 = Conv2DTranspose(filt, kernel_size = (3,3), strides=(2,2), padding='same', use_bias = False)(eact_1)
    dnorm_1 = BatchNormalization(axis = 1)(dec_1)
    dact_1 = Activation('relu')(dnorm_1)
    
    dec_2 = Conv2D(3, kernel_size = (3,3), strides=(1,1), padding='same')(dact_1)
    dact_2 = Activation('tanh')(dec_2)
    
    outputs = (dact_2)
    return keras.Model(inputs,outputs)

model = get_model()
model.compile(loss = 'mse', optimizer='Adam', metrics=['accuracy'])
model.summary()

#model.fit(img_generator(img_path = 'D:\\vidz\\neuro\\', batch_size=6), epochs=5, steps_per_epoch = 50)
def load_weights():
    model.load_weights('D:\\vidz\\Projects\\test_cnn.h5')

def test_images(in_path, out_path):
    file_list = make_file_list(in_path, file_names = ['jpg', 'bmp'])
    test_mass=[]
    
    for img in file_list:
        img_x = Image.open(in_path + img)
        arr_x = np.array(img_x)
        test_mass.append(arr_x)
    test_set = np.stack(test_mass, axis=0)
    test_set = test_set.astype('float32') / 127.5 - 1  
    result = model.predict(test_set)
    result = result * 0.5 + 0.5 #нормализуем изображение

    num = 0
    for img in result:
        out = img*255
        out = out.astype(np.uint8)
        out = Image.fromarray(out)
        out.save(out_path + '0' * (8 - len(str(num))) + str(num) + '.jpg', quality=100)
        num = num + 1