import os, random
from PIL import Image
import numpy as np

def make_dotes_images(in_path, out_path):
    lst = make_file_list(in_path, file_names = ['jpg'])
    for image in lst:
        img_out = put_dotes(in_path + image)
        img_out.save(out_path + image, quality=100)
    
def put_dotes(img_x):
    img_x = Image.open(img_x)
    img_width, img_height = img_x.size
    img_x = np.array(img_x).astype(np.float32)
    for x in range(8,img_width,8):
        for y in range(8,img_width,8):
            img_x[x,y,:] = 0
            img_x[x,y,0:1] = 255
            img_x[x-1,y,:] = 0
            img_x[x-1,y,0:1] = 255 
    out = img_x.astype(np.uint8)
    return Image.fromarray(out)

def make_img_matrix(img_path):
    img_path = img_path
    lst = make_file_list(img_path, file_names = ['jpg'])
    img_matrix = list(chunked(lst,9))
    return(img_matrix)

def chunked(s, n):
    current = 0
    while True:
        chunk = s[current : current+n]
        current += n
        if chunk:
            yield chunk
        else:
            break
    
def mess_img(img_x, img_y):
    img_x = Image.open(img_x)
    img_y = Image.open(img_y)
    if img_x.size != img_y.size:
        print('Размеры изображений не одинаковы!')
        return
    img_heigh, img_width = img_x.size
    img_x = np.array(img_x).astype(np.float32)
    img_y = np.array(img_y).astype(np.float32)
    
    out = np.zeros([img_heigh, img_width, 3])
    out[0:64,:,:] = img_x[0:64,:,:]
    out[64:,:,:] = (img_y[64:,:,:] + img_x[64:,:,:])/2
    
    out = out.astype(np.uint8)
    img_out = Image.fromarray(out)
    img_out.save('D:\\vidz\\train_set\\test\\out.jpg', quality=100)
    
def img_generator(img_path, batch_size=5):
    
    img_list = make_file_list(img_path + 'in_train\\', file_names = ['jpg'])
    batch_per_epoch = len(img_list)//batch_size
    counter = 0
    
    while True:
        
        if counter >= batch_per_epoch:
            counter = 0
            random.shuffle(img_list)
        
        x_mass = []
        y_mass = []
        for i in range(batch_size):
            img_x = Image.open(img_path + 'in_train\\' + str(img_list[counter * batch_size + i]))
            img_y = Image.open(img_path + 'out_train\\' + str(img_list[counter * batch_size + i]))
            arr_x = np.array(img_x)
            arr_y = np.array(img_y)
            x_mass.append(arr_x)
            y_mass.append(arr_y)
        
        x_train=np.stack(x_mass, axis=0)
        y_train=np.stack(y_mass, axis=0)
        x_train = x_train.astype('float32') / 127.5 - 1
        y_train = y_train.astype('float32') / 127.5 - 1
        counter += 1
        yield x_train, y_train

def make_train_set(in_path, out_path, box_size = 128, step = 2):
    file_list = make_file_list(in_path, file_names = ['jpg', 'bmp'])
    if file_list == None:
        return
    for file in file_list:
        image_splice(in_path + file, 
                     out_path + file[:len(file)-3],
                     box_size,
                     step)

def make_file_list(in_path, file_names):
    in_path = os.path.normpath(in_path) +'\\'
    try:
        files = os.listdir(in_path)
    except:
        print('Неправильно задана входная директория!')
        return
    
    file_list = []
    for file in files:
        ending = file[len(file)-3:].lower()
        if os.path.isfile(in_path + file) == True and ending in file_names:
            file_list.append(file)
    return file_list
    
def image_splice(image_in, image_out, box_size, step):
    img = Image.open(image_in)
    img_heigh, img_width = img.size
    num = 0
    for x in range(0, img_heigh - (box_size // step), box_size // step):
        for y in range(0, img_width - (box_size // step), box_size // step):
            img_crop = img.crop((x, y, x + box_size, y + box_size))
            img_crop.save(fp = image_out + str(num) + '.jpg', quality=100)
            num = num + 1

def crop_image(in_path, x = 0, y = 0, out_path = ''):
    """
    Параметры
    ----------
    in_path : string
        Путь к директории с картинками.
    x : int, optional
        Количество пикселов для обрезки сверху и снизу картинки.
    y : int, optional
        Количество пикселов для обрезки слева и справа картинки.
    out_path : string, optional
        Директория для обработанных картинок. По умолчанию создаём
        новую папку с именем 'исходная директория_cropped'.
    """
    graph_names = ['jpg', 'bmp']
    
    in_path = os.path.normpath(in_path) +'\\'
    if out_path == '':
        out_path = in_path[:-1] + '_cropped'
    out_path = os.path.normpath(out_path) +'\\'
      
    try:
        files = os.listdir(in_path)
    except:
        print('Неправильно задана входная директория!')
        return
    
    try:        
        os.mkdir(out_path)
    except:
        pass
    
    try:
        o_files = os.listdir(out_path)
    except:
        print('Неправильно задана выходная директория!')
        return
    
    file_list = []
    for file in files:
        ending = file[len(file)-3:].lower()
        if os.path.isfile(in_path + file) == True and ending in graph_names:
            file_list.append(file)

    for file in file_list:
        try:
            img_x = Image.open(in_path + file)
            img_x = np.array(img_x)
            img_heigh = img_x.shape[0]
            img_width = img_x.shape[1]
            if x < (img_heigh // 2) and y < (img_width // 2):
                img_res = img_x[x : img_heigh - x, y : (img_width - y)]
                img_out = Image.fromarray(img_res)
                img_out.save(out_path + file, quality=100)
        except Exception as e:
            print(e)
            pass

def renamer(path, prefix = ''):    
    try:
        files = os.listdir(path)
    except:
        print('Неправильно задана директория!')
        return
    
    path = os.path.normpath(path) +'\\'
    file_list = []
    for file in files:
        if os.path.isfile(path + file) == True:
            file_list.append(file)
    
    try:
        os.mkdir(path + 'test' + prefix)
        os.rmdir(path + 'test' + prefix)
    except:
        print('Неправильно задан префикс!')
        return
        
    
    if file_list == []:
        print('Директория пуста!')
        return
        
    start = prefix
    num_prefix = len(str(len(file_list)))
    num = 0
    for file in files:
        src = path + file
        try:
            end_file_i = file.rindex('.')
            end_file = file[end_file_i:]
        except:
            end_file = ''
        prefix = '0' * (num_prefix - len(str(num)))        
        dst = path + start + prefix + str(num) + end_file
        try:
            os.rename(src, dst)
        except Exception as e:
            print('Произошла ошибка!')
            print(e)
            return
        num = num + 1
    print('Файлов переименовано: ' + str(num) + '.')