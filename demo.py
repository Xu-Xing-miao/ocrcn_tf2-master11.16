"""
inference on a single Chinese character
image and recognition the meaning of it
"""
import os
import cv2
import sys
import numpy as np
import tensorflow as tf
import glob
from alfred.dl.tf.common import mute_tf
mute_tf()
from alfred.utils.log import logger as logging
from dataset.casia_hwdb import load_ds, load_characters, load_val_ds
from models.cnn_net import CNNNet, build_net_002, build_net_003



target_size = 64
characters = load_characters()
num_classes = len(characters)
#use_keras_fit = False
use_keras_fit = True
ckpt_path = './checkpoints/cn_ocr-{epoch}.ckpt'


def show(name, src):
    """
        展示图片 show
        :param name:图片
        :param src:宽度
        :return:展示的图片
    """
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):  #
    """
    使用插值方法对图片 resize
    :param image:图片
    :param width:宽度
    :param height:高度
    :param inter:插值方法
    :return:调整大小后的图片
    """
    dim = None
    (h, w) = image.shape[:2]  #
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # print(dim)
    resized = cv2.resize(image, dim, interpolation=inter)  # interpo;ation为插值方法，这里选用的是
    return resized


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    # original is 64x64, add a channel dim
    x['image'] = tf.expand_dims(x['image'], axis=-1)
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] = (x['image'] - 128.) / 128.
    return x['image'], x['label']


def get_model():
    # init model
    model = build_net_003((64, 64, 1), num_classes)
    logging.info('model loaded.')

    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {} at epoch: {}'.format(latest_ckpt, start_epoch))
        return model
    else:
        logging.error('can not found any checkpoints matched: {}'.format(ckpt_path))


def predict(model, img_f, is_processing):
    ori_img = cv2.imread(img_f)
    show("r", ori_img)
    img = tf.expand_dims(ori_img[:, :, 0], axis=-1)
    img = tf.image.resize(img, (target_size, target_size))
    img = (img - 128.)/128.
    img = tf.expand_dims(img, axis=0)
    print(img.shape)
    out = model(img).numpy()
    print('{}'.format(characters[np.argmax(out[0])]))
    name = '{}'.format(characters[np.argmax(out[0])])
    # 采用ASCll码代替无法创建文件的符号
    if name == '\\' or name == '/' or name == ':' or name == '*' or name == '?' \
            or name == '"' or name == '<' or name == '>' or name == '|':
        name = str(ord(name))
    pre_name = "pred_" + name + ".png"

    # support data
    path = 'assets/results/'
    if not os.path.exists(path):
        os.makedirs(path)
    if not is_processing:
        cv2.imencode('.png', ori_img)[1].tofile(path + pre_name)
    else:
        path = 'assets/results/pending/'
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imencode('.png', ori_img)[1].tofile(path + pre_name)


def get_file_name(filename):
    (filepath, tempFileName) = os.path.split(filename)
    (shotName, extension) = os.path.splitext(tempFileName)
    print("name", shotName)
    return shotName


def data_preprocess(img_file, processing):
    # 预处理后的文件路径
    after_process = []
    for imgF in img_file:
        print("imgF", imgF)
        # 得到目录下的文件名
        name = get_file_name(imgF)
        img = cv2.imread(imgF)
        img = resize(img, 65, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
        # show("thresh", thresh)
        # 预处理后的图片路径
        path = "assets/after_pending/" + str(name) + ".png"
        print("path", path)
        cv2.imwrite(path, thresh)
        after_process.append(path)
    print("after_process", after_process)

    return after_process


if __name__ == '__main__':
    img_files = glob.glob('assets/data/*.png')
    print("img_files", img_files)
    is_processing = True
    if is_processing:
        img_file = glob.glob('assets/Pending_pictures/*.png')
        after_processing = data_preprocess(img_file, is_processing)
        after_process = glob.glob('assets/after_pending/*.png')
        model = get_model()
        for img_f in after_process:
            a = cv2.imread(img_f)
            predict(model, img_f, is_processing)

    else:
        model = get_model()
        for img_f in img_files:
            a = cv2.imread(img_f)
            predict(model, img_f, is_processing)




