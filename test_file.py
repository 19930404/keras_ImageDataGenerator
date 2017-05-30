# coding: utf-8

import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
 


HOME_DIR = "/home/kobayashi/keras/examples/pp"
AFTER_DIR = "/home/kobayashi/keras/examples/pp2"


if __name__ == '__main__':

    # ディレクトリを作成
    result_dir = "{0}".format(AFTER_DIR)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 入力画像ファイルの格納ディレクトリを指定
    input_dir = '{0}'.format(HOME_DIR)


    # 入力画像ディレクトリ内のファイル一覧を取得
    filenames = os.listdir(input_dir)


    # 100度回転するジェネレータを生成
    #datagen = ImageDataGenerator(width_shift_range=0.2)
    #datagen = ImageDataGenerator(height_shift_range=0.2)
    datagen = ImageDataGenerator(shear_range=0.7)
#pi/4=0.78
#sample only disease shear 0.7 
    '''
    1 0.5 
    2 0.7
    3 0.8
    4 0.9
    5 0.4
    6 0.3
    7 0.2
    datagen = ImageDataGenerator(ertical_flip=True)
    '''
    
    for filename in filenames:
        print filename
        
        # ファイルを読み込み
        img = load_img("{0}/{1}".format(input_dir, filename))
               
        # ファイルを配列に格納
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # ファイルを保存
        gen = datagen.flow(x, batch_size=1, save_to_dir=result_dir, save_prefix='img', save_format='jpg')
        gen.next()

'''
    # print(x.shape)
    画像を指定角度の範囲でランダムに回転する。
    #datagen = ImageDataGenerator(channel_shift_range=100)
    #draw_images(datagen, x, "result_rotation.jpg")
    画像を水平方向にランダムに移動する。
    #datagen = ImageDataGenerator(width_shift_range=0.2)
    #draw_images(datagen, x, "result_width_shift.jpg")
    画像垂直方向にランダムに移動する。
    #datagen = ImageDataGenerator(height_shift_range=0.2)
    #draw_images(datagen, x, "result_height_shift.jpg")
　　シアー変換
    #datagen = ImageDataGenerator(shear_range=0.78)  # pi/4
    #draw_images(datagen, x, "result_shear.jpg")
    画像をランダムにズームする。
    datagen = ImageDataGenerator(zoom_range=0.5)
    draw_images(datagen, x, "result_zoom.jpg"
    画像のチャンネルをランダムに移動する。RGBの値をランダムに加えるのかな？要調査。
    #datagen = ImageDataGenerator(channel_shift_range=200)
    #draw_images(datagen, x, "result_channel_shift.jpg")
    画像を水平方向にランダムに反転する。
    #datagen = ImageDataGenerator(horizontal_flip=True)
    #draw_images(datagen, x, "result_horizontal_flip.jpg")
    画像を垂直方向にランダムに反転する。
    #datagen = ImageDataGenerator(vertical_flip=True)
    #draw_images(datagen, x, "result_vertical_flip.jpg")
    サンプル平均を0にする。平均の正規化なんだろうけど平均0にしてimshow()しても大丈夫なのかな？imshow()でとりあえず画像出たけどあとで仕様を確認。
    #datagen = ImageDataGenerator(samplewise_center=True)
    #draw_images(datagen, x, "result_samplewise_center.jpg")
    サンプルを標準偏差で割る。画像はなんか真っ黒になっちゃった。
    上のと組み合わせると平均0、標準偏差1になる正規化をかけられるようだ。これもアルゴリズムを要確認。
    #datagen = ImageDataGenerator(samplewise_std_normalization=True)
    #draw_images(datagen, x, "result_samplewise_std_normalization.jpg")

'''

