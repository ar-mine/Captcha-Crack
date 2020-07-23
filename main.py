import myCaptcha as cha
import im2rec as i2r
import numpy as np
import random
import os
import SSD as ssd
import armine as am
import cv2 as cv
import config

if __name__ == "__main__":
    # '''******************************  Step-1 Generate images and labels  *********************************'''
    # # img_dir = "./asset/imgset/"
    # # save_dir = "./"
    # # file_prefix = "test"
    img_dir = config.img_dir
    save_dir = config.save_dir
    file_prefix = config.file_prefix

    cap = cha.MyCaptcha(width=config.img_w, height=config.img_h, normalized=True)
    # 四字符
    for i in range(25000):
        word = ""
        for j in range(4):
            word += cap.dictset[random.randint(0, config.num_classes)]
        cap.write(word, img_dir+'%d.png' % (i))

    poslist = np.array(cap.poslist)


    '''******************************  Step-2 Compress images set to RecordIO file  ***********************'''
    rec = i2r.Im2rec(img_path=img_dir, save_path=save_dir, fname=file_prefix)
    rec.save_file(poslist)

    '''*******************************  Step-3 Load the RecordIO file  ************************************'''

    # batch_size = 4
    # train_iter = am.load_data_test(batch_size, save_dir, file_prefix)
    # batch = train_iter.next()
    # print(batch.data[0].shape, batch.label[0].shape)
    #
    # batch = train_iter.next()
    # img = batch.data[0][0].transpose((1, 2, 0)).asnumpy().astype(np.uint8)
    # cv.imshow('test', img)
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''