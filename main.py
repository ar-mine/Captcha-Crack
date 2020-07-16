import myCaptcha as cha
import im2rec as i2r
import numpy as np
import SSD as ssd
import armine as am
import cv2 as cv

if __name__ == "__main__":
    '''******************************  Step-1 Generate images and labels  *********************************'''
    # img_dir = "./asset/imgset/"
    # save_dir = "./"
    # file_prefix = "test"
    img_dir = "E:/Dataset/Captcha/img/"
    save_dir = "E:/Dataset/Captcha/rec/"
    file_prefix = "rec_256_256"

    cap = cha.MyCaptcha(width=256, height=256)
    for i in range(1000):
        cap.write('1', img_dir+'%s.png' % i)
    poslist = np.array(cap.poslist)

    '''******************************  Step-2 Compress images set to RecordIO file  ***********************'''
    rec = i2r.Im2rec(img_path=img_dir, save_path=save_dir, fname=file_prefix)
    rec.save_file(poslist)

    '''*******************************  Step-3 Load the RecordIO file  ************************************'''
    '''
    batch_size = 4
    train_iter = ssd.load_data_test(batch_size)
    batch = train_iter.next()
    print(batch.data[0].shape, batch.label[0].shape)

    label_data = batch.label[0]
    label_test = label_data.reshape((label_data.shape[0], int(label_data.shape[2]/5), 5))

    imgs = (batch.data[0][0:].transpose(0, 3, 2, 1))   # 显示的时候必须改成opencv格式数据，其他时候都用w/h的格式
    # show = am.cv_multishow([imgs[0]], 1, 1)
    # show = imgs[0].astype(np.uint8)
    show = np.array(imgs[0], dtype=np.uint8)
    am.cv_rectangle_normalized(show, label_test[0, :, 1:])

    cv.imshow('img', show)
    cv.waitKey(0)
    cv.destroyAllWindows()'''
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''