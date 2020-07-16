import myCaptcha as cha
import im2rec as i2r
import numpy as np
import SSD as ssd
import armine as am
import cv2 as cv

if __name__ == "__main__":
    '''******************************  Step-1 Generate images and labels  *********************************'''
    img_dir = "./asset/imgset/"

    cap = cha.MyCaptcha()
    for i in range(5):
        cap.write('1234', img_dir+'%s.png' % i)
    poslist = np.array(cap.poslist)

    '''******************************  Step-2 Compress images set to RecordIO file  ***********************'''
    rec = i2r.Im2rec(img_path=img_dir, save_path="./", fname="test")
    rec.save_file(poslist)

    '''*******************************  Step-3 Load the RecordIO file  ************************************'''
    batch_size = 4
    train_iter = ssd.load_data_pikachu(batch_size)
    batch = train_iter.next()
    print(batch.data[0].shape, batch.label[0].shape)

    label_test = batch.label[0].reshape((batch.label[0].shape[0], int(batch.label[0].shape[2]/5), 5))
    imgs = (batch.data[0][0:4].transpose(0, 3, 2, 1))
    show = am.cv_multishow([imgs[0]], 1, 1)
    am.cv_rectangle_normalized(show, label_test[0, :, 1:])
    cv.imshow('img', show)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''
    '''*****************************************************************************************'''