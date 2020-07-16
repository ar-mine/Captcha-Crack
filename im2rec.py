import mxnet as mx
import cv2 as cv
import numpy as np
import os



class Im2rec:
    def __init__(self, img_path, save_path, fname):
        self.img_path = img_path
        self.save_path = save_path
        self.fname = fname

    def _label_generate(self, img, label):
        w = img.shape[0]
        h = img.shape[1]
        ret_array = np.zeros((4+label.size, ))
        ret_array[0] = 4
        ret_array[1] = label.size
        ret_array[2], ret_array[3] = w, h
        ret_array[4:] = label.flatten()
        return ret_array

    def save_file(self, labels):
        write_record = mx.recordio.MXIndexedRecordIO(self.save_path+"%s.idx" % self.fname,
                                                     self.save_path+"%s.rec" % self.fname, 'w')
        file_name = os.listdir(self.img_path)
        for i, file in enumerate(file_name):
            img = cv.imread(self.img_path+file)
            img = img.transpose(1, 0, 2)
            label = self._label_generate(img, labels[i])
            header = mx.recordio.IRHeader(flag=0, label=label, id=i, id2=0)
            s = mx.recordio.pack_img(header, img, quality=95, img_fmt=".png")
            #将数据写入到rec文件中
            write_record.write_idx(i, s)
        write_record.close()

    def save_data(self, datas, labels):
        write_record = mx.recordio.MXIndexedRecordIO(self.save_path+"%s.idx" % self.fname,
                                                     self.save_path+"%s.rec" % self.fname, 'w')
        for i, data in enumerate(datas):
            img = data.transpose(1, 0, 2)
            label = np.array([4, 5, img.shape[0], img.shape[1], labels[i]])
            header = mx.recordio.IRHeader(flag=0, label=label, id=i, id2=0)
            s = mx.recordio.pack_img(header, img, quality=95, img_fmt=".png")
            #将数据写入到rec文件中
            write_record.write_idx(i, s)
        write_record.close()

    def read_show(self):
        read_record = mx.recordio.MXIndexedRecordIO("%s.idx" % self.fname,
                                                    "%s.rec" % self.fname, 'r')
        #遍历rec文件
        for i, idx in enumerate(read_record.keys):
            item = read_record.read()
            #解压数据
            header, s = mx.recordio.unpack(item)
            #将图片的bytes数据转换为ndarray
            img = mx.image.imdecode(s).asnumpy()
            cv.imshow('%d'%i, img)
        cv.waitKey(0)


if __name__ == "__main__":
    img_dir = "./asset/imgset/"
    rec = Im2rec(img_path=img_dir, save_path="./", fname="test")
    poslist = np.zeros((10, 5))
    rec.save_file(poslist)
    rec.read_show()