import mxnet as mx
import cv2 as cv
import numpy as np

write_record = mx.recordio.MXIndexedRecordIO("test.idx", "test.rec", 'w')
#读取图片
img_path = "./asset/imgset/"
#将图片数据转为Numpy数组
for i in range(10):
    img = cv.imread(img_path+'%s.png' % i)
    label = np.array([4,5,160,60,i,0.01,0.02,0.03,0.04])
    header = mx.recordio.IRHeader(flag=0,label=label,id=i,id2=0)
    s = mx.recordio.pack_img(header,img,quality=95,img_fmt=".png")
    #将数据写入到rec文件中
    write_record.write_idx(i,s)
write_record.close()

# read_record = mx.recordio.MXIndexedRecordIO("test.idx","test.rec","r")
# #遍历rec文件
# for idx in read_record.keys:
#     item = read_record.read()
#     #解压数据
#     header,s = mx.recordio.unpack(item)
#     #将图片的bytes数据转换为ndarray
#     img = mx.image.imdecode(s).asnumpy()
#     cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
