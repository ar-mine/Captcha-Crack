import cv2 as cv

'''
    This is a library with some functions useful to this project, some are from other open-source and some are 
    created by me. So maybe you can see some familiar code in this library.
'''
img = cv.imread("./asset/catdog.jpg")
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]


def cv_rectangle(img, pos, text=False, color=(255, 0, 0), thickness=2):
    '''  Simplify the opencv function to draw rectangle  '''
    if type(pos[0]) is not list:
        cv.rectangle(img=img, pt1=(pos[0], pos[1]), pt2=(pos[2], pos[3]),
                     color=color, thickness=thickness)
        if text:
            showtxt = "text"
            cv_showtxt(img, showtxt, (pos[0], pos[1]), color)
    else:
        for posi in pos:
            cv.rectangle(img=img, pt1=(posi[0], posi[1]), pt2=(posi[2], posi[3]),
                         color=color, thickness=thickness)
            if text:
                showtxt = "text"
                cv_showtxt(img, showtxt, (posi[0], posi[1]), color)


def cv_showtxt(img, text, org, color, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2):
    cv.putText(img=img, text=text, org=org, fontFace=fontFace,
               fontScale=fontScale, color=color, thickness=thickness)


cv_rectangle(img, [dog_bbox, cat_bbox], True)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
