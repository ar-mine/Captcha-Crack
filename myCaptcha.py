from captcha.image import ImageCaptcha
import os
import random
from PIL import Image
from PIL.ImageDraw import Draw
import cv2 as cv
import armine as am
import numpy as np

table = []
for i in range(256):
    table.append(i * 1.97)

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'asset', 'font')
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono.ttf')]


class MyCaptcha(ImageCaptcha):
    def __init__(self, width=160, height=60, fonts=None, font_sizes=None, normalized=False):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []
        self.poslist = []
        self.normalized = normalized

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)
        global getposlist

        poslist = []
        flaglist = []

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        for c in chars:
            if random.random() > 0.5:
                images.append(_draw_character(" "))
                flaglist.append(0)
            images.append(_draw_character(c))
            flaglist.append(1)

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        str_count = 0
        for i, im in enumerate(images):
            w, h = im.size
            y = int((self._height - h) / 2)
            if(flaglist[i]):
                test = chars[str_count]
                poslist.append([int(chars[str_count]), offset, y, w+offset, h+y])
                str_count += 1
            mask = im.convert('L').point(table)
            image.paste(im, (offset, y), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))
            divtemp = width / self._width
            for l in poslist:
                l[1] = int(l[1] / divtemp)
                l[3] = int(l[3] / divtemp)
        if self.normalized:
            for l in poslist:
                l[1], l[3] = l[1]/self._width, l[3]/self._width
                l[2], l[4] = l[2]/self._height, l[4]/self._height
        self.poslist.append(poslist)

        return image

if __name__ == "__main__":
    img_dir = "./asset/imgset/"
    image = MyCaptcha(width=256, height=256, normalized=True)
    # for i in range(10):
    #     data = image.generate('1234')
    #     image.write('1234', img_dir+'%s.png' % i)
    # data = image.generate('1234')
    image.write('1234', "out.png")
    x = cv.imread("out.png")

    poslist = np.array(image.poslist)
    for ls in poslist:
        for l in ls:
            #cv.rectangle(x, (l[1], l[2]), (l[3], l[4]), (255, 0, 0))
            am.cv_rectangle_normalized(x, ls[:, 1:], normallized=True)
    cv.imshow('image', x)
    cv.waitKey()

