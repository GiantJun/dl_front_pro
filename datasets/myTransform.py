import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

class CenterCrop(object):
    def __init__(self, scale=20, p=0.6):
        assert isinstance(p, float) and isinstance(scale, int)
        self.p = p
        self.scale = scale

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            result = img.copy()
            if random.uniform(0,1) < 0.5:
                w, h = result.size
                result = transforms.CenterCrop((h,w-random.randint(1, self.scale)))(result)
            if random.uniform(0,1) < 0.5:
                w, h = result.size
                result = transforms.CenterCrop((h-random.randint(1, self.scale),w))(result)
            return result
        else:
            return img