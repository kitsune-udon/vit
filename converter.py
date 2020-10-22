import os
import glob
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, Resize


def convert(target_dir):
    it = glob.iglob(os.path.join(target_dir, '**', '*.JPEG'), recursive=True)

    transform = Compose([
        Resize(224),
        CenterCrop(224)
    ])

    for filepath in it:
        print(filepath)
        img = Image.open(filepath).convert('RGB')
        img = transform(img)
        img.save(filepath)


convert('/imagenet/ILSVRC2012_img_val')
convert('/imagenet/ILSVRC2012_img_train')
