from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import Image
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa



def get_augmented_image(image, mask, bbox, i):
    """ Generate image from the source image. 
    Bbox and mask will be generated accordingly.
    See also: https://github.com/aleju/imgaug.git.
    Args: ndarray
        image: ndarray, (512, 512, 3)
            An image to generate new image
        mask: ndarray, (512, 512, 1)
            mask of the image
        bbox: list, [x1, y1, w, h]
            boundary boxes of the image
    Returns: 
        aug_img: ndarray, (512, 512, 3)
        mask_aug: ndaarray, (512, 512, 1)
        bbox_aug: list, [x1, y1, w, h]

    """
    if i==1: mode = iaa.AdditiveGaussianNoise(scale=(0.14, 0.2*255))
    elif i==2: mode = iaa.SaltAndPepper(0.17) # Replace 17% of all pixels with salt and pepper noise
    elif i==3: mode = iaa.Affine(shear=(-20, 20)) # Shear images by -20 to 20 degrees
    elif i==4: mode = iaa.PerspectiveTransform(scale=(0.9, 0.13)) # Apply perspective transformations using a random scale
    elif i==5: mode = iaa.ElasticTransformation(alpha=(0.6, 0.9), sigma=1.0) # Distort images locally by moving individual pixels
    elif i==6: mode = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
    elif i==7: mode = iaa.Sharpen(alpha=(0.25, 0.75), lightness=(0.75, 1.5))
    elif i==8: mode = iaa.Superpixels(p_replace=0.3, n_segments=144)
    elif i==9: mode = iaa.Rot90((1, 3))
    elif i==10: mode = iaa.Fliplr(1)

    bbs = [ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])]

    seq = iaa.Sequential([mode])

    image_aug, mask_aug, bbox_aug = seq(images=[image], segmentation_maps=[mask.reshape(512, 512)], bounding_boxes=bbs)
    width = bbox_aug[0].x2 - bbox_aug[0].x1
    height = bbox_aug[0].y2 - bbox_aug[0].y1

    aug_img = np.array(image_aug[0])
    mask_aug = mask_aug[0].reshape(512, 512, 1)
    bbox_aug = [bbox_aug[0].x1, bbox_aug[0].y1, width, height]

    return  aug_img, mask_aug, bbox_aug
