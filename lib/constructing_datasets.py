import numpy as np
from mrcnn import utils
from mrcnn.config import Config
import json
import os
import time
import numpy as np
from PIL import Image, ImageDraw
from lib.augmentation import *
from lib.utils import *
import pandas as pd
import progressbar


class buttsDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name'][:-4] + '.jpg'
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations)


    def start_augmentation(self, save_dir, ratio=10):
        """ Generate augmented images for each image in dataset
        and include them into this dataset. Use after load_data().
        Args:
            save_dir: Directory to save new images
            ratio: Number of images to generate per source image.
        """
        # print('Augmentation started...')
        bar = progressbar.ProgressBar().start() # Progress bar
        annotations = {"annotations": [], "categories": [{"supercategory": "litter", "id": 1, "name": "cig_butt"}], "images": []}
        number_of_new_images = 0
        number_of_original_images = len(self.image_ids)
        for img_id in self.image_ids:
            bar.update(img_id)
            img = self.load_image(img_id)
            mask = self.load_mask(img_id)[0]
            bbox = self.image_info[img_id]['annotations'][0]['bbox']
            original_image_path = os.path.abspath(os.path.join(save_dir, 'augmented', f'{img_id:08}' + '.jpg'))
            Image.fromarray(img.astype(np.uint8)).save(original_image_path)
            for i in range(1, ratio + 1):
                # Generate new image
                img_aug, mask_aug, bbox_aug = get_augmented_image(img, mask, bbox, i)

                new_id = number_of_original_images + number_of_new_images
                number_of_new_images += 1
                image_path = os.path.abspath(os.path.join(save_dir, 'augmented', str(new_id) + '.jpg'))
                Image.fromarray(img_aug.astype(np.uint8)).save(image_path)

                # Create annotation for the image
                image_info = self.image_info[img_id]
                image_annotations = image_info ['annotations'][0].copy()
                image_annotations['id'] = new_id
                image_annotations['image_id'] = new_id               
                image_annotations['iscrowd'] = 1
                image_annotations['bbox'] = bbox_aug
                image_annotations['segmentation'] = [encode_rle(mask_aug)]
                image_width = image_info['width']
                image_height = image_info['height']

                # Annotations of generated images
                annotations['annotations'].append(image_annotations)
                annotations['images'].append({"file_name": str(new_id) + ".jpg",
                                              "width": int(image_width), 
                                              "height": int(image_height),
                                              "id": int(new_id)})

            # Annotations of original images
            annotations['annotations'].append(self.image_info[img_id]['annotations'][0])
            annotations['images'].append({"file_name": f"{img_id:08}" + ".jpg",
                                          "width": int(image_width), 
                                          "height": int(image_height),
                                          "id": int(img_id)})

        # Save annotations of all images in json file
        with open(os.path.abspath(os.path.join(save_dir, 'coco_annotations_augmented.json')), 'w') as json_file:
            print(json.dumps(annotations), file=json_file)

        bar.finish()
        print(f'Augmentation ended. Check {save_dir} directory.')   


    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        # Augmented image annotations was written in RLE format
        # and 'iscrowd' == 1 for them
        for annotation in annotations:
            class_id = annotation['category_id']
            if annotation['iscrowd'] == 0:
                mask = Image.new('1', (image_info['width'], image_info['height']))
                mask_draw = ImageDraw.ImageDraw(mask, '1')
                for segmentation in annotation['segmentation']:
                    mask_draw.polygon(segmentation, fill=1)
                    bool_array = np.array(mask) > 0
                    instance_masks.append(bool_array)
                    class_ids.append(class_id)
            if annotation['iscrowd'] == 1:
                mask = decode_rle(annotation['segmentation'][0])
                bool_array = mask > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)

        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids
