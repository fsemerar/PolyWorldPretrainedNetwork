from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import json


PROJECT_PATH = "/Users/federico.semeraro/Documents/PolyWorldPretrainedNetwork"
IMAGES_DIRECTORY = os.path.join(PROJECT_PATH, "data", "test")
ANNOTATIONS_PATH = os.path.join(IMAGES_DIRECTORY, "annotation-test.json")
PREDICTIONS_PATH = os.path.join(PROJECT_PATH, "out", "predictions.json")

predictions = json.loads(open(PREDICTIONS_PATH).read())

coco = COCO(ANNOTATIONS_PATH)

category_ids = coco.loadCats(coco.getCatIds())
image_ids = coco.getImgIds(catIds=coco.getCatIds())

img = coco.loadImgs(image_ids[0])[0]
image_path = os.path.join(IMAGES_DIRECTORY, img["file_name"])
I = io.imread(image_path)
plt.imshow(I)

coco.showAnns(predictions, draw_bbox=False)
plt.show()


# convert to raster
# plt.imshow(I)
# for prediction in predictions:
#     rle = cocomask.frPyObjects(prediction['segmentation'], img['height'], img['width'])
#     m = cocomask.decode(rle)
#     m = m.reshape((img['height'], img['width']))
#     masked_data = np.ma.masked_where(m < 0.5, m)
#     plt.imshow(masked_data)
# plt.show()
