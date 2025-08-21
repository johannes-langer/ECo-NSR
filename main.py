from src import log
from src.data import generate_raven
from src.segmentation import segment

if __name__ == "__main__":
    log.init()

    generate_raven.generate_raven()
    generate_raven.generate_coco_json()

    segment.train_segmentation_model()