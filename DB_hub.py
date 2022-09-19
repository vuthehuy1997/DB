import torch
import yaml
import cv2

class DB:
    def __init__(self, args):
        self.model = torch.hub.load('modules/DB', 'DB', source='local',  pretrained=False, args=args)

    def predict(self, img):
        bboxes = self.model.predict(img)
        return bboxes


if __name__ == '__main__':
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    model = DB(config['db'])
    img = cv2.imread('/data/publication_safety/ocr_trc/data/demo/000_001.jpg')
    print(model.predict(img))
    print(model)
    # while True:
    #     i = 1
