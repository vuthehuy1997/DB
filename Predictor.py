#!python3
import argparse
import os
import torch
import cv2
import numpy as np
import sys
import math

from experiment import Structure, Experiment
from concern.config import Configurable, Config, State

class Predictor():
    def __init__(self, config):
        cwd = os.getcwd()
        
        cmd = dict()
        cmd['exp'] = os.path.join(cwd, config['exp'])
        # cmd['resume'] = 'model/final_totaltext_50_sanh_train_mix'
        cmd['resume'] = os.path.join(cwd, config['weight_path'])
        cmd['box_thresh'] = 0.5
        # cmd['thresh'] = 0.5
        cmd['polygon'] = False
        cmd['visualize'] = True
        cmd['image_short_side'] = 736

        os.chdir(os.path.join(cwd, 'modules/DB'))
        conf = Config()
        args = conf.compile(conf.load(cmd['exp']))['Experiment']
        # print(args)
        args.update(cmd=cmd)
        experiment = Configurable.construct_class_from_config(args)
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        # experiment.load('evaluation', **args)
        self.args = cmd
        # model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

        self.init_torch_tensor(config['device'])
        self.model = self.init_model()
        self.resume(self.model, self.model_path)
        self.model.eval()
        print('hi')

        os.chdir(cwd)


    def init_torch_tensor(self, device):
        # Use gpu or not
        self.device = torch.device(device)
        if 'cuda' in device:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
    
    def preprocess_image(self, image):
        img = image.astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def predict(self, image):
        '''
        input: image (format: cv2 numpy)
        output: list box (format: list of (x1,y1,x2,y2,x3,y3,x4,y4,score))

        '''
        batch = dict()
        img, original_shape = self.preprocess_image(image)
        # print('image shape: ', img.shape)
        # print('original_shape: ', original_shape)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = self.model.forward(batch, training=False)
            # print('pred: ', pred.shape)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 

            batch_boxes, batch_scores = output
            results = []
            for index in range(batch['image'].size(0)):
                original_shape = batch['shape'][index]
                boxes = batch_boxes[index]
                scores = batch_scores[index]
                if self.args['polygon']:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        # result = [int(x) for x in box]
                        result = [[int(box[x]), int(box[x+1])] for x in range(0,len(box),2)]
                        score = scores[i]
                        result.append(float(score))
                        results.append(result)
                else:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        # result = [int(x) for x in box]
                        result = [[int(box[x]), int(box[x+1])] for x in range(0,len(box),2)]
                        result.append(float(score))
                        results.append(result)
        return results
    
    def min_area(self, rect): # b is rectangle
        # print('rect: ', rect)
        box = rect
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        x1, x2, x3, x4 = box
        d1 = (x1[0]-x2[0])*(x1[0]-x2[0])+(x1[1]-x2[1])*(x1[1]-x2[1]) # first left
        d2 = (x1[0]-x4[0])*(x1[0]-x4[0])+(x1[1]-x4[1])*(x1[1]-x4[1]) # last left
        # if d1 > d2:
        #     box = np.array([x2, x3, x4, x1])
        # else:
        box = np.array([x1, x2, x3, x4])

        x1, x2, x3, x4 = box
        #top
        x2n = [0,0]
        x3n = [0,0]
        x2n[0] = x2[0]+int((x2[0]-x1[0])/20)
        x2n[1] = x2[1]+int((x2[1]-x1[1])/20)
        x3n[0] = x3[0]+int((x3[0]-x4[0])/20)
        x3n[1] = x3[1]+int((x3[1]-x4[1])/20)

        # bot
        x1n = [0,0]
        x4n = [0,0]
        x1n[0] = x1[0]+int((x1[0]-x2[0])/20)
        x1n[1] = x1[1]+int((x1[1]-x2[1])/20)
        x4n[0] = x4[0]+int((x4[0]-x3[0])/20)
        x4n[1] = x4[1]+int((x4[1]-x3[1])/20)

        box = np.array([x1n, x2n, x3n, x4n])
        # ar = Polygon(box)
        # return box, ar.area
        return box

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        
    def inference(self, image_path, visualize=False):
        
        model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        print('image shape: ', img.shape)
        print('original_shape: ', original_shape)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = model.forward(batch, training=False)
            print('pred: ', pred.shape)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            # print('output: ', output)
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)
                print('finished')
if __name__ == '__main__':
    main()
