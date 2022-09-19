import torch
import yaml


with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
model = torch.hub.load('modules/DB', 'DB', source='local',  pretrained=False, args=config['db'])
print(model)