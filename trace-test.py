import torch.nn as nn
import torch
import math
from src.model import EfficientDet
from src.utils import BBoxTransform, ClipBoxes, Anchors
from src.loss import FocalLoss
from torchvision.ops.boxes import nms as nms_torch
import os
print("Torch version: ", torch.__version__)

# from tensorboardX import SummaryWriter
# def count_parameters(model): 
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# modell = EfficientDet(num_classes=80)
# print (count_parameters(modell))

print ("\nPerform trace test")

## Load test data
from torchvision import transforms
""" from src.dataset import CocoDataset, Resizer, Normalizer
 """
## Load model
PATH = "/home/ljosefs/Desktop/EffDet/efficientdet/trained_models/signatrix_efficientdet_coco.pth"

model = EfficientDet(num_classes=80)
state_dict = torch.load(PATH).module.state_dict()

## convert state dictionary to cpu tensors
for k, v in state_dict.items():
  state_dict[k] = v.cpu()

model.cpu()
# model.cuda()
model.load_state_dict(state_dict)


## Save CPU model
cpu_dict_path = "/home/ljosefs/Desktop/EffDet/efficientdet/trained_models/state_dict_cpu_effdet.pth"
cpu_model_path = "/home/ljosefs/Desktop/EffDet/efficientdet/trained_models/cpu_effdet.pth"
gpu_model_path = "/home/ljosefs/Desktop/EffDet/efficientdet/trained_models/signatrix_efficientdet_coco.pth"
no_numpy_cpu_model_path = "/home/ljosefs/Desktop/EffDet/efficientdet/trained_models/no_numpy_cpu_effdet.pth"

""" torch.save(model.state_dict(), cpu_dict_path)
torch.save(model, cpu_model_path) """

## load CPU model
model = torch.load(no_numpy_cpu_model_path)
model.eval()

print("Successfully loaded dataparallel model: ", isinstance(model,nn.DataParallel))


## load CoCo data
""" dataset = CocoDataset(root_dir="/home/ljosefs/Desktop/EffDet/efficientdet/data/COCO", set='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
data = dataset[0]
scale = data['scale'] """

# print(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0).size()) returns: torch.Size([1, 3, 512, 512])

# print(data['img'])
# with torch.no_grad():
#     scores, labels, boxes = model(data['img'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
#     boxes /= scale
# # print(scores)
## Trace test for just-in-time compilation

## random input tensor
input = torch.randn([1, 3, 512, 512])

# print(model(input.to('cuda:0')))
#print(model(input))

## trace checking
check_inputs = [(torch.randn([1, 3, 512, 512])).to('cpu'), (torch.randn([1, 3, 512, 512])).to('cpu')]

## trace test
#traced_module = torch.jit.trace(model, input.to('cpu'))
traced_module = torch.jit.trace(model, input.to('cpu'), check_inputs=check_inputs)
##print(traced_module.code)