from pytorch_nndct.apis import torch_quantizer
import torch

## Import model
PATH_CPU = "./trained_models/cpu_effdet.pth"
PATH_CUDA = "/home/ljosefs/Desktop/EffDet/efficientdet/trained_models/signatrix_efficientdet_coco.pth"
PATH_NO_NUMPY_CPU = "/home/ljosefs/Desktop/EffDet/efficientdet/trained_models/no_numpy_cpu_effdet.pth"
model = torch.load(PATH_NO_NUMPY_CPU)
# put model in evaluation mode
model.eval()


# Test changing from memoryefficient swish to ordinary swish
model.backbone_net.model.set_swish(memory_efficient=False)

device = torch.device("cpu")
quant_mode = 'calib'

## random input tensor
input = torch.randn([1, 3, 512, 512])

## new api
####################################################################################
quantizer = torch_quantizer(
    quant_mode, model, (input.to(('cpu'))), device=torch.device('cpu'))

quant_model = quantizer.quant_model
#####################################################################################
