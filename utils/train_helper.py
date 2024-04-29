import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from graphs.models.deeplab101_IN import DeeplabMulti101_IN

from graphs.models.deeplab50_ClassINW import Res50_ClassINW

from graphs.models.deeplab50_bn import Deeplab50_bn


def get_model(args):

    

    
    model = Res50_ClassINW(args,num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
    params = model.optim_parameters(args)
    

    
    

    return model, params

