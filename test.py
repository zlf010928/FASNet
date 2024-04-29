from net import Deeplab50_bn
import torch 
import SAN
model =  ResNetMulti(Bottleneck, [3, 4, 23, 3], 2)

output=torch.randn(2,3,512,512)
temp=model(output)
san=SAN(2)
t=san(temp[0])
print(t.shape)