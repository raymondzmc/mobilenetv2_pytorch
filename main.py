import torch
from network import MobileNetv2
import importlib, pdb

if __name__ == "__main__":
    cfg = importlib.import_module('config')
    net = MobileNetv2(cfg)
    net.eval()
    test = torch.Tensor(3, 224, 224).unsqueeze(0)
    out = net(test)

    saved = torch.load('mobilenetv2_pretrained.pth')
    weights = net.state_dict()

    
    
    _dict = {
        'features.0.0.weight': 'features.0.0.weight',
        'features.0.1.weight': 'features.0.1.weight',
        'features.0.1.bias': 'features.0.1.bias',
        'features.0.1.running_mean': 'features.0.1.running_mean',
        'features.0.1.running_var': 'features.0.1.running_var',
        'features.0.1.num_batches_tracked': 'features.0.1.num_batches_tracked',

         'features.1.block.0.weight': 'features.1.conv.0.weight',
         'features.1.block.1.weight': 'features.1.conv.1.weight',
         'features.1.block.1.bias': 'features.1.conv.1.bias',
         'features.1.block.1.running_mean': 'features.1.conv.1.running_mean',
         'features.1.block.1.running_var': 'features.1.conv.1.running_var',
         'features.1.block.1.num_batches_tracked': 'features.1.conv.1.num_batches_tracked',

         'features.1.block.3.weight': 'features.2.block.0.weight',
         'features.1.block.4.weight': 'features.2.block.1.weight',
         'features.1.block.4.bias': 'features.2.block.1.bias',
         'features.1.block.4.running_mean': 'features.2.block.1.running_mean',
         'features.1.block.4.running_var': 'features.2.block.1.running_var',
         'features.1.block.4.num_batches_tracked': 'features.2.block.1.num_batches_tracked',
          
         'features.1.block.6.weight': 'features.2.block.3.weight',
         'features.1.block.7.weight': 'features.2.block.4.weight',
         'features.1.block.7.bias': 'features.2.block.4.bias',
         'features.1.block.7.running_mean': 'features.2.block.4.running_mean',    
         'features.1.block.7.running_var': 'features.2.block.4.running_var',   
         'features.1.block.7.num_batches_tracked': 'features.2.block.4.num_batches_tracked',

          'features.2.block.0.weight': 'features.2.conv.3.weight',
         'features.2.block.1.weight': 'features.2.conv.4.weight',
         'features.2.block.1.bias': 'features.2.conv.4.bias',
         'features.2.block.1.running_mean': 'features.2.conv.4.running_mean',
         'features.2.block.1.running_var': 'features.2.conv.4.running_var',
         'features.2.block.1.num_batches_tracked': 'features.2.conv.4.num_batches_tracked',
         'features.2.block.3.weight': 'features.2.conv.6.weight',
         'features.2.block.4.weight': 'features.2.conv.7.weight',
         'features.2.block.4.bias': 'features.2.conv.7.bias',
         'features.2.block.4.running_mean': 'features.2.conv.7.running_mean',
         'features.2.block.4.running_var': 'features.2.conv.7.running_var',
         'features.2.block.4.num_batches_tracked': 'features.2.conv.7.num_batches_tracked',
    }
    for key, val in _dict.items():
        print(weights[key].size(), saved[val].size())
        if weights[key].size() != saved[val].size():
            pdb.set_trace()
    pdb.set_trace()

