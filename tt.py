import torch

state_dict = torch.load('x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    print(k)
    name = k.replace('backbone.','')
    print(name)
    new_state_dict[name] = v

torch.save(new_state_dict, 'modified_x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth')

