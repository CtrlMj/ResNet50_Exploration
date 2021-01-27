import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from imgaug import augmenters as iaa
from cv2 import resize
import shutil

class AugTransform:
    def __init__(self, augpipeline):
        self.augpipeline = augpipeline
  
    def __call__(self, images):
        images = images.permute(0, 2, 3, 1)
        images = images.numpy()
        images = self.augpipeline(images=images)
        images = torch.from_numpy(images)
        return images.permute(0, 3, 1, 2)





def init_weights_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_normal_(m.weight)

def init_weights_normal(m, std=1, mean=0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.normal_(m.weight, mean=mean, std=std)

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)



def get_lr(optimizer):
    for param_grp in optimizer.param_groups:
        return param_grp['lr']

def get_activation(name):
    def hook(module, input, output):
        activation[name] = output
    return hook
    


def save_checkpoint(state_dict, is_best, path):
    checkpoint_path = path + '/checkpoint.pt'
    torch.save(state_dict, checkpoint_path)
    if is_best:
        best_path = path + "/best_state.pt"
        shutil.copyfile(checkpoint_path, best_path)
        

def aggrAugs(*batches, approach='mean'):
    predictions = 0
    if approach == 'mean':
        for batch in batches:
            predictions += batch
        return predictions / len(batches)
    elif approach == 'max':
        predictions = torch.stack(batches, dim=0)
        return torch.max(predictions, dim=0).values 
      

