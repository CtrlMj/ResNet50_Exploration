import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from imgaug import augmenters as iaa
from cv2 import resize
import shutil
import numpy as np
sys.path.insert(1, "./LSUV-pytorch")   # For LSUV initialization method. from https://github.com/ducha-aiki/LSUV-pytorch/
from LSUV import LSUVinit


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
 
def init_weights_LSUV(data, model, n_points=64):
    data = []
    iterator = iter(data)
    for i in range(n_points):
        image = next(iterator)[0]
        image = image.permute(1, 2, 0)
        image = image.numpy()
        image = resize(image, (224, 224))
        image = torch.from_numpy(image).unsqueeze(0)
        data.append(image)
    model = LSUVinit(model, torch.stack(data, dim=0).to(device), needed_std = 1.0, std_tol = 0.1, max_attempts = 10, needed_mean = 0., do_orthonorm = True, cuda=True)
    return model


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
      

def train(Res50, train_dataloader, augment = None, aug_prob=0.0):
    Res50.train()
    total_loss = 0
    total_preds = 0
    for step, batch in enumerate(train_dataloader):
        if step % 512 == 0 and step != 0:
          print(f"batch {step} ----loss: {loss.item()}")
          #writer.add_scalar("optimizer/lr", get_lr(optimizer), step)
    
        images, labels = batch[0], batch[1]
        Res50.zero_grad()
        predictions = Res50(images.to(device))
        loss = loss_fn(predictions.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_preds += 1
       
        if np.random.rand() < aug_prob:
          for i in range(5):
            images = augment(images)
            Res50.zero_grad()
            predictions = Res50(images.to(device))
            loss = loss_fn(predictions.to(device), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_preds += 1

    return total_loss / total_preds



def val(Res50, test_dataloader, test_augments: List):
    Res50.eval()

    total_loss = 0
    total_steps = 0
    total_corrects = 0
    total_preds = 0

    for step, batch in enumerate(test_dataloader):
    # if step % 512 == 0 and step != 0:
    #   print(f"test step {step} --- loss: {loss.item()}")

        images, labels = batch[0], batch[1]
        with torch.no_grad():
            predictions = []
            predictions.append(Res50(images.to(device)))
            for aug in test_augments:
                predictions.append(aug(image).to(device))
#             rotated = rotate(images)
#             predictions1 = Res50(rotated.to(device))
#             scaledx = scaleX(images)
#             predictions2 = Res50(scaledx.to(device))
#             scaledx2 = scaleX2(images)
#             predictions3 = Res50(scaledx2.to(device))
#             scaledy = scaleY(images)
#             predictions4 = Res50(scaledy.to(device))
#             scaledy2 = scaleY2(images)
#             predictions5 = Res50(scaledy2.to(device))
#             flipped = flip(images)
#             predictions6 = Res50(flipped.to(device))
#             translatedx = translateX(images)
#             predictions7 = Res50(translatedx.to(device))
#             translatedy = translateY(images)
#             predictions8 = Res50(translatedy.to(device))
#             cropped = cropandpad(images)
#             predictions9 = Res50(cropped.to(device))

            prediction = aggrAugs(predictions, approach='mean')

            loss = loss_fn(prediction.to(device), labels.to(device))
            total_loss += float(loss.item())
            total_steps += 1


            total_corrects += (torch.argmax(predictions, 1) == labels.to(device)).sum().item()
            total_preds += labels.size(0)

    mean_loss = total_loss / total_steps
    mean_accuracy = total_corrects / total_preds
  
    return mean_loss, mean_accuracy
