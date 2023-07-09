from src.model import *
from src.utilities import *
from src.data_loader import *
from src.Adam import *
from src.dice_score import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch import optim
import logging

################################################################
# tensor_to_image
################################################################

def tensor_to_image(tensor):
    if tensor.dim()==4:
        tensor=tensor.squeeze(0)  ###去掉batch维度
    tensor=tensor.permute(1,2,0) ##将c,h,w 转换为h,w,c
    tensor=tensor.mul(255).clamp(0,255)  ###将像素值转换为0-255之间
    tensor=tensor.cpu().detach().numpy().astype('uint8')  ###
    return tensor

################################################################
# configs
################################################################

ntrain = 80
ntest = 20
batch_size = 1
gpu = 0
learning_rate = 0.0005
weight_decay: float = 1e-8
momentum: float = 0.999
amp: bool = False
gradient_clipping: float = 1.0

################################################################
# load data and data normalization
################################################################
device = torch.device(
    "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu"
)


data_types = ['_current', '_eff_dist', '_ir_drop_map', '_pdn_density']
dataset_root = 'dataset'
dataset_csv = os.path.join(dataset_root, 'train_data.csv')
dataset_path = os.path.join(dataset_root, 'train')

datas = DPDataset(dataset_csv, dataset_path)
train_set, val_set = random_split(datas, [ntrain, ntest], generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

tdl = train_loader.__iter__()
epoch_loss = 0

for i in tqdm(range(int(ntrain / batch_size))):
    current_map, eff_dist_map, ir_drop_map_map, pdn_densit_map = next(tdl)
    print(current_map.size())
    print(eff_dist_map.size())
    print(ir_drop_map_map.size())
    print(pdn_densit_map.size())
    merged_map = np.stack((current_map, eff_dist_map, pdn_densit_map), axis=1)
    input_tensor = torch.from_numpy(merged_map)
    double_tensor = input_tensor.to(torch.double).to(device)
    model = UNet(3 , 1).to(device)
    model = model.double()
    outputs = model.forward(double_tensor)
    labels = ir_drop_map_map

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.out_channels > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        masks_pred = outputs.to(device)
        true_masks = labels.to(device)
        if model.out_channels == 1:
            loss = criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
        else:
            loss = criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.softmax(masks_pred, dim=1).float(),
                F.one_hot(true_masks, model.out_channels).permute(0, 3, 1, 2).float(),
                multiclass=True
            )

    # criterion = nn.CrossEntropyLoss().to(device)
    # outputs = tensor_to_image(outputs)
    outputs = np.squeeze(outputs[0], 0)
    labels = np.squeeze(labels[0], 0)
    # loss = criterion(outputs, labels).to(device)
    optimizer.zero_grad(set_to_none=True)
    grad_scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    epoch_loss += loss.item()
    print(loss.item())


    outputs = outputs.cpu().detach().numpy()	#transfer tensor to array
    plt.imshow(outputs)
    plt.show()
    for i in range(5):
        torch.cuda.empty_cache()



# ################################################################
# # model
# ################################################################

# # TODO: define your model here
# # for epoch in range(2):  # loop over the dataset multiple times

# #     running_loss = 0.0
# #     for i, data in enumerate(train_loader, 0):
# #         # get the inputs; data is a list of [inputs, labels]
# #         inputs, labels = data
# #         inputs = inputs.cuda()
# #         labels = labels.cuda()
# #         labels.to('cuda:0')
# #         # zero the parameter gradients

# #         # forward + backward + optimize
        
# #         outputs = unet(inputs)
# #         loss = criterion(outputs, labels)
# #         loss_list.append(loss.item())
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         correct = 0
# #         total = 0
# #         _, predictions = torch.max(outputs, 1)
# #         # collect the correct predictions for each class
# #         for label, prediction in zip(labels, predictions):
# #             if label == prediction:
# #                 correct += 1
# #             total += 1

# #         accuracy = correct / total
# #         accuracy_list.append(accuracy)

# #         # print statistics
# #         running_loss += loss.item()
# #         if i % 2000 == 1999:  # print every 2000 mini-batches
# #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
# #             running_loss = 0.0

# # print('Finished Training')
