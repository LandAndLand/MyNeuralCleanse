import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
import time
import os
import logging
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data import get_data


def train(model, target_label, train_loader, param, logger):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    '''
    trigger image的大小和原图相等，mask标记trigger的像素位置
    '''
    trigger = torch.rand((3, width, height), requires_grad=True)
    # ？
    trigger = trigger.to(device).detach().requires_grad_(True)
    '''
    先初始化trigger的位置是全图
    '''
    mask = torch.rand((width, height), requires_grad=True)
    # ？
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    '''
    需要被训练的参数：
        {"params": trigger},{"params": mask}
    '''
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    '''
    ！！！这里注意，attacked model在这里不进行参数的更新
    '''
    model.eval()
    epoch_all_time = 0
    for epoch in range(Epochs):
        logger.info(f'----------------label: {target_label}--------epoch: {epoch} ---------------------')
        trainstart_time = time.time()
        norm = 0.0
        '''
        用于reverse trigger的train_loader是由对应的测试集组成的
        '''
        index = 0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            index += 1
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            # 构造一个有10个元素（和y_pred一样）的一维tensor，所有元素都是10
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            # logger.info(f'index: {index}, loss: {loss.item()}')
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                # 限制trigger和mask的元素只能取[0, 1]区间内的值
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))
        logger.info(f'\t\t mask abs norm: {norm}')
        every_time = time.time()-trainstart_time
        epoch_all_time +=  every_time
        # logger.info(f'\t\t reverse time for {epoch} epoch: {every_time}')

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break
    logger.info(f'\t\t***avg*** avg time epoch {epoch} avg time: {epoch_all_time / float(Epochs)}')

    return trigger.cpu(), mask.cpu()



def reverse_engineer():
    param = {
        "root": "/home/ay3/data/results/MyNC",
        "dataset": "cifar10",
        "Epochs": 300,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": (32, 32),
        "trigger_pattern": "bottom_right_white_4"
    }
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = 'model_9.pkl'
    model_path = os.path.join(param['root'], param['dataset'], 'models', param["trigger_pattern"])
    model = torch.load(f'{model_path}/{model_name}').to(device)
    logging.basicConfig(filename="reverse.log", filemode='a',
                        format='%(asctime)s, [%(levelname)s]: %(message)s', datefmt="%Y-%m-%d, %H: %M: %S", level=logging.DEBUG)
    logger = logging.getLogger()
    logger.info(param)
    logger.info(f'device: {device}, model: {model_name}')
    #  with the testing data used for reverse engineering
    _, _, x_test, y_test = get_data(param)
    x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)
    trigger_dir= os.path.join(param['root'], param['dataset'], 'triggers', str(param["Epochs"]))
    mask_dir = os.path.join(trigger_dir, 'mask')
    if not os.path.exists(trigger_dir):
        os.makedirs(trigger_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    norm_list = []
    # 遍历所有的label， 将遍历到的label视为潜在的target label
    label_start_time = time.time()
    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param, logger)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1,2,0))
        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig(f'{trigger_dir}/trigger_{label}.png', bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)
        plt.savefig(f'{mask_dir}/mask_{label}.png', bbox_inches='tight', pad_inches=0.0)

    print(norm_list)
    logger.info(f'norm_list: {norm_list}')
    all_time = time.time()-label_start_time
    logger.info(f'all reverse time: {all_time}')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reverse_engineer()
