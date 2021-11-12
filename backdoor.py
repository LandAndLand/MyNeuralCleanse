import torchvision
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import transforms
from data import get_data, poison, fill_param
from model import get_model, weight_init
import tqdm
import numpy as np
import os
import time
import logging


def train(param, logger):
    x_train, y_train, x_test, y_test = get_data(param)
    num_p = int(param["injection_rate"] * x_train.shape[0])
    # 毒害一部分的训练集合数据集
    x_train[:num_p], y_train[:num_p] = poison(
        x_train[:num_p], y_train[:num_p], param)
    x_test_pos, y_test_pos = poison(x_test.copy(), y_test.copy(), param)

    # make dataset
    # 转换数据集为tensor类型
    x_train, y_train = torch.from_numpy(
        x_train)/255., torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    x_test_pos, y_test_pos = torch.from_numpy(
        x_test_pos)/255., torch.from_numpy(y_test_pos)

    train_loader = DataLoader(TensorDataset(
        x_train, y_train), batch_size=param["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(
        x_test, y_test), batch_size=param["batch_size"], shuffle=False)
    test_pos_loader = DataLoader(TensorDataset(
        x_test_pos, y_test_pos), batch_size=param["batch_size"], shuffle=False)

    # train model
    model = get_model(param).to(device)
    model.apply(weight_init)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.001, weight_decay=1e-6, eps=1e-6)
    saved_dir = os.path.join(param['root'], param['dataset'], 'models', param["trigger_pattern"])
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    for epoch in range(param["Epochs"]):
        logger.info(f"-----------------\t epoch: {epoch}---------------------")
        trainstart_time = time.time()
        model.train()
        adjust_learning_rate(optimizer, epoch)
        train_correct = 0
        train_total = 0
        for images, labels in tqdm.tqdm(train_loader, desc='Training Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            y_out = model(images)
            loss = criterion(y_out, labels)
            loss.backward()
            optimizer.step()
            y_out = torch.argmax(y_out, dim=1)
            train_correct += (y_out == labels).sum().item()
            train_total += images.size(0)
        logging.info(f"epoch {epoch} train time: {time.time()-trainstart_time}")
        model.eval()
        with torch.no_grad():
            valstart_time = time.time()
            correct = 0
            total = 0
            for images, labels in tqdm.tqdm(test_loader, desc="Testing..."):
                images, labels = images.to(device), labels.to(device)
                y_out = model(images)
                y_out = torch.argmax(y_out, dim=1)
                correct += torch.sum(y_out == labels).item()
                total += images.size(0)

            correct_trojan = 0
            all_trojan = len(test_pos_loader)*param["batch_size"]
            for images, labels in tqdm.tqdm(test_pos_loader, desc="Testing..."):
                images, labels = images.to(device), labels.to(device)
                y_out = model(images)
                y_out = torch.argmax(y_out, dim=1)
                correct_trojan += torch.sum(y_out == labels).item()
            print(all_trojan,  correct_trojan)
            acc_train =  100. * train_correct / train_total
            acc_test = 100. * correct/ total
            asr_test = 100. * correct_trojan / all_trojan
            
            logger.info(f"epoch {epoch} valid time: {time.time()-valstart_time}")
            print(f"Epoch: {epoch}, Training Accuracy: {acc_train}%, "
                  f"Testing Accuracy: {acc_test}%, Testing ASR: {asr_test}%")
            logger.info(
                f"\t Epoch: {epoch}, Training Accuracy: {acc_train}%, Testing Accuracy: {acc_test}%, Testing ASR: {asr_test}%")
        torch.save(model, f"{saved_dir}/model_{epoch}.pkl")


def adjust_learning_rate(optimizer, epoch):
    if epoch < 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    elif 80 <= epoch < 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param = {
        "root": "/mnt/data/results/MyNC",
        "dataset": "cifar10",
        "model": "default",
        "poisoning_method": "badnet",
        "injection_rate": 0.02,
        "target_label": 8,
        "Epochs": 130,
        "batch_size": 64,
        "trigger_pattern": "bottom_right_white_4"
    }
    fill_param(param)
    logging.basicConfig(filename="backdoor-train.log", filemode='a',
                        format='%(asctime)s, [%(levelname)s]: %(message)s', datefmt="%Y-%m-%d, %H: %M: %S", level=logging.DEBUG)
    logger = logging.getLogger()
    logger.info(param)
    logger.info(f'device: {device}')
    train(param, logger)


'''
pytorch 效果差原因：
1. PyTorch有自己的一套初始化方案，而不是Glorot uniform(也即xavier_initializer)
2. PyTorch的各层默认参数和keras不同，这可能导致结果不正确
3. PyTorch的优化器可能有一点问题(可能还是划归为默认参数, 如eps, lambda, beta这些)
4. 学习率没有动态调整

'''
