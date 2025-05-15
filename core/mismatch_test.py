import os
import argparse
import time

import torch
from torch import nn
import collections

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter

from thop import profile

import os
import shutil
from glob import glob



def filter_selected_classes(source_dir, target_dir, class_indices):

    os.makedirs(target_dir, exist_ok=True)


    class_folders = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

    class_indices = list(class_indices)
    print(class_indices)


    selected_classes = [class_folders[i] for i in class_indices if i < len(class_folders)]
    print(selected_classes)

    for class_name in selected_classes:
        src_path = os.path.join(source_dir, class_name)
        dst_path = os.path.join(target_dir, class_name)
        shutil.copytree(src_path, dst_path)
        print(f"Copied: {class_name}")

def merge_datasets(dataset1_path, dataset2_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Sort class folders alphabetically
    dataset1_classes = sorted(os.listdir(dataset1_path))
    dataset2_classes = sorted(os.listdir(dataset2_path))

    print(f"Found {len(dataset1_classes)} classes in Dataset 1 and {len(dataset2_classes)} in Dataset 2.")

    def copy_dataset(source_path, classes, offset):
        for i, class_name in enumerate(classes):
            label = i + offset
            src_dir = os.path.join(source_path, class_name)
            dst_dir = os.path.join(save_path, f"{label:03d}")
            os.makedirs(dst_dir, exist_ok=True)

            for img_path in glob(os.path.join(src_dir, '*')):
                shutil.copy(img_path, dst_dir)
            print(f"Copied class '{class_name}' as '{label:03d}'")

    print("Copying Dataset 1...")
    copy_dataset(dataset1_path, dataset1_classes, offset=0)

    print("Copying Dataset 2...")
    copy_dataset(dataset2_path, dataset2_classes, offset=len(dataset1_classes))

    print(f" Merge Complete! Combined dataset saved at: {save_path}")
    
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name') #name of model to be tested 
    parser.add_argument('--data_dir', default='', type=str, help='data directory') #dataset
    parser.add_argument('--data_name', default='', type=str, help='data name') #data nume
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path') #specific model class
    parser.add_argument('--classes', default='', nargs='+', type=int, help='classes') #list of classes to be split into 
    
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-' * 50)
    print('TEST ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    state = torch.load(args.model_path,weights_only = False)
    if isinstance(state, collections.OrderedDict):
        model = models.load_model(args.model_name, num_classes=args.num_classes)
        model.load_state_dict(state)
    else:
        model = state
    model.to(device)
    
    
    classes = args.classes
    num = len(classes)
    
    splits = [range(i,i+num) for i in range(0, args.num_classes-num, num)]
    source_test_dir = '/mainfs/lyceum/abc1g22/abc1g22/DeepLearningCW/Data_for_Lego/cifar100_images/test'
    target_test_dir = '/mainfs/lyceum/abc1g22/abc1g22/DeepLearningCW/Data_for_Lego/cifar100_images_subset/test_'+str(classes[0])+'_'+str(classes[-1])

    if os.path.exists(target_test_dir)==False:
        filter_selected_classes(
            source_dir=source_test_dir,
            target_dir=target_test_dir,
            class_indices=range(classes[0],classes[-1]+1))
    image_splits = []
    orig_dir = target_test_dir
    for split in splits:
        start = split[0]
        end = split[-1]
        
        class_start = classes[0]
        class_end = classes[-1]
        
        # Case 1: Entire split is before or after classes range - no change needed
        if end < class_start or start > class_end:
            split = split
        # Case 2: Partial overlap - remove overlapping range
        else:
            split = [x for x in split if x < class_start or x > class_end]
        if len(split)==0:
            continue 
        print("split range")
        print(split)
        target_test_dir = '/mainfs/lyceum/abc1g22/abc1g22/DeepLearningCW/Data_for_Lego/cifar100_images_subset/test_'+str(split[0])+'_'+str(split[-1])
        if os.path.exists(target_test_dir)==False:
            filter_selected_classes(
              source_dir=source_test_dir,
              target_dir=target_test_dir,
              class_indices=split)
        image_splits.append(target_test_dir)
    #originial loop 
    print("Original Accuracy" )
    test_loader = loaders.load_data(orig_dir, args.data_name, data_type='test')
    
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------
    # speed
    # ----------------------------------------
    speed(model, device)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    # since = time.time()

    loss, acc1, acc5, class_acc = test(test_loader, model, criterion, device)

    orig_acc=acc1.avg
    print("Orig acc", orig_acc)
    accs=[]
    # print('TIME CONSUMED', time.time() - since)
    for split in image_splits:  
        test_loader = loaders.load_data(split, args.data_name, data_type='test')
    
        criterion = nn.CrossEntropyLoss()
    
        # ----------------------------------------
        # speed
        # ----------------------------------------
        speed(model, device)
    
        # ----------------------------------------
        # each epoch
        # ----------------------------------------
        # since = time.time()
    
        loss, acc1, acc5, class_acc = test(test_loader, model, criterion, device)
    
        accs.append(acc1.avg)
        print("Split acc",split,acc1.avg)
        # print('TIME CONSUMED', time.time() - since)
    print("Results ")
    print("Average accuracy",sum(accs)/len(accs))
    print("Improved by", orig_acc - sum(accs)/len(accs))
    print("\n\n\n\n\n\n")
def test(test_loader, model, criterion, device):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
                             meters=[loss_meter, acc1_meter, acc5_meter])
    class_acc = metrics.ClassAccuracy()
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 1))
            class_acc.update(outputs, labels)

            loss_meter.update(loss.item(), inputs.size(0))
            acc1_meter.update(acc1.item(), inputs.size(0))
            acc5_meter.update(acc5.item(), inputs.size(0))

            progress.display(i)

    return loss_meter, acc1_meter, acc5_meter, class_acc


def speed(model, device):
    model.eval()

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),)) # changed to 64,64 for vgg tinyimagenet
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


if __name__ == '__main__':
    main()
