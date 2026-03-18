import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from synthesize.utils import load_model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate_model(dataset_name, arch_name, data_dir, input_size, nclass):
    print(f"\nEvaluating: {arch_name} on {dataset_name}")
    
    # Standard normalization for pretrained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Minimal transform: strictly matching input size
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = load_model(model_name=arch_name, dataset=dataset_name, pretrained=True, classes=list(range(nclass)))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    top1 = 0.0
    total = 0
    with torch.no_grad():
        for images, target in val_loader:
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            prec1, = accuracy(output, target, topk=(1,))
            
            top1 += prec1.item() * images.size(0)
            total += images.size(0)

    top1_acc = top1 / total
    print(f"Result -> Dataset: {dataset_name:15s} | Backbone: {arch_name:20s} | Top1-Acc: {top1_acc:.2f}% | Size: {input_size}x{input_size}")
    return top1_acc


def main():
    experiments = [
        ("cifar10", "resnet18_modified", 32, 10, "./data/cifar10/val/"),
        ("cifar10", "conv3", 32, 10, "./data/cifar10/val/"),
        ("cifar100", "resnet18_modified", 32, 100, "./data/cifar100/val/"),
        ("cifar100", "conv3", 32, 100, "./data/cifar100/val/"),
        ("tinyimagenet", "resnet18_modified", 64, 200, "./data/tinyimagenet/val/"),
        ("tinyimagenet", "conv4", 64, 200, "./data/tinyimagenet/val/"),
        ("imagenet-nette", "resnet18", 224, 10, "./data/imagenet-nette/val/"),
        ("imagenet-nette", "conv5", 128, 10, "./data/imagenet-nette/val/"),
        ("imagenet-woof", "resnet18", 224, 10, "./data/imagenet-woof/val/"),
        ("imagenet-woof", "conv5", 128, 10, "./data/imagenet-woof/val/"),
        ("imagenet-10", "resnet18", 224, 10, "./data/imagenet-10/val/"),
        ("imagenet-10", "conv5", 128, 10, "./data/imagenet-10/val/"),
        ("imagenet-100", "resnet18", 224, 100, "./data/imagenet-100/val/"),
        ("imagenet-100", "conv6", 128, 100, "./data/imagenet-100/val/"),
        ("imagenet-1k", "conv4", 64, 1000, "./data/imagenet-1k/val/"),
    ]

    print(f"{'Dataset':<15} | {'Backbone':<20} | {'Top1-accuracy':<13} | {'Input Size':<10}")
    print("-" * 65)
    
    results = []
    for ds_name, arch_name, input_size, nclass, data_dir in experiments:
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} missing.")
            continue
        try:
            acc = evaluate_model(ds_name, arch_name, data_dir, input_size, nclass)
            results.append((ds_name, arch_name, acc, input_size))
        except Exception as e:
            print(f"Error evaluating {arch_name} on {ds_name}: {e}")

    print("\n\nFINAL RESULT TABLE")
    print(f"{'Dataset':<15} | {'Backbone':<20} | {'Top1-accuracy':<13} | {'Input Size':<10}")
    print("-" * 65)
    for ds_name, arch_name, acc, input_size in results:
        print(f"{ds_name:<15} | {arch_name:<20} | {acc:<13.2f} | {input_size} x {input_size}")

if __name__ == "__main__":
    main()
