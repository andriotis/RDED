import torch, torch.nn as nn, sys, collections
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
sys.argv = ['x']
from synthesize.utils import load_model


def ev(model, val_dir, size):
    tfm = T.Compose([
        T.Resize(size // 7 * 8, antialias=True), T.CenterCrop(size), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(root=val_dir, transform=tfm)
    ld = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()
    cnt = collections.Counter(); tot = collections.Counter()
    pred_for = collections.defaultdict(collections.Counter)
    with torch.no_grad():
        for x, y in ld:
            x = x.cuda()
            p = model(x).argmax(1).cpu()
            for pi, yi in zip(p.tolist(), y.tolist()):
                tot[yi] += 1
                if pi == yi:
                    cnt[yi] += 1
                pred_for[yi][pi] += 1
    c = sum(cnt.values()); t = sum(tot.values())
    print('  classes:', ds.classes)
    print('  class_to_idx:', ds.class_to_idx)
    print('  overall top1: %.2f' % (100 * c / t))
    for k in sorted(tot):
        mp = pred_for[k].most_common(1)[0]
        print('   cls %d: acc %.1f  (mostpred=%d:%d/%d)' % (k, 100 * cnt[k] / tot[k], mp[0], mp[1], tot[k]))


for subset, arch, size in [('imagenet-nette', 'resnet18', 224), ('imagenet-woof', 'resnet18', 224)]:
    print('==', subset, arch, '==')
    m = load_model(model_name=arch, dataset=subset, pretrained=True, classes=list(range(10)))
    m = nn.DataParallel(m).cuda()
    ev(m, f'./data/{subset}/val', size)
    del m
    torch.cuda.empty_cache()
