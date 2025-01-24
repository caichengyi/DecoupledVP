import argparse
import sys
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Lambda, ToTensor, CenterCrop, RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.append(".")
from cfg import *
from tools import *
from datasets import *
from methods.vp import PaddingVR

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--backbone', type=str, default="vitb16", choices=['rn50', 'vitb16', 'rn101', 'vitb32', 'vitl14'])
    p.add_argument('--dataset', choices=['caltech101' ,'dtd' ,'eurosat' ,'fgvc' ,'food101', 'oxford_flowers' ,'oxford_pets','stanford_cars' ,'sun397' ,'ucf101', 'resisc45'], default='dtd')
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    exp = f"results/fs_dvp_cse"
    save_path = os.path.join(exp, args.dataset + args.backbone + str(args.seed))

    # Load the Pretrained Model
    if args.backbone == "rn50":
        model, _ = clip.load("RN50")
    elif args.backbone == 'vitb16':
        model, _ = clip.load("ViT-B/16")
    elif args.backbone == 'rn101':
        model, _ = clip.load("RN101")
    elif args.backbone == 'vitb32':
        model, _ = clip.load("ViT-B/32")
    convert_models_to_fp32(model)

    model.eval()
    model.requires_grad_(False)

    # Data Argument
    train_process = Compose([
        Resize(param['dvp-cse']['input_size'], interpolation=InterpolationMode.BICUBIC),
        RandomResizedCrop(param['dvp-cse']['input_size'], interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    preprocess = Compose([
        Resize(param['dvp-cse']['input_size'], interpolation=InterpolationMode.BICUBIC),
        CenterCrop(param['dvp-cse']['input_size']),
        Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        ToTensor(),
    ])

    # Load the Downstream Tasks
    bs = 64
    if args.dataset in ['caltech101' ,'dtd' ,'eurosat' ,'fgvc' ,'food101' ,'oxford_flowers' ,'oxford_pets'
                          ,'stanford_cars' ,'sun397' ,'ucf101', 'resisc45']:
        trainloader, testloader, classes = build_loader(args.dataset, DOWNSTREAM_PATH, train_process, preprocess, batch_size=bs, shot=param['dvp-cse']['shot'])

    causes_emb_list = []
    causes_txt_list = []
    causes_list = param['dvp-cse']['cause_no'].split(',')
    for i in range(param['dvp-cse']['causes']):
        causes_emb, causes_txt = clip_txt(classes, model, 'causes/' + args.dataset + '_cse_' + str(causes_list[i]) + '.json', param['dvp-cse']['m'])
        causes_emb = causes_emb.to(device)
        causes_emb_list.append(causes_emb)
        causes_txt_list.append(causes_txt)

    def network(x, vplist):
        single_logits = []
        for i in range(param['dvp-cse']['causes']):
            x_emb = model.encode_image(vplist[i](x))
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
            exp = model.logit_scale.exp()
            single_logits.append((exp * x_emb @ causes_emb_list[i].T).float().reshape(-1, len(classes), param['dvp-cse']['m']))
        logits = torch.cat(single_logits, dim=2).reshape(-1, len(classes) * param['dvp-cse']['m'] * param['dvp-cse']['causes'])
        return logits

    # Visual Prompt
    vplist = []
    optlist = []
    schelist = []
    for i in range(param['dvp-cse']['causes']):
        vplist.append(PaddingVR(224, param['dvp-cse']['input_size']).to(device))

    for i in range(param['dvp-cse']['causes']):
        optimizer = torch.optim.SGD(vplist[i].parameters(), lr=param['dvp-cse']['lr'], momentum=0.9)
        t_max = param['dvp-cse']['epoch'] * len(trainloader)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
        optlist.append(optimizer)
        schelist.append(scheduler)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)

    # Train
    prob_y_s = initialize_prm(len(classes), param['dvp-cse']['m'] * param['dvp-cse']['causes'], device)
    for i in range(param['dvp-cse']['causes']):
        vplist[i].train()
    best_acc = 0.
    progress_bar = tqdm(total=param['dvp-cse']['epoch'], desc='Training', leave=True)

    for epoch in range(param['dvp-cse']['epoch']):

        total_num = 0
        true_num = 0
        loss_sum = 0
        vres = []
        yres = []
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            if epoch:
                for i in range(param['dvp-cse']['causes']):
                    optlist[i].zero_grad()
                with autocast():
                    fx = network(x, vplist)
                    out = fx @ prob_y_s
                    loss = F.cross_entropy(out, y, reduction='mean')

                # Store Intermediate Data
                with torch.no_grad():
                    newfx = fx.reshape(-1, len(classes), param['dvp-cse']['m'] * param['dvp-cse']['causes'])
                    _, indices = torch.topk(newfx[torch.arange(newfx.size(0)), y], param['dvp-cse']['k'], dim=1)
                loss.backward()

                # Update
                for i in range(param['dvp-cse']['causes']):
                    optlist[i].step()
                model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

                # Results
                total_num += y.size(0)
                true_num += torch.argmax(out, 1).eq(y).float().sum().item()
                loss_sum += loss.item() * fx.size(0)

                # Schedule
                for i in range(param['dvp-cse']['causes']):
                    schelist[i].step()
            else:
                # In the first epoch, only calculate PRM
                with torch.no_grad():
                    fx = network(x, vplist)
                    newfx = fx.reshape(-1, len(classes), param['dvp-cse']['m'] * param['dvp-cse']['causes'])
                    _, indices = torch.topk(newfx[torch.arange(newfx.size(0)), y], param['dvp-cse']['k'], dim=1)

            # Store Intermediate Data
            vres.append(indices.detach().cpu().numpy())
            yres.append(y.detach().cpu().numpy().reshape(-1, 1))

        # Update PRM in each epoch
        vall = np.vstack(vres)
        yall = np.vstack(yres)
        prob_y_s = calculate_prm(vall, yall, param['dvp-cse']['m'] * param['dvp-cse']['causes'], device)

        if epoch:
            acc = true_num / total_num
            progress_bar.set_postfix({'Epoch': epoch + 1, 'Train Acc': acc})

            # Test
            for i in range(param['dvp-cse']['causes']):
                vplist[i].eval()
            total_num = 0
            true_num = 0
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    fx = network(x, vplist)
                    out = fx @ prob_y_s
                total_num += y.size(0)
                true_num += torch.argmax(out, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            progress_bar.set_postfix({'Epoch': epoch + 1, 'Test Acc': acc, 'Best Acc': best_acc}, refresh=False)

            # Save CKPT
            state_dict = {
                "visual_prompt_dict": [vplist[i].state_dict() for i in range(param['dvp-cse']['causes'])],
                "epoch": epoch,
                "prob": prob_y_s.detach().cpu().numpy(),
                "best_acc": best_acc,
            }
            if acc > best_acc:
                best_acc = acc
                state_dict['best_acc'] = best_acc
                torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            progress_bar.update(1)
    progress_bar.close()