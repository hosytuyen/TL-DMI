import torch.nn.init as init
import os, models.facenet as facenet, sys
import json, time, random, torch
from models import classify
from models.classify import *
from models.discri import *
from models.generator import *
from models.vgg import vggr
import models.resnet as resnet
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvls
from torchvision import transforms
from datetime import datetime
import dataloader
from torch.autograd import grad

device = "cuda"

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()

class LinearWeightNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = utils.Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = utils.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = utils.Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1 
    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim = 1, keepdim = True))
        return F.linear(x, W, self.bias)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', weight_scale=' + str(self.weight_scale) + ')'

def weights_init(m):
    if isinstance(m, model.MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def init_dataloader(args, file_path, batch_size=64, mode="gan", iterator=False):
    tf = time.time()

    if mode == "attack":
        shuffle_flag = False
    else:
        shuffle_flag = True

    # if args['dataset']['name'] == "celeba" or :
    #     data_set = dataloader.ImageFolder(args, file_path, mode)
    try:
        batch_size = args[args["dataset"]["model_name"]]["batch_size"]
    except:
        batch_size = args["dataset"]["batch_size"]

    if args['dataset']['name'] == 'mnist' or  args['dataset']['name'] == 'xray':
        data_set = dataloader.GrayFolder(args, file_path, mode)
    else:
        data_set = dataloader.ImageFolder(args, file_path, mode)

    if iterator:
        data_loader = torch.utils.data.DataLoader(data_set,
                                batch_size=batch_size,
                                shuffle=shuffle_flag,
                                drop_last=True,
                                num_workers=4,
                                pin_memory=True).__iter__()
    else:
        data_loader = torch.utils.data.DataLoader(data_set,
                                batch_size=batch_size,
                                shuffle=shuffle_flag,
                                drop_last=True,
                                num_workers=4,
                                pin_memory=True)
        interval = time.time() - tf
        print('Initializing data loader took %ds' % interval)
    
    return data_set, data_loader

def load_pretrain(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name.startswith("module.fc_layer"):
            continue
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_params(self, model):
    own_state = self.state_dict()
    for name, param in model.named_parameters():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param.data)

def load_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def print_params(info, params, dataset=None):
    print('-----------------------------------------------------------------')
    print("Running time: %s" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    for i, (key, value) in enumerate(info.items()):
        print("%s: %s" % (key, str(value)))
    for i, (key, value) in enumerate(params.items()):
        print("%s: %s" % (key, str(value)))
    print('-----------------------------------------------------------------')

def save_tensor_images(images, filename, nrow = None, normalize = True):
    if not nrow:
        tvls.save_image(images, filename, normalize = normalize, padding=0)
    else:
        tvls.save_image(images, filename, normalize = normalize, nrow=nrow, padding=0)


def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    #print(state_dict)
    for name, param in state_dict.items():
        if name not in own_state:
            print(name)
            continue
        #print(param.data.shape)
        own_state[name].copy_(param.data)

# add module to keys
def load_module_state_dict(net, state_dict, add=None, strict=False):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.
    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if add is not None:
            name = add + name
        if name in own_state:
            print(name)
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} and '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin : ymin + h, xmin : xmin + w]

def get_center_mask(img_size, bs):
    mask = torch.zeros(img_size, img_size).cuda()
    scale = 0.15
    l = int(img_size * scale)
    u = int(img_size * (1.0 - scale))
    mask[l:u, l:u] = 1
    mask = mask.expand(bs, 1, img_size, img_size)
    return mask

def get_train_mask(img_size, bs):
    mask = torch.zeros(img_size, img_size).cuda()
    typ = random.randint(0, 1)
    if typ == 0:
        scale = 0.25
        l = int(img_size * scale)
        u = int(img_size * (1.0 - scale))
        mask[l:, l:u] = 1
    else:
        u, d = 10, 52
        l, r = 25, 40
        mask[l:r, u:d] = 0
        u, d = 26, 38
        l, r = 40, 63
        mask[l:r, u:d] = 0

    mask = mask.repeat(bs, 1, 1, 1)
    return mask

def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)

def get_deprocessor():
    # resize 112,112
    proc = []
    proc.append(transforms.Resize((112, 112)))
    proc.append(transforms.ToTensor())
    return transforms.Compose(proc)

def low2high(img):
    # 0 and 1, 64 to 112
    bs = img.size(0)
    proc = get_deprocessor()
    img_tensor = img.detach().cpu().float()
    img = torch.zeros(bs, 3, 112, 112)
    for i in range(bs):
        img_i = transforms.ToPILImage()(img_tensor[i, :, :, :]).convert('RGB')
        img_i = proc(img_i)
        img[i, :, :, :] = img_i[:, :, :]
    
    img = img.cuda()
    return img

def calc_feat(img):
    I = FaceNet(1000)
    BACKBONE_RESUME_ROOT = './target_model/target_ckp/FaceNet_95.88.tar'
    print("Loading Backbone Checkpoint ")
    I.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    I = torch.nn.DataParallel(I).cuda()
    img = low2high(img)
    feat, res = I(img)
    return feat

def get_model(attack_name, classes):
    if attack_name.startswith("VGG16"):
        T = classify.VGG16(classes)
    elif attack_name.startswith("IR50"):
        T = classify.IR50(classes)
    elif attack_name.startswith("IR152"):
        T = classify.IR152(classes)
    elif attack_name.startswith("FaceNet64"):
        T = facenet.FaceNet64(classes)
    else:
        print("Model doesn't exist")
        exit()

    T = torch.nn.DataParallel(T).cuda()
    return T

def get_augmodel(args, model_name, nclass, path_T=None, dataset='celeba'):
    if model_name=="VGG16":
        model = VGG16(nclass)   
    elif model_name=="FaceNet":
        model = FaceNet(nclass)
    elif model_name=="FaceNet64":
        model = FaceNet64(nclass)
    elif model_name=="IR152":
        model = IR152(nclass)
    elif model_name =="efficientnet_b0":
        model = classify.EfficientNet_b0(nclass)   
    elif model_name =="efficientnet_b1":
        model = classify.EfficientNet_b1(nclass)   
    elif model_name =="efficientnet_b2":
        model = classify.EfficientNet_b2(nclass)  
    elif model_name == "VGG-R":
        model = vggr(args["train"]["type"], nclass)
    elif model_name == "Resnet-R":
        model = resnet._resnetR(layers=args["train"]["type"], num_classes=nclass, plain=True)
    model = torch.nn.DataParallel(model).cuda()
    if path_T is not None: 
        ckp_T = torch.load(path_T)        
        model.load_state_dict(ckp_T['state_dict'], strict=True)
    return model

def calc_psnr(img1, img2):
    bs, c, h, w = img1.size()
    ten = torch.tensor(10).float().cuda()
    mse = (img1 - img2) ** 2
    # [bs, c, h, w]
    mse = torch.sum(mse, dim = 1)
    mse = torch.sum(mse, dim = 1)
    mse = torch.sum(mse, dim = 1).view(-1, 1) / (c * h * w)
    maxI = torch.ones(bs, 1).cuda()
    psnr = 20 * torch.log(maxI / torch.sqrt(mse)) / torch.log(ten)
    return torch.mean(psnr)

def calc_acc(net, img, iden):
    #img = (img - 0.5) / 0.5
    img = low2high(img)
    __, ___, out_iden = net(img)
    iden = iden.view(-1, 1)
    bs = iden.size(0)
    acc = torch.sum(iden == out_iden).item() * 1.0 / bs
    return acc

def calc_center(feat, iden, path="feature"):
    iden = iden.long()
    feat = feat.cpu()
    center = torch.from_numpy(np.load(os.path.join(path, "center.npy"))).float()
    bs = feat.size(0)
    true_feat = torch.zeros(feat.size()).float()
    for i in range(bs):
        real_iden = iden[i].item()
        true_feat[i, :] = center[real_iden, :]
    dist = torch.sum((feat - true_feat) ** 2) / bs
    return dist.item()
    
def calc_knn(feat, iden, path="feature"):
    iden = iden.cpu().long()
    feat = feat.cpu()
    true_feat = torch.from_numpy(np.load(os.path.join(path, "feat.npy"))).float()
    info = torch.from_numpy(np.load(os.path.join(path, "info.npy"))).view(-1).long()
    bs = feat.size(0)
    tot = true_feat.size(0)
    knn_dist = 0
    for i in range(bs):
        knn = 1e8
        for j in range(tot):
            if info[j] == iden[i]:
                dist = torch.sum((feat[i, :] - true_feat[j, :]) ** 2)
                knn = min(knn, dist)
        knn_dist += knn
    return (knn_dist / bs).item()


def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

# define "soft" cross-entropy with pytorch tensor operations
def softXEnt (input, target):
    targetprobs = nn.functional.softmax (target, dim = 1)
    logprobs = nn.functional.log_softmax (input, dim = 1)
    return  -(targetprobs * logprobs).sum() / input.shape[0]

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
    
def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y, DG):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_GAN(dataset, gan_type, gan_model_dir, n_classes, z_dim, target_model):

    G = Generator(z_dim)
    gan_condition = (gan_type == True) and ('-R' not in target_model)# and ('VGG16' not in target_model)
    if gan_condition:
        D = MinibatchDiscriminator(n_classes=n_classes)
    else:
        D = DGWGAN(3)

    if gan_condition:
        path = os.path.join(os.path.join(gan_model_dir, dataset), target_model)
        path_G = os.path.join(path, "improved_{}_G.tar".format(dataset))
        path_D = os.path.join(path, "improved_{}_D.tar".format(dataset))
    else:
        path = os.path.join(gan_model_dir, dataset)
        path_G = os.path.join(path, "{}_G.tar".format(dataset))
        path_D = os.path.join(path, "{}_D.tar".format(dataset)) 

    print('path_G',path_G)
    print('path_D',path_D)

    G = torch.nn.DataParallel(G).to(device)
    D = torch.nn.DataParallel(D).to(device)
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=True)
    ckp_D = torch.load(path_D)
    D.load_state_dict(ckp_D['state_dict'], strict=True)
  
    return G, D


def get_attack_model(args, args_json, eval_mode=False):
    now = datetime.now() # current date and time
    
    if not eval_mode:
        log_file = "invertion_logs_{}_{}.txt".format(args.loss, now.strftime("%m_%d_%Y_%H_%M_%S"))
        utils.Tee(os.path.join(args.log_path, log_file), 'w')
    
    n_classes=args_json['dataset']['n_classes']
    
    
    model_types_ = args_json['train']['model_types'].split(',')
    checkpoints = args_json['train']['cls_ckpts'].split(',')

    G, D = get_GAN(args_json['dataset']['name'],gan_type=args.improved_flag, 
                    gan_model_dir=args_json['train']['gan_model_dir'],
                    n_classes=n_classes,z_dim=100,target_model=model_types_[0])

    dataset = args_json['dataset']['name']
    cid = args.classid.split(',')

    # target and student classifiers
    for i in range(len(cid)):
        id_ = int(cid[i])
        model_types_[id_] = model_types_[id_].strip()
        checkpoints[id_] = checkpoints[id_].strip()
        model = get_augmodel(args_json, model_types_[id_], n_classes, checkpoints[id_], dataset)
        model = model.to(device)
        model = model.eval()
        if i==0:
            targetnets = [model]
        else:
            targetnets.append(model)
    
        # p_reg 
        if args.loss=='logit_loss' and ("-R" not in args_json['dataset']['model_name']):
            if model_types_[id_] == "IR152" or model_types_[id_]=="VGG16" or model_types_[id_]=="FaceNet64":
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"], '{}_{}_p_reg.pt'.format(dataset,model_types_[id_])) #'./p_reg/{}_{}_p_reg.pt'.format(dataset,model_types_[id_])
            else:
                p_reg = os.path.join(args_json["dataset"]["p_reg_path"], '{}_{}_{}_p_reg.pt'.format(dataset,model_types_[0],model_types_[id_])) #'./p_reg/{}_{}_{}_p_reg.pt'.format(dataset,model_types_[0],model_types_[id_])
            # print('p_reg',p_reg)
            if not os.path.exists(p_reg):
                _, dataloader_gan = init_dataloader(args_json, args_json['dataset']['gan_file_path'], 50, mode="gan")
                from attack import get_act_reg
                fea_mean_,fea_logvar_ = get_act_reg(dataloader_gan,model,device)
                torch.save({'fea_mean':fea_mean_,'fea_logvar':fea_logvar_},p_reg)
            else:
                fea_reg = torch.load(p_reg)
                fea_mean_ = fea_reg['fea_mean']
                fea_logvar_ = fea_reg['fea_logvar']
            if i == 0:
                fea_mean = [fea_mean_.to(device)]
                fea_logvar = [fea_logvar_.to(device)]
            else:
                fea_mean.append(fea_mean_)
                fea_logvar.append(fea_logvar_)
            # print('fea_logvar_',i,fea_logvar_.shape,fea_mean_.shape)
            
        else:
            fea_mean, fea_logvar = 0,0

    # evaluation classifier
    E = get_augmodel(args_json, args_json['train']['eval_model'], n_classes,args_json['train']['eval_dir'])    
    E.eval()
    G.eval()
    D.eval()

    return targetnets, E, G, D, n_classes, fea_mean, fea_logvar
