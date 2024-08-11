from utils import *
from models.classify import *
from models.generator import *
from models.discri import *
import torch
import os
import numpy as np
from attack import inversion, dist_inversion
from argparse import  ArgumentParser


torch.manual_seed(9)

parser = ArgumentParser(description='Inversion')
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/celeba_standard.json')
parser.add_argument('--mode', type=str, choices=['specific', 'general'], default='specific')    
parser.add_argument('--model_name', type=str, default=None)  
parser.add_argument('--model_cfg', type=str, default=None) 
parser.add_argument('--ckp_dir', type=str, default=None)
parser.add_argument('--ckp_epoch', type=str, default='')
parser.add_argument('--gan_model_dir', type=str, default=None)
parser.add_argument('--method', type=str, default=None)
parser.add_argument('--model_types', type=str, default=None)
parser.add_argument('--root_path', type=str, default=None)


args = parser.parse_args()



def init_attack_args(cfg):

    if args.ckp_dir is not None:
        cfg["train"]["cls_ckpts"] = args.ckp_dir
    
    if args.method is not None:
        cfg["attack"]["method"] = args.method

    if args.model_types is not None:
        cfg["train"]["model_types"] = args.model_types
        cfg["dataset"]["model_name"] = args.model_types

    if cfg["attack"]["method"] =='kedmi':
        args.improved_flag = True
        args.clipz = True
        args.num_seeds = 1
    else:
        args.improved_flag = False
        args.clipz = False
        args.num_seeds = 5
        args.istart = 0

    if cfg["attack"]["variant"] == 'L_logit' or cfg["attack"]["variant"] == 'ours':
        args.loss = 'logit_loss'
    else:
        args.loss = 'cel'

    if cfg["attack"]["variant"] == 'L_aug' or cfg["attack"]["variant"] == 'ours':
        args.classid = '0,1,2,3'
    else:
        args.classid = '0'



    



if __name__ == "__main__":
    # global args, logger

    cfg = load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)


    if args.model_name is not None:
        cfg["dataset"]["model_name"] = args.model_name
        cfg["train"]["model_types"] = args.model_name
        args.model_cfg = args.model_cfg.replace("'", '"')
        cfg["train"]["type"] = json.loads(args.model_cfg) 

        model_cfg = str(json.loads(args.model_cfg)).replace(",", "")
        model_cfg = model_cfg.replace("'", "")
        model_cfg = model_cfg.replace(" ", "-")
        model_path_name = os.path.join(model_cfg, "{}_best.tar".format(args.model_name))
        cfg["train"]["cls_ckpts"] = os.path.join(args.ckp_dir, model_path_name)


    if args.root_path is None:
        root_path = os.path.dirname(cfg["train"]["cls_ckpts"])
    else:
        root_path = args.root_path 
    # Save dir
    if args.improved_flag == True:
        prefix = os.path.join(root_path, "kedmi_300ids{}".format(args.ckp_epoch)) 
    else:
        prefix = os.path.join(root_path, "gmi_300ids{}".format(args.ckp_epoch)) 
    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]), cfg["attack"]["variant"])
    prefix = os.path.join(prefix, save_folder)
    save_dir = os.path.join(prefix, "latent")
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))
    args.log_path = os.path.join(prefix, "invertion_logs")

    os.makedirs(prefix, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    
    # Load models
    targetnets, E, G, D, n_classes, fea_mean, fea_logvar = get_attack_model(args, cfg)
    N = 5
    bs = 60
    

    # Begin attacking
    for i in range(1):
        iden = torch.from_numpy(np.arange(bs))

        # evaluate on the first 300 identities only
        target_cosines = 0
        eval_cosines = 0
        for idx in range(5):
            iden = iden %n_classes
            print("--------------------- Attack batch [%s]------------------------------" % idx)
            print('Iden:{}'.format(iden))
            save_dir_z = '{}/{}_{}'.format(save_dir,i,idx)
            
            if args.improved_flag == True:
                #KEDMI
                print('kedmi')

                dist_inversion(G, D, targetnets, E, iden,  
                                        lr=cfg["attack"]["lr"], iter_times=cfg["attack"]["iters_mi"],
                                        momentum=0.9, lamda=100,  
                                        clip_range=1, improved=args.improved_flag, 
                                        num_seeds=args.num_seeds, 
                                        used_loss=args.loss,
                                        prefix=save_dir_z,
                                        save_img_dir=os.path.join(save_img_dir, '{}_'.format(idx)),
                                        fea_mean=fea_mean,
                                        fea_logvar=fea_logvar,
                                        lam=cfg["attack"]["lam"],
                                        clipz=args.clipz)
            else:
                #GMI
                print('gmi')
                if cfg["attack"]["same_z"] =='':
                    inversion(G, D, targetnets, E, iden,  
                                            lr=cfg["attack"]["lr"], iter_times=cfg["attack"]["iters_mi"], 
                                            momentum=0.9, lamda=100, 
                                            clip_range=1, improved=args.improved_flag,
                                            used_loss=args.loss,
                                            prefix=save_dir_z,
                                            save_img_dir=save_img_dir,
                                            num_seeds=args.num_seeds,                                        
                                            fea_mean=fea_mean,
                                            fea_logvar=fea_logvar,lam=cfg["attack"]["lam"],
                                            istart=args.istart)
                else:
                    inversion(G, D, targetnets, E, iden,  
                                            lr=args.lr, iter_times=args.iters_mi, 
                                            momentum=0.9, lamda=100, 
                                            clip_range=1, improved=args.improved_flag,
                                            used_loss=args.loss,
                                            prefix=save_dir_z,
                                            save_img_dir=save_img_dir,
                                            num_seeds=args.num_seeds,                                        
                                            fea_mean=fea_mean,
                                            fea_logvar=fea_logvar,lam=cfg["attack"]["lam"],
                                            istart=args.istart,
                                            same_z='{}/{}_{}'.format(args.same_z,i,idx))
            iden = iden + bs 

