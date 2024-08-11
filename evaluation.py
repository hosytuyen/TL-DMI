from argparse import  ArgumentParser
from metrics.KNN_dist import eval_KNN
from metrics.eval_accuracy import eval_accuracy, eval_acc_class
from metrics.fid import eval_fid
from utils import load_json, get_attack_model
import os, csv, json

parser = ArgumentParser(description='Evaluation')
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/celeba_standard.json')    
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


if __name__ == '__main__':
    # Load Data
    cfg = load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)

    if args.ckp_dir == None:
        args.ckp_dir = cfg["train"]["cls_ckpts"]
    
    # os.makedirs(args.ckp_dir, exist_ok=True)

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

    # Load models
    _, E, G, _, _, _, _ = get_attack_model(args, cfg, eval_mode=True)

    # Metrics
    metric = cfg["attack"]["eval_metric"].split(',')
    fid = 0
    aver_acc, aver_acc5, aver_std, aver_std5 = 0, 0, 0, 0
    knn = 0, 0
    nsamples = 0 
    dataset, model_types = '', ''


    
    for metric_ in metric:
        metric_ = metric_.strip()
        if metric_ == 'fid':
            fid, nsamples = eval_fid(G=G, E=E, save_dir=save_dir, cfg=cfg, args=args)
        elif metric_ == 'acc':
            aver_acc, aver_acc5, aver_std, aver_std5 = eval_accuracy(G=G, E=E, save_dir=save_dir, args=args)
        elif metric_ == 'knn':
            knn = eval_KNN(G=G, E=E, save_dir=save_dir, KNN_real_path=cfg["dataset"]["KNN_real_path"], args=args)
        elif metric_ == 'eval_acc_class':
            eval_acc_class(G=G, E=E, save_dir=save_dir, prefix=prefix, args=args)

 
    csv_file = os.path.join(os.path.dirname(args.ckp_dir), 'Eval_results.csv') 
    if not os.path.exists(csv_file):
        header = ['Save_dir', 'Method', 'Succesful_samples',                    
                    'acc','std','acc5','std5',
                    'knn', 'fid']
        with open(csv_file, 'w') as f:                
            writer = csv.writer(f)
            writer.writerow(header)
    
    if args.model_cfg is not None:
        saved_column = args.model_cfg
    else:
        saved_column = save_dir

    # fields=['{}'.format(saved_column), 
    #         '{}'.format(cfg["attack"]["method"]),
    #         '{}'.format(cfg["attack"]["variant"]),
    #         '{:.3f}'.format(aver_acc),
    #         '{:.3f}'.format(aver_std),
    #         '{:.3f}'.format(aver_acc5),
    #         '{:.3f}'.format(aver_std5),
    #         '{:.3f}'.format(knn),
    #         '{:.3f}'.format(fid)]
    
    print("---------------Evaluation---------------")
    print('Method: {} '.format(cfg["attack"]["method"]))

    print('Variant: {}'.format(cfg["attack"]["variant"]))
    print('Top 1 attack accuracy:{:.2f} +/- {:.2f} '.format(aver_acc, aver_std))
    print('Top 5 attack accuracy:{:.2f} +/- {:.2f} '.format(aver_acc5, aver_std5))
    # print('KNN distance: {:.3f}'.format(knn))
    # print('FID score: {:.3f}'.format(fid))      
    
    # print("----------------------------------------")  
    # with open(csv_file, 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(fields)
