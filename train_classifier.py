import torch, os, engine, utils
import torch.nn as nn
from argparse import  ArgumentParser
from models import classify
import json
from engine import test

parser = ArgumentParser(description='Train Classifier')
parser.add_argument('--configs', type=str, default='./config/celeba/training_classifiers/classify.json')
parser.add_argument('--model_name', type=str, default=None)  
parser.add_argument('--model_cfg', type=str, default=None) 
parser.add_argument('--ckp_dir', type=str, default=None)  

args = parser.parse_args()



def main(args, model_name, trainloader, testloader, model_path):
    n_classes = args["dataset"]["n_classes"]
    mode = args["dataset"]["mode"]

    resume_path = args[args["dataset"]["model_name"]]["resume"]
    net = classify.get_classifier(model_name=model_name, mode=mode, n_classes=n_classes, resume_path=resume_path, args=args)
    


    if args["dataset"]["model_name"] in ["VGG-R"]:
        optimizer = torch.optim.AdamW(params=net.parameters(),
                                    lr=args[model_name]['lr'],
                                    betas=(0.5, 0.999))
    else:
        optimizer = torch.optim.SGD(params=net.parameters(),
                                    lr=args[model_name]['lr'], 
                                    momentum=args[model_name]['momentum'], 
                                    weight_decay=args[model_name]['weight_decay'])
	

    criterion = nn.CrossEntropyLoss().cuda()

    net = torch.nn.DataParallel(net).to(args['dataset']['device'])
    

    # if resume_path is not "":
    #     ckp_T = torch.load(args[model_name]["resume"]) 
    #     net.load_state_dict(ckp_T['state_dict'], strict=True)

    print("Acc from previous training: {:.2f}".format(test(net, criterion, testloader)[0]))
    print("Number of parameters: {}".format(utils.count_parameters(net)))

    mode = args["dataset"]["mode"]
    n_epochs = args[model_name]['epochs']
    print("Start Training!")
	
    if mode == "reg":
        best_model, best_acc = engine.train_reg(args, net, criterion, optimizer, trainloader, testloader, n_epochs, model_path=model_path)
    elif mode == "vib":
        best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs, model_path=model_path)
	
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_best.tar").format(model_name))


if __name__ == '__main__':
    
    cfg = utils.load_json(json_file=args.configs)

    if args.ckp_dir is not None:
        cfg['root_path'] = args.ckp_dir
    model_path = cfg["root_path"]

    # model_path = os.path.join(root_path, "target_ckp")
    if args.model_name is not None:
        cfg['dataset']['model_name'] = args.model_name
        args.model_cfg = args.model_cfg.replace("'", '"')
        cfg[args.model_name]['type'] = json.loads(args.model_cfg) 
        print("Model configuration: {}".format(cfg[args.model_name]['type']))
        
        model_cfg = str(cfg[args.model_name]['type']).replace(",", "")
        model_cfg = model_cfg.replace("'", "")
        model_cfg = model_cfg.replace(" ", "-")
        model_path = os.path.join(model_path, model_cfg)
        
    os.makedirs(model_path, exist_ok=True)
    model_name = cfg['dataset']['model_name']
    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(model_path, log_file), 'w')

    print("TRAINING %s" % model_name)

    train_file = cfg['dataset']['train_file_path']
    test_file = cfg['dataset']['test_file_path']
    _, trainloader = utils.init_dataloader(cfg, train_file, mode="train")
    _, testloader = utils.init_dataloader(cfg, test_file, mode="test")

    # for x, y in trainloader:
    #     from utils import save_tensor_images
    #     save_tensor_images(x, 'test.png')
    #     exit()

    main(cfg, model_name, trainloader, testloader, model_path=model_path)
