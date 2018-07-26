import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from utils.net_utils import weights_init
from utils.load_configs import yaml2cfg
from models.model_factory import build_ssd
from layers.modules import MultiBoxLoss
from data.make_dataset import make_dataset
from data import detection_collate
import logging
def train(args):
    ## initilize net 
    ssd_net = build_ssd('train', args.ssd_dim, args.num_classes, args.backbone_arch)
    if args.resume:
        ssd_net.load_weights(args.resume)
    else:
        print("Initializing weights")
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    if args.use_cuda:
        ssd_net = ssd_net.cuda()
    if args.optim_type == 'SGD':
        optimizer = optim.SGD(ssd_net.parameters(), lr = args.init_lr, 
                            momentum = args.momentum, weight_decay = args.weight_decay)
    print(type(args.use_cuda))
    criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.use_cuda)
    dataset = make_dataset(args.dataset, args.train_root, imageSize = args.ssd_dim)
    print(len(dataset))
    epoch_size = len(dataset) // args.batch_size
    step_index = 0
    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers  = 4, 
                                shuffle = True, pin_memory = True, collate_fn = detection_collate )
    batch_iterator = None
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    for iteration in range(args.start_iteration, args.iterations):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(data_loader)
            epoch += 1
            loc_loss = 0
            conf_loss = 0

        with torch.no_grad():
            images, targets = next(batch_iterator)
            if args.use_cuda:
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]
        out = ssd_net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = 0.01*loss_l + loss_c
        loss.backward()
        optimizer.step()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 100 == 0:
            logging.warning('Iter' + repr(iteration) + '|| Loss_loc: %.4f Loss_conf %.4f ||'%(loss_l.data[0], loss_c.data[0]))
        if iteration % 10000 == 0:
            print('Saving state, iter:', iteration)
            save_dir = os.path.join(args.exp_dir, 'weights')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(ssd_net.state_dict(), os.path.join(save_dir, 'ssd_base'+repr(iteration)+'.pth'))

        
         

if __name__ =='__main__':
    import sys
    yaml_file = sys.argv[1]
    args = yaml2cfg(yaml_file)
    train(args)
