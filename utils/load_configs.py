import yaml


def yaml2cfg(yaml_file):
    with open(yaml_file, 'r') as f:
        cfg_dict = yaml.load(f)
    cfg = base_cfg(cfg_dict)
    return cfg

class base_cfg(object):
    def __init__(self, args):
        self.detector_type = args['detector_type']
        print(self.detector_type)
        if self.detector_type == 'base_ssd':
            self.ssd_dim = args['ssd_params']['ssd_dim']
            self.backbone_arch = args['ssd_params']['backbone_arch']
            self.batch_size = args['ssd_params']['batch_size']
            self.anchor_version = args['ssd_params']['anchor_version']
        ## detect settings
        self.num_classes = args['detector_params']['num_classes']
        ## checkpoint settings
        self.resume = args['checkpoint_params']['resume']
        ## save experimental logs, models...
        self.exp_dir = args['checkpoint_params']['exp_dir']
        ## dataset related
        self.dataset = args['dataset']['name']
        self.train_root = args['dataset']['train_root']
        self.val_root = args['dataset']['val_root']
        
        ## optimizer settings
        self.optim_type = args['optim_params']['type']
        self.init_lr = args['optim_params']['init_lr']
        self.momentum = args['optim_params']['momentum']
        self.weight_decay = args['optim_params']['weight_decay']

        ## running settings
        self.use_cuda = args['running_params']['use_cuda']
        self.start_iteration = args['running_params']['start_iterations']
        self.iterations = args['running_params']['iterations']
        ## transfom 
def test_load_cfg():
    import sys
    yaml_file = sys.argv[1]
    cfg = yaml2cfg(yaml_file)
if __name__ == '__main__':
    test_load_cfg()