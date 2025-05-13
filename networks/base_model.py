import os, ntpath
import torch
from collections import OrderedDict
from util import util
from . import base_function
import numpy as np


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = opt.run_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.value_names = []
        self.image_names = []
        self.optimizers = []
        self.schedulers = []

    def name(self):
        return 'BaseModel'

    @staticmethod
    def modify_options(parser, is_train):
        """Add new options and rewrite default values for existing options"""
        return parser

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps"""
        pass

    def setup(self, opt):
        """Load networks, create schedulers"""
        if self.isTrain:
            self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_iter)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()

    def get_image_names(self):
        """Return image names that are used to load current data"""
        return self.image_names

    def update_learning_rate(self):
        """Update learning rate"""
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate=%.7f' % lr)

    def get_current_errors(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_ret[name] = util.tensor2im(value[-1].data)
                else:
                    visual_ret[name] = util.tensor2im(value.data)
        return visual_ret

    def get_current_dis(self):
        """Return the distribution of encoder features"""
        dis_ret = OrderedDict()
        value = getattr(self, 'distribution')
        for i in range(1):
            for j, name in enumerate(self.value_names):
                if isinstance(name, str):
                    dis_ret[name+str(i)] =util.tensor2array(value[i][j].data)

        return dis_ret

    # save model
    def save_networks(self, which_epoch):
        """Save all the networks to the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    # load models
    def load_networks(self, which_epoch):
        """Load all the networks from the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                filename = '%s_net_%s.pth' % (which_epoch, name)
                path = os.path.join(self.save_dir, filename)
                net = getattr(self, 'net_' + name)
                try:
                    net.load_state_dict(torch.load(path))
                except:
                    pretrained_dict = torch.load(path)
                    model_dict = net.state_dict()
                    try:
                        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                        net.load_state_dict(pretrained_dict)
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % name)
                    except:
                        print('Pretrained network %s has fewer layers; The following are not initialized:' % name)
                        not_initialized = set()
                        for k, v in pretrained_dict.items():
                            if v.size() == model_dict[k].size():
                                model_dict[k] = v

                        for k, v in model_dict.items():
                            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                                not_initialized.add(k.split('.')[0])
                        print(sorted(not_initialized))
                        net.load_state_dict(model_dict)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()
                if not self.isTrain:
                    net.eval()

    def save_results(self, save_data, iter=None, sample=None, score=None, data_name='none', save_path=None):
        """Save the training or testing results to disk"""
        img_names = self.get_image_names()

        for i in range(save_data.size(0)):
            print('process image ...... %s' % img_names[i])
            name = os.path.splitext(img_names[i])[0]
            if type(score) == type(None):
                img_name = '%s_%s.png' % (name, data_name)
            else:
                img_name = '%s_%s_%s.png' % (name, data_name, str(score))
            if type(sample) != type(None):
                name = os.path.splitext(img_name)[0]
                img_name = f'{name}_sample{sample:02d}.png'
            if type(iter) != type(None):
                name = os.path.splitext(img_name)[0]
                img_name = f'{name}_{iter:05d}.png'
            # save predicted image with discriminator score
            if save_path is None:
                if self.isTrain:
                    save_path = os.path.join(self.opt.run_dir, 'images')
                else:
                    save_path = self.opt.results_dir
            util.mkdir(save_path)
            img_path = os.path.join(save_path, img_name)
            if data_name == 'mask':
                output = save_data[i]
                replacement = torch.ones_like(output) * -1
                result = torch.where(output == 0, replacement, output)
                img_numpy = util.tensor2im(result.data)
            else:
                img_numpy = util.tensor2im(save_data[i].data)
            util.save_image(img_numpy, img_path)
