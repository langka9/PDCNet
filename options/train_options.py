from .base_options import BaseOptions
import torch
import re
import os


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each images')

        # training epoch
        parser.add_argument('--iter_count', type=int, default=0, help='the starting epoch count')
        parser.add_argument('--niter', type=int, default=5000000, help='# of iter with initial learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to decay learning rate to zero')
        parser.add_argument('--run_dir', type=str, default='', help='continue training: load the latest model from this checkpoint path')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # learning rate and loss weight
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy[lambda|step|plateau]')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_flow', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='wgangp', choices=['wgangp', 'hinge', 'lsgan'])

        # display the results
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_iters_freq', type=int, default=10000, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')

        self.isTrain = True

        return parser

    def parse(self):
        """Parse the options"""

        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        desc = ''
        desc += f'{opt.model}'
        desc += f'{opt.dataset_type}'
        desc += f'-{opt.fineSize}'
        desc += f'-{opt.lr}'
        
        self.opt.name = desc

        # Pick output directory.
        if self.opt.run_dir == '':
            prev_run_dirs = []
            if os.path.isdir(self.opt.checkpoints_dir):
                prev_run_dirs = [x for x in os.listdir(self.opt.checkpoints_dir) if os.path.isdir(os.path.join(self.opt.checkpoints_dir, x))]
            prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
            prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
            cur_run_id = max(prev_run_ids, default=-1) + 1
            self.opt.run_dir = os.path.join(self.opt.checkpoints_dir, f'{cur_run_id:05d}-{self.opt.name}')

        self.print_options(opt)

        return self.opt

