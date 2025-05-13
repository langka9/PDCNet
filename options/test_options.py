from .base_options import BaseOptions
import torch


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--run_dir', type=str, default='', help='continue training: load the latest model from this checkpoint path')
        parser.add_argument('--how_many', type=int, default=10, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--nsampling', type=int, default=5, help='ramplimg # times for each images')
        parser.add_argument('--save_number', type=int, default=10, help='choice # reasonable results based on the discriminator score')

        self.isTrain = False

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
        desc += f'-{opt.dataset_type}'
        desc += f'-{opt.fineSize}'
        if opt.no_use_mask:
            desc += '-noise'
        
        self.opt.name = desc

        # Pick output directory.
        assert self.opt.run_dir != ''

        self.print_options(opt)

        return self.opt


