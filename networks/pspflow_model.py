import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import network, base_function, external_function
from util import task
import itertools

from torch import nn
from math import log, sqrt, pi
from .PID import PIDControl


def kl_divergence_norm(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), -1)
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), -1)

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    
    return total_kld, dimension_wise_kld, mean_kld

def kl_divergence_regular(mu1, logvar1, mu2, logvar2):
    batch_size = mu1.size(0)
    assert batch_size != 0
    if mu1.data.ndimension() == 4:
        mu1 = mu1.view(mu1.size(0), -1)
    if logvar1.data.ndimension() == 4:
        logvar1 = logvar1.view(logvar1.size(0), -1)
    if mu2.data.ndimension() == 4:
        mu2 = mu2.view(mu2.size(0), -1)
    if logvar2.data.ndimension() == 4:
        logvar2 = logvar2.view(logvar2.size(0), -1)

    klds = -0.5*(1 + logvar1 - logvar2 - logvar1.exp() / logvar2.exp() - (mu1 - mu2).pow(2) / logvar2.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    
    return total_kld, dimension_wise_kld, mean_kld


class Pspflow(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "Pspflow"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=5, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths')
            parser.add_argument('--max_grad_clip', type=float, default=5.0, help='max_grad_clip for flow network')
            parser.add_argument('--max_grad_norm', type=float, default=100.0, help='max_grad_norm for flow network')
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=20.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')
            parser.add_argument('--lambda_gp', type=float, default=10.0, help='weight for wgan-gp loss')
            parser.add_argument('--lambda_per', type=float, default=30.0, help='weight for image perceptual loss')
            parser.add_argument('--lambda_sty', type=float, default=1000.0, help='weight for style loss')

        return parser

    def __init__(self, opt):
        """Initial the pluralistic model"""
        BaseModel.__init__(self, opt)
        self.n_bins = 2.0 ** opt.n_bits

        self.loss_names = ['kl_rec', 'kl_g', 'kl_flow', 'app_rec', 'app_g', 'per_rec', 'per_g', 'sty_rec', 'sty_g', 'ad_g', 'img_d', 'ad_rec', 'img_d_rec']
        self.visual_names = ['img_m', 'img_c', 'img_truth', 'img_out', 'img_g', 'img_rec']
        self.value_names = ['u_m', 'sigma_m', 'u_post', 'sigma_post', 'u_prior', 'sigma_prior']
        self.model_names = ['E', 'Flow', 'G', 'D', 'D_rec']
        self.distribution = []

        # define the inpainting model
        self.net_E = network.define_e(ngf=32, z_nc=128, img_f=128, layers=6, norm='none', activation='SiLU',
                                      init_type='orthogonal', gpu_ids=opt.gpu_ids)  # fineSize // 2 ** layer = 输入flow的size = 256 // 2**6 = 4
        self.net_Flow = network.define_flow(in_channel=128,  n_flow=8, n_block=2, affine=not opt.no_affine, no_lu=False, 
                                            activation='SiLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_G = network.define_g(ngf=32, z_nc=128, img_f=128, L=1, layers=6, output_scale=opt.output_scale,
                                      norm='instance', activation='SiLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_R = network.define_r(output_nc=3, ngf=64, z_nc=512, img_f=512, L=1, layers=5, norm='instance', activation='SiLU', init_type='orthogonal', gpu_ids=opt.gpu_ids):
        
        # define the discriminator model
        self.net_D = network.define_d(ndf=32, img_f=128, layers=6, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_D_rec = network.define_d(ndf=32, img_f=128, layers=6, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        if self.opt.is_PID:
            ## init PID control
            self.PID = PIDControl()
            self.Kp = 0.01
            self.Ki = -0.0001
            self.Kd = 0.0

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.resnet_loss = ResNetPL(weights_path='pretrained')
            self.vgg = VGG19()
            if len(self.gpu_ids) > 0:
                self.resnet_loss = self.resnet_loss.cuda()
                self.vgg = self.vgg.cuda()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()), filter(lambda p: p.requires_grad, self.net_R.parameters()),
                        filter(lambda p: p.requires_grad, self.net_E.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_Flow = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_Flow.parameters()),), lr=opt.lr_flow, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                                filter(lambda p: p.requires_grad, self.net_D_rec.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_Flow)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.setup(opt)

    def get_z_sample(self):
        input_size = self.opt.fineSize // (2**6)
        z_samples = []
        z_shapes = self.calc_z_shapes(128, input_size, n_flow=8, n_block=2)
        for z in z_shapes:
            z_new = torch.randn(self.opt.batchSize, *z) * self.opt.temp
            if len(self.gpu_ids) > 0:
                z_sample = z_new.cuda(self.gpu_ids[0])
            z_samples.append(z_sample)
        return z_samples
            
    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_names = self.input['name']
        self.img = input['to_im']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img
        self.img_m = (1 - self.mask) * self.img_truth
        self.img_c = self.mask * self.img_truth

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)

    def test(self, iter=None, save_path=None):
        """Forward function used in test time"""
        # save the groundtruth and masked image
        self.save_results(self.img_truth, iter=iter, data_name='truth', save_path=save_path)
        self.save_results(self.img_m, iter=iter, data_name='masked', save_path=save_path)

        # encoder process
        distribution, f = self.net_E(self.img_m)
        q_distribution = torch.distributions.Normal(distribution[-1][0], distribution[-1][1].exp())
        print(distribution[-1][1].max(), distribution[-1][1].min())
        print(distribution[-1][0].max(), distribution[-1][0].min())
        scale_mask = task.scale_img(self.mask, size=[f[3].size(2), f[3].size(3)])

        # decoder process
        for i in range(self.opt.nsampling):
            z_q = q_distribution.sample()
            z = self.get_z_sample()
            z_p = self.net_Flow.module.reverse(torch.cat([z, z_q], dim=1))
            z_in = z_q + z_p
            
            self.img_g, attn = self.net_G(z_in, f_m=f[-1], f_e=f[3], mask=scale_mask.chunk(3, dim=1)[0])
            out = self.net_R(self.img_g[-1])
            self.img_out = (1 - self.mask) * self.img_truth + self.mask * out
            self.save_results(self.img_out, iter=iter, sample=i, data_name='out', save_path=save_path)

    def val(self):
        """Forward function used in test time"""

        # encoder process
        distribution, f = self.net_E(self.img_m)
        q_distribution = torch.distributions.Normal(distribution[-1][0], distribution[-1][1].exp())
        scale_mask = task.scale_img(self.mask, size=[f[3].size(2), f[3].size(3)])

        # decoder process
        img_outs = []
        for i in range(3):
            z_q = q_distribution.sample()
            z = self.get_z_sample()
            z_p = self.net_Flow.module.reverse(z)
            z_in = z_q + z_p
            
            self.img_g, attn = self.net_G(z_in, f_m=f[-1], f_e=f[3], mask=scale_mask.chunk(3, dim=1)[0])
            out = self.net_R(self.img_g[-1])
            self.img_out = self.mask * out.detach() + (1 - self.mask) * self.img_m
            self.score = self.net_D(self.img_out)
            img_outs.append(self.img_out)
        return self.img_truth, self.img_m, img_outs

    def get_distribution(self, distributions):
        """Calculate encoder distribution for img_m, img_c"""
        # get distribution
        p_distribution, q_distribution, kl_rec, kl_g = 0, 0, 0, 0
        self.distribution = []
        for distribution in distributions:
            p_mu, p_logsigma, q_mu, q_logsigma = distribution
            p_distribution = torch.distributions.Normal(p_mu, p_logsigma.exp())
            q_distribution = torch.distributions.Normal(q_mu, q_logsigma.exp())

            # kl divergence
            kl_rec += kl_divergence_regular(p_mu, p_logsigma, torch.zeros_like(p_mu), (torch.ones_like(p_logsigma)).log())[0]
            if self.opt.train_paths == "one":
                kl_g += kl_divergence_regular(q_mu, q_logsigma, torch.zeros_like(p_mu), (torch.ones_like(p_logsigma)).log())[0]
            elif self.opt.train_paths == "two":
                kl_g += kl_divergence_regular(q_mu, q_logsigma, p_mu.detach(), p_logsigma.detach())[0]
            self.distribution.append([torch.zeros_like(p_mu), (torch.ones_like(p_logsigma)).log(), p_mu, p_logsigma, q_mu, q_logsigma])

        return p_distribution, q_distribution, kl_rec, kl_g
    
    def calc_loss(self, log_p, logdet, image_size, n_bins):
        # log_p = calc_log_p([z_list])
        n_pixel = image_size * image_size * 3

        loss = -log(n_bins) * n_pixel
        loss = loss + logdet + log_p

        return (
            (-loss / (log(2) * n_pixel)).mean(),
            (log_p / (log(2) * n_pixel)).mean(),
            (logdet / (log(2) * n_pixel)).mean(),
        )

    def calc_z_shapes(self, n_channel, input_size, n_flow, n_block):
        z_shapes = []

        for i in range(n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes

    def get_G_inputs(self, p_distribution, q_distribution, f):
        """Process the encoder feature and distributions for generation network"""
        f_m = torch.cat([f[-1].chunk(2)[0], f[-1].chunk(2)[0]], dim=0)
        f_e = torch.cat([f[3].chunk(2)[0], f[3].chunk(2)[0]], dim=0)
        scale_mask = task.scale_img(self.mask, size=[f_e.size(2), f_e.size(3)])
        mask = torch.cat([scale_mask.chunk(3, dim=1)[0], scale_mask.chunk(3, dim=1)[0]], dim=0)
        z_p = p_distribution.rsample()
        z_q = q_distribution.rsample()
        z_q = z_q + z_p
        z = torch.cat([z_p, z_q], dim=0)
        return z_p, z_q, z, f_m, f_e, mask

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        distributions, f = self.net_E(self.img_m, self.img_c)
        p_distribution, q_distribution, self.kl_rec, self.kl_g = self.get_distribution(distributions)

        # decoder process
        z_p, z_q, z, f_m, f_e, mask = self.get_G_inputs(p_distribution, q_distribution, f)
        results, attn = self.net_G(z, f_m, f_e, mask)
        log_p, logdet, _ = self.net_Flow(torch.cat([z_p, z_q], dim=1))
        logdet = logdet.mean()
        self.kl_flow, log_p, log_det = self.calc_loss(log_p, logdet, self.opt.fineSize // 2**6, self.n_bins)
        self.img_rec = []
        self.img_g = []
        for result in results:
            img_rec, img_g = result.chunk(2)
            self.img_rec.append(img_rec)
            self.img_g.append(img_g)
        out = self.net_R(self.img_g[-1])
        self.img_out = out.detach() * self.mask + (1 - self.mask) * self.img_truth

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty * self.opt.lambda_gp

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D, self.net_D_rec)
        self.loss_img_d = self.backward_D_basic(self.net_D, self.img_truth, self.img_g[-1])
        self.loss_img_d_rec = self.backward_D_basic(self.net_D_rec, self.img_truth, self.img_rec[-1])

    def backward_G(self, total_iteration):
        """Calculate training loss for the generator"""

        # encoder kl loss
        C = min(total_iteration // 100000 * 5, 50)
        if self.opt.is_PID:
            self.beta_kl_rec, _ = self.PID.pid(self.opt.KL_Loss, self.kl_rec.item(), self.Kp, self.Ki, self.Kd)
            self.beta_kl_g, _ = self.PID.pid(self.opt.KL_Loss, self.kl_g.item(), self.Kp, self.Ki, self.Kd)
            self.loss_kl_rec = self.kl_rec * self.beta_kl_rec * self.opt.lambda_rec
            self.loss_kl_g = self.kl_g * self.beta_kl_g * self.opt.lambda_rec
        else:
            self.loss_kl_rec = self.L1loss(self.kl_rec, C * torch.ones_like(self.kl_rec)) * self.opt.lambda_kl
            self.loss_kl_g = self.L1loss(self.kl_g, C * torch.ones_like(self.kl_g)) * self.opt.lambda_kl

        self.loss_kl_flow = self.kl_flow.mean()

        # generator adversarial loss
        base_function._freeze(self.net_D, self.net_D_rec)
        # g loss fake
        D_fake = self.net_D(self.img_g[-1])
        self.loss_ad_g = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        D_fake = self.net_D(self.out)
        self.loss_ad_g += self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # rec loss fake
        D_fake = self.net_D_rec(self.img_rec[-1])
        D_real = self.net_D_rec(self.img_truth)
        self.loss_ad_rec = self.L2loss(D_fake, D_real) * self.opt.lambda_g

        # calculate l1 loss for multi-scale outputs
        loss_app_rec, loss_app_g = 0, 0
        for i, (img_rec_i, img_fake_i, img_real_i, mask_i) in enumerate(zip(self.img_rec, self.img_g, self.scale_img, self.scale_mask)):
            loss_app_rec += self.L1loss(img_rec_i, img_real_i)
            if self.opt.train_paths == "one":
                loss_app_g += self.L1loss(img_fake_i, img_real_i)
            elif self.opt.train_paths == "two":
                loss_app_g += self.L1loss(img_fake_i*(1-mask_i), img_real_i*(1-mask_i))
        loss_app_g += self.L1loss(self.img_out, self.img_truth)
        self.loss_app_rec = loss_app_rec * self.opt.lambda_rec
        self.loss_app_g = loss_app_g * self.opt.lambda_rec

        # calculate perceptual loss
        self.loss_per_rec = self.resnet_loss(self.img_rec[-1], self.img_truth) * self.opt.lambda_per
        self.loss_per_g = self.resnet_loss(self.img_g[-1], self.img_truth) * self.opt.lambda_per
        self.loss_per_g += self.resnet_loss(self.img_out, self.img_truth) * self.opt.lambda_per
        # calculate style loss
        self.loss_sty_rec = self.style_loss(self.img_rec[-1], self.img_truth, self.vgg) * self.opt.lambda_sty
        self.loss_sty_g = self.style_loss(self.img_g[-1], self.img_truth, self.vgg) * self.opt.lambda_sty
        self.loss_sty_g = self.style_loss(self.img_out, self.img_truth, self.vgg) * self.opt.lambda_sty

        # if one path during the training, just calculate the loss for generation path
        if self.opt.train_paths == "one":
            self.loss_app_rec = self.loss_app_rec * 0
            self.loss_ad_rec = self.loss_ad_rec * 0
            self.loss_kl_rec = self.loss_kl_rec * 0

        total_loss = 0

        for name in self.loss_names:
            if name != 'img_d' and name != 'img_d_rec' and name != 'kl_flow':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()
        self.loss_kl_flow.backward()
        if self.opt.max_grad_clip is not None and self.opt.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(self.net_Flow.parameters(), self.opt.max_grad_clip)
        if self.opt.max_grad_norm is not None and self.opt.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net_Flow.parameters(), self.opt.max_grad_norm)

    def style_loss(self, x_r, x_gt, vgg):
        x_gt_vgg = vgg((x_gt + 1.0) / 2.0)
        x_r_vgg = vgg((x_r + 1.0) / 2.0)
        loss = F.l1_loss(self.compute_gram(x_r_vgg['relu2_2']), self.compute_gram(x_gt_vgg['relu2_2'])) + \
                F.l1_loss(self.compute_gram(x_r_vgg['relu3_4']), self.compute_gram(x_gt_vgg['relu3_4'])) + \
                F.l1_loss(self.compute_gram(x_r_vgg['relu4_4']), self.compute_gram(x_gt_vgg['relu4_4'])) + \
                F.l1_loss(self.compute_gram(x_r_vgg['relu5_2']), self.compute_gram(x_gt_vgg['relu5_2']))

        return loss

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def optimize_parameters(self, total_iteration):
        """update network weights"""
        # compute the image completion results
        if total_iteration <= 1:
            with torch.no_grad():
                self.forward()
        else:
            self.forward()
            # optimize the discrinimator network parameters
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            # optimize the completion network parameters
            self.optimizer_G.zero_grad()
            self.optimizer_Flow.zero_grad()
            self.backward_G(total_iteration)
            self.optimizer_G.step()
            self.optimizer_Flow.step()


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]
from .ade20k import ModelBuilder

class ResNetPL(nn.Module):
    def __init__(self, weight=1,
                 weights_path=None, arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        print('*'*10, weights_path)
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target, spatial_discounting_mask_tensor=None):
        # pred and target is [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        if spatial_discounting_mask_tensor is not None:
            H = spatial_discounting_mask_tensor.size()[2]
            mse_list = []
            for cur_pred, cur_target in zip(pred_feats, target_feats):
                h = cur_pred.size()[2]
                scale_factor = h / H
                scale_sd_mask_tensor = F.interpolate(spatial_discounting_mask_tensor, scale_factor=scale_factor,
                              mode='bilinear', align_corners=True)
                mse = F.mse_loss(cur_pred * scale_sd_mask_tensor, cur_target * scale_sd_mask_tensor)
                mse_list.append(mse)

            result = torch.stack(mse_list).sum() * self.weight
        else:
            result = torch.stack([F.mse_loss(cur_pred, cur_target) for cur_pred, cur_target in zip(pred_feats, target_feats)]).sum() * self.weight
        return result


import torchvision.models as models
# Assume input range is [0, 1]
class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
