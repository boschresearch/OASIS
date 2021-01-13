import os
import numpy as np
import torch
import time
from scipy import linalg # For numpy FID
from pathlib import Path
from PIL import Image
import models.models as models
from utils.fid_folder.inception import InceptionV3
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------#
# This code is an adapted version of https://github.com/mseitzer/pytorch-fid
# --------------------------------------------------------------------------#

class fid_pytorch():
    def __init__(self, opt, dataloader_val):
        self.opt = opt
        self.dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model_inc = InceptionV3([block_idx])
        if opt.gpu_ids != "-1":
            self.model_inc.cuda()
        self.val_dataloader = dataloader_val
        self.m1, self.s1 = self.compute_statistics_of_val_path(dataloader_val)
        self.best_fid = 99999999
        self.path_to_save = os.path.join(self.opt.checkpoints_dir, self.opt.name, "FID")
        Path(self.path_to_save).mkdir(parents=True, exist_ok=True)

    def compute_statistics_of_val_path(self, dataloader_val):
        print("--- Now computing Inception activations for real set ---")
        pool = self.accumulate_inception_activations()
        mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
        print("--- Finished FID stats for real set ---")
        return mu, sigma

    def accumulate_inception_activations(self):
        pool, logits, labels = [], [], []
        self.model_inc.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image = data_i["image"]
                if self.opt.gpu_ids != "-1":
                    image = image.cuda()
                image = (image + 1) / 2
                pool_val = self.model_inc(image.float())[0][:, :, 0, 0]
                pool += [pool_val]
        return torch.cat(pool, 0)

    def compute_fid_with_valid_path(self, netG, netEMA):
        pool, logits, labels = [], [], []
        self.model_inc.eval()
        netG.eval()
        if not self.opt.no_EMA:
            netEMA.eval()
        with torch.no_grad():
            for i, data_i in enumerate(self.val_dataloader):
                image, label = models.preprocess_input(self.opt, data_i)
                if self.opt.no_EMA:
                    generated = netG(label)
                else:
                    generated = netEMA(label)
                generated = (generated + 1) / 2
                pool_val = self.model_inc(generated.float())[0][:, :, 0, 0]
                pool += [pool_val]
            pool = torch.cat(pool, 0)
            mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
            answer = self.numpy_calculate_frechet_distance(self.m1, self.s1, mu, sigma)
        netG.train()
        if not self.opt.no_EMA:
            netEMA.train()
        return answer

    def numpy_calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        Taken from https://github.com/bioinf-jku/TTUR
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1, sigma1, mu2, sigma2 = mu1.detach().cpu().numpy(), sigma1.detach().cpu().numpy(), mu2.detach().cpu().numpy(), sigma2.detach().cpu().numpy()

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            #print('wat')
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #print('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return out

    def update(self, model, cur_iter):
        print("--- Iter %s: computing FID ---" % (cur_iter))
        cur_fid = self.compute_fid_with_valid_path(model.module.netG, model.module.netEMA)
        self.update_logs(cur_fid, cur_iter)
        print("--- FID at Iter %s: " % cur_iter, "{:.2f}".format(cur_fid))
        if cur_fid < self.best_fid:
            self.best_fid = cur_fid
            is_best = True
        else:
            is_best = False
        return is_best

    def update_logs(self, cur_fid, epoch):
        try :
            np_file = np.load(self.path_to_save + "/fid_log.npy")
            first = list(np_file[0, :])
            sercon = list(np_file[1, :])
            first.append(epoch)
            sercon.append(cur_fid)
            np_file = [first, sercon]
        except:
            np_file = [[epoch], [cur_fid]]

        np.save(self.path_to_save + "/fid_log.npy", np_file)

        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(self.path_to_save + "/plot_fid", dpi=600)
        plt.close()


def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
