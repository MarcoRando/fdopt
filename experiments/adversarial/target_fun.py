import torch

import torchvision
import torchvision.transforms as transforms


class AdversarialTarget:
    def __init__(self, net, x, y, lam = 1.0, num_classes= 10, cw=False) -> None:
        self.net = net
        self.x = x # Image to attack
        self.y = y # Label to attack
        self.cw = cw
        self.lam = lam # Regularization parameter
        self.num_classes = num_classes
        self.inds_to_attack = torch.arange(self.num_classes) != self.y
        
    def apply_perturbation(self, v):
        # v is a batch of perturbations (n x d)
        im_arct = torch.arctanh(1.9999999 * self.x)
        im_pert = (0.5 * (im_arct.flatten() + v).tanh())
#        return self.normalize(im_pert.view( v.shape[0], *self.x.shape)) 
        return im_pert.view( v.shape[0], *self.x.shape)

    def get_distortion(self, v):
        with torch.no_grad():
            perturbed_imgs = self.apply_perturbation(v)
            return torch.linalg.norm(perturbed_imgs.view(v.shape[0], -1) - self.x.view(1, -1), dim=1).square()


    def __call__(self, v):
        # v is a batch of perturbations (n x d)
        with torch.no_grad():
            perturbed_imgs = self.apply_perturbation(v)
#            F_v = (torch.functional.F.softmax(self.net(perturbed_imgs), dim=1)).log()
            if self.cw:
                F_v = self.net(perturbed_imgs)
            else:
                F_v = (torch.functional.F.softmax(self.net(perturbed_imgs), dim=1)).log()
            score_target = F_v[:, self.y]# torch.maximum(F_v[:, self.y], torch.full((v.shape[0],), 1e-20, dtype=v.dtype, device=v.device))
            score_others = F_v[:, self.inds_to_attack].max(dim=1).values#torch.maximum(F_v[:, self.inds_to_attack].max(dim=1).values, torch.full((v.shape[0],), 1e-20, dtype=v.dtype, device=v.device))
#            print("probabilities:", self.net(perturbed_imgs), torch.functional.F.softmax(self.net(perturbed_imgs), dim=1))
#            # print(F_v[:, self.y], F_v[:, self.inds_to_attack].max(dim=1).values)
            attack_ls= torch.maximum( score_target - score_others, torch.zeros(1, dtype=v.dtype, device=v.device) - 1.0)
        #    attack_ls= torch.maximum(F_v[:, self.y] - F_v[:, self.inds_to_attack].max(dim=1).values, torch.zeros(1, dtype=v.dtype, device=v.device))
            reg = 0.5 * self.lam * torch.linalg.norm(perturbed_imgs.view(v.shape[0], -1) - self.x.view(1, -1), dim=1).square()
#            print(attack_ls, reg)
            return attack_ls  + reg
