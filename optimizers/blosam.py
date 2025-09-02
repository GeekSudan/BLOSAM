import torch
from torch.optim.optimizer import Optimizer

class BLOSAM(Optimizer):
    def __init__(self, params, lr=0.001, rho=0.05, p=2,
        xi_lr_ratio=3, momentum_theta=0.9, momentum_xi=0.9, weight_decay=0.0):
        if lr <= 0.0 or rho <= 0.0:
            raise ValueError("Invalid learning rate or rho")
        if p not in [2, float('inf')]:
            raise ValueError("Only p=2 and p=inf supported")

        defaults = dict(
            lr=lr, rho=rho, p=p,
            xi_lr_ratio=xi_lr_ratio,
            momentum_theta=momentum_theta,
            momentum_xi=momentum_xi,
            weight_decay=weight_decay
        )
        super(BLOSAM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['xi'] = torch.zeros_like(p.data)
                state['v_theta'] = torch.clone(p.grad).detach()
                state['v_xi'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            eta_theta = group['lr']
            eta_xi = eta_theta * group['xi_lr_ratio']
            rho = group['rho']
            p_norm = group['p']
            mu_theta = group['momentum_theta']
            mu_xi = group['momentum_xi'] * group['xi_lr_ratio']
            weight_decay = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state[p]
            xi = state['xi']
            v_theta = state['v_theta']
            v_xi = state['v_xi']    

            # Step 1: compute g(ξ + ∇_ξ f)
            u = xi + grad
            if p_norm == 2:
                norm = torch.norm(u)
                projected = u if norm <= rho else rho * u / norm
            else: # p == ∞
                projected = torch.clamp(u, -rho, rho)

            # Step 2: ξ update (fast dynamics)
            v_xi_new = mu_xi * v_xi + eta_xi * (-xi + projected)
            xi_new = xi + v_xi_new
            state['v_xi'] = v_xi_new
            state['xi'] = xi_new

            # Step 3: θ update (slow dynamics) + weight decay
            if weight_decay != 0:
                grad = grad.add(p.data, alpha=weight_decay)

            v_theta_new = mu_theta * v_theta - eta_theta * grad
            p.data.add_(v_theta_new)
            state['v_theta'] = v_theta_new