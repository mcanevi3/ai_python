import torch
import torch.nn as nn

class LTINetwork(nn.Module):
    def __init__(self, n_states=2, n_inputs=1, n_outputs=1):
        super().__init__()
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # Learnable state update matrices
        self.A = nn.Parameter(torch.eye(n_states))  # (n,n)
        self.B = nn.Parameter(torch.randn(n_states, n_inputs) * 0.1)  # (n,m)
        # Output matrix C: either learnable or fixed
        C_init = torch.zeros(n_outputs, n_states)
        C_init[0, 0] = 1.0  # default C = [1 0] if not learned
        self.C = nn.Parameter(C_init)  # Learnable
        # Internal state: not a parameter
        self.register_buffer("x_hat", torch.zeros(n_states))  # (n,)

    def reset_state(self, x0=None):
        if x0 is None:
            self.x_hat = torch.zeros(self.n_states)
        else:
            self.x_hat = x0.detach().clone()

    def forward(self, u):
        # u: shape (n_inputs,)
        self.x_hat = self.A @ self.x_hat + self.B @ u
        y_hat = self.C @ self.x_hat  # Output computation
        return y_hat