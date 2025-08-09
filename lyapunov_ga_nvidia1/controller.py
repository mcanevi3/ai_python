import torch
import torch.nn as nn

from device import *

class Controller(nn.Module):
    def __init__(self,n=3):
        super().__init__()
        self.Fs = nn.Sequential(
        # nn.Linear(n, 3, bias=True),
        # nn.Linear(3, 3, bias=True),
        nn.Linear(3, 1, bias=False,device=device)
        )

    def forward(self,x):
        return self.Fs(x)

    def get_controller(self):
        with torch.no_grad():
            return torch.cat([p.view(-1) for p in self.Fs.parameters()])
        
    def controller_param_count(self):
        total_params = sum(p.numel() for p in self.Fs.parameters())
        return total_params
    
    def set_controller(self,fvec):
        with torch.no_grad():
            fvec_t = torch.as_tensor(fvec, dtype=torch.float32, device=device)
            idx = 0
            for p in self.Fs.parameters():
                numel = p.numel()
                p.copy_(fvec_t[idx:idx + numel].view_as(p))
                idx += numel
            if idx != fvec_t.numel():
                raise ValueError("Size of fvec does not match total number of model parameters")
    
    def print_controller(self):
        for i, layer in enumerate(self.Fs):
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:
                    print(f"Layer {i} - ({layer.in_features}x{layer.out_features} with bias)")
                else:
                    print(f"Layer {i} - ({layer.in_features}x{layer.out_features})")

                print("  Weights:")
                print(layer.weight.data.detach().numpy())
                if layer.bias is not None:
                    print("  Biases:")
                    print(layer.bias.data.detach().numpy())
            else:
                print(f"Layer {i} - {layer.__class__.__name__}")
    
    def save_controller(self,filename):
        torch.save(self.Fs.state_dict(), filename)
    
    def load_controller(self,filename):
        self.Fs.load_state_dict(torch.load(filename))
        self.Fs.eval()

if __name__=="__main__":
    con=Controller()
    print(f"The controller has {con.controller_param_count()} parameters")
    con.print_controller()
    with torch.no_grad():
        x=torch.tensor([1.0,1.0,1.0])
        u=con(x)
        print(u)
