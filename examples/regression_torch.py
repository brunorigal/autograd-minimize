import numpy as np
from autograd_minimize.torch_wrapper import torch_function_factory
from autograd_minimize import minimize 
import torch.nn as nn
import torch

#### Prepares data
X = np.random.random((200, 2))
y = X[:,:1]*2+X[:,1:]*0.4-1

#### Creates model
model = nn.Sequential(nn.Linear(2, 1))

# Transforms model into a function of its parameter
func, params, names = torch_function_factory(model, nn.MSELoss(), X, y)

# Minimization
res = minimize(func, params, method='trust-constr', backend='torch')

print('Fitted parameters:')
print(res.x)

mae = np.abs(model(torch.tensor(X, dtype=torch.float32)).cpu().detach().numpy()-y).mean()
print(f'mae: {mae}')