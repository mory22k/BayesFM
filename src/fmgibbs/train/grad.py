import torch
import torch.nn as nn
import torch.optim as optim
import warnings

class FactorizationMachineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b, w, v):
        q = torch.matmul(x, v)  # (batch_size, k)
        ctx.save_for_backward(q, x, v)

        linear_part = torch.sum(w * x, dim=1)
        interaction_part = 0.5 * torch.sum(q ** 2 - torch.matmul(x ** 2, v ** 2), dim=1)
        output = b + linear_part + interaction_part
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, x, v = ctx.saved_tensors
        batch_size = x.size(0)

        grad_bias = grad_output.sum().unsqueeze(0)
        grad_linear = torch.einsum('ti,t->i', x, grad_output)

        grad_v = torch.zeros_like(v)
        for f in range(v.size(1)):
            q_f = q[:, f]
            grad_v[:, f] = torch.einsum('t,tj->j', grad_output, (q_f.unsqueeze(1) - x * v[:, f].unsqueeze(0)) * x)

        grad_v = grad_v / batch_size
        grad_bias /= batch_size
        grad_linear /= batch_size

        return None, grad_bias, grad_linear, grad_v

class FactorizationMachineModel(nn.Module):
    def __init__(self, num_features, dim_hidden):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.randn(num_features, dim_hidden))

    def forward(self, x):
        return FactorizationMachineFunction.apply(x, self.b, self.w, self.v)

class FactorizationMachineGradRegressor:
    def __init__(self, dim_hidden=5, max_iter=100, warm_start=True, seed=None, optimizer_class=None, criterion=None):
        self.dim_hidden = dim_hidden
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.seed = seed
        self.model = None
        self.error_history_ = None

        if optimizer_class is None:
            optimizer_class = optim.SGD
        if criterion is None:
            criterion = nn.MSELoss()
        self.optimizer_class = optimizer_class
        self.optimizer = None
        self.criterion = criterion

        if seed is not None:
            torch.manual_seed(seed)

    def initialize_model(self, num_features):
        self.model = FactorizationMachineModel(num_features, self.dim_hidden)
        self.optimizer = self.optimizer_class(self.model.parameters())

    def fit(self, X, y, optimizer_class=None, criterion=None, batch_size=None, shuffle=False, logger=None, record_error=False):
        if self.model is None or not self.warm_start:
            self.initialize_model(X.shape[1])

        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        if batch_size is None:
            if shuffle:
                warnings.warn("Shuffle is set to True, but batch_size is None. Shuffle will be ignored.")
            data_loader = [(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))] # batch_size=len(X)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        self.error_history_ = train_model(self.model, data_loader, self.optimizer, self.criterion, self.max_iter, logger, record_error)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()

    def set_params(self, b, w, v):
        self.model = FactorizationMachineModel(w.shape[0], v.shape[1])
        self.model.b.data = torch.tensor(b)
        self.model.w.data = torch.tensor(w)
        self.model.v.data = torch.tensor(v)
        self.optimizer = self.optimizer_class(self.model.parameters())

    def get_params(self):
        return self.model.b.data.detach().numpy().item(), self.model.w.data.detach().numpy(), self.model.v.data.detach().numpy()

def train_model(model, data_loader, optimizer, criterion, max_iter, logger=None, record_error=False):
    error_history = []
    for epoch in range(max_iter):
        total_loss = 0
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if logger is not None or record_error:
            error = torch.sqrt(torch.mean(torch.tensor(total_loss / len(data_loader))))
        if record_error:
            error_history.append(error)
        if logger:
            logger.info(f'{epoch:4d} | {error:10.3e}')
    return error_history
