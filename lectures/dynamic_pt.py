import torch

# N is batch size;
# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensors to hold input and outputs.
dtype, device = torch.float, torch.device("cpu")
x = torch.randn(N, D_in,
                device=device,
                dtype=dtype)
y = torch.randn(N, D_out,
                device=device,
                dtype=dtype)

# create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device,
                 dtype=dtype,
                 requires_grad=True)
w2 = torch.randn(H, D_out, device=device,
                 dtype=dtype,
                 requires_grad=True)

# setup training
learning_rate = 1e-6

# execute the graph
for t in range(500):
    # forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # compute loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # backward pass
    loss.backward()

    # update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # manually zero the gradients
        # after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
