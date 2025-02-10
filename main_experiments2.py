import torch


logits = torch.randn(4, requires_grad=True)

target_entropy = torch.FloatTensor([0.0001])

optimizer = torch.optim.Adam([logits], lr=5e-1)
for i in range(100):
    distribution = torch.distributions.Categorical(logits=logits)
    loss = torch.nn.functional.mse_loss(target_entropy, distribution.entropy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)

distribution = torch.distributions.Categorical(logits=logits)
probs = distribution.probs
print(distribution.probs.min())
print(distribution.probs.max())
print(probs[probs.argsort()[-10:]])
print()
