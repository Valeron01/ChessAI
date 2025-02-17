import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, in_channels, dim_feedforward):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, dim_feedforward),
            nn.LayerNorm([dim_feedforward]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_feedforward, in_channels)
        )

    def forward(self, x):
        return self.block(x)


class DenseResBlock(nn.Module):
    def __init__(self, in_channels, dim_feedforward):
        super().__init__()
        self.blocks = nn.Sequential(
            FeedForward(in_channels, dim_feedforward),
            FeedForward(in_channels, dim_feedforward)
        )

    def forward(self, x):
        return self.blocks(x) + x


class BasicTransformerModel(nn.Module):
    def __init__(self, input_dim: int, dim_model: int, n_heads: int, dim_feedforward: int, n_layers: int, n_layers_head: int):
        super().__init__()
        self.dim_model = dim_model
        self.input_projection = nn.Linear(input_dim, dim_model)

        self.position_encoding = nn.Parameter(
            torch.randn(1, 64, dim_model)
        )

        self.conv_layer = nn.Sequential(
            nn.Conv2d(dim_model, dim_model, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_model, dim_model, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_model, dim_model, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_model, dim_model, 3, 1, 1),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_model, n_heads, dim_feedforward, batch_first=True),
            num_layers=n_layers
        )

        self.transformer_source = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_model, n_heads, dim_feedforward, batch_first=True),
            num_layers=n_layers_head
        )
        self.transformer_target = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_model, n_heads, dim_feedforward, batch_first=True),
            num_layers=n_layers_head
        )
        self.transformer_critic = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_model, n_heads, dim_feedforward, batch_first=True),
            num_layers=n_layers_head
        )
        self.critic_cls_token = nn.Parameter(
            torch.randn(1, 1, dim_model)
        )

        self.mlp_actor = nn.Sequential(
            DenseResBlock(dim_model * 2, dim_feedforward),
            nn.Linear(dim_model * 2, 1)
        )

        self.mlp_value = nn.Sequential(
            FeedForward(dim_model, dim_feedforward),
            FeedForward(dim_model, dim_feedforward),
            nn.Linear(dim_model, 1)
        )

        nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        nn.init.normal_(self.critic_cls_token, mean=0, std=0.02)

    def forward(self, field_tensor: torch.Tensor):
        assert field_tensor.ndim == 4
        assert field_tensor.shape[2] == 8
        assert field_tensor.shape[3] == 8

        with torch.autocast(field_tensor.device.type, torch.float16), torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True, enable_cudnn=True
        ):
            field_tensor_flattened = field_tensor.permute(0, 2, 3, 1)
            input_projection = self.input_projection(field_tensor_flattened).permute(0, 3, 1, 2)
            conv = self.conv_layer(input_projection).flatten(2).permute(0, 2, 1)
            conv_with_pos_encoding = self.position_encoding + conv
            transformer = self.transformer(conv_with_pos_encoding)
            source_embeddings = self.transformer_source(transformer)
            target_embeddings = self.transformer_target(transformer)

            actor_features = torch.cat([
                source_embeddings[:, :, None, :].tile(1, 1, 64, 1).flatten(1, 2),
                target_embeddings[:, None, :, :].tile(1, 64, 1, 1).flatten(1, 2)
            ], dim=-1)

            critic_embeddings = self.transformer_critic(torch.cat([
                transformer, self.critic_cls_token.tile([field_tensor.shape[0], 1, 1])
            ], dim=1))[:, -1, :]

        actor_logits = self.mlp_actor(actor_features).squeeze(-1)
        value = self.mlp_value(critic_embeddings).squeeze(-1)
        return torch.distributions.Categorical(logits=actor_logits), value


if __name__ == '__main__':
    model = BasicTransformerModel(**{
        "dim_model": 128,
        "n_heads": 8,
        "dim_feedforward": 256,
        "n_layers": 4,
        "n_layers_head": 1,
        "input_dim": 96
    }).cuda()
    field = torch.randn([3, 96, 8, 8]).cuda()
    actor, value = model(field)

    print(actor.entropy())
    print(value)
    print(actor.probs.max())
    print(actor.probs.min())

