import torch


class ResBlock(torch.nn.Module):
    def __init__(self, n_dims_channel_in: int, n_dims_channel_out: int, activate: torch.nn.Module, label: str) -> None:
        super().__init__()
        self.skip = torch.nn.Conv2d(n_dims_channel_in, n_dims_channel_out, kernel_size=1)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_dims_channel_in, n_dims_channel_out, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(n_dims_channel_out),
            activate,
            torch.nn.Conv2d(n_dims_channel_in, n_dims_channel_out, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(n_dims_channel_out),
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if label == "down" else torch.nn.ConvTranspose2d(n_dims_channel_out, n_dims_channel_out, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, xs):
        return self.pool(self.cnn(xs) + self.skip(xs))


class Encoder(torch.nn.Module):
    def __init__(self, n_layers: int, n_dims_hidden: int, n_params: int, activate: torch.nn.Module):
        super().__init__()

        self.activate = activate
        self.n_dims_hidden = n_dims_hidden

        self.n_layers = n_layers
        for i in range(self.n_layers):
            setattr(self, f"kernel{i}", ResBlock(self.n_dims_hidden, self.n_dims_hidden, self.activate, "down"))

        self.estimator_mean = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_dims_hidden, n_params)
        )
        self.estimator_log_std = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.n_dims_hidden, n_params)
        )

    def forward(self, images: torch.Tensor):
        for i in range(self.n_layers):
            images = self.activate(getattr(self, f"kernel{i}")(images))
        return self.estimator_mean(images), self.estimator_log_std(images)



class Decoder(torch.nn.Module):
    def __init__(self, n_layers: int, n_dims_hidden: int, n_params: int, activate: torch.nn.Module):
        super().__init__()

        self.activate = activate
        self.n_dims_hidden = n_dims_hidden

        self.estimator_inv = torch.nn.Sequential(
            torch.nn.Linear(n_params, self.n_dims_hidden),
            torch.nn.Unflatten(1, (self.n_dims_hidden, 1, 1))
        )

        self.n_layers = n_layers
        for i in range(self.n_layers):
            setattr(self, f"kernel{i}", ResBlock(self.n_dims_hidden, self.n_dims_hidden, self.activate, "up"))

    def forward(self, params: torch.Tensor):
        images = self.estimator_inv(params)
        for i in range(self.n_layers):
            images = self.activate(getattr(self, f"kernel{i}")(images))
        return images


class VAE(torch.nn.Module):
    def __init__(self, n_params: int) -> None:
        super().__init__()

        self.activate = torch.nn.GELU()
        self.n_layers = 10
        self.n_dims_hidden = 32

        self.lift = torch.nn.Linear(3, self.n_dims_hidden)

        self.encoder = Encoder(self.n_layers, self.n_dims_hidden, n_params, self.activate)
        self.decoder = Decoder(self.n_layers, self.n_dims_hidden, n_params, self.activate)

        self.project = torch.nn.Sequential(
            torch.nn.Linear(self.n_dims_hidden, 64),
            self.activate,
            torch.nn.Linear(64, 3)
        )

    def forward(self, images: torch.Tensor):
        images = self.lift(images).permute(0, 3, 1, 2) # (#batch, n_dims_hidden, height, width)

        means, log_stds = self.encoder(images)
        params = (torch.exp(log_stds) * torch.randn_like(means) + means)
        images = self.decoder(params).permute(0, 2, 3, 1) # (#batch, height, width, 3)

        images = self.project(images)
        return images, means, log_stds
