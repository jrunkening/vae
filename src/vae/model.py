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
    def __init__(self, height: int, width: int, n_params: int):
        super().__init__()

        self.activate = torch.nn.ReLU()

        self.lift = torch.nn.Linear(3, 64)

        self.n_layers = 6
        for i in range(self.n_layers):
            setattr(self, f"kernel{i}", ResBlock(64, 64, self.activate, "down"))

        self.project = torch.nn.Sequential(
            torch.nn.Linear(64, 512),
            self.activate,
            torch.nn.Linear(512, 3)
        )

        self.estimator_mean = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16*16*3, n_params)
        )
        self.estimator_log_std = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16*16*3, n_params)
        )

    def forward(self, vertices: torch.Tensor):
        vertices = self.lift(vertices).permute(0, 3, 1, 2)

        for i in range(self.n_layers):
            vertices = self.activate(getattr(self, f"kernel{i}")(vertices))

        vertices = self.project(vertices.permute(0, 2, 3, 1))
        return self.estimator_mean(vertices), self.estimator_log_std(vertices)


class Decoder(torch.nn.Module):
    def __init__(self, height: int, width: int, n_params: int):
        super().__init__()

        self.height = height
        self.width = width
        self.n_layers = 6

        self.activate = torch.nn.ReLU()

        self.estimator_inv = torch.nn.Linear(n_params, 16*16*3)

        self.project_inv = torch.nn.Sequential(
            torch.nn.Linear(3, 512),
            self.activate,
            torch.nn.Linear(512, 64),
        )

        for i in range(self.n_layers):
            setattr(self, f"kernel{i}", ResBlock(64, 64, self.activate, "up"))

        self.lift_inv = torch.nn.Linear(64, 3)


    def forward(self, params: torch.Tensor):
        vertices = self.estimator_inv(params).reshape(params.shape[0], 16, 16, -1)
        vertices = self.project_inv(vertices).permute(0, 3, 1, 2)

        for i in range(self.n_layers):
            vertices = self.activate(getattr(self, f"kernel{i}")(vertices))

        vertices = self.lift_inv(vertices.permute(0, 2, 3, 1))
        return vertices


class VAE(torch.nn.Module):
    def __init__(self, height: int, width: int, n_params: int) -> None:
        super().__init__()
        self.encoder = Encoder(height, width, n_params)
        self.decoder = Decoder(height, width, n_params)

    def forward(self, xs: torch.Tensor):
        means, log_stds = self.encoder(xs)
        params = torch.exp(log_stds) * torch.randn_like(means) + means
        xs = self.decoder(params)
        return xs, means, log_stds
