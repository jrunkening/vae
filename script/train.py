from pathlib import Path

from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim
from tqdm import tqdm

from vae.data import CelebAHQ
from vae.model import VAE


def calc_loss_kld(means: torch.Tensor, log_stds: torch.Tensor):
    return (-0.5 * (1 + 2*log_stds - means.square() - (2*log_stds).exp()).sum(dim=-1)).mean()


class Trainer(torch.nn.Module):
    def __init__(
        self,
        model: VAE,
        dtype=torch.float32, device=torch.device("cuda", index=0)
    ):
        super().__init__()

        self.dtype = dtype
        self.device = device

        self.model = model

    def train(
        self,
        params: torch.nn.Parameter | List[torch.nn.Parameter],
        dataset: Dataset,
        lr: float = 1e-4, n_epoch: int = 500, size_batch: int = 4,
        verbose: bool = False, verbose_image: bool = False, path_image: Optional[Path] = None,
        path_params: Optional[List[Path]] = None, save_on_every_k_iter: int = 0,
    ):
        optimizer = torch.optim.Adam(params, lr=lr)

        self.model.train(True)
        for i_epoch in range(n_epoch):
            dataloader = DataLoader(dataset, size_batch, shuffle=True, num_workers=4, persistent_workers=True)
            for i, images_gt in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()

                images_gt = images_gt.to(self.dtype).to(self.device)

                images_predicted, means, log_stds = self.model.forward(images_gt)

                loss_colors = torch.nn.functional.l1_loss(images_predicted, images_gt)
                loss_ssim = (1 - ssim(images_predicted.permute(0, 3, 1, 2), images_gt.permute(0, 3, 1, 2), size_average=False, nonnegative_ssim=True).mean())/2
                loss_reconstruction = sum([
                    loss_colors,
                    loss_ssim
                ])
                loss_kld = calc_loss_kld(means, log_stds)
                loss = loss_reconstruction + loss_kld

                loss.backward()
                optimizer.step()

                i_iter = i_epoch*len(dataset) + i + 1
                log = []
                if verbose:
                    log = [
                        "\n",
                        f"\r\033[0K [{type(self.model).__name__}] iter: {i_iter}\n",
                        f"\r\033[0K loss: {loss}\n",
                        f"\r\033[0K loss_colors: {loss_colors}\n",
                        f"\r\033[0K loss_ssim: {loss_ssim}\n",
                        f"\r\033[0K loss_kld: {loss_kld}\n",
                    ]
                    print(*log, len(log)*"\033[1F", end="", flush=True)
                if save_on_every_k_iter > 0 and i_iter % save_on_every_k_iter == 0:
                    torch.save(self.model, path_params)
            print(len(log)*"\n", end="", flush=True)


if __name__ == "__main__":
    dtype = torch.float32
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using {device} device")

    root_params_celebahq = Path.home().joinpath("Data", "CelebAMask-HQ")
    root_result = Path(__file__).parent.parent.joinpath("result")
    root_result.mkdir(parents=True, exist_ok=True)

    dataset = CelebAHQ(root_params_celebahq)

    name_model = "vae"

    if root_result.joinpath(f"{name_model}.pt").exists() and input("train from scratch? y/[n]: ") != "y":
        print(f"continue training...")
        model = torch.load(root_result.joinpath(f"{name_model}.pt"), weights_only=False).to(dtype).to(device)
    else:
        print(f"train from scratch!!!")
        model = VAE(dataset.height, dataset.width, 300).to(dtype).to(device)

    trainer = Trainer(model, dtype=dtype, device=device)

    trainer.train(
        model.parameters(), dataset,
        verbose = True,
        verbose_image = True, path_image = root_result.joinpath(f"{name_model}.png"),
        path_params = root_result.joinpath(f"{name_model}.pt"), save_on_every_k_iter = 500,
    )
