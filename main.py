import hydra

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from dataloader.dataset import CLS_Dataset, custom_collate_fn
from torchvision.transforms import v2

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    train_set = CLS_Dataset(cfg.db.train_path, cfg.db.pre_path, transform=v2.Compose([
        v2.Resize((cfg.db.width, cfg.db.height), antialias=True)
    ]))

    dataload = DataLoader(train_set,
                          batch_size=cfg.db.batch_size,
                          shuffle=cfg.db.shuffle,
                          drop_last=True,
                          num_workers=1)
    for i, data in enumerate(dataload):
        img, label = data
        print(img.shape)
        print(label.shape)
    # print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()