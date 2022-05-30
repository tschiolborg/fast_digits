from fastai.vision.all import (
    untar_data,
    URLs,
    vision_learner,
    aug_transforms,
    accuracy,
    ShowGraphCallback,
)
from fastai.vision.data import (
    DataBlock,
    ImageBlock,
    CategoryBlock,
    get_image_files,
    RandomSplitter,
    parent_label,
)
from fastai.vision.models import resnet18


def train():

    # get data
    path = untar_data(URLs.MNIST)

    # loader
    block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2),
        get_y=parent_label,
        batch_tfms=aug_transforms(mult=2.0, do_flip=False, size=28),
    )
    loaders = block.dataloaders(path / "training", bs=64)

    # resnet18
    learn = vision_learner(
        loaders,
        arch=resnet18,
        metrics=accuracy,
    )

    # train
    learn.fine_tune(5, base_lr=1e-2, cbs=[ShowGraphCallback()])

    learn.export("models/model_1.pkl")


if __name__ == "__main__":
    train()
