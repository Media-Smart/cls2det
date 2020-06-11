import torch

from .model import resnet18


def model_builder(cfg):
    if cfg.gpu is not None and torch.cuda.is_available():
        print('=> use GPU: {}'.format(cfg.gpu))
        device = torch.device(f'cuda:{cfg.gpu}')
    else:
        print('=> use CPU')
        device = torch.device('cpu')

    print('=> building pre-trained model resnet18')
    model = resnet18(pretrained=True)

    model = model.to(device)
    model.eval()

    return model, device
