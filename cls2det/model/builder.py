import torch

from .model import resnet18


def model_builder(cfg):
    if cfg.gpu and torch.cuda.is_available():
        print('=> use GPU: {}'.format(cfg.gpu))
        device = torch.device(f'cuda:{cfg.gpu}')
    else:
        print('=> use CPU')
        device = torch.device('cpu')

    if cfg.model.arch != 'resnet18':
        raise Exception('currently this tool only support model "resnent18".')
    print('=> building pre-trained model {}'.format(cfg.model.arch))
    model = resnet18(pretrained=cfg.model.pre_trained)

    model = model.to(device)
    model.eval()

    return model, device
