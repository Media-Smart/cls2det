import torch
import torch.nn.functional as F
from torchvision import transforms

from cls2det.detection.util import get_label
from cls2det.model import model_builder


class Classifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.labels = get_label(cfg.data.class_txt)
        self.trns = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.norm_cfg.mean,
                                 self.cfg.norm_cfg.std)])
        self.model, self.device = model_builder(self.cfg)

    def predict(self, img, type):
        with torch.no_grad():
            img = self.trns(img).unsqueeze(0)
            img = img.to(self.device)
            if type == 'fm':
                fm = self.model(img, type)
                fm = F.softmax(fm, dim=3)
                return fm[0]
            else:
                output, output_pre = self.model(img, type)
                output = F.softmax(output, dim=1)
                value, index = output.max(dim=1)
                label = self.labels[index]
                if type == 'cls':
                    return label, value.item()
                else:
                    return output_pre
