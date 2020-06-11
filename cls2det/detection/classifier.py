import torch
import torch.nn.functional as F
from torchvision import transforms

from .utils import get_label
from cls2det.model import model_builder


class Classifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.labels = get_label(cfg.data.class_txt)
        self.trns = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.model, self.device = model_builder(self.cfg)

    def predict(self, img, mode):
        with torch.no_grad():
            img = self.trns(img).unsqueeze(0)
            img = img.to(self.device)
            if mode == 'fm':
                feature_map = self.model(img, mode)
                feature_map = F.softmax(feature_map, dim=3)
                return feature_map[0]
            else:
                output, output_pre = self.model(img, mode)
                output = F.softmax(output, dim=1)
                value, index = output.max(dim=1)
                label = self.labels[index]
                if mode == 'cls':
                    return label, value.item()
                else:
                    return output_pre
