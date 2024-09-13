import copy
import urllib.request
from transformers import (
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters
from torch import nn
import pandas as pd
from src.config import ex
from src.modules import ITRTransformerSS
from demo import collator_func, get_image, get_text, get_pretrained_tokenizer

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")

    plt.savefig('vis/demo.png')

    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


def get_suit(text, img_path, tokenizer, _config, device):
    ret = dict()
    ret.update(get_image(img_path, _config, device))
    ret.update(get_text(text, tokenizer, _config))
    return ret


#@markdown To visualize which parts of the image activate for a given caption, we use the caption as the target label and backprop through the network using the image as the input.
#@markdown In the case of CLIP models with resnet encoders, we save the activation and gradients at the layer before the attention pool, i.e., layer4.
class Hook:
    """Attaches to a module and records its activations and gradients.
        Our gradCAM implementation registers a forward hook on the model at the specified layer. 
        This allows us to save the intermediate activations and gradients at that layer.
        """

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        output = output[0]
        self.data = output

        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def gradCAM(
    model: nn.Module,
    image_input: torch.Tensor,
    txt_target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the image_input.
    if image_input.grad is not None:
        image_input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.image_transformer.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model.encode_image(image_input)
        output.backward(txt_target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.

        B, D = grad.size(0), grad.size(-1)
        grad = grad[:, 1:].transpose(1, 2).view(B, D, 14, 14)
        act = act[:, 1:].transpose(1, 2).view(B, D, 14, 14)

        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        image_input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        pass
        # param.requires_grad_(requires_grad[name])
        
    return gradcam


@ex.automain
def main(_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    #@markdown #### CLIP model settings
    model = ITRTransformerSS(_config)
    model = model.cuda()

    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    vocabulary_list = list(tokenizer.vocab.keys())
    vocab_size = 30522
    counter = torch.zeros(vocab_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #@markdown #### Visualization settings
    blur = True #@param {type:"boolean"}

    df = pd.read_csv(f"../data/F30K/f30k_test.tsv", sep="\t")
    
    captions = df["title"].tolist()
    images = df["filepath"].tolist()

    for idx in range(1000):
        img_path = '../data/' + str(images[idx][3:])
        text = str(captions[idx]).lower()
        print(text)
        # text = 'women'
        # text = '[unused660]'
        text = '[PAD]'
        batch = get_suit(text, img_path, tokenizer, _config, device)

        mlm_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        # import pdb; pdb.set_trace()
        input_batch = collator_func([batch], mlm_collator, device)

        ret = model.forward(input_batch)

        # GradCam
        # image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_np = load_image(img_path, 224)
        # text_input = clip.tokenize([image_caption]).to(device)
        image_input = batch[f"image_features"]
        text_reps = ret['text_bottleneck_repre']

        attn_map = gradCAM(
            model,
            image_input,
            text_reps.float(),
            model.image_transformer.beit.encoder.layer[0] # getattr(model.image_transformer, saliency_layer)
        )
        attn_map = attn_map.squeeze().detach().cpu().numpy()

        viz_attn(image_np, attn_map, blur)
        import pdb; pdb.set_trace()
        


    _, index = torch.topk(counter, 10)
    index = index.cpu().detach().numpy().tolist()
    out = {}
    for ids in index:
        out[vocabulary_list[ids]] = counter[ids]



    