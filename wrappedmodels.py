import open_clip
import torch
from abc import abstractmethod, ABC
from torchvision.transforms import v2
from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification, AutoModelForVision2Seq, BartForConditionalGeneration, BartTokenizer, BlipForConditionalGeneration, BlipProcessor
from timm.data.transforms_factory import create_transform
from transformers import SiglipModel, SiglipProcessor
import warnings
from transformers import GPTNeoXForCausalLM, AutoTokenizer, MambaForCausalLM, LlamaForCausalLM, LlamaTokenizer, GPT2Model, GPT2Tokenizer, \
      AutoModelForZeroShotObjectDetection , AutoProcessor, AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms

warnings.filterwarnings("ignore", message=".*xFormers is.*available.*")

class WrappedModel(torch.nn.Module, ABC):
    """Abstract Class, defining behavior for a model having its activations recorded"""
    def __init__(self, layers,*args,**kwargs):
        super().__init__()
        self.activations = {}
        self.model, self.transform = self.setup_model(**kwargs)
        self.layers = layers
        self.register_layers()

    @abstractmethod
    def setup_model(self, **kwargs):
        """Return:
            - model (nn.Module): the actual model
            - transform (Callable): preprocessing transformation applied to data
        """
        pass

    def forward(self, x):
        return self.model(x)

    @property
    @abstractmethod
    def layers_to_record(self):
        """Property: List of layers  (usually tranformer blocks) having their activations recorded"""
        pass

    @abstractmethod
    def d_vit(self, layer_id):
        """Size of the n-th layer, required to create SAEs"""
        pass

    def register_layers(self):
        for layer in self.layers:
            self.layers_to_record[layer].register_forward_hook(
                self.get_activations(layer)
            )

    def get_activations(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple) and len(output) == 1:
                self.activations[name] = output[0].detach()
            elif isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

class ClipVision(WrappedModel):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def setup_model(self, model_size='B', clip_model_name='ViT-B/32-quickgelu', clip_pretrained='openai',**kwargs):
        if   model_size=='B': clip_model_name='ViT-B/32-quickgelu'
        elif model_size=='L': clip_model_name='ViT-L/14-quickgelu'

        if "hf-hub" in clip_model_name: clip_pretrained=None
        clip, transform = open_clip.create_model_from_pretrained(
            clip_model_name, pretrained=clip_pretrained
        )

        return clip.visual, transform

    @property
    def layers_to_record(self):
        return self.model.transformer.resblocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp[0].in_features
    

class CocaVision(WrappedModel):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def setup_model(self, model_size='B',coca_model_name='coca_ViT-L-14', coca_pretrained='mscoco_finetuned_laion2B-s13B-b90k',**kwargs):
        if model_size=='B': coca_model_name='coca_ViT-B-32'
        if model_size=='L': coca_model_name='coca_ViT-L-14'
        clip, transform = open_clip.create_model_from_pretrained(
            coca_model_name, pretrained=coca_pretrained
        )

        return clip.visual, transform

    @property
    def layers_to_record(self):
        return self.model.transformer.resblocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp[0].in_features
    

class CocaText(WrappedModel):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def setup_model(self, coca_model_name='coca_ViT-L-14', coca_pretrained='mscoco_finetuned_laion2B-s13B-b90k',**kwargs):
        clip, transform = open_clip.create_model_from_pretrained(
            coca_model_name, pretrained=coca_pretrained
        )
        tokenizer = open_clip.get_tokenizer(coca_model_name.replace('/','-')) 

        return clip.text, tokenizer

    @property
    def layers_to_record(self):
        return self.model.text.transformer.resblocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp[0].in_features
    

class ClipText(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, clip_model_name='ViT-B/32-quickgelu', clip_pretrained='openai',**kwargs):
        if "hf-hub" in clip_model_name: clip_pretrained=None
        clip, _image_transform = open_clip.create_model_from_pretrained(
            clip_model_name, pretrained=clip_pretrained
        )
        try:
            tokenizer = open_clip.get_tokenizer(clip_model_name.replace('/','-'))            
        except: tokenizer = open_clip.get_tokenizer(clip_model_name)
        return clip.transformer, lambda txt: clip.token_embedding(tokenizer(txt)) + clip.positional_embedding
    
    @property
    def layers_to_record(self):
        return self.model.resblocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp[0].in_features


class DinoV2(WrappedModel):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def setup_model(self, model_size='B', dino_backbone='dinov2_vitl14', **kwargs):
        if model_size == 'B': dino_backbone='dinov2_vitb14'
        elif model_size == 'L': dino_backbone='dinov2_vitl14'
        model = torch.hub.load("facebookresearch/dinov2", dino_backbone)
        transform = v2.Compose([
            # TODO: I bet this should be 256, 256, which is causing localization issues in non-square images.
            v2.Resize(size=256),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        return model, transform
    
    @property
    def layers_to_record(self):
        return self.model.blocks
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].norm1.normalized_shape[0]


class MambaVision(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers,  *args, **kwargs)

    def setup_model(self, mambavision_model="nvidia/MambaVision-S-1K", **kwargs):

        mv = AutoModel.from_pretrained(mambavision_model, trust_remote_code=True)

        input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions, using same as DinoV2
        transform = create_transform(
                            input_size=input_resolution,
                            is_training=False,
                            mean=mv.config.mean,
                            std=mv.config.std,
                            crop_mode=mv.config.crop_mode,
                            crop_pct=mv.config.crop_pct
                        )
        return mv.model, transform
    
    @property
    def layers_to_record(self):
        #return self.model.levels
        layers = torch.nn.ModuleList()
        for level in self.model.levels:
            for block in level.blocks:
                layers.append(block)
        return layers
    
    def d_vit(self, layer_id):
        layer = self.layers_to_record[layer_id]
        d = None
        # ConvBlock
        if hasattr(layer, 'conv1'):
            D0 = self.layers_to_record[0].norm2.num_features
            n = layer.norm2.num_features
            if layer_id < 3: # First MambaVisionBlock
                #print("=n", end="")
                d = 56*56 # H/4 * H/4
            else: d = 28 * 28 # H/16 * H/16

        # Block
        else:
            d = layer.mlp.fc2.out_features
        print("DIM", layer_id, d)
        return d
        

class ViT(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, vit_model_name="google/vit-base-patch16-224", **kwargs):
        processor = AutoImageProcessor.from_pretrained(vit_model_name, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(vit_model_name)
        return model.vit, lambda img: processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
    
    @property
    def layers_to_record(self):
        return self.model.encoder.layer

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features


class SigLIP2Vision(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, siglip2_ckpt="google/siglip2-base-patch16-224", **kwargs):
        model = SiglipModel.from_pretrained(siglip2_ckpt, local_files_only=True).vision_model
        processor = SiglipProcessor.from_pretrained(siglip2_ckpt, local_files_only=True)
        return model, lambda img: processor(images=img)['pixel_values'].squeeze(0)
    
    @property
    def layers_to_record(self):
        return self.model.encoder.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].layer_norm2.normalized_shape[0]
    


class SigLIP2Text(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, siglip2_ckpt="google/siglip2-base-patch16-224", **kwargs):
        model = SiglipModel.from_pretrained(siglip2_ckpt).text_model
        processor = SiglipProcessor.from_pretrained(siglip2_ckpt)
        #tokenizer = SiglipTokenizer.from_pretrained(siglip2_ckpt)
        return model, processor# lambda txt: processor(text=txt, padding=True)['input_ids'].squeeze(0)
    
    @property
    def layers_to_record(self):
        return self.model.encoder.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].layer_norm2.normalized_shape[0]
    

class Pythia(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, pythia_pretrained="EleutherAI/pythia-160m",**kwargs):
        pythia_model = GPTNeoXForCausalLM.from_pretrained(pythia_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(pythia_pretrained)
        tokenizer.pad_token = tokenizer.eos_token
        return pythia_model.base_model, lambda txt: tokenizer(text=txt, padding=True, return_tensors='pt')['input_ids'].squeeze(0)
    
    @property
    def layers_to_record(self):
        return self.model.layers
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp.dense_4h_to_h.out_features


class MambaText(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, mamba_pretrained="state-spaces/mamba-130m-hf", **kwargs):
        model = MambaForCausalLM.from_pretrained(mamba_pretrained)
        tokenizer = AutoTokenizer.from_pretrained(mamba_pretrained)
        tokenizer.pad_token = tokenizer.eos_token
        return model.base_model, lambda txt: tokenizer(txt, return_tensors="pt", padding=True)["input_ids"].squeeze(0)
    
    @property
    def layers_to_record(self):
        return self.model.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mixer.out_proj.out_features


class Llama3(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, llama_pretrained="openlm-research/open_llama_3b_v2",**kwargs):
        tokenizer = LlamaTokenizer.from_pretrained(llama_pretrained)
        model = LlamaForCausalLM.from_pretrained(
            llama_pretrained, torch_dtype=torch.float16, device_map='auto',
        )
        tokenizer.pad_token = tokenizer.eos_token
        return model.base_model, lambda txt: tokenizer(txt, return_tensors="pt", padding=True).input_ids
    
    @property
    def layers_to_record(self):
        return self.model.layers
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp.down_proj.out_features
    

class GPT2(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, gpt2_pretrained="gpt2",**kwargs):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        return model.base_model, lambda txt: tokenizer(txt, return_tensors='pt', padding=True)
    
    @property
    def layers_to_record(self):
        return self.base_model.h
    
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp.c_proj.nf

class GroundingDinoText(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)
    
    def setup_model(self, gdino_pretrained="IDEA-Research/grounding-dino-base", **kwargs):
        processor = AutoProcessor.from_pretrained(gdino_pretrained)
        grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_pretrained)

        return grounding_dino.model.text_backbone, processor.tokenizer # lambda txt: processor(text=txt, return_tensors='pt', padding=True)
    
    @property
    def layers_to_record(self):
        return self.model.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features
    
    def forward(self, input_tensor):

        return self.model(**input_tensor)

class GroundingDinoVision(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)
    
    def setup_model(self, gdino_pretrained="IDEA-Research/grounding-dino-tiny", **kwargs):
        #processor = AutoImageProcessor.from_pretrained(gdino_pretrained, padding=True)
        grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_pretrained)

        preprocess = transforms.Compose([
            transforms.Resize((800, 800)),  # Resize to match model input size
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as expected by the model
        ])

        # grounding_dino.model.backbone.conv_encoder.model.encoder
        return grounding_dino.model.backbone, preprocess
    
    @property
    def layers_to_record(self):
        layers = torch.nn.ModuleList()
        for layer in self.model.conv_encoder.model.encoder.layers:
            for block in layer.blocks:
                layers.append(block)
        
        return layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features
    
    def forward(self, input_tensor):
        model_device = next(self.model.parameters()).device

        pixel_values = input_tensor.to(torch.float32).to(model_device)
        pixel_mask = torch.ones((1, *pixel_values.shape[-2:]), dtype=torch.float32).to(model_device)
        inputs = {"pixel_values":pixel_values, "pixel_mask":pixel_mask}
        return self.model(**inputs)
    
class Bert(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)
    def setup_model(self, **kwargs):
        model = AutoModel.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        #tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    @property
    def layers_to_record(self):
        return self.model.encoder.layer
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features
    def forward(self, inputs):
        return self.model(**inputs)
    

class Deberta(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)
    def setup_model(self, model_size='B', deberta_model_name="microsoft/deberta-base", **kwargs):
        if model_size == 'B': deberta_model_name="microsoft/deberta-base"
        elif model_size == 'L': deberta_model_name="microsoft/deberta-large"
        model = AutoModel.from_pretrained(deberta_model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(deberta_model_name, local_files_only=True)
        tokenizer_fn = lambda x: tokenizer(x, return_tensors='pt', padding=True)
        return model, tokenizer_fn

    @property
    def layers_to_record(self):
        return self.model.encoder.layer
    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features
    def forward(self, inputs):
        return self.model(**inputs)


class Blip(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, **kwargs):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return model, lambda raw_image: processor(raw_image, return_tensors="pt")
    
    def forward(self, inputs):
        #print(inputs.keys())
        inputs['pixel_values'] = inputs['pixel_values'].squeeze()
        return self.model.vision_model(**inputs)

    @property
    def layers_to_record(self):
        return self.model.vision_model.encoder.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].mlp.fc2.out_features

class AST(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)
    
    def setup_model(self, **kwargs):
        extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        return model, extractor
    
    def forward(self, inputs):
        return self.model(input_values=inputs)

    @property
    def layers_to_record(self):
        return self.model.audio_spectrogram_transformer.encoder.layer

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].output.dense.out_features

class Donut(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, **kwargs):
        processor = AutoProcessor.from_pretrained("naver-clova-ix/donut-base")
        model = AutoModelForVision2Seq.from_pretrained("naver-clova-ix/donut-base")
        #print("HIHAN")
        #print(model.encoder.encoder.layers)
        return model, lambda raw_image: processor(raw_image, return_tensors="pt")
    
    def forward(self, inputs):
        #print(inputs)
        #print(inputs.keys())
        #inputs['pixel_values'] = inputs['pixel_values'].squeeze()
        return self.model.encoder(**inputs)

    @property
    def layers_to_record(self):
        return self.model.encoder.encoder.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].blocks[-1].output.dense.out_features


class BartEncoder(WrappedModel):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(layers, *args, **kwargs)

    def setup_model(self, **kwargs):
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", local_files_only=True)
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", local_files_only=True)
        #print("HIHAN")
        #print(model.encoder.encoder.layers)
        return model, lambda text: tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    def forward(self, inputs):
        return self.model.generate(inputs["input_ids"])

    @property
    def layers_to_record(self):
        return self.model.model.encoder.layers

    def d_vit(self, layer_id):
        return self.layers_to_record[layer_id].fc2.out_features

if __name__ == '__main__':
    wm = Deberta([-1], model_size='B')
    wm = Deberta([-1], model_size='L')
    wm = Blip([-1])
    print(wm)
    wm = AST([-1])

    #wm = Donut([-1])
    wm = BartEncoder([-1])
