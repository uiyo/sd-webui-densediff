import modules.scripts as scripts
import gradio as gr
import os
import torch

from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from PIL import Image
from scripts import pre
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction
from contextlib import nullcontext
from packaging import version
from modules import images, scripts, script_callbacks, shared
from scripts.logging import logger

import modules.scripts as scripts
import gradio as gr
import os
import numpy as np
import pdb
import torch.nn.functional as F

MAX_COLORS = 12
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
COUNT = 0

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

class DenseDiff(scripts.Script):
    def __init__(self):
        super().__init__()
        # self.MAX_COLORS = 12
        self.colors = []
        # self.color_row = [None] * self.MAX_COLORS
        self.prompts = []
        self.post_sketch = None
        self.sketch_button = None
    
    # Extension title in menu UI
    def title(self):
        return "DenseDiffusion"

    # Decide to show menu in txt2img or img2img
    # - in "txt2img" -> is_img2img is `False`
    # - in "img2img" -> is_img2img is `True`
    #
    # below code always show extension menu
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # Setup menu ui detail
    def ui(self, is_img2img):
        color_row = [None] * MAX_COLORS
        prompts = []
        colors = []
        binary_matrixes = gr.State([])
        # self.steps = 50
        # self.stepsIMG = 50
        masks = None
        with gr.Accordion('DenseDiffusion', open=False):
            enabled = gr.Checkbox(label="Disabled ❌", value=False, elem_id='densenet_enable', interactive=True)
    
            gr.HTML('<p style="margin-bottom:0.8em"> **If DenseDiffusion is enabled, the default sampling steps can only be larger than 30, please use main prompt box only for loading negative prompts and lora, max length is 75. </p>')
                

            with gr.Row(visible=True): 
                with gr.Box():
                    if not is_img2img:
                        with gr.Tabs():
                            with gr.Tab(label="Sketch"):
                                masks_sk = gr.Image(source="canvas", tool="color-sketch", type="numpy", shape=(512, 512)).style(width=400, height=400)
                                sketch_button_sk = gr.Button("Start Processing Sketch", interactive=True)
                            with gr.Tab(label="Canvas"):
                                masks_cv = gr.Image(source="upload", tool="color-sketch", type="numpy", shape=(512, 512)).style(width=400, height=400)
                                sketch_button_cv = gr.Button("Start Processing Sketch", interactive=True)
                    else:
                        image = self.original_image
                        masks = self.skecth
                        sketch_button = gr.Button("Start Processing Sketch", interactive=True)

                    with gr.Column(visible=False) as post_sketch:
                        for n in range(MAX_COLORS):
                            if n == 0:
                                with gr.Row(visible=False) as color_row[n]:
                                    colors.append(gr.Image(shape=(100, 100), label="background", type="pil", image_mode="RGB").style(width=100, height=100))
                                    prompts.append(gr.Textbox(label="Prompt for the background (white region)", value=""))
                            else:
                                with gr.Row(visible=False) as color_row[n]:
                                    colors.append(gr.Image(shape=(100, 100), label="segment "+str(n), type="pil", image_mode="RGB").style(width=100, height=100))
                                    prompts.append(gr.Textbox(label="Prompt for the segment "+str(n)))
                    with gr.Column():
                        get_genprompt_run = gr.Button("Formulate Prompts", interactive=True)
                    with gr.Column(visible=False) as gen_prompt_vis:
                        general_prompt = gr.Textbox(value='', label="Textual Description for the entire image", interactive=True)
                        txt2img_tab_name = ["txt2img_densediff_creg", "txt2img_densediff_sreg", "txt2img_densediff_sizereg"]
                        img2img_tab_name = ["img2img_densediff_creg", "img2img_densediff_sreg", "img2img_densediff_sizereg"]
                        if not is_img2img:
                            creg_id, sreg_id, sizereg_id = txt2img_tab_name
                        else:
                            creg_id, sreg_id, sizereg_id = img2img_tab_name
                        
                        with gr.Accordion("Tune the hyperparameters", open=False):
                            creg_ = gr.Slider(label=" w\u1D9C (The degree of attention modulation at cross-attention layers) ", minimum=0, maximum=2., value=1.0, step=0.1, elem_id=creg_id)
                            sreg_ = gr.Slider(label=" w \u02E2 (The degree of attention modulation at self-attention layers) ", minimum=0, maximum=2., value=0.6, step=0.1, elem_id=sreg_id)
                            sizereg_ = gr.Slider(label="The degree of mask-area adaptive adjustment", minimum=0, maximum=1., value=1., step=0.1, elem_id=sizereg_id)
                    with gr.Row():
                        set_default_btn = gr.Button("Default Setting Reset", interactive=True) 
                           
                
            enabled.change(pre.switchEnableLabel, [enabled], [enabled, post_sketch, gen_prompt_vis, general_prompt, binary_matrixes, *prompts])
            
            if not is_img2img:
                sketch_button_sk.click(pre.process_sketch, inputs=[enabled, masks_sk], outputs=[post_sketch, binary_matrixes, *color_row, *colors], queue=False)
                sketch_button_cv.click(pre.process_sketch, inputs=[enabled, masks_cv], outputs=[post_sketch, binary_matrixes, *color_row, *colors], queue=False)
            else:
                
                sketch_button.click(pre.process_sketch, inputs=[enabled, masks, image], outputs=[post_sketch, binary_matrixes, *color_row, *colors], queue=False)
            
            get_genprompt_run.click(pre.process_prompts, inputs=[enabled, binary_matrixes, *prompts], outputs=[gen_prompt_vis, general_prompt], queue=False)
            
            if not is_img2img:
                if enabled:
                    set_default_btn.click(pre.default_setting, outputs=[self.steps, self.creg, self.sreg, self.sizereg])
            else:
                if enabled:    
                    set_default_btn.click(pre.default_setting, outputs=[self.stepsIMG, self.cregIMG, self.sregIMG, self.sizeregIMG])
 
            # TODO: add more UI components (cf. https://gradio.app/docs/#components)
                        

        return [enabled, general_prompt, binary_matrixes, creg_, sreg_, sizereg_, *prompts]


    def before_component(self, general_output, **kwargs): 
        if kwargs.get("elem_id") == "txt2img_prompt":
            self.boxx = general_output
        
        if kwargs.get("elem_id") == "img2img_prompt":
            self.boxxIMG = general_output 
        
        if kwargs.get("elem_id") == "img2img_image":
            self.original_image = general_output
        
        if kwargs.get("elem_id") == "img2img_sketch":
            self.skecth = general_output
        if kwargs.get("elem_id") == "txt2img_neg_prompt":
            self.negative_prompt = general_output
        if kwargs.get("elem_id") == "img2img_neg_prompt":
            self.negative_promptIMG = general_output
        if kwargs.get("elem_id") == "txt2img_steps":
            self.steps = general_output
        if kwargs.get("elem_id") == "img2img_steps":
            self.stepsIMG = general_output
        if kwargs.get("elem_id") == "txt2img_densediff_creg":
            self.creg = general_output
        if kwargs.get("elem_id") == "txt2img_densediff_sreg":
            self.sreg = general_output
        if kwargs.get("elem_id") == "txt2img_densediff_sizereg":
            self.sizereg = general_output
        if kwargs.get("elem_id") == "img2img_densediff_creg":
            self.cregIMG = general_output
        if kwargs.get("elem_id") == "img2img_densediff_sreg":
            self.sregIMG = general_output
        if kwargs.get("elem_id") == "img2img_densediff_sizereg":
            self.sizeregIMG = general_output     
    
    def process(self, p, enabled, general_prompt, binary_matrixes, creg_, sreg_, sizereg_, *prompts):
        if enabled:
            p.steps = p.steps if p.steps >= 30 else 30      
            global creg, sreg, sizereg #? any choice better than global 
            creg, sreg, sizereg = creg_, sreg_, sizereg_
            master_prompt = general_prompt
            
            clipped_prompts = prompts[:len(binary_matrixes)]
            prompts = [master_prompt] + list(clipped_prompts)
            device = p.sd_model.device
            
            if p.sd_model.is_sdxl:
                sp_sz = 128
            else:
                sp_sz = p.sd_model.image_size
            bsz = p.batch_size

            layouts = torch.cat([pre.preprocess_mask(mask_, sp_sz, sp_sz, device) for mask_ in binary_matrixes])

            p.prompts = prompts
            p.negative_prompts = [p.negative_prompt]
            p.setup_prompts()
            p.setup_conds()
            all_cond_embeddings = []
            for i in range(len(prompts)):
                if p.sd_model.is_sdxl:
                    embeddings = p.c.batch[i][0].schedules[0].cond['crossattn']
                else:    
                    embeddings = p.c.batch[i][0].schedules[0].cond
                all_cond_embeddings.append(embeddings.unsqueeze(0))
            cond_embeddings = torch.cat(all_cond_embeddings)
            if p.sd_model.is_sdxl:
                uncond_embeddings = p.uc[0][0].cond['crossattn'].unsqueeze(0)
            else:
                uncond_embeddings = p.uc[0][0].cond.unsqueeze(0)
            
            if not p.sd_model.is_sdxl:
                text_input = p.sd_model.cond_stage_model.tokenizer(
                                        prompts, padding="max_length", return_length=True, 
                                        return_overflowing_tokens=False, 
                                        max_length=p.sd_model.cond_stage_model.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                                        )
                # cond_embeddings = p.sd_model.cond_stage_model(prompts)
                # 0: general prompt embedding
                # 1: bg embedding
                # i: object embedding
                
            else:
                
                text_input = p.sd_model.conditioner.embedders[0].tokenizer(
                                        prompts, padding="max_length", return_length=True, 
                                        return_overflowing_tokens=False, 
                                        max_length=p.sd_model.conditioner.embedders[0].tokenizer.model_max_length, truncation=True, return_tensors="pt"
                )

            
            ###########################
            ###### prep for sreg ###### 
            ###########################
            global sreg_maps, reg_sizes #? any choice better than global 

            sreg_maps = {}
            reg_sizes = {}
            for r in range(4):
                layouts_s = F.interpolate(layouts,(np.power(2,r+3),np.power(2,r+3)),mode='nearest')
                layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1)

                reg_sizes[np.power(2,(r+3)*2)] = 1-sizereg*layouts_s.sum(-1, keepdim=True)/(np.power(2,(r+3)*2))
                sreg_maps[np.power(2,(r+3)*2)] = layouts_s
            
            
            pww_maps = torch.zeros(1,77,sp_sz, sp_sz).to(device)
            
            for i in range(1,len(prompts)):
                wlen = text_input['length'][i] - 2
                widx = text_input['input_ids'][i][1:1+wlen]
                for j in range(77):
                    try:
                        text_input['input_ids'][0][j:j+wlen] == widx
                    except Exception as e:
                        logger.error(e)
                        raise ValueError('Segment lables are unmatched with formulated prompts!')
                        
                    if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
    
                        pww_maps[:,j:j+wlen,:,:] = layouts[i-1:i]
                        cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen]
                        break #

            global creg_maps #
            creg_maps = {}
            for r in range(4):
                layout_c = F.interpolate(pww_maps,(np.power(2,r+3),np.power(2,r+3)),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(bsz,1,1)
                creg_maps[np.power(2,(r+3)*2)] = layout_c
            
            ###########################    
            #### prep for text_emb ####
            ###########################
            
            global text_cond, timesteps
            
            timesteps = torch.arange(1, 1001, 1000//p.steps).flip(0)
            if p.sampler_name in ['DDIM', 'PLMS', 'UniPC']:
                if p.sd_model.is_sdxl:
                    print('SDXL inapplicable!')
                    model = 'sdxl'
                    ddim = False
                    
                
                else:
                    ddim = True
                    model = ''
                text_embed = torch.cat([uncond_embeddings,cond_embeddings[:1].repeat(bsz,1,1)], dim=0)
            else:
                ddim = False

                if p.sd_model.is_sdxl:
                    model = 'sdxl'
                else:
                    model = ''
                text_embed = torch.cat([cond_embeddings[:1].repeat(bsz,1,1),uncond_embeddings], dim=0)
            text_cond = {
                            'model':model,
                            'ddim-series': ddim, 
                            'text_cond': text_embed,
                        }
            global COUNT
            COUNT = 0
            torch.compile(mod_forward) # this is so said acceleration 
            # modify the attention operation
            for _module in p.sd_model.model.named_modules():
                if _module[1].__class__.__name__ == 'CrossAttention':
                    _module[1].forward = mod_forward.__get__(_module[1], _module[1].__class__)
            if p.sd_model.is_sdxl:
                text_cond['model'] = 'sdxl'
                p.prompts = prompts[0]
            return 
        else:
            for _module in p.sd_model.model.named_modules():
                if _module[1].__class__.__name__ == 'CrossAttention':
                    _module[1].forward = original_forward.__get__(_module[1], _module[1].__class__)
        
        return super().process(p)
   
    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def run(self, p, *args):
        # TODO: get UI info through UI object angle, checkbox
        proc = process_images(p)
        # TODO: add image edit process via Processed object proc
        return proc

def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def mod_forward(self, x, context=None, mask=None, additional_tokens=None,n_times_crossframe_attn_in_self=0):
    h = self.heads
    
    if additional_tokens is not None:
        # get the number of masked tokens at the beginning of the output sequence
        n_tokens_to_mask = additional_tokens.shape[1]
        # add additional token
        x = torch.cat([additional_tokens, x], dim=1)
    
    q = self.to_q(x)
    global text_cond, timesteps
    
    
    layers = 140 if text_cond['model'] == 'sdxl' else 32 # v2.1 v1.5: 32 xl:140 
    
    global sreg, creg, COUNT, creg_maps, sreg_maps, reg_sizes
    COUNT += 1
    
    rate = 0.3

        
    text_context = text_cond['text_cond'] if context is not None else x

    
    k = self.to_k(text_context)
    v = self.to_v(text_context)

    if n_times_crossframe_attn_in_self:
        # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
        assert x.shape[0] % n_times_crossframe_attn_in_self == 0
        n_cp = x.shape[0] // n_times_crossframe_attn_in_self
        k = repeat(
            k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
        )
        v = repeat(
            v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
        )

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
    
    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION =="fp32":
        with torch.autocast(enabled=False, device_type = 'cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    
    if COUNT/layers < len(timesteps) * rate:
        pass
        if not text_cond['ddim-series']:
            index = list(range(sim.size(0)//2))
        else:
            index = list(range(sim.size(0)//2, sim.size(0)))
        treg = torch.pow(timesteps[COUNT//layers]/1000, 5)
        
        if sim.size(1) in sreg_maps:
            # p.height != p.width is not recommand, may lose controllability
            ## reg at self-attn
            if context is None:

                min_value = sim[index].min(-1)[0].unsqueeze(-1)
                max_value = sim[index].max(-1)[0].unsqueeze(-1)
                
                segmask = sreg_maps[sim.size(1)].repeat(h,1,1)
                size_reg = reg_sizes[sim.size(1)].repeat(h,1,1)
                
                sim[index] += (segmask>0)*size_reg*sreg*treg*(max_value-sim[index])
                sim[index] -= ~(segmask>0)*size_reg*sreg*treg*(sim[index]-min_value)
                if text_cond['model'] != 'sdxl':    
                    mask = sreg_maps[sim.size(1)]
    
            ## reg at cross-attn
            else:
                min_value = sim[index].min(-1)[0].unsqueeze(-1)
                max_value = sim[index].max(-1)[0].unsqueeze(-1)
    
                segmask = creg_maps[sim.size(1)].repeat(h,1,1)
            
                size_reg = reg_sizes[sim.size(1)].repeat(h,1,1)
    
                sim[index] += (segmask>0)*size_reg*creg*treg*(max_value-sim[index])
                sim[index] -= ~(segmask>0)*size_reg*creg*treg*(sim[index]-min_value)
                
                # to augmentation the semantics of each mask
                if text_cond['model'] == 'sdxl':
                    mask = creg_maps[sim.size(1)]
    del q, k
    if exists(mask):
        # mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        # min_value = torch.finfo(sim.dtype).min
        # mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~(mask>0), max_neg_value)
    
    sim = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    if additional_tokens is not None:
        # remove additional token
        out = out[:, n_tokens_to_mask:]

    return self.to_out(out)


def original_forward(self, x, context=None, mask =None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
    # sgm.modules.attention CrossAttention.forward() 
    # ./repositories/generative-models/sgm/modules/attention.py
    h = self.heads

    if additional_tokens is not None:
        # get the number of masked tokens at the beginning of the output sequence
        n_tokens_to_mask = additional_tokens.shape[1]
        # add additional token
        x = torch.cat([additional_tokens, x], dim=1)

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    if n_times_crossframe_attn_in_self:
        # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
        assert x.shape[0] % n_times_crossframe_attn_in_self == 0
        n_cp = x.shape[0] // n_times_crossframe_attn_in_self
        k = repeat(
            k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
        )
        v = repeat(
            v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
        )

    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
    
    with sdp_kernel(**BACKEND_MAP[self.backend]):
        # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )  # scale is dim_head ** -0.5 per default

    del q, k, v
    out = rearrange(out, "b h n d -> b n (h d)", h=h)

    if additional_tokens is not None:
        # remove additional token
        out = out[:, n_tokens_to_mask:]
    return self.to_out(out)
