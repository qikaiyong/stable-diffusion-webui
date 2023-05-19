import copy
import math
import os
import random
import sys
import traceback
import shlex

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers
from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None, #翻译过来是 生成对抗模型
    "outpath_samples": process_string_tag, #翻译过来是 输出样本路径
    "outpath_grids": process_string_tag, #翻译过来是 输出网格路径
    "prompt_for_display": process_string_tag, #翻译过来是 显示提示
    "prompt": process_string_tag, #翻译过来是 提示
    "negative_prompt": process_string_tag, #翻译过来是 负提示
    "styles": process_string_tag, #翻译过来是 样式
    "seed": process_int_tag, #翻译过来是 种子
    "subseed_strength": process_float_tag, #翻译过来是 子种子强度
    "subseed": process_int_tag, #翻译过来是 子种子
    "seed_resize_from_h": process_int_tag, #翻译过来是 从高度调整种子大小
    "seed_resize_from_w": process_int_tag, #翻译过来是 从宽度调整种子大小
    "sampler_index": process_int_tag, #翻译过来是 采样器索引
    "sampler_name": process_string_tag, #翻译过来是 采样器名称
    "batch_size": process_int_tag, #翻译过来是 批量大小
    "n_iter": process_int_tag, #翻译过来是 迭代次数
    "steps": process_int_tag, #翻译过来是 步数
    "cfg_scale": process_float_tag, #翻译过来是 配置比例
    "width": process_int_tag, #翻译过来是 宽度
    "height": process_int_tag, #翻译过来是 高度
    "restore_faces": process_boolean_tag, #翻译过来是 恢复面部
    "tiling": process_boolean_tag, #翻译过来是 平铺
    "do_not_save_samples": process_boolean_tag, #翻译过来是 不保存样本
    "do_not_save_grid": process_boolean_tag #翻译过来是 不保存网格
}

ooo = "Black hair," 

def cmdargs(line): #函数功能是  命令行参数
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}' #翻译过来是 必须以“--”开头
        assert pos+1 < len(args), f'missing argument for command line option {arg}' #翻译过来是 命令行选项缺少参数

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt": #翻译过来是 提示或负提示
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"): #翻译过来是 以“--”开头
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue


        func = prompt_tags.get(tag, None) #翻译过来是 提示标签
        assert func, f'unknown commandline option: {arg}' #翻译过来是 未知的命令行选项
        
        val = args[pos+1] #翻译过来是 值
        if tag == "sampler_name": #翻译过来是 采样器名称
            val = sd_samplers.samplers_map.get(val.lower(), None) #翻译过来是 采样器映射

        res[tag] = func(val) #翻译过来是 函数

        pos += 2 #翻译过来是 位置

    return res  #上面这个函数的功能是  通过命令行参数来设置参数


def load_prompt_file(file):
    if file is None:
        lines = []
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]

    return None, "\n".join(lines), gr.update(lines=7)

def ddd(num):
    if num == 0:
        return ""
    else:
        return '(' * num + 'Sense of speed' + ')' * num
    
def qqq(num):
    if num == 0:
        return ""
    elif num < 0:
        return '(' * abs(num) + '(Cute art style),' + ')' * abs(num)
    else:
        return '(' * num + 'Realistic style,' + ')' * num


ppt = ",Movie s hots, "   
dddf = ",stone, " 
mtt = ",kkk, " 
cccctt = "Left, right, up, down, forward, backward, north, south, east, west, northeast, northwest, southeast, southwest, horizontal, vertical, diagonal, ascending, descending, clockwise" 
mtt1 = ",Split black and white manuscript，Black and white comics, Black and white stories," 
mtt2 = ",Craig Mullins, Dynamic segmentation,Best composition,Best story expression, Best visual expression,Best composition, " 
ggg = "" 
dddtt = "Perspective, Horizon, Vanishing point, Foreshortening, Depth, Distance, Angle, Scale, Proportion, Overlapping, Shadow, Light source, Reflection, Refraction, Transparency, Opacity, Gradient, Texture, Contrast, Saturation, Hue, Tint, Shade, Tone, Highlight, Midtone, fast shadow, Ambient light, Specular highlight, Rim light" 





class Script(scripts.Script):
    def title(self):    
        return "AI黑白漫画助手"
    
    def ui(self, is_img2img):       
        CX1 = gr.Checkbox(label="随机镜头【开启后，每个图的输出，镜头角度都会随机变化，微调】", value=False,display="inline", elem_id=self.elem_id("CX1"))
        CXt = gr.Checkbox(label="随机表达【这个会画面变化较大，有一点影响角色稳定性，但是会增强视觉效果】", value=False,display="inline", elem_id=self.elem_id("CXt"))
        ttm = gr.Slider(minimum=-6, maximum = 6, step=1, label='写实程度【往左边是卡通，往右边是写实】', value=0, elem_id=self.elem_id("ttm"))
        seedX = gr.Number(label="固定seed值,-1是每张图随机，主动输入seed值会固定输出一致性，如果要输出连贯统一的画风，必须手动输入seed值，随便什么数字都行。", value=222, precision=2, elem_id="seedX") #如何缩短框框 答案是 
        CX4 = gr.Checkbox(label="十连抽【开启这个，会连续跑10遍，适合单句剧本抽卡】（单句词，固定种子，随机镜头，可以稳定抽不同角度的卡）", value=False,display="inline-block",elem_id=self.elem_id("CX3"))
        PTX = gr.Textbox(label="列表输入，这里输入批处理文本或者剧本，每行会输出一张图。【推荐使用GPT来写分镜，一行一个分镜，无限输出】", lines=1, elem_id=self.elem_id("PTX"))
        CX2 = gr.Checkbox(label="漫画模式【开启后会输出为漫画的图】（建议用二次元模型）", value=False,display="inline-block",elem_id=self.elem_id("CX2"))  
        CX3 = gr.Checkbox(label="电影分镜【开启后，会输出电影分镜效果，不要和漫画模式同时使用】（建议用偏2.5d的模型）", value=False,display="inline-block",elem_id=self.elem_id("CX3"))
        fast = gr.Slider(minimum=0, maximum = 6, step=1, label='动态强度【值越大，画面越动感，太大角色会崩】', value=0, elem_id=self.elem_id("fast"))
        ow_text = gr.Textbox(label="画风（输入画风，构图等等控制）", lines=1, elem_id=self.elem_id("CX4"))
        style_txt = gr.Textbox(label="时代背景（时间，时代背景等.）", lines=1, elem_id=self.elem_id("style")) 
        flow_text = gr.Textbox(label="其他（可以补充任意词，会对每张图产生作用）", lines=1, elem_id=self.elem_id("flow_text"))
        file = gr.File(label="上传文件来载入任务列表，注意每行会产生一张图的任务。", type='binary', elem_id=self.elem_id("file"))
        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, PTX, PTX])
        PTX.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[PTX], outputs=[PTX])  
        return [ PTX, style_txt, flow_text,ow_text,seedX,CX1,CX2,CX3,CXt,CX4,fast,ttm]

    def run(self, p, PTX: str, style_txt: str, flow_text: str, ow_text: str,seedX: int,CX1: bool,CX2: bool,CX3: bool,CXt: bool,CX4: bool,fast: int,ttm: int):
        global ppt, dddf
        lines = [x.strip() for x in PTX.splitlines()] 
        lines = [x for x in lines if len(x) > 0] 
        p.do_not_save_grid = True 
        ppt = f",{style_txt},"
        dddf = f",{flow_text}," 
        mtt = f",{ow_text},"

        job_count = 0
        jobs = []
        jobs = []
        job_count = 0
        
        ggg = ppt + dddf
        
        if CX2:
            ggg = ggg + mtt1
        if CX3:
            ggg = ggg + mtt2
        if CX4:    
            lines = lines * 10  
            
        for line in lines:
            args = {"prompt":mtt + ooo + line + ggg + qqq(ttm) + ddd(fast)}  
            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

        print(f"准备 处理 {len(lines)} 行 在 {job_count} 任务列表，整个任务开始.")
        if seedX != -1:
           p.seed = seedX

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"
            if CX1:
                args["prompt"] ="(("+random.choice(cccctt.split(","))+"))"+","+ args["prompt"]
            if CXt:
                args["prompt"] = "(("+random.choice(dddtt.split(","))+"))"+","+ args["prompt"] 
            
            print("-----准备开始处理任务：",n+1)
            copy_p = copy.copy(p)   
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images
            print("处理了一张图. seed值为:",copy_p.seed)

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)   
    
   
