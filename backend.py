"""
Stable-Hair 后端服务
接收发型照片和参数，运行推理并返回生成的图片
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import io
import uuid
import torch
import numpy as np
from PIL import Image
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from diffusers import UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel

from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import adapter_injection, set_scale
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline

# 创建 FastAPI 应用
app = FastAPI(title="Stable-Hair API", description="发型迁移后端服务")

# 添加 CORS 中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
model = None


class StableHairModel:
    """Stable-Hair 模型封装类"""
    
    def __init__(self, config="./configs/hair_transfer.yaml", weight_dtype=torch.float16):
        print("正在初始化 Stable-Hair 模型...")
        self.config = OmegaConf.load(config)
        self.weight_dtype = weight_dtype
        
        # 多 GPU 分配（与 infer_full.py 保持一致）
        self.device_main = "cuda:0"
        self.device_ref = "cuda:1"
        self.device_bald = "cuda:2"
        self.device_unet = "cuda:3"
        
        # 加载 UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_path, subfolder="unet"
        ).to(self.device_unet)
        
        # 加载 ControlNet
        controlnet = ControlNetModel.from_unet(unet).to(self.device_main)
        _state_dict = torch.load(
            os.path.join(self.config.pretrained_folder, self.config.controlnet_path)
        )
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)
        
        # 创建主 Pipeline
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(self.device_main)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # 加载 Hair Encoder/Adapter
        self.hair_encoder = ref_unet.from_pretrained(
            self.config.pretrained_model_path, subfolder="unet"
        ).to(self.device_ref)
        _state_dict = torch.load(
            os.path.join(self.config.pretrained_folder, self.config.encoder_path)
        )
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_encoder.to(weight_dtype)
        
        self.hair_adapter = adapter_injection(
            self.pipeline.unet, 
            device=self.device_ref, 
            dtype=weight_dtype, 
            use_resampler=False
        )
        _state_dict = torch.load(
            os.path.join(self.config.pretrained_folder, self.config.adapter_path)
        )
        self.hair_adapter.load_state_dict(_state_dict, strict=False)
        self.hair_adapter.to(weight_dtype)
        
        # 加载 Bald Converter
        bald_converter = ControlNetModel.from_unet(unet).to(self.device_bald)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet
        
        # 创建去除头发的 Pipeline
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.remove_hair_pipeline.scheduler.config
        )
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(self.device_bald)
        
        print("模型初始化完成！")
    
    def get_bald(self, id_image, scale=0.9):
        """生成光头图像"""
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]
        return image
    
    def crop_to_square(self, image: Image.Image) -> Image.Image:
        """将图片裁剪成正方形（居中裁剪）"""
        width, height = image.size
        if width == height:
            return image
        
        # 取短边作为正方形边长
        size = min(width, height)
        
        # 计算裁剪区域（居中）
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        return image.crop((left, top, right, bottom))
    
    def hair_transfer(
        self,
        source_image: Image.Image,
        reference_image: Image.Image,
        random_seed: int = -1,
        step: int = 50,
        guidance_scale: float = 1.3,
        scale: float = 0.65,
        controlnet_conditioning_scale: float = 0.85,
        size: int = 512
    ) -> Image.Image:
        """
        发型迁移
        
        Args:
            source_image: 源图像（人脸）
            reference_image: 参考图像（发型）
            random_seed: 随机种子，-1 表示随机
            step: 推理步数
            guidance_scale: 引导强度
            scale: adapter 缩放系数
            controlnet_conditioning_scale: ControlNet 条件缩放
            size: 图像尺寸
            
        Returns:
            生成的图像
        """
        # 预处理图像：先裁剪成正方形，再 resize
        source_image = self.crop_to_square(source_image.convert("RGB")).resize((size, size))
        reference_image_np = np.array(
            self.crop_to_square(reference_image.convert("RGB")).resize((size, size))
        )
        
        # 生成光头图像
        source_image_bald = np.array(self.get_bald(source_image, scale=0.9))
        H, W, C = source_image_bald.shape
        
        # 设置随机种子
        if random_seed == -1:
            random_seed = np.random.randint(0, 2**32 - 1)
        
        # 将 hair_encoder 和 hair_adapter 移到 device_main（与 infer_full.py 一致）
        self.hair_encoder.to(self.device_main)
        self.hair_adapter.to(self.device_main)
        
        # 设置 adapter scale
        set_scale(self.pipeline.unet, scale)
        
        # 创建生成器
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        
        # 生成图像
        sample = self.pipeline(
            "",
            negative_prompt="",
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image_bald,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image_np,
        ).samples
        
        # 转换为 PIL Image
        result_image = Image.fromarray((sample * 255.).astype(np.uint8))
        return result_image


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    global model
    print("正在加载模型...")
    try:
        model = StableHairModel(
            config="./configs/hair_transfer.yaml",
            weight_dtype=torch.float16
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise e


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "Stable-Hair API 服务正在运行"}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/transfer")
async def transfer_hair(
    source_image: UploadFile = File(..., description="源图像（人脸照片）"),
    reference_image: UploadFile = File(..., description="参考图像（发型照片）"),
    random_seed: int = Form(default=-1, description="随机种子，-1表示随机"),
    step: int = Form(default=50, description="推理步数"),
    guidance_scale: float = Form(default=1.3, description="引导强度"),
    scale: float = Form(default=0.65, description="Adapter缩放系数"),
    controlnet_conditioning_scale: float = Form(default=0.85, description="ControlNet条件缩放"),
    size: int = Form(default=512, description="输出图像尺寸")
):
    """
    发型迁移接口
    
    接收源图像（人脸）和参考图像（发型），返回迁移后的图像
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    
    try:
        # 读取上传的图片
        source_bytes = await source_image.read()
        reference_bytes = await reference_image.read()
        
        source_pil = Image.open(io.BytesIO(source_bytes))
        reference_pil = Image.open(io.BytesIO(reference_bytes))
        
        # 执行发型迁移
        result_image = model.hair_transfer(
            source_image=source_pil,
            reference_image=reference_pil,
            random_seed=random_seed,
            step=step,
            guidance_scale=guidance_scale,
            scale=scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            size=size
        )
        
        # 将结果转换为字节流返回
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=result.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/transfer_base64")
async def transfer_hair_base64(
    source_image: UploadFile = File(..., description="源图像（人脸照片）"),
    reference_image: UploadFile = File(..., description="参考图像（发型照片）"),
    random_seed: int = Form(default=-1, description="随机种子，-1表示随机"),
    step: int = Form(default=50, description="推理步数"),
    guidance_scale: float = Form(default=1.3, description="引导强度"),
    scale: float = Form(default=0.65, description="Adapter缩放系数"),
    controlnet_conditioning_scale: float = Form(default=0.85, description="ControlNet条件缩放"),
    size: int = Form(default=512, description="输出图像尺寸")
):
    """
    发型迁移接口（返回Base64编码的图片）
    
    接收源图像（人脸）和参考图像（发型），返回Base64编码的结果图像
    """
    import base64
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    
    try:
        # 读取上传的图片
        source_bytes = await source_image.read()
        reference_bytes = await reference_image.read()
        
        source_pil = Image.open(io.BytesIO(source_bytes))
        reference_pil = Image.open(io.BytesIO(reference_bytes))
        
        # 执行发型迁移
        result_image = model.hair_transfer(
            source_image=source_pil,
            reference_image=reference_pil,
            random_seed=random_seed,
            step=step,
            guidance_scale=guidance_scale,
            scale=scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            size=size
        )
        
        # 将结果转换为Base64
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return JSONResponse(content={
            "success": True,
            "image": img_base64,
            "format": "png"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("Stable-Hair 后端服务")
    print("=" * 50)
    print("API 文档地址: http://0.0.0.0:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
