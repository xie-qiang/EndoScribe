# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Literal

import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    InterpolationMode,
)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from translate import Translator


# ==================== Device helpers ====================
def set_visible_devices(
    ascend_rt_visible_devices: Optional[str] = None,
    cuda_visible_devices: Optional[str] = None,
) -> None:
    if ascend_rt_visible_devices is not None:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(ascend_rt_visible_devices)
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)


# 可见设备号
set_visible_devices(
    ascend_rt_visible_devices="1",  # NPU 可见设备号
    cuda_visible_devices=None,      # GPU 可见设备号
)


# ==================== Translator ====================
def translate_text_english(text: str) -> str:
    try:
        translator = Translator(from_lang="zh", to_lang="en")
        return translator.translate(text) if translator else text
    except Exception:
        return text


# ==================== Reproducibility ====================
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==================== Dataset ====================
def _convert_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


class FolderImageDataset(Dataset):
    """
    Returns:
      image_tensor: FloatTensor [3,H,W]
      image_path: str
    """

    def __init__(self, image_paths: Sequence[str], resolution: int = 224):
        super().__init__()
        self.lab2id = {
            "食道": 0,
            "贲门": 1,
            "胃底": 2,
            "胃体": 3,
            "胃角": 4,
            "胃窦和幽门": 5,
            "十二指肠球部": 6,
            "十二指肠降部": 7,
        }
        self.id2lab = {v: k for k, v in self.lab2id.items()}
        self.image_paths = list(image_paths)
        self.transform = self._build_transform(resolution)

    @staticmethod
    def _build_transform(resolution: int) -> Compose:
        return Compose(
            [
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path)
        image = self.transform(image)
        return image, path


# ==================== Layers & Models ====================
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class AnomalyConvNeXt(nn.Module):
    """二分类：0=normal, 1=abnormal"""

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            LayerNorm2d((1024,), eps=1e-6, elementwise_affine=True),
            nn.Flatten(1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)


class PartConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 8, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            LayerNorm2d((1024,), eps=1e-6, elementwise_affine=True),
            nn.Flatten(1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        self.lowlevel_getter = IntermediateLayerGetter(
            self.backbone.features,
            return_layers={"1": "stage0", "3": "stage1", "5": "stage2", "7": "stage3"},
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)

    def extract_lowlevel_features(self, x: Tensor) -> Tensor:
        outputs = self.lowlevel_getter(x)
        feat = outputs["stage1"]  # (B, 256, H, W)
        feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # (B, 256)
        return feat


class PartResNet50(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained_weights_path: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()

        if pretrained_weights_path:
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            _ = self.backbone.load_state_dict(state_dict, strict=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)


class PartSwinB(nn.Module):
    def __init__(self, num_classes: int = 8, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = swin_b(weights=Swin_B_Weights.DEFAULT)
        self.backbone.head = nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        return self.classifier(feat)


# ==================== Checkpoint ====================
def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: str = "cpu") -> None:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model_state", ckpt)  # 兼容 dict / 纯 state_dict
    model.load_state_dict(state, strict=False)


# ==================== Config ====================
@dataclass
class InferenceConfig:
    num_classes: int = 8
    resolution: int = 224
    batch_size: int = 256
    num_workers: int = 8
    confidence_threshold: float = 0.4
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass
class ModelPaths:
    part_res50_ckpt: Optional[str] = None
    part_convnext_ckpt: Optional[str] = None
    part_swin_ckpt: Optional[str] = None
    anomaly_ckpt: Optional[str] = None
    res50_backbone_pretrained: Optional[str] = None  # RN50 backbone 预训练（可选）


DevicePref = Literal["auto", "npu", "cuda", "cpu"]


def get_device(device_pref: DevicePref = "auto") -> torch.device:
    if device_pref == "npu":
        if getattr(torch, "npu", None) and torch.npu.is_available():
            return torch.device("npu")
        raise RuntimeError("device_pref='npu' 但 torch.npu 不可用。")

    if device_pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("device_pref='cuda' 但 CUDA 不可用。")

    if device_pref == "cpu":
        return torch.device("cpu")

    if getattr(torch, "npu", None) and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==================== IO helpers ====================
def collect_image_paths(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    image_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.startswith("."):
                continue
            if not fn.lower().endswith(exts):
                continue
            image_paths.append(os.path.abspath(os.path.join(dirpath, fn)))
    image_paths.sort()
    return image_paths


def list_subfolders(root_dir: str) -> List[str]:
    """返回 root_dir 下的直接子文件夹（按名称排序）"""
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")
    subfolders: List[str] = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            subfolders.append(p)
    subfolders.sort()
    return subfolders


# ==================== Part/Anomaly inference (folder) ====================
@torch.inference_mode()
def infer_folder_images(
    image_root_dir: str,
    part_models: Sequence[nn.Module],
    anomaly_model: nn.Module,
    config: InferenceConfig,
    device: torch.device,
) -> List[Dict]:
    """
    推理一个文件夹下所有图片，返回 List[Dict]：
      {
        "filename": basename,
        "part_name": 类别名 or "reject",
        "part_confidence": float,
        "anomaly_score": float  # abnormal 概率
      }
    """
    image_paths = collect_image_paths(root_dir=image_root_dir, exts=config.exts)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_root_dir}")

    dataset = FolderImageDataset(image_paths, resolution=config.resolution)
    id2lab = dataset.id2lab

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=(device.type != "cpu"),
    )

    results: List[Dict] = []
    thr = config.confidence_threshold

    for images, paths in tqdm(loader, desc=f"Infer[{os.path.basename(image_root_dir)}]", leave=False):
        images = images.to(device, non_blocking=True)

        logits_sum = None
        for m in part_models:
            out = m(images)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        logits = logits_sum / len(part_models)

        probs = logits.softmax(dim=1)
        conf, label = probs.max(dim=1)

        ac_logits = anomaly_model(images)
        ac_probs = F.softmax(ac_logits, dim=1)

        conf_cpu = conf.detach().cpu()
        label_cpu = label.detach().cpu()
        ac_probs_cpu = ac_probs.detach().cpu()

        for i, p in enumerate(paths):
            ok = float(conf_cpu[i].item()) >= thr
            part_name = id2lab[int(label_cpu[i].item())] if ok else "reject"
            anomaly_score = float(ac_probs_cpu[i, 1].item())  # abnormal prob

            results.append(
                {
                    "filename": os.path.basename(p),
                    "part_name": part_name,
                    "part_confidence": round(float(conf_cpu[i].item()), 5),
                    "anomaly_score": round(anomaly_score, 5),
                }
            )

    return results


def build_part_outputs(
    results: List[Dict],
    part_order: Sequence[str],
    base_dir: str,
    topk_per_part: int = 5,
    include_reject: bool = False,
    make_abs: bool = True,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    返回值全部是可读文件路径（而不是 filename）：
    - part_to_images: {部位: [img_path, ...]}
    - ordered_topk_images: [img_path, ...]  (按部位顺序，每部位取 anomaly_score topK)
    """

    def to_path(filename: str) -> str:
        p = os.path.join(base_dir, filename)
        return os.path.abspath(p) if make_abs else p

    part_to_images: Dict[str, List[str]] = {part: [] for part in part_order}
    if include_reject:
        part_to_images["reject"] = []

    for item in results:
        part_name = item.get("part_name")
        fn = item.get("filename")
        if not fn:
            continue
        img_path = to_path(fn)

        if part_name in part_to_images:
            part_to_images[part_name].append(img_path)
        elif include_reject and part_name == "reject":
            part_to_images["reject"].append(img_path)

    ordered_topk_images: List[str] = []
    for part in part_order:
        cur_items = [x for x in results if x.get("part_name") == part and x.get("filename")]
        cur_items = sorted(cur_items, key=lambda x: x.get("anomaly_score", 0.0), reverse=True)
        topk = cur_items[:topk_per_part]
        ordered_topk_images.extend([to_path(x["filename"]) for x in topk])

    return part_to_images, ordered_topk_images


# ==================== Qwen inference (multi-image) ====================
def build_endoscopy_qc_prompt(site: str, image_count: int) -> str:
    """
    生成内镜图像质控与选图助手的提示词文本。
    注意：为保持原有行为，提示词内容（包含引号样式）保持不变。
    """
    if not isinstance(site, str) or not site.strip():
        raise ValueError("site 必须是非空字符串")
    if not isinstance(image_count, int) or image_count < 1:
        raise ValueError("image_count 必须是 >= 1 的整数")

    placeholders = " or ".join(f"<image{i}>" for i in range(1, image_count + 1))

    text = (
        f"”作为内镜图像质控与选图助手，从视觉细节信息出发进行分析，"
        f"在同一部位{site}的候选胃镜图片（忽略离群图像）中选择一张在临床记录中信息量最大、可读性最高的代表图；"
        f"若存在病灶，优先选择最有利于病灶观察与记录的一张。你会选择哪一张？"
        f"{placeholders}\n"
        f"遵循以下格式输出：\n"
        f"<think>\n"
        f"请逐步分析图像、思考、比较，提出问题、验证结论。\n"
        f"</think>\n"
        f"<answer>\n"
        f"最终选择的图像编号。例如：<image1>\n"
        f"</answer>\n"
        f"“"
    )
    return text, placeholders


@torch.inference_mode()
def infer_select_images_qwen(
    model: Qwen2_5_VLForConditionalGeneration,
    images_dict: Dict[str, List[str]],
    processor: AutoProcessor,
    max_new_tokens: int = 2048,
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for part, img_list in images_dict.items():
        if not img_list or part == "reject":
            continue
        
        text, placeholders = build_endoscopy_qc_prompt(site=part, image_count=len(img_list))

        mapping = {f"<image{i+1}>": img_path for i, img_path in enumerate(img_list)}

        content = [{"type": "image", "image": img} for img in img_list] + [
            {"type": "text", "text": text}
        ]
        messages = [{"role": "user", "content": content}]
        print(f"Qwen input messages for part '{part}': {messages}")

        texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = process_vision_info([messages])

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
        pred = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results[part] = pred.strip()

    return results, mapping


@torch.inference_mode()
def infer_gi_endoscopy_qwen(
    model: Qwen2_5_VLForConditionalGeneration,
    images: Sequence[str],
    text: str,
    processor: AutoProcessor,
    max_new_tokens: int = 1024,
) -> str:
    content = [{"type": "image", "image": img} for img in images] + [
        {"type": "text", "text": text.replace("<image>\n", "")}
    ]
    messages = [{"role": "user", "content": content}]
    print(f"Qwen input messages: {messages}")

    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = process_vision_info([messages])

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
    pred = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return pred


# ==================== Model builders ====================
def build_part_models(
    model_paths: ModelPaths,
    enabled_part_models: Sequence[str],
    num_classes: int,
    device: torch.device,
) -> List[nn.Module]:
    part_models: List[nn.Module] = []
    enabled = set(enabled_part_models)

    if "res50" in enabled:
        if not model_paths.part_res50_ckpt:
            raise ValueError("part_res50_ckpt is None but 'res50' is enabled.")
        m = PartResNet50(
            num_classes=num_classes,
            pretrained_weights_path=model_paths.res50_backbone_pretrained,
            freeze_backbone=False,
        )
        load_checkpoint(m, model_paths.part_res50_ckpt, map_location="cpu")
        m.to(device).eval()
        part_models.append(m)

    if "conv" in enabled:
        if not model_paths.part_convnext_ckpt:
            raise ValueError("part_convnext_ckpt is None but 'conv' is enabled.")
        m = PartConvNeXt(num_classes=num_classes, freeze_backbone=False)
        load_checkpoint(m, model_paths.part_convnext_ckpt, map_location="cpu")
        m.to(device).eval()
        part_models.append(m)

    if "swin" in enabled:
        if not model_paths.part_swin_ckpt:
            raise ValueError("part_swin_ckpt is None but 'swin' is enabled.")
        m = PartSwinB(num_classes=num_classes, freeze_backbone=False)
        load_checkpoint(m, model_paths.part_swin_ckpt, map_location="cpu")
        m.to(device).eval()
        part_models.append(m)

    if not part_models:
        raise RuntimeError("No part model is enabled. Choose from: ('res50','conv','swin').")
    return part_models


def build_anomaly_model(anomaly_ckpt: str, device: torch.device) -> nn.Module:
    anomaly_model = AnomalyConvNeXt(freeze_backbone=True).to(device)
    anomaly_state = torch.load(anomaly_ckpt, map_location="cpu")
    anomaly_model.load_state_dict(anomaly_state, strict=False)
    anomaly_model.eval()
    return anomaly_model


def build_qwen(
    qwen_checkpoint_path: str,
    max_pixels: int,
    torch_dtype=torch.bfloat16,
    device_map: str = "auto",
) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_checkpoint_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2"
    )
    qwen_processor = AutoProcessor.from_pretrained(qwen_checkpoint_path, max_pixels=max_pixels)
    return qwen_model, qwen_processor


# ==================== Pipeline (single case / batch) ====================
def run_one_case(
    case_dir: str,
    part_models: Sequence[nn.Module],
    anomaly_model: nn.Module,
    qwen_select_model: Qwen2_5_VLForConditionalGeneration,
    qwen_select_processor: AutoProcessor,
    qwen_diag_model: Qwen2_5_VLForConditionalGeneration,
    qwen_diag_processor: AutoProcessor,
    diag_prompt_text: str,
    part_order: Sequence[str],
    config: InferenceConfig,
    device: torch.device,
    topk_per_part: int = 5,
    select_max_new_tokens: int = 512,
    diag_max_new_tokens: int = 1024,
) -> Tuple[Dict[str, str], str]:
    results = infer_folder_images(
        image_root_dir=case_dir,
        part_models=part_models,
        anomaly_model=anomaly_model,
        config=config,
        device=device,
    )

    part_to_images, ordered_topk_images = build_part_outputs(
        results=results,
        part_order=part_order,
        base_dir=case_dir,
        topk_per_part=topk_per_part,
        include_reject=True,
        make_abs=True,
    )

    candidate_imgs = [p for p in ordered_topk_images if os.path.isfile(p)]
    part_to_images = {part: [p for p in paths if os.path.isfile(p)] for part, paths in part_to_images.items()}

    if len(candidate_imgs) == 0:
        return {}, ""

    pred_select, mapping = infer_select_images_qwen(
        model=qwen_select_model,
        images_dict=part_to_images,
        processor=qwen_select_processor,
        max_new_tokens=select_max_new_tokens,
    )

    pred_diag = infer_gi_endoscopy_qwen(
        model=qwen_diag_model,
        images=candidate_imgs,
        text=diag_prompt_text,
        processor=qwen_diag_processor,
        max_new_tokens=diag_max_new_tokens,
    )

    return pred_select, pred_diag, mapping


def run_batch(
    sample_data_root: str,
    model_paths: ModelPaths,
    enabled_part_models: Sequence[str],
    part_order: Sequence[str],
    qwen_select_checkpoint_path: str,
    qwen_select_max_pixels: int,
    qwen_select_max_new_tokens: int,
    diag_prompt_text: str,
    qwen_diag_checkpoint_path: str,
    qwen_diag_max_pixels: int,
    qwen_diag_max_new_tokens: int,
    config: InferenceConfig,
    seed: int,
    device_pref: DevicePref,
    topk_per_part: int,
) -> None:
    set_seed(seed)
    device = get_device(device_pref=device_pref)
    print("device =", device)

    print("Building part models...")
    part_models = build_part_models(
        model_paths=model_paths,
        enabled_part_models=enabled_part_models,
        num_classes=config.num_classes,
        device=device,
    )

    if not model_paths.anomaly_ckpt:
        raise ValueError("anomaly_ckpt is None. It is required.")

    print("Building anomaly model...")
    anomaly_model = build_anomaly_model(model_paths.anomaly_ckpt, device=device)
    print("Weights loaded.")

    print("Loading Qwen (select) model/processor...")
    qwen_select_model, qwen_select_processor = build_qwen(
        qwen_checkpoint_path=qwen_select_checkpoint_path,
        max_pixels=qwen_select_max_pixels,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading Qwen (diagnosis) model/processor...")
    qwen_diag_model, qwen_diag_processor = build_qwen(
        qwen_checkpoint_path=qwen_diag_checkpoint_path,
        max_pixels=qwen_diag_max_pixels,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    case_dirs = list_subfolders(sample_data_root)
    if not case_dirs:
        raise RuntimeError(f"No subfolders found under: {sample_data_root}")
    print(f"Found {len(case_dirs)} cases under '{sample_data_root}'")

    for case_dir in tqdm(case_dirs, desc="Batch", leave=True):
        case_name = os.path.basename(case_dir)
        try:
            pred_select, pred_diag, mapping = run_one_case(
                case_dir=case_dir,
                part_models=part_models,
                anomaly_model=anomaly_model,
                qwen_select_model=qwen_select_model,
                qwen_select_processor=qwen_select_processor,
                qwen_diag_model=qwen_diag_model,
                qwen_diag_processor=qwen_diag_processor,
                diag_prompt_text=diag_prompt_text,
                part_order=part_order,
                config=config,
                device=device,
                topk_per_part=topk_per_part,
                select_max_new_tokens=qwen_select_max_new_tokens,
                diag_max_new_tokens=qwen_diag_max_new_tokens,
            )

            print("\n" + "=" * 80)
            print(f"[{case_name}]")

            print("代表性图像选择（中文版）：")
            for part, sel in pred_select.items():
                print(f"  {part}: {sel}")
                print("\n")

            print("Representative Image Selection (English):")
            for part, sel in pred_select.items():
                part_en = zh2en.get(part, part)
                print(f"  {part_en}: {sel}")
                print("\n")

            print("代表性图像路径映射（Representative Image Path Mapping）：")
            for placeholder, img_path in mapping.items():
                print(f"  {placeholder}: {img_path}")

            print("\n报告生成（中文版）：")
            print(pred_diag)

            print("\nClinical Report Generation (English):")
            print(translate_text_english(pred_diag))
            print("=" * 80 + "\n")

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"[{case_name}] ERROR: {type(e).__name__}: {str(e)}")
            print("=" * 80 + "\n")


# ======================================================================================
#                                  参数区
# ======================================================================================

SAMPLE_DATA_ROOT = "./sample_data"

model_paths = ModelPaths(
    part_res50_ckpt="./output/part_res50_best.pth",
    part_convnext_ckpt="./output/part_convnext_best.pth",
    part_swin_ckpt="./output/part_swin_best.pth",
    anomaly_ckpt="./output/anomaly_convnext_freeze_ep30.pth",
    res50_backbone_pretrained="./output/res50_backbone_dino_swsl_gastronet.pth",
)

config = InferenceConfig(
    num_classes=8,
    resolution=224,
    batch_size=256,
    num_workers=8,
    confidence_threshold=0.4,
)

PART_ORDER = [
    "食道",
    "贲门",
    "胃底",
    "胃体",
    "胃角",
    "胃窦和幽门",
    "十二指肠球部",
    "十二指肠降部",
]

PART_ORDER_EN = [
    "Esophagus",
    "Cardia",
    "Gastric Fundus",
    "Gastric Body",
    "Gastric Angle",
    "Gastric Antrum and Pylorus",
    "Duodenal Bulb",
    "Descending Duodenum",
]

zh2en = dict(zip(PART_ORDER, PART_ORDER_EN))
en2zh = dict(zip(PART_ORDER_EN, PART_ORDER))

ENABLED_PART_MODELS = ("conv", "swin", "res50")  # 可选：("res50","conv","swin")
TOPK_PER_PART = 5

QWEN_SELECT_CHECKPOINT_PATH = "output/global_step_50"
QWEN_SELECT_MAX_PIXELS = 224 * 224
QWEN_SELECT_MAX_NEW_TOKENS = 4096

DIAG_PROMPT_TEXT = "这是一组上消化道内镜检查图像，请你据此得出各部位描述和诊断结论。"
QWEN_DIAG_CHECKPOINT_PATH = "output/checkpoint-1660"
QWEN_DIAG_MAX_PIXELS = 224 * 224
QWEN_DIAG_MAX_NEW_TOKENS = 1024

SEED = 42
DEVICE_PREF: DevicePref = "auto"  # "npu" / "cuda" / "cpu" / "auto"

run_batch(
    sample_data_root=SAMPLE_DATA_ROOT,
    model_paths=model_paths,
    enabled_part_models=ENABLED_PART_MODELS,
    part_order=PART_ORDER,
    # --- 选图 ---
    qwen_select_checkpoint_path=QWEN_SELECT_CHECKPOINT_PATH,
    qwen_select_max_pixels=QWEN_SELECT_MAX_PIXELS,
    qwen_select_max_new_tokens=QWEN_SELECT_MAX_NEW_TOKENS,
    # --- 文本 ---
    diag_prompt_text=DIAG_PROMPT_TEXT,
    qwen_diag_checkpoint_path=QWEN_DIAG_CHECKPOINT_PATH,
    qwen_diag_max_pixels=QWEN_DIAG_MAX_PIXELS,
    qwen_diag_max_new_tokens=QWEN_DIAG_MAX_NEW_TOKENS,
    # --- 其它 ---
    config=config,
    seed=SEED,
    device_pref=DEVICE_PREF,
    topk_per_part=TOPK_PER_PART,
)
