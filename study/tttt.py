# 最小外接矩形ではなく、細長さを面積ベースで判定
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue
    
    # 骨格化して長さを測る方が曲がった棒に向いてる
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # 細線化（骨格抽出）
    from skimage.morphology import skeletonize
    skeleton = skeletonize(mask // 255)
    length = np.sum(skeleton)  # 骨格のピクセル数≒長さ
    
    width = area / length  # 面積÷長さ≒太さ
    
    if width < 20:  # 細いものだけ
        rods.append(cnt)




import cv2
import numpy as np

def count_rods(image_path, min_area=50, max_aspect_ratio=0.3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rods = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 傾いた最小外接矩形
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        
        if w == 0 or h == 0:
            continue
        
        aspect_ratio = min(w, h) / max(w, h)  # 細長いほど0に近い
        
        if aspect_ratio < max_aspect_ratio:  # 細長いものだけ
            rods.append({
                "center": (cx, cy),
                "angle": angle,
                "width": min(w, h),
                "length": max(w, h),
                "aspect_ratio": aspect_ratio
            })
    
    print(f"検出数: {len(rods)}")
    return rods

rods = count_rods("input.tif")



import cv2
import numpy as np

def count_blobs(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 二値化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # 小さすぎるノイズを除外
            continue
        
        # 楕円フィット（5点以上必要）
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (minor, major), angle = ellipse
            
            aspect_ratio = minor / major  # 1に近いほど丸、0に近いほど細長い
            
            # 条件を緩めに（丸〜細長い楕円まで許容）
            if aspect_ratio > 0.2:  # 極端に細長いものは除外
                blobs.append({
                    "center": (cx, cy),
                    "angle": angle,
                    "aspect_ratio": aspect_ratio
                })
    
    print(f"検出数: {len(blobs)}")
    return blobs

blobs = count_blobs("input.tif")



import cv2
import numpy as np

def count_circles(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 二値化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 丸っぽいものだけフィルタ
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)  # 1に近いほど丸
        if circularity > 0.7 and area > 50:  # 閾値は調整
            circles.append(cnt)
    
    print(f"検出数: {len(circles)}")
    return len(circles)

count_circles("input.tif")




# sam2_count.py
# SAM2（Segment Anything Model 2）で物体をセグメントしてカウントする
#
# 【インストール】
#   pip install torch torchvision
#   pip install git+https://github.com/facebookresearch/sam2.git
#
# 【モデルのダウンロード】
#   以下のいずれかを手動でダウンロードしてください（SAM2.1が最新推奨）
#
#   SAM2.1（推奨）:
#     ViT-T（超軽量）: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
#     ViT-S（軽量）  : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
#     ViT-B（バランス）: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
#     ViT-L（高精度）: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
#
# 【使い方】
#   python sam2_count.py 画像.jpg --checkpoint sam2.1_hiera_small.pt --model-type small
#
# 【細長い物体向けフィルタの調整】
#   --min-area  : 小さすぎるマスクを除外（ノイズ対策）
#   --max-area  : 大きすぎるマスクを除外（背景対策）
#   --min-ratio : 縦横比の最小値（細長さフィルタ。1対5なら 3.0〜5.0 を推奨）
#   --max-ratio : 縦横比の最大値

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os


# SAM2のモデル設定ファイル名マッピング
MODEL_CONFIG_MAP = {
    'tiny'      : 'configs/sam2.1/sam2.1_hiera_t.yaml',
    'small'     : 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'base_plus' : 'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'large'     : 'configs/sam2.1/sam2.1_hiera_l.yaml',
}


def load_sam2(checkpoint, model_type, device):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print('[ERROR] sam2がインストールされていません。')
        print('        以下を実行してください:')
        print('        pip install git+https://github.com/facebookresearch/sam2.git')
        raise

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {checkpoint}\n"
            f"以下のURLからダウンロードしてください:\n"
            f"  tiny  : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt\n"
            f"  small : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt\n"
            f"  base+ : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt\n"
            f"  large : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        )

    if model_type not in MODEL_CONFIG_MAP:
        raise ValueError(f"model-type は {list(MODEL_CONFIG_MAP.keys())} のいずれかを指定してください")

    config = MODEL_CONFIG_MAP[model_type]

    import torch
    sam2 = build_sam2(config, checkpoint, device=device)
    sam2.eval()

    # 自動マスク生成の設定
    # points_per_side     : 大きいほど細かく検出（重くなる）
    # pred_iou_thresh     : マスクの品質閾値
    # stability_score_thresh: マスクの安定性閾値
    mask_generator = SAM2AutomaticMaskGenerator(
        model                   = sam2,
        points_per_side         = 32,
        pred_iou_thresh         = 0.88,
        stability_score_thresh  = 0.95,
        crop_n_layers           = 0,
        min_mask_region_area    = 100,
    )
    return mask_generator


def get_bbox_ratio(mask):
    """マスクのバウンディングボックスの縦横比（長辺/短辺）を返す"""
    x, y, w, h = mask['bbox']  # xywh形式
    if w == 0 or h == 0:
        return 0
    return max(w, h) / min(w, h)


def filter_masks(masks, image_area, min_area_ratio, max_area_ratio, min_ratio, max_ratio):
    """
    面積・縦横比でマスクをフィルタする

    Args:
        masks          : SAM2が生成したマスクのリスト
        image_area     : 画像の総ピクセル数
        min_area_ratio : 画像面積に対する最小マスク面積の割合
        max_area_ratio : 画像面積に対する最大マスク面積の割合
        min_ratio      : バウンディングボックスの縦横比の最小値
        max_ratio      : バウンディングボックスの縦横比の最大値
    """
    filtered = []
    for m in masks:
        area  = m['area']
        ratio = get_bbox_ratio(m)

        if area < image_area * min_area_ratio:
            continue
        if area > image_area * max_area_ratio:
            continue
        if ratio < min_ratio:
            continue
        if ratio > max_ratio:
            continue

        filtered.append(m)

    return filtered


def visualize(img_pil, masks_all, masks_filtered, output_path):
    """元画像・全マスク・フィルタ後マスクを並べて表示"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    img_np = np.array(img_pil)

    # --- 元画像 ---
    axes[0].imshow(img_np)
    axes[0].set_title('元画像', fontsize=13)
    axes[0].axis('off')

    # --- 全マスク ---
    overlay_all = img_np.copy().astype(np.float32)
    rng = random.Random(42)
    for m in masks_all:
        color = np.array([rng.random() * 255 for _ in range(3)], dtype=np.float32)
        overlay_all[m['segmentation']] = (
            overlay_all[m['segmentation']] * 0.4 + color * 0.6
        )
    axes[1].imshow(overlay_all.astype(np.uint8))
    axes[1].set_title(f'全マスク: {len(masks_all)} 個', fontsize=13)
    axes[1].axis('off')

    # --- フィルタ後マスク ---
    overlay_filt = img_np.copy().astype(np.float32)
    rng2 = random.Random(0)
    for m in masks_filtered:
        color = np.array([rng2.random() * 255 for _ in range(3)], dtype=np.float32)
        overlay_filt[m['segmentation']] = (
            overlay_filt[m['segmentation']] * 0.4 + color * 0.6
        )
        x, y, w, h = m['bbox']
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.5, edgecolor='yellow', facecolor='none'
        )
        axes[2].add_patch(rect)

    axes[2].imshow(overlay_filt.astype(np.uint8))
    axes[2].set_title(f'フィルタ後（カウント）: {len(masks_filtered)} 個', fontsize=13)
    axes[2].axis('off')

    plt.suptitle('SAM2 物体カウント', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] 結果を保存: {output_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='SAM2で物体をセグメントしてカウント')
    parser.add_argument('image',
                        help='入力画像のパス')
    parser.add_argument('--checkpoint',  required=True,
                        help='SAM2モデルのチェックポイント (.pt)')
    parser.add_argument('--model-type',  default='small',
                        choices=['tiny', 'small', 'base_plus', 'large'],
                        help='モデルの種類 (default: small)\n'
                             '  tiny      : 最軽量・CPU向け\n'
                             '  small     : 軽量・バランス型（推奨）\n'
                             '  base_plus : 高精度\n'
                             '  large     : 最高精度・GPU推奨')
    parser.add_argument('--min-area',    type=float, default=0.001,
                        help='最小面積（画像面積の割合 default: 0.001 = 0.1%%）')
    parser.add_argument('--max-area',    type=float, default=0.3,
                        help='最大面積（画像面積の割合 default: 0.3 = 30%%）')
    parser.add_argument('--min-ratio',   type=float, default=1.0,
                        help='縦横比の最小値（細長さ下限 default: 1.0）\n'
                             '1対5の細長い物なら 3.0〜5.0 を推奨')
    parser.add_argument('--max-ratio',   type=float, default=20.0,
                        help='縦横比の最大値（細長さ上限 default: 20.0）')
    parser.add_argument('--output',      default='result_sam2.png',
                        help='結果画像の保存先 (default: result_sam2.png)')
    parser.add_argument('--cpu',         action='store_true',
                        help='CPUを強制使用（遅いので注意）')
    args = parser.parse_args()

    # デバイス設定
    import torch
    if args.cpu:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'[INFO] デバイス: {device}')
    print(f'[INFO] モデル  : SAM2.1 {args.model_type}')

    # モデル読み込み
    mask_generator = load_sam2(args.checkpoint, args.model_type, device)

    # 画像読み込み
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f'画像が見つかりません: {args.image}')
    img_pil    = Image.open(args.image).convert('RGB')
    img_np     = np.array(img_pil)
    image_area = img_np.shape[0] * img_np.shape[1]
    print(f'[INFO] 画像サイズ: {img_pil.size[0]}x{img_pil.size[1]}')

    # SAM2推論
    print('[INFO] SAM2推論中...')
    masks_all = mask_generator.generate(img_np)
    print(f'[INFO] 生成マスク数: {len(masks_all)}')

    # フィルタ
    masks_filtered = filter_masks(
        masks_all, image_area,
        min_area_ratio = args.min_area,
        max_area_ratio = args.max_area,
        min_ratio      = args.min_ratio,
        max_ratio      = args.max_ratio,
    )

    # 結果表示
    print(f"\n{'='*40}")
    print(f'  全マスク数          : {len(masks_all)}')
    print(f'  フィルタ後（カウント）: {len(masks_filtered)}')
    print(f"{'='*40}\n")

    if len(masks_filtered) == 0:
        print('[HINT] カウントが0の場合は以下を試してください:')
        print('       --min-ratio を下げる（例: --min-ratio 1.0）')
        print('       --min-area  を下げる（例: --min-area 0.0005）')
        print('       --max-area  を上げる（例: --max-area 0.5）')

    # 可視化
    visualize(img_pil, masks_all, masks_filtered, args.output)


if __name__ == '__main__':
    main()



# sam_count.py
# SAM（Segment Anything Model）で物体をセグメントしてカウントする
#
# 【インストール】
#   pip install torch torchvision
#   pip install git+https://github.com/facebookresearch/segment-anything.git
#
# 【モデルのダウンロード】
#   以下のいずれかを手動でダウンロードしてください
#   ViT-H（高精度・重い）: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
#   ViT-L（バランス型）  : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
#   ViT-B（軽量・速い）  : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#
# 【使い方】
#   python sam_count.py 画像.jpg --checkpoint sam_vit_b_01ec64.pth --model-type vit_b
#
# 【細長い物体向けフィルタの調整】
#   --min-area     : 小さすぎるマスクを除外（ノイズ対策）
#   --max-area     : 大きすぎるマスクを除外（背景対策）
#   --min-ratio    : 縦横比の最小値（細長さフィルタ。1対5なら5.0を指定）
#   --max-ratio    : 縦横比の最大値

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os


def load_sam(checkpoint, model_type, device):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print('[ERROR] segment-anythingがインストールされていません。')
        print('        以下を実行してください:')
        print('        pip install git+https://github.com/facebookresearch/segment-anything.git')
        raise

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {checkpoint}\n"
            f"以下のURLからダウンロードしてください:\n"
            f"  ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
            f"  ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
            f"  ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )

    import torch
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    sam.eval()

    # 自動マスク生成の設定
    # points_per_side: 大きいほど細かく検出（重くなる）
    # pred_iou_thresh: マスクの品質閾値
    # stability_score_thresh: マスクの安定性閾値
    mask_generator = SamAutomaticMaskGenerator(
        model                  = sam,
        points_per_side        = 32,
        pred_iou_thresh        = 0.88,
        stability_score_thresh = 0.95,
        crop_n_layers          = 0,
        min_mask_region_area   = 100,
    )
    return mask_generator


def get_bbox_ratio(mask):
    """マスクのバウンディングボックスの縦横比（長辺/短辺）を返す"""
    x, y, w, h = mask['bbox']  # xywh形式
    if w == 0 or h == 0:
        return 0
    return max(w, h) / min(w, h)


def filter_masks(masks, image_area, min_area_ratio, max_area_ratio, min_ratio, max_ratio):
    """
    面積・縦横比でマスクをフィルタする

    Args:
        masks          : SAMが生成したマスクのリスト
        image_area     : 画像の総ピクセル数
        min_area_ratio : 画像面積に対する最小マスク面積の割合
        max_area_ratio : 画像面積に対する最大マスク面積の割合
        min_ratio      : バウンディングボックスの縦横比の最小値
        max_ratio      : バウンディングボックスの縦横比の最大値
    """
    filtered = []
    for m in masks:
        area = m['area']
        ratio = get_bbox_ratio(m)

        # 面積フィルタ
        if area < image_area * min_area_ratio:
            continue
        if area > image_area * max_area_ratio:
            continue

        # 縦横比フィルタ（細長さ）
        if ratio < min_ratio:
            continue
        if ratio > max_ratio:
            continue

        filtered.append(m)

    return filtered


def visualize(img_pil, masks_all, masks_filtered, output_path):
    """元画像・全マスク・フィルタ後マスクを並べて表示"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    img_np = np.array(img_pil)

    # --- 元画像 ---
    axes[0].imshow(img_np)
    axes[0].set_title('元画像', fontsize=13)
    axes[0].axis('off')

    # --- 全マスク ---
    overlay_all = img_np.copy()
    rng = random.Random(42)
    for m in masks_all:
        color = [int(rng.random() * 255) for _ in range(3)]
        overlay_all[m['segmentation']] = (
            overlay_all[m['segmentation']] * 0.4 +
            np.array(color) * 0.6
        ).astype(np.uint8)
    axes[1].imshow(overlay_all)
    axes[1].set_title(f'全マスク: {len(masks_all)} 個', fontsize=13)
    axes[1].axis('off')

    # --- フィルタ後マスク ---
    overlay_filt = img_np.copy()
    rng2 = random.Random(0)
    for m in masks_filtered:
        color = [int(rng2.random() * 255) for _ in range(3)]
        overlay_filt[m['segmentation']] = (
            overlay_filt[m['segmentation']] * 0.4 +
            np.array(color) * 0.6
        ).astype(np.uint8)
        # バウンディングボックスも描画
        x, y, w, h = m['bbox']
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.5, edgecolor='yellow', facecolor='none'
        )
        axes[2].add_patch(rect)

    axes[2].imshow(overlay_filt)
    axes[2].set_title(f'フィルタ後（カウント）: {len(masks_filtered)} 個', fontsize=13)
    axes[2].axis('off')

    plt.suptitle('SAM 物体カウント', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] 結果を保存: {output_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='SAMで物体をセグメントしてカウント')
    parser.add_argument('image',
                        help='入力画像のパス')
    parser.add_argument('--checkpoint',  required=True,
                        help='SAMモデルのチェックポイント (.pth)')
    parser.add_argument('--model-type',  default='vit_b',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='モデルの種類 (default: vit_b)')
    parser.add_argument('--min-area',    type=float, default=0.001,
                        help='最小面積（画像面積の割合 default: 0.001 = 0.1%%）')
    parser.add_argument('--max-area',    type=float, default=0.3,
                        help='最大面積（画像面積の割合 default: 0.3 = 30%%）')
    parser.add_argument('--min-ratio',   type=float, default=1.0,
                        help='縦横比の最小値（細長さ下限 default: 1.0）\n'
                             '1対5の細長い物なら 3.0〜5.0 を推奨')
    parser.add_argument('--max-ratio',   type=float, default=20.0,
                        help='縦横比の最大値（細長さ上限 default: 20.0）')
    parser.add_argument('--output',      default='result_sam.png',
                        help='結果画像の保存先 (default: result_sam.png)')
    parser.add_argument('--cpu',         action='store_true',
                        help='CPUを強制使用（遅いので注意）')
    args = parser.parse_args()

    # デバイス設定
    import torch
    if args.cpu:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'[INFO] デバイス: {device}')

    # モデル読み込み
    mask_generator = load_sam(args.checkpoint, args.model_type, device)

    # 画像読み込み
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f'画像が見つかりません: {args.image}')
    img_pil = Image.open(args.image).convert('RGB')
    img_np  = np.array(img_pil)
    image_area = img_np.shape[0] * img_np.shape[1]
    print(f'[INFO] 画像サイズ: {img_pil.size[0]}x{img_pil.size[1]}')

    # SAM推論
    print('[INFO] SAM推論中... (初回は少し時間がかかります)')
    masks_all = mask_generator.generate(img_np)
    print(f'[INFO] 生成マスク数: {len(masks_all)}')

    # フィルタ
    masks_filtered = filter_masks(
        masks_all, image_area,
        min_area_ratio = args.min_area,
        max_area_ratio = args.max_area,
        min_ratio      = args.min_ratio,
        max_ratio      = args.max_ratio,
    )

    # 結果表示
    print(f"\n{'='*40}")
    print(f'  全マスク数        : {len(masks_all)}')
    print(f'  フィルタ後（カウント）: {len(masks_filtered)}')
    print(f"{'='*40}\n")

    if len(masks_filtered) == 0:
        print('[HINT] カウントが0の場合は以下を試してください:')
        print('       --min-ratio を下げる（例: --min-ratio 1.0）')
        print('       --min-area  を下げる（例: --min-area 0.0005）')
        print('       --max-area  を上げる（例: --max-area 0.5）')

    # 可視化
    visualize(img_pil, masks_all, masks_filtered, args.output)


if __name__ == '__main__':
    main()

# faster_rcnn_coco.py
# COCO学習済みFaster R-CNNでお試し推論（学習不要）
#
# 使い方:
#   python faster_rcnn_coco.py 画像.jpg
#   python faster_rcnn_coco.py 画像.jpg --thresh 0.7
#   python faster_rcnn_coco.py 画像.jpg --classes 44 45 46  # 特定クラスのみ
#
# 初回実行時にCOCO学習済み重み（約160MB）を自動ダウンロードします。
# 基本（これだけでOK）
python faster_rcnn_coco.py 画像.jpg

# 検出が少ない場合 → 閾値を下げる
python faster_rcnn_coco.py 画像.jpg --thresh 0.3

# 検出が多すぎる場合 → 閾値を上げる
python faster_rcnn_coco.py 画像.jpg --thresh 0.7

# 結果画像の保存先を指定
python faster_rcnn_coco.py 画像.jpg --output 結果.png

# COCOの80クラス一覧を確認する
python faster_rcnn_coco.py 画像.jpg --list-classes


import argparse
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import random

# COCO 80クラスのラベル一覧
COCO_LABELS = [
    '__background__',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(device):
    print('[INFO] COCO学習済みモデルを読み込み中...')
    print('       (初回は約160MBのダウンロードが発生します)')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    print('[INFO] モデル準備完了')
    return model


def predict(model, image_path, device, score_thresh=0.5, target_classes=None):
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    boxes  = output['boxes'].cpu()
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()

    # スコア閾値フィルタ
    keep = scores >= score_thresh
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # 特定クラスフィルタ（指定がある場合）
    if target_classes:
        mask = torch.tensor([l.item() in target_classes for l in labels])
        boxes  = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

    return img_pil, boxes.numpy(), labels.numpy(), scores.numpy()


def visualize(img_pil, boxes, labels, scores, output_path='result.png'):
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.imshow(img_pil)

    # クラスごとに色を固定
    color_map = {}
    rng = random.Random(42)

    # クラスごとのカウント集計
    count_per_class = {}

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        class_name = COCO_LABELS[label] if label < len(COCO_LABELS) else str(label)

        if class_name not in color_map:
            color_map[class_name] = (
                rng.random(), rng.random(), rng.random()
            )
        color = color_map[class_name]

        count_per_class[class_name] = count_per_class.get(class_name, 0) + 1

        # バウンディングボックス
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # ラベル
        text = ax.text(
            x1, y1 - 4,
            f'{class_name} {score:.2f}',
            color='white', fontsize=8, fontweight='bold',
        )
        text.set_path_effects([
            pe.Stroke(linewidth=2, foreground=color),
            pe.Normal()
        ])

    total = len(boxes)
    title = f'検出総数: {total}個'
    if count_per_class:
        detail = '  |  ' + '  '.join(
            f'{k}: {v}' for k, v in sorted(count_per_class.items())
        )
        title += detail

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] 結果を保存: {output_path}')
    plt.show()

    # コンソールにもサマリー表示
    print(f\"\
{'='*40}\")
    print(f'  検出総数: {total} 個')
    if count_per_class:
        print('  クラス別内訳:')
        for k, v in sorted(count_per_class.items(), key=lambda x: -x[1]):
            print(f'    {k:20s}: {v} 個')
    print(f\"{'='*40}\
\")


def main():
    parser = argparse.ArgumentParser(
        description='COCO学習済みFaster R-CNNで物体を検出・カウントします'
    )
    parser.add_argument('image',
                        help='入力画像のパス')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='検出スコア閾値 (default: 0.5)  低くすると検出が増える')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='検出するクラスIDを絞り込む (例: --classes 1 で人のみ)')
    parser.add_argument('--output', default='result.png',
                        help='結果画像の保存先 (default: result.png)')
    parser.add_argument('--list-classes', action='store_true',
                        help='COCOクラス一覧を表示して終了')
    parser.add_argument('--cpu', action='store_true',
                        help='CPUを強制使用')
    args = parser.parse_args()

    # クラス一覧表示モード
    if args.list_classes:
        print('\
COCO クラス一覧:')
        for i, name in enumerate(COCO_LABELS):
            if i == 0:
                continue
            print(f'  {i:3d}: {name}')
        return

    device = get_device(args.cpu)
    print(f'[INFO] デバイス: {device}')

    model = load_model(device)

    img_pil, boxes, labels, scores = predict(
        model, args.image, device,
        score_thresh=args.thresh,
        target_classes=args.classes,
    )

    visualize(img_pil, boxes, labels, scores, output_path=args.output)


if __name__ == '__main__':
    main()


# faster_rcnn_count.py
# Faster R-CNN による細長い物体のカウント
#
# 【CSRNetとの違い】
#   CSRNet  : 密度マップで「何個あるか」を推定（境界ボックスなし）
#   Faster R-CNN : 1本ずつ検出してカウント（位置・バウンディングボックスあり）
#
# 【データ構成例】
# dataset/
#   images/
#     img_001.jpg
#   annotations/
#     img_001.json  ← バウンディングボックスアノテーション（後述）
#
# 【アノテーションJSON フォーマット】
# {
#   "boxes": [
#     {"x1": 10, "y1": 20, "x2": 80, "y2": 35},  ← 左上(x1,y1) 右下(x2,y2)
#     ...
#   ]
# }

import os
import json
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# COCO 80クラスのラベル一覧\r\nCOCO_LABELS = [\r\n    '__background__',\r\n    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',\r\n    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',\r\n    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',\r\n    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',\r\n    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',\r\n    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',\r\n    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',\r\n    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',\r\n    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',\r\n    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',\r\n    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',\r\n    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',\r\n    'scissors', 'teddy bear', 'hair drier', 'toothbrush',\r\n]\r\n\r\n\r\n# ==================== モデル構築 ====================
def build_model(num_classes=2, pretrained=True):
    """
    Faster R-CNN (ResNet-50 + FPN) を構築する。

    num_classes: 背景(0) + 物体クラス数
                 物体が1種類なら num_classes=2
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # ヘッド部分を対象クラス数に差し替え
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ==================== データセット ====================

class RodDataset(Dataset):
    """
    バウンディングボックスアノテーション付きデータセット。
    細長い物体（棒・麺・製品など）向け。
    """

    def __init__(self, images_dir, annotations_dir, augment=True):
        self.images_dir      = images_dir
        self.annotations_dir = annotations_dir
        self.augment         = augment

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_paths = sorted(
            p for ext in exts
            for p in glob.glob(os.path.join(images_dir, ext))
        )
        if not self.image_paths:
            raise RuntimeError(f"画像が見つかりません: {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_annotation(self, img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(self.annotations_dir, stem + '.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"アノテーションが見つかりません: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        return data.get('boxes', [])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes_raw = self._load_annotation(img_path)

        # ボックス座標を float32 テンソルに変換
        boxes = []
        for b in boxes_raw:
            x1, y1 = float(b['x1']), float(b['y1'])
            x2, y2 = float(b['x2']), float(b['y2'])
            # 幅・高さが0以下のボックスをスキップ
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        # ---- データ拡張 ----
        if self.augment:
            # 水平反転
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
            # 垂直反転（俯瞰撮影では有効）
            if torch.rand(1) > 0.5:
                img = TF.vflip(img)
                boxes = [[b[0], h - b[3], b[2], h - b[1]] for b in boxes]
            # 輝度・コントラスト変動
            img = TF.adjust_brightness(img, 0.8 + torch.rand(1).item() * 0.4)
            img = TF.adjust_contrast(img,   0.8 + torch.rand(1).item() * 0.4)

        # ---- テンソル変換 ----
        img_tensor = TF.to_tensor(img)  # (C, H, W) in [0,1]

        if boxes:
            boxes_tensor  = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones(len(boxes), dtype=torch.int64)  # クラス1=物体
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        target = {
            'boxes' : boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    """DataLoader 用コレート関数（サイズが違う画像に対応）"""
    return tuple(zip(*batch))


# ==================== 学習ループ ====================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def train_one_epoch(model, loader, optimizer, device, print_freq=10):
    model.train()
    loss_meter = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # 勾配クリッピング（学習安定化）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(losses.item(), len(images))

        if (i + 1) % print_freq == 0:
            print(f"  step [{i+1}/{len(loader)}]  loss: {loss_meter.avg:.4f}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, score_thresh=0.5):
    """
    検証: 予測カウント数 vs GT カウント数 の MAE を返す
    """
    model.eval()
    mae_total = 0.0
    count = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, target in zip(preds, targets):
            # スコア閾値でフィルタ
            keep = pred['scores'] >= score_thresh
            pred_count = keep.sum().item()
            gt_count   = len(target['boxes'])
            mae_total += abs(pred_count - gt_count)
            count += 1

    return mae_total / max(count, 1)


def plot_history(train_losses, val_maes, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(val_maes, color='orange', label='Val MAE')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE')
    ax2.set_title('Validation MAE'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] 学習曲線: {save_path}")


# ==================== 学習メイン ====================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # デバイス
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[INFO] デバイス: {device}")

    # データセット
    full_ds = RodDataset(args.images, args.annotations, augment=True)
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    # モデル
    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)

    # オプティマイザ（バックボーンは小さいLR）
    params = [
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad], 'lr': args.lr},
    ]
    optimizer = optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_mae = float('inf')
    train_losses, val_maes = [], []
    start_epoch = 0

    # チェックポイント再開
    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_mae    = ckpt.get('best_mae', float('inf'))
        print(f"[INFO] 再開: epoch {start_epoch}, best MAE={best_mae:.2f}")

    print(f"\n{'='*50}")
    print(f"  Faster R-CNN 学習開始")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        mae  = validate(model, val_loader, device, score_thresh=args.score_thresh)
        scheduler.step()

        train_losses.append(loss)
        val_maes.append(mae)
        print(f"Epoch [{epoch+1:>3}/{args.epochs}]  Loss: {loss:.4f}  Val MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → ベストモデル更新 (MAE={best_mae:.2f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pth'))

    print(f"\n学習完了！  Best Val MAE: {best_mae:.2f}")
    plot_history(train_losses, val_maes,
                 os.path.join(args.output_dir, 'training_history.png'))


# ==================== 推論メイン ====================

# COCO 80クラスのラベル一覧
COCO_LABELS = [
    '__background__',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def predict(args):
    # デバイス
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # --weights なし → COCOお試しモード
    coco_mode = (args.weights is None)

    if coco_mode:
        print('[INFO] --weights 未指定: COCO学習済みモデルで動作します（お試しモード）')
        print('       (初回は約160MBのダウンロードが発生します)')
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(device)
    else:
        model = build_model(num_classes=2, pretrained=False).to(device)
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"重みが見つかりません: {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        print(f"[INFO] 重みを読み込みました: {args.weights}")

    model.eval()

    # 画像読み込み
    img_pil = Image.open(args.image).convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    keep   = output['scores'] >= args.score_thresh
    boxes  = output['boxes'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy() if coco_mode else None
    count  = len(boxes)

    print(f"\n{'='*40}")
    print(f"  検出数（カウント）: {count}")
    if coco_mode and count > 0:
        from collections import Counter
        class_counts = Counter(
            COCO_LABELS[l] if l < len(COCO_LABELS) else str(l)
            for l in labels
        )
        print('  クラス別内訳:')
        for name, cnt in class_counts.most_common():
            print(f'    {name:20s}: {cnt} 個')
    print(f"{'='*40}\n")

    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_pil)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        label_str = ''
        if coco_mode and labels is not None:
            l = labels[i]
            label_str = (COCO_LABELS[l] if l < len(COCO_LABELS) else str(l)) + ' '
        ax.text(x1, y1 - 4, f'{label_str}{score:.2f}',
                color='red', fontsize=7, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))

    mode_str = 'COCOお試しモード' if coco_mode else '自前学習モデル'
    ax.set_title(
        f'Faster R-CNN [{mode_str}]  検出数: {count}  (threshold={args.score_thresh})',
        fontsize=13, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 結果を保存: {args.output}")
    plt.show()


# ==================== エントリポイント ====================

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN 物体カウント')
    sub = parser.add_subparsers(dest='mode', required=True)

    # --- train ---
    t = sub.add_parser('train', help='学習')
    t.add_argument('--images',          required=True)
    t.add_argument('--annotations',     required=True)
    t.add_argument('--output-dir',      default='checkpoints')
    t.add_argument('--epochs',          type=int,   default=60)
    t.add_argument('--batch-size',      type=int,   default=2,
                   help='CPU では 1〜2 推奨')
    t.add_argument('--lr',              type=float, default=5e-3)
    t.add_argument('--val-ratio',       type=float, default=0.1)
    t.add_argument('--score-thresh',    type=float, default=0.5)
    t.add_argument('--no-pretrained',   action='store_true',
                   help='COCO 事前学習重みを使わない')
    t.add_argument('--pretrained-ckpt', type=str, default=None,
                   help='学習再開用チェックポイント')
    t.add_argument('--cpu',             action='store_true')

    # --- predict ---
    p = sub.add_parser('predict', help='推論')
    p.add_argument('image',             help='入力画像パス')
    p.add_argument('--weights',         default=None,
                   help='学習済み重みファイル (.pth)  省略するとCOCOお試しモードで動作')
    p.add_argument('--score-thresh',    type=float, default=0.5,
                   help='検出スコア閾値（上げると厳しく、下げると甘く）')
    p.add_argument('--output',          default='result.png')
    p.add_argument('--cpu',             action='store_true')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()




# Step1: バウンディングボックスでアノテーション（ドラッグして囲む）
python annotate_bbox.py --images dataset/images --output dataset/annotations

# Step2: 学習（CPU向け・batch=1〜2推奨）
python faster_rcnn_count.py train \
  --images      dataset/images \
  --annotations dataset/annotations \
  --epochs      60 \
  --batch-size  1 \
  --lr          5e-3 \
  --cpu

# Step3: 推論
python faster_rcnn_count.py predict \
  入力画像.jpg \
  --weights checkpoints/best_model.pth \
  --score-thresh 0.5


# faster_rcnn_count.py
# Faster R-CNN による細長い物体のカウント
#
# 【CSRNetとの違い】
#   CSRNet  : 密度マップで「何個あるか」を推定（境界ボックスなし）
#   Faster R-CNN : 1本ずつ検出してカウント（位置・バウンディングボックスあり）
#
# 【データ構成例】
# dataset/
#   images/
#     img_001.jpg
#   annotations/
#     img_001.json  ← バウンディングボックスアノテーション（後述）
#
# 【アノテーションJSON フォーマット】
# {
#   "boxes": [
#     {"x1": 10, "y1": 20, "x2": 80, "y2": 35},  ← 左上(x1,y1) 右下(x2,y2)
#     ...
#   ]
# }

import os
import json
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ==================== モデル構築 ====================

def build_model(num_classes=2, pretrained=True):
    """
    Faster R-CNN (ResNet-50 + FPN) を構築する。

    num_classes: 背景(0) + 物体クラス数
                 物体が1種類なら num_classes=2
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # ヘッド部分を対象クラス数に差し替え
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ==================== データセット ====================

class RodDataset(Dataset):
    """
    バウンディングボックスアノテーション付きデータセット。
    細長い物体（棒・麺・製品など）向け。
    """

    def __init__(self, images_dir, annotations_dir, augment=True):
        self.images_dir      = images_dir
        self.annotations_dir = annotations_dir
        self.augment         = augment

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_paths = sorted(
            p for ext in exts
            for p in glob.glob(os.path.join(images_dir, ext))
        )
        if not self.image_paths:
            raise RuntimeError(f"画像が見つかりません: {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_annotation(self, img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(self.annotations_dir, stem + '.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"アノテーションが見つかりません: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        return data.get('boxes', [])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes_raw = self._load_annotation(img_path)

        # ボックス座標を float32 テンソルに変換
        boxes = []
        for b in boxes_raw:
            x1, y1 = float(b['x1']), float(b['y1'])
            x2, y2 = float(b['x2']), float(b['y2'])
            # 幅・高さが0以下のボックスをスキップ
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        # ---- データ拡張 ----
        if self.augment:
            # 水平反転
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
            # 垂直反転（俯瞰撮影では有効）
            if torch.rand(1) > 0.5:
                img = TF.vflip(img)
                boxes = [[b[0], h - b[3], b[2], h - b[1]] for b in boxes]
            # 輝度・コントラスト変動
            img = TF.adjust_brightness(img, 0.8 + torch.rand(1).item() * 0.4)
            img = TF.adjust_contrast(img,   0.8 + torch.rand(1).item() * 0.4)

        # ---- テンソル変換 ----
        img_tensor = TF.to_tensor(img)  # (C, H, W) in [0,1]

        if boxes:
            boxes_tensor  = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones(len(boxes), dtype=torch.int64)  # クラス1=物体
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        target = {
            'boxes' : boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    """DataLoader 用コレート関数（サイズが違う画像に対応）"""
    return tuple(zip(*batch))


# ==================== 学習ループ ====================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def train_one_epoch(model, loader, optimizer, device, print_freq=10):
    model.train()
    loss_meter = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # 勾配クリッピング（学習安定化）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(losses.item(), len(images))

        if (i + 1) % print_freq == 0:
            print(f"  step [{i+1}/{len(loader)}]  loss: {loss_meter.avg:.4f}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, score_thresh=0.5):
    """
    検証: 予測カウント数 vs GT カウント数 の MAE を返す
    """
    model.eval()
    mae_total = 0.0
    count = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, target in zip(preds, targets):
            # スコア閾値でフィルタ
            keep = pred['scores'] >= score_thresh
            pred_count = keep.sum().item()
            gt_count   = len(target['boxes'])
            mae_total += abs(pred_count - gt_count)
            count += 1

    return mae_total / max(count, 1)


def plot_history(train_losses, val_maes, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(val_maes, color='orange', label='Val MAE')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE')
    ax2.set_title('Validation MAE'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] 学習曲線: {save_path}")


# ==================== 学習メイン ====================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # デバイス
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[INFO] デバイス: {device}")

    # データセット
    full_ds = RodDataset(args.images, args.annotations, augment=True)
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    # モデル
    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)

    # オプティマイザ（バックボーンは小さいLR）
    params = [
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad], 'lr': args.lr},
    ]
    optimizer = optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_mae = float('inf')
    train_losses, val_maes = [], []
    start_epoch = 0

    # チェックポイント再開
    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_mae    = ckpt.get('best_mae', float('inf'))
        print(f"[INFO] 再開: epoch {start_epoch}, best MAE={best_mae:.2f}")

    print(f"\n{'='*50}")
    print(f"  Faster R-CNN 学習開始")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        mae  = validate(model, val_loader, device, score_thresh=args.score_thresh)
        scheduler.step()

        train_losses.append(loss)
        val_maes.append(mae)
        print(f"Epoch [{epoch+1:>3}/{args.epochs}]  Loss: {loss:.4f}  Val MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → ベストモデル更新 (MAE={best_mae:.2f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pth'))

    print(f"\n学習完了！  Best Val MAE: {best_mae:.2f}")
    plot_history(train_losses, val_maes,
                 os.path.join(args.output_dir, 'training_history.png'))


# ==================== 推論メイン ====================

def predict(args):
    # デバイス
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # モデル
    model = build_model(num_classes=2, pretrained=False).to(device)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"重みが見つかりません: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] 重みを読み込みました: {args.weights}")

    # 画像読み込み
    img_pil = Image.open(args.image).convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    keep   = output['scores'] >= args.score_thresh
    boxes  = output['boxes'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()
    count  = len(boxes)

    print(f"\n{'='*40}")
    print(f"  検出数（カウント）: {count}")
    print(f"{'='*40}\n")

    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_pil)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f'{score:.2f}',
                color='red', fontsize=7, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))

    ax.set_title(f'Faster R-CNN  検出数: {count}  (threshold={args.score_thresh})',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 結果を保存: {args.output}")
    plt.show()


# ==================== エントリポイント ====================

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN 物体カウント')
    sub = parser.add_subparsers(dest='mode', required=True)

    # --- train ---
    t = sub.add_parser('train', help='学習')
    t.add_argument('--images',          required=True)
    t.add_argument('--annotations',     required=True)
    t.add_argument('--output-dir',      default='checkpoints')
    t.add_argument('--epochs',          type=int,   default=60)
    t.add_argument('--batch-size',      type=int,   default=2,
                   help='CPU では 1〜2 推奨')
    t.add_argument('--lr',              type=float, default=5e-3)
    t.add_argument('--val-ratio',       type=float, default=0.1)
    t.add_argument('--score-thresh',    type=float, default=0.5)
    t.add_argument('--no-pretrained',   action='store_true',
                   help='COCO 事前学習重みを使わない')
    t.add_argument('--pretrained-ckpt', type=str, default=None,
                   help='学習再開用チェックポイント')
    t.add_argument('--cpu',             action='store_true')

    # --- predict ---
    p = sub.add_parser('predict', help='推論')
    p.add_argument('image',             help='入力画像パス')
    p.add_argument('--weights',         required=True, help='学習済み重みファイル (.pth)')
    p.add_argument('--score-thresh',    type=float, default=0.5,
                   help='検出スコア閾値（上げると厳しく、下げると甘く）')
    p.add_argument('--output',          default='result.png')
    p.add_argument('--cpu',             action='store_true')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()
    
    
# annotate_bbox.py
# バウンディングボックスアノテーションツール（Faster R-CNN用）
#
# 使い方:
#   python annotate_bbox.py --images dataset/images --output dataset/annotations
#
# 操作:
#   ドラッグ    : ボックスを描く
#   右クリック  : 最後のボックスを削除
#   S キー      : 保存して次の画像へ
#   Q キー      : 終了

import os
import sys
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button


def annotate_images(images_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(images_dir, ext))
    )
    if not image_paths:
        print(f"画像が見つかりません: {images_dir}")
        sys.exit(1)

    print(f"画像数: {len(image_paths)}")
    print("操作: ドラッグ=ボックス描画 / 右クリック=最後を削除 / S=保存して次へ / Q=終了")

    idx = 0
    while idx < len(image_paths):
        img_path = image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(output_dir, stem + '.json')

        # 既存アノテーション読み込み
        if os.path.isfile(json_path):
            with open(json_path) as f:
                data = json.load(f)
            boxes = data.get('boxes', [])
            print(f"[{idx+1}/{len(image_paths)}] {stem} — 既存 {len(boxes)} ボックス")
        else:
            boxes = []
            print(f"[{idx+1}/{len(image_paths)}] {stem} — 新規")

        from PIL import Image as PILImage
        img = np.array(PILImage.open(img_path).convert('RGB'))

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.canvas.manager.set_window_title(f"{stem}  ({idx+1}/{len(image_paths)})")
        ax.imshow(img)

        # 既存ボックスを描画
        rect_patches = []
        for b in boxes:
            r = mpatches.Rectangle(
                (b['x1'], b['y1']), b['x2'] - b['x1'], b['y2'] - b['y1'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(r)
            rect_patches.append(r)

        state = {'x0': None, 'y0': None, 'cur_rect': None}
        saved = {'done': False, 'quit': False}

        def update_title():
            ax.set_title(
                f"ボックス数: {len(boxes)}  |  ドラッグ=描画  右クリック=削除  S=保存  Q=終了",
                fontsize=11
            )
            fig.canvas.draw_idle()

        update_title()

        def on_press(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                state['x0'] = event.xdata
                state['y0'] = event.ydata
                r = mpatches.Rectangle(
                    (event.xdata, event.ydata), 0, 0,
                    linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
                )
                ax.add_patch(r)
                state['cur_rect'] = r
                fig.canvas.draw_idle()
            elif event.button == 3:
                if boxes:
                    boxes.pop()
                    if rect_patches:
                        rect_patches.pop().remove()
                    update_title()

        def on_motion(event):
            if event.inaxes != ax or state['x0'] is None:
                return
            r = state['cur_rect']
            if r is None:
                return
            x0, y0 = state['x0'], state['y0']
            x1, y1 = event.xdata, event.ydata
            r.set_xy((min(x0, x1), min(y0, y1)))
            r.set_width(abs(x1 - x0))
            r.set_height(abs(y1 - y0))
            fig.canvas.draw_idle()

        def on_release(event):
            if event.button != 1 or state['x0'] is None:
                return
            r = state['cur_rect']
            if r is None:
                return
            x0, y0 = state['x0'], state['y0']
            x1, y1 = event.xdata, event.ydata
            if abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
                box = {
                    'x1': float(min(x0, x1)),
                    'y1': float(min(y0, y1)),
                    'x2': float(max(x0, x1)),
                    'y2': float(max(y0, y1)),
                }
                boxes.append(box)
                # 確定ボックスを赤で再描画
                r.set_edgecolor('red')
                r.set_linestyle('-')
                rect_patches.append(r)
            else:
                r.remove()
            state['x0'] = state['y0'] = state['cur_rect'] = None
            update_title()

        def on_key(event):
            if event.key == 's':
                with open(json_path, 'w') as f:
                    json.dump({
                        'image': os.path.basename(img_path),
                        'boxes': boxes
                    }, f, indent=2)
                print(f"  保存: {json_path}  ({len(boxes)} ボックス)")
                saved['done'] = True
                plt.close(fig)
            elif event.key == 'q':
                saved['quit'] = True
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event',   on_press)
        fig.canvas.mpl_connect('motion_notify_event',  on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('key_press_event',      on_key)

        plt.tight_layout()
        plt.show()

        if saved['quit']:
            print("終了します。")
            break
        idx += 1

    print("アノテーション完了！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='バウンディングボックスアノテーションツール')
    parser.add_argument('--images', required=True, help='画像ディレクトリ')
    parser.add_argument('--output', required=True, help='アノテーション保存先')
    args = parser.parse_args()
    annotate_images(args.images, args.output)


# csrnet_count.py
# PyTorchを使ったCSRNetによる物体カウント
# 基本的な使い方
#python csrnet_count.py crowd.jpg
# 学習済み重みを使う場合（推奨）
#python csrnet_count.py crowd.jpg --weights partA_pre.pth
# 結果を別ファイルに保存
#python csrnet_count.py crowd.jpg --weights partA_pre.pth --output result.png

#入力画像
#   ↓
#[フロントエンド] VGG-16 (conv1〜conv3) → 特徴マップ
 #  ↓
#[バックエンド]  Dilated Conv (dilation=2) → 密度マップ
 #  ↓
#カウント数 = 密度マップの全ピクセルの合計値

# csrnet_count.py
#https://github.com/leeyeehoo/CSRNet-pytorch
## partB をダウンロードして実行
#python csrnet_count.py 入力画像.jpg --weights partB_pre.pth




# PyTorchを使ったCSRNetによる物体カウント

# csrnet_count.py
# PyTorchを使ったCSRNetによる物体カウント

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os


# ==================== モデル定義 ====================

class CSRNet(nn.Module):
    """
    CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
    論文: https://arxiv.org/abs/1802.10062

    構造:
      - フロントエンド: VGG-16の最初の10層（特徴抽出）
      - バックエンド: 拡張畳み込み（Dilated Conv）によるデンスマップ推定
    """

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        # フロントエンド: VGG-16 の conv1_1 〜 pool3 まで
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat)
        self.backend  = self._make_layers(self.backend_feat, in_channels=512, dilation=True)

        # 最終出力: 密度マップ（1チャンネル）
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            self._load_vgg_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _load_vgg_weights(self):
        """VGG-16 の事前学習済み重みをフロントエンドに転送"""
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_features = list(vgg16.features.children())

        frontend_children = list(self.frontend.children())

        vgg_idx = 0
        for layer in frontend_children:
            if isinstance(layer, nn.Conv2d):
                while vgg_idx < len(vgg_features) and not isinstance(vgg_features[vgg_idx], nn.Conv2d):
                    vgg_idx += 1
                if vgg_idx < len(vgg_features):
                    layer.weight.data = vgg_features[vgg_idx].weight.data
                    layer.bias.data   = vgg_features[vgg_idx].bias.data
                    vgg_idx += 1
        print("[INFO] VGG-16 事前学習済み重みをフロントエンドに読み込みました。")


# ==================== 前処理 ====================

def get_transform():
    """ImageNetの統計量で正規化"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image_path: str):
    """
    画像を読み込み、テンソルに変換する。
    CSRNetは任意サイズの入力を受け付けるが、
    メモリ節約のため長辺を1024pxにリサイズする。
    """
    img = Image.open(image_path).convert('RGB')

    # 長辺を 1024px に制限（任意）
    max_size = 1024
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    transform = get_transform()
    tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
    return tensor, img


# ==================== 推論 ====================

def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    密度マップを推論し numpy 配列で返す。
    カウント数 = 密度マップの総和
    """
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        density_map = model(tensor)

    # (1, 1, H, W) → (H, W)
    density_map = density_map.squeeze().cpu().numpy()
    density_map = np.maximum(density_map, 0)  # 負値をゼロクリップ
    return density_map


# ==================== 可視化 ====================

def visualize(original_img: Image.Image,
              density_map: np.ndarray,
              count: float,
              save_path: str = None):
    """元画像と密度マップを並べて表示・保存"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 元画像 ---
    axes[0].imshow(original_img)
    axes[0].set_title("入力画像", fontsize=14)
    axes[0].axis('off')

    # --- 密度マップ ---
    im = axes[1].imshow(density_map, cmap=cm.jet, interpolation='bilinear')
    axes[1].set_title(f"密度マップ  (推定カウント: {count:.1f})", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("CSRNet — 物体カウント", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] 結果を保存しました: {save_path}")

    plt.show()


# ==================== メイン ====================

def main():
    parser = argparse.ArgumentParser(
        description="CSRNet で画像中の物体数をカウントします"
    )
    parser.add_argument('image', type=str,
                        help='入力画像のパス (JPG / PNG など)')
    parser.add_argument('--weights', type=str, default=None,
                        help='学習済み CSRNet 重みファイル (.pth/.pt)')
    parser.add_argument('--no-vgg-init', action='store_true',
                        help='VGG-16 事前学習重みを使わない')
    parser.add_argument('--output', type=str, default='output.png',
                        help='結果画像の保存先 (default: output.png)')
    parser.add_argument('--cpu', action='store_true',
                        help='CPU を強制使用')
    args = parser.parse_args()

    # --- デバイス設定 ---
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[INFO] 使用デバイス: {device}")

    # --- モデル構築 ---
    load_vgg = not args.no_vgg_init
    model = CSRNet(load_weights=load_vgg).to(device)

    # CSRNet 学習済み重みがあれば読み込む
    if args.weights:
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"重みファイルが見つかりません: {args.weights}")
        state = torch.load(args.weights, map_location=device)
        # checkpoint が 'state_dict' キーを持つ場合に対応
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        print(f"[INFO] 重みを読み込みました: {args.weights}")
    else:
        print("[WARN] CSRNet 学習済み重みが指定されていません。")
        print("       VGG-16 初期化のみで推論します（精度は低くなります）。")
        print("       公式モデルは以下から入手できます:")
        print("       https://github.com/leeyeehoo/CSRNet-pytorch")

    # --- 前処理 ---
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {args.image}")

    tensor, original_img = preprocess_image(args.image)
    print(f"[INFO] 入力画像サイズ: {original_img.size[0]}x{original_img.size[1]}")

    # --- 推論 ---
    density_map = predict(model, tensor, device)
    count = float(density_map.sum())
    print(f"\n{'='*40}")
    print(f"  推定カウント数: {count:.1f}")
    print(f"{'='*40}\n")

    # --- 可視化 ---
    visualize(original_img, density_map, count, save_path=args.output)


if __name__ == '__main__':
    main()



import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os

# ==================== モデル定義 ====================

class CSRNet(nn.Module):
“””
CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
論文: https://arxiv.org/abs/1802.10062

```
構造:
  - フロントエンド: VGG-16の最初の10層（特徴抽出）
  - バックエンド: 拡張畳み込み（Dilated Conv）によるデンスマップ推定
"""

def __init__(self, load_weights=False):
    super(CSRNet, self).__init__()

    # フロントエンド: VGG-16 の conv1_1 〜 pool3 まで
    self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    self.backend_feat  = [512, 512, 512, 256, 128, 64]

    self.frontend = self._make_layers(self.frontend_feat)
    self.backend  = self._make_layers(self.backend_feat, in_channels=512, dilation=True)

    # 最終出力: 密度マップ（1チャンネル）
    self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    if load_weights:
        self._load_vgg_weights()

def forward(self, x):
    x = self.frontend(x)
    x = self.backend(x)
    x = self.output_layer(x)
    return x

def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _load_vgg_weights(self):
    """VGG-16 の事前学習済み重みをフロントエンドに転送"""
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg_features = list(vgg16.features.children())

    frontend_children = list(self.frontend.children())

    vgg_idx = 0
    for layer in frontend_children:
        if isinstance(layer, nn.Conv2d):
            while vgg_idx < len(vgg_features) and not isinstance(vgg_features[vgg_idx], nn.Conv2d):
                vgg_idx += 1
            if vgg_idx < len(vgg_features):
                layer.weight.data = vgg_features[vgg_idx].weight.data
                layer.bias.data   = vgg_features[vgg_idx].bias.data
                vgg_idx += 1
    print("[INFO] VGG-16 事前学習済み重みをフロントエンドに読み込みました。")
```

# ==================== 前処理 ====================

def get_transform():
“”“ImageNetの統計量で正規化”””
return transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std =[0.229, 0.224, 0.225]
)
])

def preprocess_image(image_path: str):
“””
画像を読み込み、テンソルに変換する。
CSRNetは任意サイズの入力を受け付けるが、
メモリ節約のため長辺を1024pxにリサイズする。
“””
img = Image.open(image_path).convert(‘RGB’)

```
# 長辺を 1024px に制限（任意）
max_size = 1024
w, h = img.size
if max(w, h) > max_size:
    scale = max_size / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

transform = get_transform()
tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
return tensor, img
```

# ==================== 推論 ====================

def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
“””
密度マップを推論し numpy 配列で返す。
カウント数 = 密度マップの総和
“””
model.eval()
with torch.no_grad():
tensor = tensor.to(device)
density_map = model(tensor)

```
# (1, 1, H, W) → (H, W)
density_map = density_map.squeeze().cpu().numpy()
density_map = np.maximum(density_map, 0)  # 負値をゼロクリップ
return density_map
```

# ==================== 可視化 ====================

def visualize(original_img: Image.Image,
density_map: np.ndarray,
count: float,
save_path: str = None):
“”“元画像と密度マップを並べて表示・保存”””
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

```
# --- 元画像 ---
axes[0].imshow(original_img)
axes[0].set_title("入力画像", fontsize=14)
axes[0].axis('off')

# --- 密度マップ ---
im = axes[1].imshow(density_map, cmap=cm.jet, interpolation='bilinear')
axes[1].set_title(f"密度マップ  (推定カウント: {count:.1f})", fontsize=14)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.suptitle("CSRNet — 物体カウント", fontsize=16, fontweight='bold')
plt.tight_layout()

if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] 結果を保存しました: {save_path}")

plt.show()
```

# ==================== メイン ====================

def main():
parser = argparse.ArgumentParser(
description=“CSRNet で画像中の物体数をカウントします”
)
parser.add_argument(‘image’, type=str,
help=‘入力画像のパス (JPG / PNG など)’)
parser.add_argument(’–weights’, type=str, default=None,
help=‘学習済み CSRNet 重みファイル (.pth/.pt)’)
parser.add_argument(’–no-vgg-init’, action=‘store_true’,
help=‘VGG-16 事前学習重みを使わない’)
parser.add_argument(’–output’, type=str, default=‘output.png’,
help=‘結果画像の保存先 (default: output.png)’)
parser.add_argument(’–cpu’, action=‘store_true’,
help=‘CPU を強制使用’)
args = parser.parse_args()

```
# --- デバイス設定 ---
if args.cpu:
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"[INFO] 使用デバイス: {device}")

# --- モデル構築 ---
load_vgg = not args.no_vgg_init
model = CSRNet(load_weights=load_vgg).to(device)

# CSRNet 学習済み重みがあれば読み込む
if args.weights:
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"重みファイルが見つかりません: {args.weights}")
    state = torch.load(args.weights, map_location=device)
    # checkpoint が 'state_dict' キーを持つ場合に対応
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    print(f"[INFO] 重みを読み込みました: {args.weights}")
else:
    print("[WARN] CSRNet 学習済み重みが指定されていません。")
    print("       VGG-16 初期化のみで推論します（精度は低くなります）。")
    print("       公式モデルは以下から入手できます:")
    print("       https://github.com/leeyeehoo/CSRNet-pytorch")

# --- 前処理 ---
if not os.path.isfile(args.image):
    raise FileNotFoundError(f"画像ファイルが見つかりません: {args.image}")

tensor, original_img = preprocess_image(args.image)
print(f"[INFO] 入力画像サイズ: {original_img.size[0]}x{original_img.size[1]}")

# --- 推論 ---
density_map = predict(model, tensor, device)
count = float(density_map.sum())
print(f"\n{'='*40}")
print(f"  推定カウント数: {count:.1f}")
print(f"{'='*40}\n")

# --- 可視化 ---
visualize(original_img, density_map, count, save_path=args.output)
```

if **name** == ‘**main**’:
main()
