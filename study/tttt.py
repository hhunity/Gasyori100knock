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
