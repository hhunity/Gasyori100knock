from PIL import Image
import os

def split_image(input_path, output_dir="output", tile_height=512):
    """
    画像を高さ512pxに分割する
    
    Args:
        input_path: 入力画像のパス
        output_dir: 出力ディレクトリ
        tile_height: 分割する高さ（デフォルト512px）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img = Image.open(input_path)
    width, height = img.size
    print(f"元画像サイズ: {width} x {height}")
    
    # 分割数を計算（端数も1枚として含む）
    num_tiles = (height + tile_height - 1) // tile_height
    print(f"分割数: {num_tiles} 枚")
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    for i in range(num_tiles):
        top = i * tile_height
        bottom = min(top + tile_height, height)
        
        # クロップして保存
        cropped = img.crop((0, top, width, bottom))
        output_path = os.path.join(output_dir, f"{base_name}_{i+1:03d}.png")
        cropped.save(output_path)
        print(f"  [{i+1}/{num_tiles}] 保存: {output_path} (y: {top} ~ {bottom})")
    
    print("完了！")


# 使い方
split_image("input.jpg")

# オプション指定
# split_image("input.jpg", output_dir="splits", tile_height=256)
