from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os

def split_image(
    input_path,
    output_dir="output",
    tile_w=512, tile_h=512,
    offset_x=0, offset_y=0  # 先頭から飛ばすピクセル数
):
    """
    画像をタイル分割（オフセットで開始位置をスキップ）

    Args:
        tile_w / tile_h    : 切り出すタイルサイズ
        offset_x / offset_y: 開始位置のスキップ幅
    """
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path)
    W, H = img.size
    print(f"元画像: {W} x {H}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    count = 0

    row = 0
    for y in range(offset_y, H, tile_h):
        col = 0
        for x in range(offset_x, W, tile_w):
            right  = min(x + tile_w, W)
            bottom = min(y + tile_h, H)

            tile = img.crop((x, y, right, bottom))
            fname = f"{base}_r{row:03d}_c{col:03d}.tif"
            tile.save(os.path.join(output_dir, fname), compression="tiff_lzw")

            count += 1
            col += 1
        row += 1

    print(f"完了: {count} 枚 ({row} 行 x {col} 列)")


# --- 使い方 ---

# オフセットなし（端から分割）
split_image("input.tif", tile_w=512, tile_h=512)

# 縦横100pxスキップしてから分割
split_image("input.tif", tile_w=512, tile_h=512, offset_x=100, offset_y=100)

# 縦横で別々にスキップ
split_image("input.tif", tile_w=512, tile_h=512, offset_x=256, offset_y=128)



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
