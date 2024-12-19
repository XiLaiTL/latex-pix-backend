from PIL import Image, ImageOps
import numpy as np

def convert_image(image_path)->Image:
    image = open_image(image_path)
    image = crop_white_borders(image)
    image = resize_and_crop_to_fit(image,(500,400))
    return image

def open_image(image_path)->Image:
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert('RGB')
    return image


def crop_white_borders(image:Image, threshold=250)->Image:
    # 将图像转换为NumPy数组以便更容易处理
    np_image = np.array(image)

    # 识别非白色像素的边界
    rows = np.any(np_image < threshold, axis=2)
    cols = np.any(np_image < threshold, axis=0)

    # 计算裁剪区域
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 裁剪图像
    cropped_image = image.crop((cmin, rmin, cmax + 1, rmax + 1))

    return cropped_image


def resize_and_crop_to_fit(original_image:Image, target_size:(int,int), bg_color=(255, 255, 255))->Image:
    # 打开原始图片
    target_width, target_height = target_size
    original_width, original_height = original_image.size

    # 计算缩放比例，保持宽高比
    aspect_ratio = original_width / original_height
    if target_width / target_height > aspect_ratio:
        # 目标宽度相对于高度太宽，按高度缩放并裁剪宽度
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
        left = (target_width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = target_height
    else:
        # 目标高度相对于宽度太高，按宽度缩放并裁剪高度
        new_height = int(target_width / aspect_ratio)
        new_width = target_width
        left = 0
        top = (target_height - new_height) // 2
        right = target_width
        bottom = top + new_height

    # 缩放图片
    if new_height<original_height:
        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    else:
        resized_image = original_image.resize((new_width, new_height), Image.BICUBIC)

    # 创建一个新画布，并设置背景色
    new_image = Image.new('RGB', (target_width, target_height), bg_color)

    # 将缩放后的图片粘贴到新画布上，居中显示
    new_image.paste(resized_image, (left, top))

    return new_image

if __name__ == '__main__':
    image1 = convert_image("../img.png")
    image1.show()
    image1.save('img_.png')