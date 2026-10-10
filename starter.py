"""《图像矩阵》作业：完成四个标有 TODO 的函数。
运行方式：python starter.py
依赖库：numpy、matplotlib
"""
import numpy as np
# 一张 8×8 的人脸图像，按行优先顺序存储为一维序列。
# 换行仅用于方便阅读，并不会增加数组的维度。
FLAT_PIXELS = [
    80, 80, 80, 80, 80, 80, 80, 80,
    80, 220, 220, 220, 220, 220, 220, 80,
    80, 220, 0, 220, 220, 0, 220, 80,
    80, 220, 220, 220, 220, 220, 220, 80,
    80, 220, 0, 220, 220, 0, 220, 80,
    80, 220, 220, 0, 0, 220, 220, 80,
    80, 220, 220, 220, 220, 220, 220, 80,
    80, 80, 80, 80, 80, 80, 80, 80,
]
def build_grayscale(flat_pixels, height, width):
    """返回一个形状为 (height, width)、数据类型为 uint8 的数组，按行优先顺序排列。
    假设输入均为 0–255 范围内的有效整数，元素总数为 height * width。
    示例输入：
        flat_pixels = [0, 50, 100, 150, 200, 255]
        height = 2
        width = 3
        result = build_grayscale(flat_pixels, height, width)
    预期输出：
        print(result)
        [[  0  50 100]
         [150 200 255]]
        result.shape 为 (2, 3)，result.dtype 为 np.uint8。
        返回的是 NumPy 数组，而不是 Python 嵌套列表。
    直观理解：每行放入 width 个数值，共组成 height 行。
    前三个数值组成第一行，后三个数值组成第二行。
    数值的先后顺序不变，只改变排列形状；0 表示黑色，255 表示白色。
    """
    # TODO A：将序列转换为数组，并重塑其形状。
    img = np.array(flat_pixels, dtype=np.uint8).reshape(height, width)
    return img

def edit_pixel(gray, row, col, value):
    """返回 gray 的副本，仅修改位于 (row, col) 的像素。
    假设坐标有效，且 value 位于 [0, 255] 范围内。
    示例输入：
        gray = np.array([[0, 50, 100],
                        [150, 200, 255]], dtype=np.uint8)
        result = edit_pixel(gray, row=0, col=1, value=255)
    预期输出：
        print(result)
        [[  0 255 100]
         [150 200 255]]
        print(gray)
        [[  0  50 100]
         [150 200 255]]
        result.shape 为 (2, 3)，result.dtype 为 np.uint8。
    直观理解：行和列的索引都从 0 开始，(0, 1) 指第一行、第二列。
    只有副本中该位置的数值从 50 变为 255，原始 gray 保持不变。
    """
    # TODO B：先复制数组，再通过二维索引修改指定像素的值。
    img_copy = gray.copy()
    img_copy[row, col] = value
    return img_copy

def grayscale_to_rgb(gray):
    """返回一个新的 uint8 数组，形状为 (height, width, 3)，每个像素的 R、G、B 值相等。
    返回的数组不能与 gray 共享像素数据的存储空间。
    示例输入：
        gray = np.array([[0, 128],
                        [200, 255]], dtype=np.uint8)
        result = grayscale_to_rgb(gray)
    预期输出：
        print(result)
        [[[  0   0   0]
          [128 128 128]]
         [[200 200 200]
          [255 255 255]]]
        result.shape 为 (2, 2, 3)，result.dtype 为 np.uint8。
        result[0, 1] 为 [128, 128, 128]，依次对应 R、G、B。
    直观理解：每个灰度值变成三个相等的颜色通道值。
    行数和列数不变，新增的最后一个轴长度为 3；显示时仍是一张灰色图像。
    修改 result 中的像素值不应改变原始 gray。
    """
    # TODO C：将三份灰度值沿最后一个轴组合，构建三个颜色通道。
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb

def adjust_brightness(image, offset):
    """为每个像素值加上 offset，并将结果限制在 0–255 范围内。
    offset 为正数时变亮，为负数时变暗。
    返回新的 uint8 数组，保持原图像及数组形状不变。
    支持二维灰度图像和三维 RGB 图像；RGB 的每个通道都加上 offset。
    例子：
        image = np.array([[0, 100, 240]], dtype=np.uint8)
        result = adjust_brightness(image, offset=40)
        print(result)
        [[ 40 140 255]]
    解释：三个灰度像素分别加 40，得到 40、140、280。
    280 超过上限，因此变为 255；若结果小于 0，则变为 0。
    """
    # TODO D：先转换为有符号整数，再相加；限制数值范围后转换回原类型。
    temp = image.astype(np.int16) + offset
    temp = np.clip(temp, 0, 255)
    return temp.astype(np.uint8)

def check_functions():
    """已提供：独立检查四个函数，打印结果；全部通过时返回 True。
    请保留此检查代码。通过表示满足下列测试用例的要求。
    """
    def check_array(result, expected):
        if not isinstance(result, np.ndarray):
            raise AssertionError("应返回 NumPy 数组，请检查是否遗漏 return。")
        if result.shape != expected.shape:
            raise AssertionError(f"形状应为 {expected.shape}，实际为 {result.shape}。")
        if result.dtype != np.uint8:
            raise AssertionError(f"数据类型应为 uint8，实际为 {result.dtype}。")
        if not np.array_equal(result, expected):
            raise AssertionError(
                f"像素值不正确。\n    预期：{expected.tolist()}"
                f"\n    实际：{result.tolist()}"
            )
    def check_original(source, before, result):
        if not np.array_equal(source, before):
            raise AssertionError("原始输入被修改，应返回新的数组。")
        if np.shares_memory(source, result):
            raise AssertionError("输出与输入共享存储空间，应返回独立的数组。")
    def test_build():
        for height, width in ((2, 3), (3, 2), (1, 6)):
            for as_array in (False, True):
                source = [0, 50, 100, 150, 200, 255]
                if as_array:
                    source = np.array(source, dtype=np.uint8)
                before = np.array(source, copy=True)
                expected = np.array(source, dtype=np.uint8).reshape(height, width)
                result = build_grayscale(source, height, width)
                check_array(result, expected)
                if not np.array_equal(source, before):
                    raise AssertionError("构建图像不应改变输入序列。")
    def test_edit():
        for row, col, value in ((0, 1, 255), (1, 2, 0), (1, 0, 75)):
            source = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
            before = source.copy()
            expected = source.copy()
            expected[row, col] = value
            result = edit_pixel(source, row, col, value)
            check_array(result, expected)
            check_original(source, before, result)
    def test_rgb():
        for values in ([[0, 128], [200, 255]], [[10, 70, 240]]):
            source = np.array(values, dtype=np.uint8)
            before = source.copy()
            expected = np.empty(source.shape + (3,), dtype=np.uint8)
            for channel in range(3):
                expected[:, :, channel] = source
            result = grayscale_to_rgb(source)
            check_array(result, expected)
            check_original(source, before, result)
    def test_brightness():
        inputs = (
            np.array([[0, 20, 240], [100, 200, 255]], dtype=np.uint8),
            np.array([[[240, 20, 0], [100, 200, 255]]], dtype=np.uint8),
        )
        for image in inputs:
            for offset in (40, -60, 0, 255, -255):
                source = image.copy()
                before = source.copy()
                # 用逐元素计算生成预期值，检查两种图像及上下限。
                values = [min(255, max(0, int(value) + offset))
                          for value in before.flat]
                expected = np.array(values, dtype=np.uint8).reshape(before.shape)
                result = adjust_brightness(source, offset)
                check_array(result, expected)
                check_original(source, before, result)
    tests = [
        ("任务 A：build_grayscale", test_build),
        ("任务 B：edit_pixel", test_edit),
        ("任务 C：grayscale_to_rgb", test_rgb),
        ("任务 D：adjust_brightness", test_brightness),
    ]
    passed = 0
    print("\n========== 函数检查结果 ==========")
    for name, test in tests:
        try:
            test()
        except NotImplementedError:
            print(f"[未完成] {name}：请替换占位代码。")
        except AssertionError as error:
            print(f"[未通过] {name}：{error}")
        except Exception as error:
            print(f"[运行出错] {name}：{type(error).__name__}: {error}")
        else:
            passed += 1
            print(f"[通过] {name}")
    print(f"检查汇总：{passed}/4 个函数通过。")
    return passed == len(tests)

def show_image(ax, image, title):
    """已提供：直接显示像素，不进行平滑处理，也不自动缩放灰度显示范围。
    """
    if image.ndim == 2:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
    else:
        ax.imshow(image, interpolation="nearest")
    ax.set_title(f"{title}\nshape = {image.shape}", fontsize=10)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xticks(range(image.shape[1]))
    ax.set_yticks(range(image.shape[0]))

def main():
    """已提供：串联从一维数值序列到图像操作的完整流程。
    """
    if not check_functions():
        print("请修改未通过的函数后重新运行；全部通过后会显示图像对比。")
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("函数检查已通过。安装 matplotlib 后重新运行即可查看图像：")
        print("python -m pip install matplotlib")
        return
    flat = np.array(FLAT_PIXELS, dtype=np.uint8)
    print("1D flat sequence:", flat.shape)
    # A：将同一个数值序列排列为两种不同的空间布局。
    gray = build_grayscale(flat, height=8, width=8)
    rearranged = build_grayscale(flat, height=4, width=16)
    print("2D grayscale image:", gray.shape)
    print("Same values rearranged:", rearranged.shape)
    print("Pixel counts:", gray.size, rearranged.size)
    # B：查看一个像素和一整行，再修改图像副本。
    row, col = 2, 2
    width = gray.shape[1]
    flat_index = row * width + col
    print(f"flat[{flat_index}] = {flat[flat_index]}")
    print(f"gray[{row}, {col}] = {gray[row, col]}")
    print("One row:", gray[row, :], "shape:", gray[row, :].shape)
    edited = edit_pixel(gray, row, col, value=255)
    print("Original pixel after editing a copy:", gray[row, col])
    print("Edited pixel:", edited[row, col])
    # C：三个通道的值相等时呈现灰色；仅保留红色通道以生成红色图像。
    rgb = grayscale_to_rgb(gray)
    red_only = rgb.copy()
    red_only[:, :, 1] = 0  # 绿色通道。
    red_only[:, :, 2] = 0  # 蓝色通道。
    print("3D RGB image:", rgb.shape)
    print("One RGB pixel:", rgb[1, 1], "shape:", rgb[1, 1].shape)
    print("Red-only pixel:", red_only[1, 1])
    print("Red channel shape:", rgb[:, :, 0].shape)
    # D：对灰度数组和 RGB 数组进行算术运算。
    brighter = adjust_brightness(gray, offset=60)
    darker = adjust_brightness(gray, offset=-60)
    brighter_red = adjust_brightness(red_only, offset=40)
    print("Gray pixel 220 with +60:", brighter[1, 1])
    print("Gray pixel 0 with -60:", darker[2, 2])
    print("RGB pixel after adding 40 to ALL channels:", brighter_red[1, 1])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [
        (gray, "Original grayscale"),
        (rearranged, "Same 64 values, different arrangement"),
        (edited, "Edited pixel (2, 2)"),
        (rgb, "Equal R, G, B"),
        (red_only, "Red channel only"),
        (brighter, "Grayscale +60"),
        (darker, "Grayscale -60"),
        (brighter_red, "RGB +40 on every channel"),
    ]
    for ax, (image, title) in zip(axes.flat, panels):
        show_image(ax, image, title)
    fig.suptitle("Images are arrays: 1D → 2D → 3D → numerical operations")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as error:
        print(f"\n{error}\n请完成此函数，然后重新运行文件。")
