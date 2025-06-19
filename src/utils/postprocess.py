import cv2
import numpy as np


def post_process(pred_mask, threshold=0.5, min_area=50):
    """
    后处理步骤:
    1. 阈值化
    2. 去除小连通区域
    3. 填充孔洞
    """
    # 阈值化
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 去除小区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0

    # 填充孔洞
    contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(cleaned, [cnt], 0, 255, -1)

    return cleaned