import os
import cv2
import numpy as np

if __name__ == '__main__':
    pred_trimap_dir = 'C:/Users/moi/Desktop/ptp1/tolerance_perfect_best-s-2000-lr-1e-6'
    gt_perfect_trimap_dir = 'D:/Dataset/Matting/AIM-500/perfect_trimap'
    gt_normal_trimap_dir = 'D:/Dataset/Matting/AIM-500/trimap'

    names = os.listdir(gt_perfect_trimap_dir)

    total_wrong_perfect_u_rate = 0
    total_wrong_perfect_k_rate = 0
    total_wrong_perfect_rate = 0

    total_wrong_normal_u_rate = 0
    total_wrong_normal_k_rate = 0
    total_wrong_normal_rate = 0

    for name in names:
        pred = cv2.imread(os.path.join(pred_trimap_dir, name), cv2.IMREAD_GRAYSCALE)
        gt_perfect = cv2.imread(os.path.join(gt_perfect_trimap_dir, name), cv2.IMREAD_GRAYSCALE)
        gt_perfect = cv2.resize(gt_perfect, (pred.shape[1], pred.shape[0]))
        gt_normal = cv2.imread(os.path.join(gt_normal_trimap_dir, name), cv2.IMREAD_GRAYSCALE)
        gt_normal = cv2.resize(gt_normal, (pred.shape[1], pred.shape[0]))

        wrong_u_pixels = np.sum(np.logical_or(gt_perfect[np.logical_and(pred > 0, pred < 255)] > 200, gt_perfect[np.logical_and(pred > 0, pred < 255)] < 50))
        wrong_k_pixels = np.sum(gt_perfect[pred == 0] >= 50) + np.sum(gt_perfect[pred == 255] <= 200)
        all_pixels = pred.shape[0] * pred.shape[1]

        wrong_perfect_u_rate = wrong_u_pixels / all_pixels
        wrong_perfect_k_rate = wrong_k_pixels / all_pixels
        wrong_perfect_rate = (wrong_u_pixels + wrong_k_pixels) / all_pixels

        wrong_normal_u_pixels = np.sum(np.logical_or(gt_normal[np.logical_and(pred > 0, pred < 255)] > 200, gt_normal[np.logical_and(pred > 0, pred < 255)] < 50))
        wrong_normal_k_pixels = np.sum(gt_normal[pred == 0] >= 50) + np.sum(gt_normal[pred == 255] <= 200)
        wrong_normal_rate = (wrong_normal_u_pixels + wrong_normal_k_pixels) / all_pixels

        total_wrong_perfect_k_rate += wrong_perfect_k_rate
        total_wrong_perfect_u_rate += wrong_perfect_u_rate
        total_wrong_normal_k_rate += wrong_normal_k_pixels / all_pixels
        total_wrong_normal_u_rate += wrong_normal_u_pixels / all_pixels
        total_wrong_normal_rate += wrong_normal_rate
        total_wrong_perfect_rate += wrong_perfect_rate


    avg_wrong_perfect_u_rate = total_wrong_perfect_u_rate / len(names)
    avg_worng_perfect_k_rate = total_wrong_perfect_k_rate / len(names)
    avg_wrong_normal_u_rate = total_wrong_normal_u_rate / len(names)
    avg_worng_normal_k_rate = total_wrong_normal_k_rate / len(names)
    avg_wrong_perfect_rate = total_wrong_perfect_rate / len(names)
    avg_wrong_normal_rate = total_wrong_normal_rate / len(names)

    print(f'perfect:\n'
          f'u: {avg_wrong_perfect_u_rate * 100:2.5f}% \n'
          f'k: {avg_worng_perfect_k_rate * 100:2.5f}% \n'
          f'all: {avg_wrong_perfect_rate * 100:2.5f}% \n'
          f'normal:\n'
          f'u: {avg_wrong_normal_u_rate * 100:2.5f}% \n'
          f'k: {avg_worng_normal_k_rate * 100:2.5f}% \n'
          f'all: {avg_wrong_normal_rate * 100:2.5f}% ')