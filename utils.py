import logging
import os

import torchvision


def save_images(out_dir, names, pred_trimaps_softmax, pred_mattes_u, gt_trimap_3, logger=logging.getLogger('utils')):
    """Save a batch of images."""
    matte_path = os.path.join(out_dir, 'matte')
    matte_u_path = os.path.join(out_dir, 'matte_u')
    trimap_path = os.path.join(out_dir, 'trimap')

    os.makedirs(matte_path, exist_ok=True)
    os.makedirs(matte_u_path, exist_ok=True)
    os.makedirs(trimap_path, exist_ok=True)

    # logger.debug(f'Saving {len(names)} images to {out_dir}')

    for idx, name in enumerate(names):
        if pred_mattes_u is not None:
            matte_u = pred_mattes_u[idx]
            save_path = os.path.join(matte_u_path, name)
            torchvision.utils.save_image(matte_u, save_path)

        if pred_trimaps_softmax is not None:
            trimap = pred_trimaps_softmax[idx]
            trimap = trimap.argmax(dim=0)
            trimap = trimap / 2.
            save_path = os.path.join(trimap_path, name)
            torchvision.utils.save_image(trimap, save_path)

        if pred_mattes_u is not None:
            if pred_trimaps_softmax is None:
                trimap = gt_trimap_3[idx].argmax(dim=0)
                trimap = trimap / 2.

            matte = matte_u
            matte[(trimap == 1.).unsqueeze(0)] = 1.
            matte[(trimap == 0.).unsqueeze(0)] = 0.

            save_path = os.path.join(matte_path, name)
            torchvision.utils.save_image(matte, save_path)
