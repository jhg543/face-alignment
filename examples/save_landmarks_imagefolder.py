# coding: utf-8
import argparse
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
from skimage import io

import face_alignment


def get_args():
    parser = argparse.ArgumentParser(description='align faces')
    parser.add_argument('--src', default='C:/dev/3dface/data/celebbigside', help='source directory')
    parser.add_argument('--padding', default=0.37, type=float, help='padding')
    parser.add_argument('--split', default=0, type=float, help='split ratio')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    src_dir = args.src
    if not os.path.exists(src_dir):
        raise ValueError("src dir not exist {}".format(src_dir))

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=torch.cuda.is_available(),
                                      flip_input=True)

    file_count = 0
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.gif', '.png')) and not filename.lower().startswith("landmark"):
                absfile = os.path.join(root, filename)
                print(absfile)
                input = io.imread(absfile)
                plist = fa.get_landmarks(input)
                if plist is not None:
                    preds = fa.get_landmarks(input)[-1]

                    fig = plt.figure(figsize=plt.figaspect(.5))
                    ax = fig.add_subplot(1, 2, 1)
                    ax.imshow(input)
                    ax.plot(preds[0:17, 0], preds[0:17, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[17:22, 0], preds[17:22, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[22:27, 0], preds[22:27, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[27:31, 0], preds[27:31, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[31:36, 0], preds[31:36, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[36:42, 0], preds[36:42, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[42:48, 0], preds[42:48, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[48:60, 0], preds[48:60, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.plot(preds[60:68, 0], preds[60:68, 1], marker='o', markersize=6, linestyle='-', color='w', lw=2)
                    ax.axis('off')

                    ax = fig.add_subplot(1, 2, 2, projection='3d')
                    surf = ax.scatter(preds[:, 0] * 1.00, preds[:, 1], preds[:, 2], c="cyan", alpha=1.0, edgecolor='b')
                    ax.plot3D(preds[:17, 0] * 1.00, preds[:17, 1], preds[:17, 2], color='blue')
                    ax.plot3D(preds[17:22, 0] * 1.00, preds[17:22, 1], preds[17:22, 2], color='blue')
                    ax.plot3D(preds[22:27, 0] * 1.00, preds[22:27, 1], preds[22:27, 2], color='blue')
                    ax.plot3D(preds[27:31, 0] * 1.00, preds[27:31, 1], preds[27:31, 2], color='blue')
                    ax.plot3D(preds[31:36, 0] * 1.00, preds[31:36, 1], preds[31:36, 2], color='blue')
                    ax.plot3D(preds[36:42, 0] * 1.00, preds[36:42, 1], preds[36:42, 2], color='blue')
                    ax.plot3D(preds[42:48, 0] * 1.00, preds[42:48, 1], preds[42:48, 2], color='blue')
                    ax.plot3D(preds[48:, 0] * 1.00, preds[48:, 1], preds[48:, 2], color='blue')

                    ax.view_init(elev=90., azim=90.)
                    ax.set_xlim(ax.get_xlim()[::-1])
                    plt.savefig(os.path.join(root, "landmark" + filename))
                    plt.close(fig)
                    file_count = file_count + 1
                    if file_count % 100 == 0:
                        print(file_count)

                else:
                    print("-------"+absfile)


if __name__ == '__main__':
    main()
