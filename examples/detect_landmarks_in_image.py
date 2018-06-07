import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
from mtcnn_detector import MtcnnDetector
import mxnet as mx
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='align faces')
    parser.add_argument('--src', help='source file')
    parser.add_argument('--detmodel', help='path to face detector model')
    parser.add_argument('--model', help='path to landmark model')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    src_file = args.src
    if not os.path.isfile(src_file):
        raise ValueError("{} not exist".format(src_file))

    img = cv2.imread(src_file)
    input = io.imread(src_file)

    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False,
                                      model_path=args.model)
    detector = MtcnnDetector(model_folder=args.detmodel, ctx=mx.gpu(0), num_worker=1, accurate_landmark=True)

    # run detector
    results = detector.detect_face(img)
    box = results[0][0]

    preds = fa.get_landmarks(input, box[1], box[3], box[0], box[2])[-1]

    # TODO: Make this nice
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

    ax.view_init(elev=135., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()


if __name__ == '__main__':
    main()
