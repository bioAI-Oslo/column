import numpy as np
import pytest

from ..src.loss import CE, energy, highest_value, pixel_wise_CE, softmax


def test_highest_value():
    img = np.zeros((26, 26, 3))
    img[0, 0, 0] = 1

    assert np.isclose(highest_value(img), 0.0, atol=0.0000000001)

    img_batch = np.array([img, img, img])

    assert np.all(highest_value(img_batch) == [0.0, 0.0, 0.0])

    img[0, 0, 2] = 2

    print(highest_value(img))
    assert np.isclose(highest_value(img), 2.0, atol=0.0000000001)

    img_batch = np.array([img, img, img])

    assert np.all(highest_value(img_batch) == [2.0, 2.0, 2.0])

    img1 = np.zeros((26, 26, 3))
    img1[0, 0, 0] = 1
    img2 = np.zeros((26, 26, 3))
    img2[0, 0, 1] = 1
    img3 = np.zeros((26, 26, 3))
    img3[0, 0, 2] = 1

    img_batch = np.array([img1, img2, img3])

    assert np.all(highest_value(img_batch) == [0.0, 1.0, 2.0])


def test_pixel_wise_CE():
    img = np.zeros((26, 26, 3))
    img[0, 0, 0] = 1
    img_batch = np.array([img, img, img])
    expected = np.array([1, 0, 0])
    expected_batch = np.array([expected, expected, expected])

    assert np.isclose(
        pixel_wise_CE(img_batch, None, expected_batch), pixel_wise_CE(img, None, expected), atol=0.0000000001
    )


def test_softmax():
    img = np.zeros((26, 26, 3))

    softmax_value = softmax(img)

    assert softmax_value.shape == (26, 26, 3)
    assert np.all(softmax_value[0, 0, 0] == softmax_value)
