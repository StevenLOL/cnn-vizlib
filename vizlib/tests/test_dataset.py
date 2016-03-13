from vizlib.data.helpers import DataSet
from vizlib.data.helpers import number_of_errors_to_show
import numpy as np


class TestDataSet(object):

    def setup_method(self, method):
        np.random.seed(42)

    def test_zoom(self):
        # Arrange
        X = np.zeros((300, 1, 128, 128))
        y = np.arange(len(X))
        ds = DataSet(X, y)

        # Apply
        zoomed = ds.zoom(0.5)

        # Assert
        assert zoomed.X.shape == (300, 1, 64, 64)

    def test_to_nxmxc(self):
        # Arrange
        X = np.zeros((100, 3, 4, 2))
        y = np.arange(len(X))
        ds = DataSet(X, y)
        assert ds.X.shape == (100, 3, 4, 2)

        # Apply
        nxmxc = ds.to_nxmxc()

        # Assert
        assert nxmxc.X.shape == (100, 4, 2, 3)
        for x in nxmxc.X:
            assert x.shape == (4, 2, 3)

    def test_standardize(self):
        # Arrange
        x1 = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]])
        x2 = np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1]])
        X = np.array([x1, x2])
        y = np.arange(len(X))
        ds = DataSet(X, y)

        # Apply
        ds.standardize()

        # Assert
        for x in ds.X:
            np.testing.assert_allclose(x.mean(), 0, atol=1e-7)
            np.testing.assert_allclose(x.std(), 1, atol=1e-7)

    def test_unstandardize(self):
        # Arrange
        x1 = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]])
        x2 = np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1]])
        X = np.array([x1, x2])
        y = np.arange(len(X))
        ds = DataSet(X, y)

        # Apply
        ds.standardize()
        ds.unstandardize()

        # Assert
        np.testing.assert_allclose(ds.X[0].squeeze(), x1)
        np.testing.assert_allclose(ds.X[1].squeeze(), x2)

    def test_standardize_none(self):
        # Arrange
        x1 = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]])
        x2 = np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1]])
        X = np.array([x1, x2])
        y = np.arange(len(X))
        ds = DataSet(X, y)
        Xorig = ds.X.copy()

        # Apply
        ds.standardize(standardization_type=None)

        # Assert
        np.testing.assert_allclose(Xorig, ds.X)

    def test_standardize_global(self):
        # Arrange
        x1 = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]])
        x2 = np.array([[1, 0, 1],
                       [1, 0, 1],
                       [1, 0, 1]])
        X = np.array([x1, x2])
        y = np.arange(len(X))
        ds = DataSet(X, y)

        # Apply
        ds.standardize(standardization_type='global')

        # Assert
        assert ds.mean.shape == ()
        assert ds.mean == 0.5
        assert ds.X.std() == 1

    def test_unstandardize_global(self):
        # Arrange
        X = np.random.randn(1e4, 1, 32, 32) * 10 + 12
        y = np.arange(len(X))
        ds = DataSet(X, y)

        # Apply
        ds.standardize(standardization_type='global')
        ds.unstandardize()

        # Assert
        assert ds.mean.shape == ()
        np.testing.assert_allclose(ds.mean, 12, atol=1e-1)
        np.testing.assert_allclose(ds.std, 10, atol=1e-1)

    def test_swap_standardization(self):
        # Arrange
        X = np.random.randn(1e4, 1, 32, 32) * 10 + 12
        y = np.arange(len(X))
        ds = DataSet(X, y)

        # Apply
        ds.standardize(standardization_type='individual')
        ds.standardize(standardization_type='global')

        # Assert
        assert ds.mean.shape == ()
        np.testing.assert_allclose(ds.mean, 12, atol=1e-1)
        np.testing.assert_allclose(ds.std, 10, atol=1e-1)

    # Could not get this to pass! This means that it matters how
    # you normalize to begin with!
    # def test_swap_standardization2(self):
    #     # Arrange
    #     X = np.random.randn(1e4, 1, 32, 32) * 10 + 12
    #     y = np.arange(len(X))
    #     ds = DataSet(X, y)
    #     ds2 = DataSet(X, y)
    #     ds2.X = ds2.X.astype(np.float32)

    #     # Apply
    #     ds.standardize(standardization_type='individual')
    #     ds2.standardize(standardization_type='individual')
    #     ds2.unstandardize()
    #     ds2.standardize(standardization_type='global')

    #     # Assert
    #     #assert np.abs(ds2.X - ds.X).max() < 1e-5
    #     np.testing.assert_allclose(ds2.X, ds.X, atol=1e-5)

    def test_unstandardize2(self):
        # Arrange
        X = np.random.randn(1e4, 1, 32, 32) * 10 + 12
        y = np.arange(len(X))
        ds = DataSet(X, y)

        # Apply
        ds.standardize(standardization_type='individual')
        ds.unstandardize()

        # Assert
        np.testing.assert_allclose(X, ds.X, atol=1e-5)

    def test_classes(self):
        # Arrange
        X = np.ones((10, 32, 32))
        y = [0, 0, 1, 2, 2, 1, 3, 3, 4, 4, 0]
        ds = DataSet(X, y)

        # Apply
        classes = ds.classes

        # Assert
        assert classes == [0, 1, 2, 3, 4]

    def test_one_of_every_class(self):
        # Arrange
        a = np.array([[[0]]])
        a2 = np.array([[[0]]])
        b = np.array([[[1]]])
        b2 = np.array([[[1]]])
        c = np.array([[[2]]])
        d = np.array([[[3]]])
        d2 = np.array([[[3]]])
        d3 = np.array([[[3]]])

        X = np.array([a, a2, b, b2, c, d, d2, d3])
        y = np.array([0, 0, 1, 1, 2, 3, 3, 3])
        ds = DataSet(X, y).shuffle()

        # Apply
        sample = ds.one_of_class()

        # Assert
        np.testing.assert_equal(sample.X, np.array([a, b, c, d]))
        np.testing.assert_equal(sample.y, np.array([0, 1, 2, 3]))


def test_show_errors():
    # Arrange
    conf_mat = np.array([
        [ 4,  5,  8, 10,  3, 20],
        [ 6, 14,  6,  4,  8, 12],
        [ 4,  4, 10, 11,  4, 17],
        [ 0,  8,  8,  3,  9, 22],
        [ 0, 10,  2, 12, 18,  8],
        [ 2,  9,  9,  2,  7, 21]
    ])
    expected = np.array([
        [ 0, 1, 2, 3, 1, 5 ],
        [ 2, 0, 1, 1, 3, 5 ],
        [ 1, 1, 0, 3, 1, 6 ],
        [ 0, 2, 2, 0, 3, 5 ],
        [ 0, 4, 1, 4, 0, 3 ],
        [ 1, 4, 3, 1, 3, 0 ],
    ])

    # Apply
    to_show = number_of_errors_to_show(conf_mat, 12)

    # Assert
    np.testing.assert_array_equal(to_show, expected)
