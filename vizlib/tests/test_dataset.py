from vizlib.data.helpers import DataSet
import numpy as np


class TestDataSet(object):

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
