from substratools import metrics, Opener

import pytest


@pytest.fixture
def dummy_opener():
    # fake opener module using a class
    class FakeOpener(Opener):
        def get_X(self, folder):
            pass

        def get_y(self, folder):
            return 'y'

        def fake_X(self):
            pass

        def fake_y(self):
            pass

        def get_pred(self, path):
            return 'pred'

        def save_pred(self, pred, path):
            pass

    yield FakeOpener()


@pytest.fixture
def dummy_metrics_class(dummy_opener):
    class DummyMetrics(metrics.Metrics):
        OPENER = dummy_opener

        def score(self, y_true, y_pred):
            return y_true + y_pred

    return DummyMetrics


def test_create(dummy_metrics_class):
    dummy_metrics_class()


def test_score(dummy_metrics_class):
    m = dummy_metrics_class()
    wp = metrics.MetricsWrapper(m)
    s = wp.score()
    assert s == 'ypred'


def test_execute(dummy_metrics_class):
    s = metrics.execute(dummy_metrics_class)
    assert s == 'ypred'
