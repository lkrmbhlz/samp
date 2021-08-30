import pytest
import numpy as np

from samp.samp import pca_projections_2d, varimax_projections_2d, asymmetries_x_axis, asymmetries_y_axis, \
    min_max_asymmetries, get_min_max_varimax


def test_samp_shapes():

    pointcloud_1 = np.random.randint(10, size=(50, 3))
    pointcloud_2 = np.random.randint(10, size=(50, 3))
    pointcloud_3 = np.random.randint(10, size=(50, 3))
    pointcloud_list = [pointcloud_1, pointcloud_2, pointcloud_3]

    for pointcloud in pointcloud_list:
        assert pointcloud.shape == (50, 3)

    pca_2d_projections = pca_projections_2d(pointcloud_list)

    for projection in pca_2d_projections:
        assert projection.shape == (50, 2)

    varimax_2d_projections = varimax_projections_2d(pointcloud_list)

    for projection in varimax_2d_projections:
        assert projection.shape == (50, 2)

    asymmetries_x = asymmetries_x_axis(varimax_2d_projections)
    assert len(asymmetries_x) == 3

    asymmetries_y = asymmetries_y_axis(varimax_2d_projections)
    assert len(asymmetries_y) == 3

    min_max_asymmetry = min_max_asymmetries(asymmetries_x, asymmetries_y)
    assert len(min_max_asymmetry) == 3

    min_max_asymmetry_pipeline_1st_2nd = get_min_max_varimax(pointcloud_list)
    assert min_max_asymmetry_pipeline_1st_2nd == min_max_asymmetry

    min_max_asymmetry_pipeline_1st_3rd = get_min_max_varimax(pointcloud_list, get_1st_and_3rd_component=True)
    assert len(min_max_asymmetry_pipeline_1st_3rd) == 3

test_samp_shapes()