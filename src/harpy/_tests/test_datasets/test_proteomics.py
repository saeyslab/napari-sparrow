import pytest
from spatialdata import SpatialData

from harpy.datasets.proteomics import macsima_example, macsima_tonsil, mibi_example


# @pytest.mark.skip(reason="This test downloads a full experiment to the OS cache.")
def test_macsima_tonsil():
    sdata = macsima_tonsil()
    assert isinstance(sdata, SpatialData)


@pytest.mark.skip(reason="This test downloads a full experiment to the OS cache.")
def test_mibi_example():
    sdata = mibi_example()
    assert isinstance(sdata, SpatialData)
    assert len([*sdata.images]) != 0


@pytest.mark.skip(reason="This test downloads a full experiment to the OS cache.")
def test_macsima_example():
    sdata = macsima_example()
    assert isinstance(sdata, SpatialData)
    assert len([*sdata.images]) != 0
