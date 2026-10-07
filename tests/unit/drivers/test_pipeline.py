"""nonlinear_resistive_calculation argument forwarding (analyse_with_mre mocked; no codes run)."""
import pytest

import tearing_physics_suite.drivers.pipeline as pl


class _Stop(Exception):
    pass


def test_nonlinear_forwards_resistivity_settings(monkeypatch):
    seen = {}

    def fake_analyse(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop
    monkeypatch.setattr(pl, 'analyse_with_mre', fake_analyse)
    with pytest.raises(_Stop):
        pl.nonlinear_resistive_calculation('eq', None, None, None, None, Zeff=1.5, average_ion_mass=2.5,
                                           energy_confinement_time=0.1, Coulomb_logarithm=17.0, eta_fac=2.0)
    assert seen['Coulomb_logarithm'] == 17.0 and seen['eta_fac'] == 2.0
