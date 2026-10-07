"""nonlinear_resistive_calculation argument forwarding (analyse_with_mre mocked; no codes run)."""
import pytest

import tearing_physics_suite.drivers.pipeline as pl
from tearing_physics_suite.wrappers.run_codes import CodeResults


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


def test_analyse_with_mre_passes_te_and_ti_to_their_parameters(monkeypatch):
    seen = {}

    def fake_terms(rdcon_xr, **kwargs):
        seen.update(kwargs)
        raise _Stop
    monkeypatch.setattr(pl, 'run_resistive_calculation',
                        lambda *a, **k: CodeResults(None, None, None, False, False, False, {'nn': 1}, None, {}, {}))
    monkeypatch.setattr(pl, 'mre_terms_on_modes', fake_terms)
    with pytest.raises(_Stop):
        pl.analyse_with_mre('eq', 1, 'ni', 'ne', 'te', 'ti', energy_confinement_time=0.1)
    assert (seen['ni_spline'], seen['ne_spline'], seen['te_spline'], seen['ti_spline']) == ('ni', 'ne', 'te', 'ti')
