"""extract_scanned_xrs on real rdcon/stride outputs (fixtures; no codes run)."""
import os

import xarray as xr

from tearing_physics_suite.drivers.input_scans import extract_scanned_xrs
from tearing_physics_suite.wrappers.run_codes import CodeResults

FIX = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fixtures', 'g147131.02300_DIIID_KEFIT_')


def test_code_missing_at_one_scan_value():
    # stride failed at the second value: the message lists the codes present at every value (was a KeyError)
    r = xr.open_dataset(FIX + 'rdcon_n1.nc').load()
    s = xr.open_dataset(FIX + 'stride_n1.nc').load()
    res = [CodeResults(r, s, None, True, True, False, {'mtheta': 129, 'nn': 1}, None, {}, {}),
           CodeResults(r, None, None, True, False, False, {'mtheta': 257, 'nn': 1}, None, {}, {})]
    _, message = extract_scanned_xrs(res, 'mtheta', 0.05, 0.05, '')
    assert 'RDCON delta prime' in message and 'STRIDE delta prime' not in message
