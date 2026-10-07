"""Delta' from jGPEC (Julia GPEC, package GeneralizedPerturbedEquilibrium) as TPS codes jGPEC_galerkin / jGPEC_riccati.

jGPEC runs in one persistent Julia process per Python process (JuliaServer), so Julia compiles once per
worker. Inputs are mapped from the RDCON/STRIDE inputs so the domain and surfaces match RDCON.
"""
import atexit
import os
import selectors
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr

SOLVERS = ('galerkin', 'riccati')
SERVER_JL = Path(__file__).with_name('jgpec_server.jl')
JULIA_INSTALL = '/fusion/projects/codes/julia/julia-1.11.7/bin/julia'
# Defaults where unset (`module load julia` is not used: its purge() would drop the compiler modules)
_JULIA_DEFAULTS = {
    'JULIA_DEPOT_PATH': '/fusion/projects/tmdb/julia_depot_benjamins:',
    'JULIA_CPU_TARGET': 'generic',
    'OPENBLAS_NUM_THREADS': '1',
}
EQUILIBRIUM_KEYS = {'eq_filename', 'eq_type', 'jac_type', 'grid_type', 'psilow', 'psihigh', 'mpsi', 'mtheta',
                    'newq0', 'etol', 'force_termination', 'psi_accuracy', 'profile_source', 'use_galgrid'}
WALL_KEYS = {'shape', 'a', 'aw', 'bw', 'cw', 'dw', 'tw', 'equal_arc_wall'}


def jgpec_home():
    """jGPEC repository (Julia project): $JGPEC_HOME, else /fusion/projects/tmdb/src/GPEC."""
    return os.environ.get('JGPEC_HOME', '/fusion/projects/tmdb/src/GPEC')


def julia_exe():
    """Julia executable: $JGPEC_JULIA, else $JULIA_BINDIR/julia, else JULIA_INSTALL.

    (The module's bin/julia on PATH is a wrapper that needs the module's shell functions.)
    """
    if os.environ.get('JGPEC_JULIA'):
        return os.environ['JGPEC_JULIA']
    if os.environ.get('JULIA_BINDIR'):
        return os.path.join(os.environ['JULIA_BINDIR'], 'julia')
    return JULIA_INSTALL


def julia_env(threads=1):
    """Environment for the Julia process (module defaults filled in where unset)."""
    env = dict(os.environ)
    for k, v in _JULIA_DEFAULTS.items():
        env.setdefault(k, v)
    env['JULIA_NUM_THREADS'] = str(threads)
    env.pop('LD_LIBRARY_PATH', None)  # module libraries (e.g. PCRE) break Julia's bundled ones
    env.pop('JULIA_PROJECT', None)  # the module points it at the global environment; --project is passed
    return env


def precompile_jgpec(timeout=7200):
    """Instantiate and precompile jGPEC once (call before starting a pool of workers)."""
    cmd = [julia_exe(), f'--project={jgpec_home()}', '--startup-file=no', '-e',
           'using Pkg; Pkg.instantiate(); using GeneralizedPerturbedEquilibrium']
    subprocess.run(cmd, env=julia_env(), check=True, timeout=timeout)


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------
def _unquote(v):
    return str(v).strip().strip("'\"")


def _true(v):
    return str(v).strip().strip("'\".").lower() in ('t', 'true')


def _toml_value(v):
    if isinstance(v, (bool, np.bool_)):
        return 'true' if v else 'false'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_toml_value(x) for x in v) + ']'
    return '"' + str(v).replace('"', '\\"') + '"'


def gpec_toml_sections(eq_filename, nn, solver, inputs, **jgpec_kwargs):
    """{section: {key: value}} for gpec.toml, from the RDCON/STRIDE input dict (rdcon_stride_input_dict).

    jgpec_kwargs: '<key>_jgpec' sets <key> in [Equilibrium], [Wall] or [ForceFreeStates] (by key).
    """
    if solver not in SOLVERS:
        raise ValueError(f"jGPEC solver must be one of {SOLVERS}, not {solver!r}")
    vac = _true(inputs['vac_flag'])
    if solver == 'riccati' and not vac:
        raise ValueError("jGPEC Riccati gives no Delta' with a conducting wall on the plasma (vac_flag='f'); "
                         "use jgpec_solvers=('galerkin',).")
    profile_source = {'integrate': 'derivatives', 'values': 'values'}[_unquote(inputs.get('profile_source', 'integrate'))]
    eq = dict(eq_filename=eq_filename, eq_type=_unquote(inputs['eq_type']), jac_type=_unquote(inputs['jac_type']),
              grid_type=_unquote(inputs['grid_type']), psilow=float(inputs['psilow']), psihigh=float(inputs['psihigh']),
              mpsi=int(inputs['mpsi']), mtheta=int(inputs['mtheta']), newq0=float(inputs['newq0']),
              etol=float(inputs['etol']), profile_source=profile_source, force_termination=False)
    a_wall, ishape = float(inputs.get('a_wall', 0.0)), int(inputs.get('ishape', 6))
    if vac and ishape != 6:
        raise NotImplementedError(f"jGPEC wall for ishape={ishape}: only the conformal wall (ishape=6) is mapped.")
    wall = dict(shape='conformal', a=a_wall) if vac and a_wall < 10 else dict(shape='nowall')  # a >= 10: no wall in VACUUM
    ffs = dict(vac_flag=vac, qlow=float(inputs['qlow']), qhigh=float(inputs['qhigh']), sing_start=int(inputs['sing_start']),
               nn_low=int(nn), nn_high=int(nn), delta_mlow=int(inputs['delta_mlow']), delta_mhigh=int(inputs['delta_mhigh']),
               mthvac=int(inputs['mthvac']), singfac_min=float(inputs['singfac_min']), ucrit=float(inputs['ucrit']),
               eulerlagrange_tolerance=float(min(inputs['tol_r'], inputs['tol_nr'])),
               set_psilim_via_dmlim=_true(inputs['sas_flag']), dmlim=float(inputs['dmlim']),
               integrator=solver, kinetic_factor=0.0, local_stability_flag=False, psiedge=1.0,
               force_termination=True, HDF5_filename='gpec.h5')
    if solver == 'galerkin':
        ffs.update(gal_nx=int(inputs['nx']), gal_nq=int(inputs['nq']), gal_pfac=float(inputs['pfac']),
                   gal_dx0=float(inputs['dx0']), gal_dx1=float(inputs['dx1']), gal_dx2=float(inputs['dx2']),
                   gal_cutoff=int(inputs['cutoff']), gal_tol=float(inputs['gal_tol']),
                   gal_dx1dx2_flag=_true(inputs['dx1dx2_flag']), gal_sing_order=int(inputs['sing_order']),
                   gal_sing_order_ceiling=_true(inputs['sing_order_ceiling']), gal_solver=_unquote(inputs['solver']),
                   gal_rpec_flag=False)  # rpec drops the vacuum edge
    sections = {'Equilibrium': eq, 'Wall': wall, 'ForceFreeStates': ffs}
    for k, v in jgpec_kwargs.items():
        key = k.removesuffix('_jgpec')
        sec = 'Equilibrium' if key in EQUILIBRIUM_KEYS else 'Wall' if key in WALL_KEYS else 'ForceFreeStates'
        sections[sec][key] = v
    return sections


def write_gpec_toml(run_dir, sections):
    """Write run_dir/gpec.toml from {section: {key: value}}."""
    lines = []
    for sec, kv in sections.items():
        lines.append(f'[{sec}]')
        lines += [f'{k} = {_toml_value(v)}' for k, v in kv.items()]
        lines.append('')
    Path(run_dir, 'gpec.toml').write_text('\n'.join(lines))


# ------------------------------------------------------------------------------------------------
# Persistent Julia server
# ------------------------------------------------------------------------------------------------
class JuliaServer:
    """One Julia process running jgpec_server.jl; run(dir) runs jGPEC on dir/gpec.toml."""

    def __init__(self, threads=1, log_path=None, startup_timeout=7200):
        self.threads, self.startup_timeout = threads, startup_timeout
        self.log_path = log_path or os.path.join(os.getcwd(), f'jgpec_server_{os.getpid()}.log')
        self.proc = None

    def start(self):
        log = open(self.log_path, 'a')
        self.proc = subprocess.Popen(
            [julia_exe(), f'--project={jgpec_home()}', '--startup-file=no', f'-t{self.threads}', str(SERVER_JL)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=julia_env(self.threads))
        log.close()
        self._readline_until('TPS_READY', self.startup_timeout)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _readline_until(self, prefix, timeout):
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        end = time.monotonic() + timeout
        try:
            while True:
                left = end - time.monotonic()
                if left <= 0 or not sel.select(left):
                    self.close(kill=True)
                    raise TimeoutError(f'jGPEC server: no {prefix} within {timeout} s (log {self.log_path})')
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'jGPEC server exited (log {self.log_path})')
                if line.startswith(prefix):
                    return line.strip()
        finally:
            sel.close()

    def run(self, run_dir, timeout=7200):
        """Run jGPEC in run_dir; returns (ok, message). Restarts the server if it died."""
        if not self.alive():
            self.start()
        self.proc.stdin.write(str(run_dir) + '\n')
        self.proc.stdin.flush()
        reply = self._readline_until('TPS_DONE', timeout).split(' ', 2)
        return reply[1] == 'ok', (reply[2] if len(reply) > 2 else '')

    def close(self, kill=False):
        if self.proc is None:
            return
        if self.alive():
            if kill:
                self.proc.kill()
            else:
                try:
                    self.proc.stdin.write('TPS_QUIT\n')
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    self.proc.kill()
        self.proc = None


_SERVER = None


def get_server(threads=1):
    """This process's JuliaServer (started on first use, closed at exit)."""
    global _SERVER
    if _SERVER is None:
        _SERVER = JuliaServer(threads=threads)
        atexit.register(_SERVER.close)
    return _SERVER
