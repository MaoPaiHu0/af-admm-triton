"""Run the seven-method solve, evaluation, export and plotting pipeline."""
from pathlib import Path
from importlib.util import find_spec
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
runtime = ROOT / '.runtime'
for name in ('cache', 'temp', 'mpl'):
    (runtime / name).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('TRITON_CACHE_DIR', str(runtime / 'cache'))
os.environ.setdefault('MPLCONFIGDIR', str(runtime / 'mpl'))
if os.name == 'nt':
    os.environ['TEMP'] = os.environ['TMP'] = str(runtime / 'temp')
    # Conda can load a user-site Triton while sysconfig still points to Conda's
    # site-packages. Locate the toolchain beside the imported package in that case.
    triton_spec = find_spec('triton')
    if triton_spec and triton_spec.submodule_search_locations:
        triton_root = Path(next(iter(triton_spec.submodule_search_locations)))
        bundled_cc = triton_root / 'runtime' / 'tcc' / 'tcc.exe'
        bundled_cuda = triton_root / 'backends' / 'nvidia'
        if bundled_cc.is_file():
            os.environ.setdefault('CC', str(bundled_cc))
        if all((bundled_cuda / relative).is_file() for relative in (
            'bin/ptxas.exe', 'include/cuda.h', 'lib/x64/cuda.lib'
        )) and not (os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')):
            os.environ['CUDA_PATH'] = str(bundled_cuda)

from af_unfolding.pipeline import main

if __name__ == '__main__':
    main()
