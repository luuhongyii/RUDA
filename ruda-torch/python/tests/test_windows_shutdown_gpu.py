import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows embedding shutdown contract')
@pytest.mark.parametrize('mode', ['return', 'sys_exit', 'late_callback', 'worker_thread'])
def test_python_exit_drains_native_workers(mode):
    if os.environ.get('RUDA_REQUIRE_GPU') != '1':
        pytest.skip('requires native RUDA GPU backend')
    code = '''
import atexit, sys, threading
import torch
import ruda_torch
mode = sys.argv[1]
retained = []
def work():
    x = torch.arange(33, dtype=torch.float32, device='ruda')
    y = x * x
    torch.testing.assert_close(y.cpu(), torch.arange(33, dtype=torch.float32).square())
    retained.append((x, y))
if mode == 'worker_thread':
    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert retained
else:
    work()
if mode == 'late_callback':
    atexit.register(work)
if mode == 'sys_exit':
    sys.exit(0)
'''
    result = subprocess.run([sys.executable, '-B', '-X', 'faulthandler', '-c', code, mode],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    for failure in ('panicked', 'shutdown failed', 'Exception ignored in atexit'):
        assert failure not in result.stderr, result.stderr
