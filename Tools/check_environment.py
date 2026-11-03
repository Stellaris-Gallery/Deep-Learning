"""Check course imports, the installed lock, and real computation on every GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = {
    'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
    'pandas': 'pandas', 'scipy': 'scipy', 'matplotlib': 'matplotlib',
    'scikit-learn': 'sklearn', 'gensim': 'gensim', 'transformers': 'transformers',
    'wandb': 'wandb', 'lifelines': 'lifelines', 'statsmodels': 'statsmodels.api',
    'jupyterlab': 'jupyterlab', 'ipykernel': 'ipykernel', 'nbformat': 'nbformat',
    'nbclient': 'nbclient', 'pytest': 'pytest',
}


def locked_versions(text):
    expected = {}
    for line in text.splitlines():
        match = re.match(r'^([A-Za-z0-9_.-]+)==([^\s;\\]+)', line)
        if match:
            expected[match[1]] = match[2]
        elif line.startswith(('torch @ ', 'torchvision @ ')):
            name = line.split()[0]
            wheel = unquote(line.split(' @ ', 1)[1].split()[0]).rsplit('/', 1)[-1]
            expected[name] = wheel.split('-')[1]
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-gpus', type=int, default=1,
                        help='Minimum number of visible GPUs; use 2 on the Sony workstation.')
    parser.add_argument('--report', type=Path, default=ROOT / '.runtime/environment-report.json')
    args = parser.parse_args()
    if args.expected_gpus < 1:
        parser.error('--expected-gpus must be at least 1')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    report = {'time_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'packages': {}, 'gpus': [], 'checks': []}

    def check(name, action):
        try:
            detail = action()
            report['checks'].append({'name': name, 'passed': True, 'detail': detail})
            print(f'PASS {name}: {detail}', flush=True)
        except Exception as exc:
            report['checks'].append({'name': name, 'passed': False, 'error': str(exc)})
            print(f'FAIL {name}: {exc}', flush=True)
            traceback.print_exc()

    def runtime_check():
        assert sys.platform == 'win32' and sys.version_info[:2] == (3, 14), 'Requires Windows and Python 3.14'
        assert sys.maxsize > 2**32, 'Requires 64-bit Python'
        assert sys.prefix != sys.base_prefix, 'Use the project virtual environment'
        return f'Python {platform.python_version()}, 64-bit virtual environment'

    check('runtime', runtime_check)
    for distribution, module in IMPORTS.items():
        def load(distribution=distribution, module=module):
            importlib.import_module(module)
            version = importlib.metadata.version(distribution)
            report['packages'][distribution] = version
            return version
        check(f'import {distribution}', load)

    def lock_check():
        lock = ROOT / 'environment/requirements.lock.txt'
        raw = lock.read_bytes()
        report['lock_sha256'] = hashlib.sha256(raw).hexdigest()
        expected = locked_versions(raw.decode('utf-8'))
        assert len(expected) >= len(IMPORTS), 'Lockfile is incomplete'
        mismatches = []
        for name, version in expected.items():
            try:
                installed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = 'MISSING'
            if installed != version:
                mismatches.append(f'{name}: expected {version}, found {installed}')
        assert not mismatches, '; '.join(mismatches)
        return f'{len(expected)} locked package versions match'

    check('lockfile', lock_check)

    def dependency_check():
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                                text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.strip()

    check('pip check', dependency_check)

    def cpu_checks():
        import numpy as np
        from gensim.models import KeyedVectors
        from sklearn.ensemble import RandomForestClassifier
        from lifelines import KaplanMeierFitter
        import matplotlib.pyplot as plt
        from transformers import BertConfig, BertForSequenceClassification
        import torch
        vectors = KeyedVectors(vector_size=2)
        vectors.add_vectors(['a', 'b'], np.array([[1., 0.], [0., 1.]], dtype=np.float32))
        assert abs(vectors.similarity('a', 'b')) < 1e-6
        rf = RandomForestClassifier(n_estimators=4, random_state=0).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        assert len(rf.predict([[0], [3]])) == 2
        KaplanMeierFitter().fit([1, 2, 3], event_observed=[1, 0, 1])
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.canvas.draw()
        plt.close(fig)
        tiny = BertForSequenceClassification(BertConfig(vocab_size=32, hidden_size=16,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=2))
        out = tiny(input_ids=torch.randint(0, 32, (2, 8)), labels=torch.tensor([0, 1]))
        out.loss.backward()
        assert torch.isfinite(out.loss), 'Nonfinite BERT loss'
        return 'Gensim, sklearn, lifelines, Matplotlib, and offline BERT forward/backward'

    check('course library smoke checks', cpu_checks)

    def gpu_inventory():
        import torch
        assert torch.cuda.is_available(), 'CUDA unavailable: check the NVIDIA driver and CUDA PyTorch wheel'
        assert torch.version.cuda == '13.0', f'Expected CUDA 13.0 wheel, got {torch.version.cuda}'
        count = torch.cuda.device_count()
        assert count >= args.expected_gpus, f'Expected at least {args.expected_gpus} GPUs; found {count}'
        report['torch_cuda'] = torch.version.cuda
        for index in range(count):
            prop = torch.cuda.get_device_properties(index)
            report['gpus'].append({'index': index, 'name': prop.name,
                'memory_gib': round(prop.total_memory / 1024**3, 2),
                'compute_capability': f'{prop.major}.{prop.minor}'})
        return report['gpus']

    check('GPU inventory', gpu_inventory)
    for gpu in report['gpus']:
        def gpu_math(index=gpu['index']):
            import torch
            from torchvision.ops import nms
            device = torch.device(f'cuda:{index}')
            with torch.cuda.device(device):
                torch.manual_seed(0)
                a = torch.randn(64, 64)
                b = torch.randn(64, 64)
                torch.testing.assert_close((a.to(device) @ b.to(device)).cpu(), a @ b, atol=1e-4, rtol=1e-4)
                for kind in (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU):
                    model = kind(8, 16, batch_first=True).to(device)
                    loss = model(torch.randn(2, 8, 8, device=device))[0].square().mean()
                    loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                conv = torch.nn.Conv2d(1, 4, 3).to(device)
                conv(torch.randn(2, 1, 8, 8, device=device)).square().mean().backward()
                boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device=device)
                assert nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5).numel() == 1
                torch.cuda.synchronize(device)
            return 'Matrix multiplication, RNN/LSTM/GRU and CNN backward, torchvision CUDA NMS'
        check(f'GPU {gpu["index"]} computation', gpu_math)

    report['passed'] = all(c['passed'] for c in report['checks'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {args.report.resolve()}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
