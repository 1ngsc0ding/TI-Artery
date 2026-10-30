"""Supervise repeatable training trials. Uses validation only, never test.
Standard-library supervisor; trainers need the project's torch/cv2 dependencies.
"""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

METRIC = 'validation_global_pixel_f1_training_resolution'
TZ = timezone(timedelta(hours=7))


def now():
    return datetime.now(TZ).isoformat(timespec='seconds')


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def error_details(lines):
    text = '\n'.join(lines)
    frames = re.findall(r'File "([^"]+)", line (\d+)', text)
    files = re.findall(r'(?:Cannot read (?:image|mask) file: |(?:image|mask)=)([^,\n]+)', text)
    message = next((x.strip() for x in reversed(lines) if re.match(r'^[\w.]+(?:Error|Exception):', x.strip())), '')
    return {'source_file': frames[-1][0] if frames else None,
            'source_line': int(frames[-1][1]) if frames else None,
            'data_files': files, 'problem': message or (lines[-1] if lines else 'No output from child process')}


def stop_child(proc):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=['plaque', 'wall'], default='plaque')
    ap.add_argument('--data', help='Dataset root containing train and val folders')
    ap.add_argument('--target', type=float, default=0.90)
    ap.add_argument('--epochs', type=int, default=100, help='Epoch budget PER trial')
    ap.add_argument('--max-runs', type=int, default=6, help='0 = run until target or error/Ctrl+C')
    ap.add_argument('--sizes', type=int, nargs='+', default=[320, 512])
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=0.0003)
    ap.add_argument('--arch', choices=['unet', 'attnunet', 'unetpp'], default='unet')
    ap.add_argument('--workers', type=int, default=0, help='0 gives precise in-process data error tracebacks')
    ap.add_argument('--output', help='NEW output directory; each launch uses its own directory')
    args = ap.parse_args(argv)
    if not math.isfinite(args.target) or not 0 < args.target <= 1:
        ap.error('target must be in (0, 1]')
    if args.max_runs < 0 or args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.base < 1:
        ap.error('Invalid run/epoch/batch/worker/base value')
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error('lr must be finite and positive')
    if any(s < 32 or s % 16 for s in args.sizes):
        ap.error('sizes must be multiples of 16, at least 32')
    root = Path(__file__).resolve().parent
    data = Path(args.data or ('data/plaque_v2' if args.task == 'plaque' else 'data/cubs')).resolve()
    output = Path(args.output).resolve() if args.output else root/'runs'/f'{args.task}_{datetime.now(TZ):%Y%m%d_%H%M%S}_{os.getpid()}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {'started_at': now(), 'task': args.task, 'target': args.target, 'metric': METRIC,
             'data': str(data), 'output': str(output), 'status': 'starting', 'best_val_f1': None,
             'best_checkpoint': None, 'target_reached': False, 'trial': 0}
    events = output/'events.jsonl'
    status = output/'status.json'
    proc = None

    def event(kind, **fields):
        item = {'time': now(), 'event': kind, **fields}
        with events.open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')
        print(f"[{item['time']}] {kind}: {json.dumps(fields, ensure_ascii=False)}", flush=True)
        state['updated_at'] = item['time']
        state['elapsed_seconds'] = round(time.monotonic()-started, 1)
        write_json(status, state)

    try:
        trainer = root/('train_plaque_unet.py' if args.task == 'plaque' else 'train_wall_unet.py')
        state['trainer_file'] = str(trainer)
        state['trainer_sha256'] = hashlib.sha256(trainer.read_bytes()).hexdigest()
        write_json(output/'config.json', vars(args))
        for split in ['train', 'val']:
            if not (data/split).is_dir():
                raise FileNotFoundError(f'Missing dataset folder: {data/split}')
        event('START', metric=METRIC, target=args.target, log=str(events), status_file=str(status),
              warning='Validation target only; no test performance guarantee. Repeated selection can overfit validation.')
        trial = 0
        while args.max_runs == 0 or trial < args.max_runs:
            trial += 1
            size = args.sizes[(trial-1) % len(args.sizes)]
            # Independent starts: do not reuse old test-selected or newly split validation weights.
            # Cycle resolution and LR; advance seed between trials.
            lr = args.lr * (0.5 ** (((trial-1)//len(args.sizes)) % 3))
            trial_dir = output/f'trial_{trial:04d}'
            trial_dir.mkdir()
            best = trial_dir/'best.pt'
            last = trial_dir/'last.pt'
            raw_log = trial_dir/'console.log'
            cmd = [sys.executable, '-u', str(trainer), '--data', str(data), '--epochs', str(args.epochs),
                   '--size', str(size), '--batch', str(args.batch), '--lr', str(lr),
                   '--base', str(args.base), '--arch', args.arch, '--seed', str(args.seed+trial-1),
                   '--resize-mode', 'letterbox', '--workers', str(args.workers),
                   '--target-f1', str(args.target), '--out', str(best), '--last', str(last)]
            state.update(status='training', trial=trial, console_log=str(raw_log), last_checkpoint=str(last))
            event('TRIAL_START', trial=trial, command=cmd, model_file=str(best), console_log=str(raw_log))
            tail = deque(maxlen=150)
            metrics_seen = 0
            with raw_log.open('w', encoding='utf-8') as log:
                proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding='utf-8', errors='replace', bufsize=1)
                for raw in proc.stdout:
                    line = raw.rstrip('\r\n')
                    timestamp = now()
                    log.write(f'[{timestamp}] {line}\n'); log.flush()
                    print(f'[{timestamp}] {line}', flush=True)
                    tail.append(line)
                    if line.startswith('METRIC_JSON '):
                        m = json.loads(line[len('METRIC_JSON '):])
                        score = float(m['val_f1'])
                        if m.get('metric') != METRIC or not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError(f'Invalid metric from {trainer}: {m}')
                        if not best.is_file():
                            raise FileNotFoundError(f'Metric received but checkpoint is missing: {best}')
                        metrics_seen += 1
                        if state['best_val_f1'] is None or score > state['best_val_f1']:
                            state.update(best_val_f1=score, best_checkpoint=str(best), best_epoch=m['epoch'],
                                         best_trial=trial, best_size=size)
                        event('EPOCH', trial=trial, epoch=m['epoch'], val_f1=score, model_file=str(best))
                code = proc.wait()
            proc = None
            if code != 0:
                details = error_details(list(tail))
                state.update(status='error', error={**details, 'exit_code': code, 'time': now(),
                             'trainer_file': str(trainer), 'console_log': str(raw_log), 'trial': trial})
                event('ERROR', **state['error'])
                # Do not loop forever on corrupt files, missing dependencies, or OOM.
                return 1
            if metrics_seen == 0:
                raise RuntimeError(f'No structured metrics from trainer {trainer}; see {raw_log}')
            if state['best_val_f1'] >= args.target:
                state.update(status='target_reached', target_reached=True, ended_at=now())
                event('TARGET_REACHED', best_val_f1=state['best_val_f1'], model_file=state['best_checkpoint'])
                return 0
            event('TRIAL_END', trial=trial, best_val_f1=state['best_val_f1'])
        state.update(status='budget_exhausted', ended_at=now())
        event('STOPPED_BELOW_TARGET', best_val_f1=state['best_val_f1'], model_file=state['best_checkpoint'])
        return 2
    except KeyboardInterrupt:
        if proc is not None:
            stop_child(proc)
        state.update(status='interrupted', ended_at=now())
        event('INTERRUPTED', message='Stopped by user; completed epoch checkpoints remain available')
        return 130
    except Exception as exc:
        if proc is not None:
            stop_child(proc)
        import traceback
        trace = traceback.format_exc()
        (output/'supervisor_error.log').write_text(trace, encoding='utf-8')
        details = error_details(trace.splitlines())
        state.update(status='error', ended_at=now(), error={**details, 'time': now()})
        event('ERROR', **state['error'], traceback_file=str(output/'supervisor_error.log'))
        return 1


if __name__ == '__main__':
    # Convert termination into the same clean interrupt path on platforms supporting SIGTERM.
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
