"""Run one fresh MiT2 scale ablation on idle GPU0 without changing live31 files."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review32_mit2_train70'
REFERENCE = ROOT / 'runs/submission31_train70/plan.json'
SOURCES = ['scripts/run_scale32.py', 'scripts/review_scale32.py', 'src/train_scale32.py']
GPU = 'GPU-76da22e2-40da-ebc0-098a-fa92ba030976'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from run_submission31 import ENV, now, read, sha, verify, parsed, relative, idle, logs, training_complete


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temp.replace(path)


def prepare():
    from data import load_split, scale_for
    assert not OUT.exists(), 'Refusing to overwrite existing scale32 experiment'
    reference = read(REFERENCE)
    verify(reference)
    original = next(job for job in reference['jobs'] if job['id'] == 'mit4')
    base = original['args']
    assert (base['arch'], base['encoder'], base['epochs']) == ('Unet', 'mit_b2', 60)
    assert base['width'] == 2560 and base['scale_multiplier'] == 2.0
    assert base['crop'] == base['tile'] == 1536
    training, validation = load_split('data', 'train'), load_split('data', 'valid')
    training_names = sorted(x['name'] for x in training)
    validation_names = sorted(x['name'] for x in validation)
    assert training_names == reference['training_names'] and len(training_names) == 70
    assert validation_names == reference['validation_names'] and len(validation_names) == 20
    assert set(training_names).isdisjoint(validation_names)
    values = dict(base, scale_multiplier=1.0,
                  out=relative(OUT / 'models/mit2/base'),
                  save_oof=relative(OUT / 'valid_probs/mit2/base'))
    values = parsed(values)
    differences = {key: {'mit4': base[key], 'mit2': values[key]}
                   for key in base if values[key] != base[key]}
    assert set(differences) == {'scale_multiplier', 'out', 'save_oof'}
    scale_policies = []
    for width in sorted({x['gray'].shape[1] for x in training + validation}):
        scale2 = scale_for(width, values['width'], values['scale_multiplier'])
        scale4 = scale_for(width, base['width'], base['scale_multiplier'])
        assert scale4 == 2 * scale2
        scale_policies.append(dict(original_width=int(width), mit2_scale=float(scale2),
            mit4_scale=float(scale4), train_images=sum(x['gray'].shape[1] == width for x in training),
            validation_images=sum(x['gray'].shape[1] == width for x in validation)))
    initial_audit = ROOT / base['out'] / 'initialization_audit.json'
    audit = read(initial_audit)
    assert audit['args'] == base and audit['inherited_checkpoint'] is None
    assert audit['plan_sha256'] == sha(REFERENCE)
    assert not audit['previous_experiment_checkpoint_used']
    paths = [ROOT / name for name in SOURCES] + [REFERENCE, initial_audit, ROOT / base['out'] / 'args.json']
    hashes = dict(reference['input_hashes'])
    hashes.update({relative(path): sha(path) for path in paths})
    job = dict(id='mit2', gpu_index=0, gpu_uuid=GPU, args=values,
               training_names=training_names, validation_names=validation_names,
               microbatches_per_epoch=280, optimizer_updates_per_epoch=140,
               command=[str(ROOT / '.venv/bin/python'), 'src/train_scale32.py', '--job', 'mit2'])
    idle(job)
    plan = dict(created=now(), purpose='Controlled MiT2 versus fresh MiT4 scale comparison on provided70/20',
                jobs=[job], training_names=training_names, validation_names=validation_names,
                training_images=70, validation_images=20, input_hashes=hashes,
                image_net_cache=reference['image_net_cache'], reference_plan=relative(REFERENCE),
                reference_mit4_args=base, reference_initialization_audit=relative(initial_audit),
                reference_initial_state_sha256=audit['full_initial_state_sha256'],
                argument_differences=differences, scale_policies=scale_policies, fresh_imagenet=True,
                historical_project_checkpoints_allowed=False, previously_viewed_validation=True,
                independent_validation=False, test_images_or_labels_used=False,
                shared_alloy_ids=reference['shared_alloy_ids'],
                checkpoint_selection='Fixed epoch60 last.pt; never validation-selected best.pt',
                evaluation='Same20 TTA8 and same PrimarySi postprocess, original inference scales only; no cap6',
                comparison_reference='runs/submission31_train70/reviews/mit4/all.done',
                automatic_adoption=False, automatic_test_prediction=False,
                further_training_scheduled=False, maximum_controller_hours=8,
                expected_total_minutes=[210, 225], live_submission31_unchanged=True)
    save('plan.json', plan)
    save('state.json', dict(time=now(), stage='prepared_not_started'))
    print(json.dumps(dict(prepared=True, plan_sha256=sha(OUT / 'plan.json'),
                          changed_args=differences, gpu=GPU, input_hashes=len(hashes))), flush=True)


def run():
    plan = read(OUT / 'plan.json')
    verify(plan)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not any((OUT / name).exists() for name in ['started.json', 'error.json', 'all.done'])
    job = plan['jobs'][0]
    idle(job)
    assert not (ROOT / job['args']['out']).exists()
    streams, children = [], []
    started = time.monotonic()
    state = dict(time=now(), controller_pid=os.getpid(), gpu_index=0, gpu_uuid=GPU, job='mit2')

    def interrupted(*unused):
        raise RuntimeError('Scale32 controller interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def spawn(stage, command, use_gpu):
        stream = (OUT / (stage + '.log')).open('x')
        streams.append(stream)
        child = subprocess.Popen(command, cwd=ROOT,
            env=dict(ENV, CUDA_VISIBLE_DEVICES=GPU if use_gpu else ''),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        state.update(stage=stage, pid=child.pid, command=command, stage_started=now())
        return child

    def tick():
        assert time.monotonic() - started < plan['maximum_controller_hours'] * 3600
        state.update(time=now(), elapsed_minutes=(time.monotonic() - started) / 60)
        save('state.json', state)

    def wait_child(child, training=False):
        while child.poll() is None:
            if training:
                rows = logs(job['args'])
                assert all(math.isfinite(row['loss']) for row in rows)
                state.update(completed_epochs=len(rows), total_epochs=60, latest_epoch=rows[-1] if rows else None)
            tick()
            time.sleep(15)
        assert child.returncode == 0, f"{state['stage']} failed with exit code {child.returncode}"

    try:
        child = spawn('training', job['command'], True)
        save('started.json', dict(time=now(), controller_pid=os.getpid(),
                                 plan_sha256=sha(OUT / 'plan.json'), job=dict(state)))
        wait_child(child, training=True)
        state['training_receipt'] = training_complete(job, 'base')
        verify(plan)
        child = spawn('review', [str(ROOT / '.venv/bin/python'), 'scripts/review_scale32.py'], False)
        wait_child(child)
        assert (OUT / 'review/all.done').exists()
        state.update(stage='waiting_for_reference_review', pid=None)
        reference_done = ROOT / plan['comparison_reference']
        while not reference_done.exists():
            reference_error = REFERENCE.parent / 'error.json'
            assert not reference_error.exists(), 'MiT2 review complete but reference31 controller failed; comparison deferred'
            tick()
            time.sleep(30)
        child = spawn('compare', [str(ROOT / '.venv/bin/python'), 'scripts/review_scale32.py', '--compare'], False)
        wait_child(child)
        verify(plan)
        assert (OUT / 'all.done').exists()
        state.update(stage='complete_and_stop', pid=None, completed_epochs=60, automatic_adoption=False)
        tick()
    except BaseException as error:
        save('error.json', dict(time=now(), error=f'{type(error).__name__}: {error}', state=state))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    options = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(ENV)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    prepare() if options.mode == 'prepare' else run()
