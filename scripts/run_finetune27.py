"""Extend the fixed six-epoch interface recipe to the four remaining CV folds.

--prepare performs CPU checks only. --run explicitly starts four isolated jobs.
Existing fold0 results and submission files are read-only inputs.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
import run_crop18 as common
import finetune_final26 as prior

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review27_finetune_full'
OLD = ROOT / 'runs/review26_final_strengthening/finetune_plan'
SCALE = ROOT / 'runs/review26_final_strengthening/scale_trials'
BASE = prior.BASE
MIT = prior.MIT
SOURCES = list(dict.fromkeys(prior.SOURCES + [
    'scripts/run_finetune27.py', 'scripts/review_finetune27.py']))
GPU_UUIDS = {
    0: 'GPU-76da22e2-40da-ebc0-098a-fa92ba030976',
    1: 'GPU-3fc53ad0-271b-ffd6-c83c-84575e82d19c',
    2: 'GPU-e800362c-2a5a-0208-8d32-28107bca6017',
    3: 'GPU-dce5fed1-b1cd-7a95-9c1b-1f281a29f713',
}
ENV = {**common.ENV, 'PYTHONDONTWRITEBYTECODE': '1'}
CHILDREN = []
ALLOWED_RECIPE_DIFFERENCES = {'fold', 'init', 'out', 'save_oof', 'hard_maps'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(prior.clean(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def status(stage, **fields):
    value = dict(time=common.now(), pid=os.getpid(), stage=stage,
                 new_training_folds=[1, 2, 3, 4], reused_completed_folds=[0],
                 epochs_per_new_fold=6, total_evaluation_images=90,
                 independent_validation=False, submission_changed=False,
                 further_training_scheduled=False, **fields)
    save('state.json', value)
    print(json.dumps(prior.clean(value), ensure_ascii=False, allow_nan=False), flush=True)


def focus_map(label):
    import cv2
    import numpy as np
    a, e = (label == 1).astype(np.uint8), (label == 3).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    return ((a > 0) & (cv2.dilate(e, kernel) > 0)) | ((e > 0) & (cv2.dilate(a, kernel) > 0))


def validate_checkpoint(path, args, reference):
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == args['epochs'] and ckpt['args'] == args
    weights = ckpt['model']
    assert reference.keys() == weights.keys() and len(weights) == 350
    assert all(reference[k].shape == value.shape and reference[k].dtype == value.dtype
               for k, value in weights.items())
    assert all(torch.isfinite(value).all().item()
               for value in weights.values() if value.is_floating_point())
    return len(weights)


def prepare():
    """Pin immutable inputs, prove fold identity and create only train-label maps."""
    import numpy as np
    import torch
    from PIL import Image
    from data import load_cv
    from train import build_from_args

    assert not OUT.exists(), f'Refusing to overwrite preparation: {OUT}'
    old_plan, old_done, old_results = [read(OLD / name) for name in ['plan.json', 'all.done', 'results.json']]
    assert old_done['plan_sha256'] == common.sha256(OLD / 'plan.json')
    assert old_done['results_sha256'] == common.sha256(OLD / 'results.json')
    assert old_results['plan_sha256'] == old_done['plan_sha256']
    prior.verify(old_plan['input_hashes'])
    old_job = next(job for job in old_plan['jobs'] if job['arm'] == 'interface')
    old_receipt = next(row for row in old_results['completion_receipts'] if row['id'] == 'interface')
    prior.verify(old_receipt['hashes'])
    recomputed = prior.completed(old_job)
    assert recomputed['hashes'] == old_receipt['hashes']
    assert old_job['fold'] == 0 and old_job['args']['epochs'] == 6
    assert old_job['args']['hard_crop'] == 0.35 and old_job['args']['lr'] == 3e-5

    source0 = read(ROOT / f'runs/cv/{BASE}/fold0/args.json')
    expected = dict(arch='UnetPlusPlus', encoder='resnet34', epochs=120, fold=0,
                    width=3840, crop=1536, tile=1536, bs=4, spi=8, bw=2.0,
                    rot='rot90', seed=0, amp=True, enc_lr_mult=1.0,
                    full=False, init='', hard_crop=0.0)
    assert all(source0.get(key) == value for key, value in expected.items())
    with patch.object(socket.socket, 'connect', prior.no_network), patch.object(socket, 'create_connection', prior.no_network):
        model = build_from_args(source0, weights='imagenet').cpu()
    reference = model.state_dict()
    validate_checkpoint(common.under_root(old_job['args']['out']) / 'last.pt', old_job['args'], reference)
    fold_of = read(ROOT / 'splits/folds5.json')
    assert len(fold_of) == 90 and set(fold_of.values()) == set(range(5))
    jobs, training_items = [], {}
    paths = [ROOT / name for name in SOURCES]
    paths += [ROOT / 'splits/folds5.json']
    paths += [OLD / name for name in ['plan.json', 'all.done', 'results.json', 'completed_jobs.json', 'per_image_rows.json']]
    for fold in range(1, 5):
        source_dir = ROOT / f'runs/cv/{BASE}/fold{fold}'
        source_args = read(source_dir / 'args.json')
        assert {key for key in source0 if source0[key] != source_args[key]} == {'fold', 'out'}
        assert set(source0) == set(source_args) and source_args['fold'] == fold
        assert source_args['out'] == relative(source_dir)
        checkpoint = source_dir / 'last.pt'
        tensor_count = validate_checkpoint(checkpoint, source_args, reference)
        train_items, valid_items = load_cv('data', fold, 'splits/folds5.json')
        train_names = sorted(item['name'] for item in train_items)
        valid_names = sorted(item['name'] for item in valid_items)
        assert len(train_names) == 72 and len(valid_names) == 18
        assert len(set(train_names)) == 72 and len(set(valid_names)) == 18
        assert set(train_names).isdisjoint(valid_names)
        assert set(train_names) | set(valid_names) == set(fold_of)
        assert {item['alloy'] for item in train_items}.isdisjoint({item['alloy'] for item in valid_items})
        assert all(fold_of[name] == fold for name in valid_names)
        assert all(fold_of[name] != fold for name in train_names)
        values = dict(old_job['args'], fold=fold, init=relative(checkpoint),
                      out=relative(OUT / 'interface' / f'fold{fold}'),
                      save_oof=relative(OUT / 'oof/interface' / f'fold{fold}'),
                      hard_maps=relative(OUT / 'train_interface_maps' / f'fold{fold}'))
        values = prior.actual_parser(values)
        assert {key for key in values if values[key] != old_job['args'][key]} == ALLOWED_RECIPE_DIFFERENCES
        assert not common.under_root(values['out']).exists()
        assert not common.under_root(values['save_oof']).exists()
        jobs.append(dict(id=f'interface_fold{fold}', arm='interface', fold=fold, args=values,
                         train_names=train_names, validation_names=valid_names,
                         strict_initialization_tensors=tensor_count,
                         gpu=dict(index=fold - 1, uuid=GPU_UUIDS[fold - 1])))
        training_items[fold] = train_items
        paths += [checkpoint, source_dir / 'args.json', source_dir / 'result.json']
        if fold == 1:
            for item in train_items + valid_items:
                paths += [ROOT / f"data/{item['split']}/images/{item['name']}_image.png",
                          ROOT / f"data/{item['split']}/masks/{item['name']}_mask.png"]
    assert sorted(name for job in jobs for name in job['validation_names']) == sorted(
        name for name, fold in fold_of.items() if fold != 0)
    for name in sorted(fold_of):
        for experiment in [BASE, MIT]:
            paths.append(ROOT / f'runs/oof/{experiment}/{name}.npy')

    scale_plan, scale_results, scale_done = [read(SCALE / name) for name in ['plan.json', 'results.json', 'all.done']]
    assert scale_done['plan_sha256'] == common.sha256(SCALE / 'plan.json') == scale_results['plan_sha256']
    assert scale_done['results_sha256'] == common.sha256(SCALE / 'results.json')
    cap6 = [row for row in scale_results['output_receipts'] if row['arm'] == 'cap6']
    assert len(cap6) == 26 and len({row['name'] for row in cap6}) == 26
    assert all(row['fold'] == fold_of[row['name']] and row['actual_scale'] == 6 for row in cap6)
    cap6_hashes = {row['path']: row['sha256'] for row in cap6}
    prior.verify(cap6_hashes)
    paths += [SCALE / name for name in ['plan.json', 'results.json', 'all.done', 'per_image_rows.json']]
    inputs = {**old_plan['input_hashes'], **old_receipt['hashes'], **prior.pin(paths), **cap6_hashes}
    prior.verify(inputs)
    del model, reference
    OUT.mkdir(parents=True)
    for job in jobs:
        maps = common.under_root(job['args']['hard_maps'])
        maps.mkdir(parents=True)
        rows = []
        for item in training_items[job['fold']]:
            label = item['lab']
            focus = focus_map(label)
            path = maps / f"{item['name']}.png"
            Image.fromarray(focus.astype(np.uint8) * 255).save(path)
            assert path.stem in job['train_names'] and path.stem not in job['validation_names']
            inputs[relative(path)] = common.sha256(path)
            rows.append(dict(name=item['name'], selected_pixels=int(focus.sum()), total_pixels=int(focus.size),
                             al3ni_side=int((focus & (label == 1)).sum()),
                             eutectic_side=int((focus & (label == 3)).sum())))
        assert len(list(maps.glob('*.png'))) == 72
        job['map_rows'] = rows
        job['map_images_with_focus'] = sum(row['selected_pixels'] > 0 for row in rows)
    plan = dict(created=common.now(), jobs=jobs,
                reused_fold0=dict(job=old_job, completion_receipt=old_receipt),
                input_hashes=inputs, source_hashes=prior.pin([ROOT / name for name in SOURCES]),
                split_sha256=common.sha256(ROOT / 'splits/folds5.json'),
                cap6_output_receipts=cap6, old_completed_experiment=relative(OLD),
                initial_training_epochs=120, continuation_epochs=6, lr=3e-5,
                focus_definition=old_plan['focus_definition'], focus_probability=0.35,
                radius_original_pixels=2, map_uses_validation_labels=False, map_uses_prediction_errors=False,
                empty_map_behavior=old_plan['empty_map_behavior'],
                recipe_changes_from_fold0=sorted(ALLOWED_RECIPE_DIFFERENCES),
                optimizer_state_resumed=False, new_adamw_and_cosine_schedule=True,
                checkpoint_for_evaluation='last.pt at continuation epoch6, never best.pt',
                effective_batch=4, optimizer_updates_per_epoch=144, total_updates_per_new_fold=864,
                new_training_folds=[1, 2, 3, 4], reused_completed_folds=[0], total_evaluation_images=90,
                selection_rule='Highest full90 observed mIoU among four fixed 50:50 two-family pipelines; ties retain baseline',
                comparison_recipes=['baseline_original', 'baseline_cap6', 'interface_original', 'interface_cap6'],
                fixed_family_weights=[0.5, 0.5], unchanged_tta=8,
                no_per_fold_recipe_selection=True, no_weight_tuning=True,
                estimated_parallel_minutes=[11, 15], maximum_controller_minutes=25,
                independent_validation=False, submission_changed=False, automatic_adoption=False,
                further_training_scheduled=False)
    save('plan.json', plan)
    status('prepared_not_started', map_images_with_focus={str(job['fold']): job['map_images_with_focus'] for job in jobs},
           strict_initialization_tensors=350, expected_minutes=[11, 15])


def idle_expected_gpus():
    available = {row['index']: row['uuid'] for row in prior.inventory()}
    assert all(available.get(index) == uuid for index, uuid in GPU_UUIDS.items()), (
        'The four recorded GPUs must all be idle; existing work will not be interrupted')
    return [dict(index=index, uuid=uuid) for index, uuid in GPU_UUIDS.items()]


def job_progress(job, process, gpu, started):
    rows = common.read_epoch_logs(common.under_root(job['args']['out']) / 'log.jsonl')
    latest = rows[-1] if rows else {}
    epoch = latest.get('epoch', 0)
    assert 0 <= epoch <= 6
    times = [float(row['train_sec']) for row in rows[-3:]]
    mean = sum(times) / len(times) if times else None
    return dict(id=job['id'], fold=job['fold'], gpu=gpu, pid=process.pid,
                epoch=epoch, total_epochs=6, phase='final_tta' if epoch == 6 else 'training',
                elapsed_sec=round(time.monotonic() - started, 1), loss=latest.get('loss'),
                recent_three_mean_train_sec=mean,
                remaining_train_sec_estimate=(6 - epoch) * mean if mean is not None else None,
                remaining_total_sec_estimate=(6 - epoch) * mean + 70 if mean is not None and epoch < 6 else None,
                eta_method='recent up-to-3 training mean plus 70 seconds final validation/TTA; final-TTA ETA omitted')


def run():
    assert (OUT / 'plan.json').is_file()
    assert not any((OUT / marker).exists() for marker in ['started.json', 'results.json', 'error.json', 'all.done'])
    plan = read(OUT / 'plan.json')
    assert [job['fold'] for job in plan['jobs']] == [1, 2, 3, 4]
    prior.verify(plan['input_hashes'])
    gpus = idle_expected_gpus()
    time.sleep(3)
    assert idle_expected_gpus() == gpus
    for job in plan['jobs']:
        assert not common.under_root(job['args']['out']).exists()
        assert not common.under_root(job['args']['save_oof']).exists()
    (OUT / 'logs').mkdir()
    save('started.json', dict(time=common.now(), pid=os.getpid(),
                             plan_sha256=common.sha256(OUT / 'plan.json'), gpus=gpus))
    deadline = time.monotonic() + 25 * 60
    running, receipts = {}, []
    for job, gpu in zip(plan['jobs'], gpus):
        assert job['gpu'] == gpu
        command = [sys.executable, str(Path(__file__).resolve()), '--train-worker', *common.cli_args(job['args'])]
        with (OUT / 'logs' / f"{job['id']}.log").open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': gpu['uuid']},
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(child)
        running[job['id']] = (job, child, gpu['index'], time.monotonic())
    while running:
        assert time.monotonic() < deadline, 'Controller exceeded fixed 25-minute budget'
        for identifier, (job, child, gpu, started) in list(running.items()):
            rows = common.read_epoch_logs(common.under_root(job['args']['out']) / 'log.jsonl')
            assert all(math.isfinite(row['loss']) for row in rows)
            if child.poll() is None:
                continue
            assert child.returncode == 0, f'{identifier} exited {child.returncode}'
            receipt = prior.completed(job)
            receipt.update(fold=job['fold'], epochs=6, images=18, gpu=job['gpu'])
            receipts.append(receipt)
            save('completed_jobs.json', receipts)
            del running[identifier]
        status('four_fold_continuation' if running else 'evaluating_cpu',
               jobs=[job_progress(*values) for values in running.values()], completed_jobs=receipts)
        if running:
            time.sleep(15)
    prior.verify(plan['input_hashes'])
    with (OUT / 'logs/evaluation.log').open('x', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, 'scripts/review_finetune27.py', '--evaluate'],
                                 cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': ''}, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(child)
    while child.poll() is None:
        assert time.monotonic() < deadline, 'Controller exceeded fixed 25-minute budget during evaluation'
        time.sleep(5)
    assert child.returncode == 0, f'Evaluation exited {child.returncode}'
    done = read(OUT / 'all.done')
    assert done['results_sha256'] == common.sha256(OUT / 'results.json')
    assert done['plan_sha256'] == common.sha256(OUT / 'plan.json')
    status('complete_and_stop', results=relative(OUT / 'results.json'))


def self_test():
    import numpy as np
    label = np.zeros((9, 12), dtype=np.uint8)
    label[:, 1:6] = 1
    label[:, 6:11] = 3
    actual = focus_map(label)
    expected = np.zeros_like(label, dtype=bool)
    expected[:, 4:8] = True
    assert np.array_equal(actual, expected)
    assert not focus_map(np.ones((8, 8), dtype=np.uint8)).any()
    assert not focus_map(np.full((8, 8), 3, dtype=np.uint8)).any()
    assert len(GPU_UUIDS) == 4 and len(set(GPU_UUIDS.values())) == 4
    print(json.dumps(dict(self_test='passed', boundary_radius=2, two_sided=True,
                          empty_maps_uniform=True, gpu_launched=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true', help='CPU-only guarded preparation')
    action.add_argument('--run', action='store_true', help='Explicitly launch the four prepared jobs')
    action.add_argument('--self-test', action='store_true', help='CPU-only focused geometry checks')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(common.THREAD_ENV)
    sys.path.insert(0, str(ROOT / 'src'))
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test()
        return
    if args.prepare:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Preparation must explicitly disable CUDA'
        prepare()
        return
    assert OUT.is_dir()
    with (OUT / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def interrupted(signum, frame):
            raise RuntimeError(f'Controller received signal {signum}')

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        common.CHILDREN = CHILDREN
        try:
            run()
        except BaseException as error:
            common.cleanup_children()
            status('error', message=repr(error))
            save('error.json', dict(time=common.now(), error=repr(error)))
            raise


if __name__ == '__main__':
    if sys.argv[1:2] == ['--train-worker']:
        common.train_worker()
    else:
        main()
