"""Review fresh train70 -> valid20 runs and fixed soft-voting combinations."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/submission31_train70'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.chdir(ROOT)
JOB_IDS = ('mit4', 'unet3', 'context4')


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def fresh_path(value):
    path = (ROOT / value).resolve()
    assert path.is_relative_to(OUT.resolve()), f'Not a fresh-run artifact: {value}'
    return path


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def receipt(path):
    return dict(path=relative(path), sha256=sha(path))


def verify(plan):
    assert {job['id'] for job in plan['jobs']} == set(JOB_IDS)
    assert len(plan['jobs']) == 3
    for name, digest in plan['input_hashes'].items():
        assert sha(ROOT / name) == digest, name
    for record in plan.get('image_net_cache', []):
        assert sha(record['path']) == record['sha256'], record['path']


def dataset(job):
    from data import load_split
    args = job['args']
    expected = dict(fold=-1, train_splits=['train'], val_split='valid', full=False,
                    train_frac=1., seed=0, max_iters=0, init='', init_skip_head=False)
    assert all(args.get(key) == value for key, value in expected.items()), args
    training = load_split(args['data'], 'train')
    valid = load_split(args['data'], 'valid')
    train_names = sorted(item['name'] for item in training)
    valid_names = sorted(item['name'] for item in valid)
    assert len(training) == len(set(train_names)) == 70
    assert len(valid) == len(set(valid_names)) == 20
    assert set(train_names).isdisjoint(valid_names)
    assert all(item['lab'] is not None for item in training + valid)
    for key in ('training_names', 'train_names'):
        if key in job:
            assert job[key] == train_names
    if 'validation_names' in job:
        assert job['validation_names'] == valid_names
    return training, valid


def probability(path, item):
    import numpy as np
    stored = np.load(path, allow_pickle=False)
    assert stored.dtype == np.float16 and stored.shape == (4, *item['gray'].shape), str(path)
    assert np.isfinite(stored).all() and stored.min() >= 0 and stored.max() <= 1, str(path)
    value = stored.astype(np.float32)
    assert np.max(np.abs(value.sum(0) - 1)) < .005, str(path)
    return value


def image_row(item, prob):
    from infer import image_iou, postprocess
    return dict(tta_pp=image_iou(item['lab'], postprocess(prob)).tolist(),
                low=bool(item['low_contrast']), width=int(item['gray'].shape[1]), alloy=int(item['alloy']))


def scores(rows):
    import numpy as np
    from data import CLASSES
    result = {}
    for label, records in rows.items():
        values = np.array([record['tta_pp'] for record in records.values()], dtype=float)
        assert values.shape == (20, 4)
        assert np.isfinite(values).any(axis=1).all()
        result[label] = dict(images=20, mIoU=float(np.nanmean(values, axis=1).mean() * 100),
            class_IoU={name: float(np.nanmean(values[:, i]) * 100) for i, name in enumerate(CLASSES)},
            both_absent_one_mIoU=float(np.where(np.isnan(values), 1., values).mean() * 100),
            both_absent_class_image_pairs=int(np.isnan(values).sum()))
    return result


def verify_stage(args, job, training, valid, initial_checkpoint=None):
    import torch
    from hybrid30 import install
    train = install()
    base = job['args']
    for key in ('data', 'train_splits', 'val_split', 'fold', 'full', 'train_frac', 'seed',
                'max_iters', 'arch', 'encoder', 'width', 'scale_multiplier', 'crop', 'tile'):
        assert args[key] == base[key], key
    assert not args['init_skip_head']
    if initial_checkpoint is None:
        assert args['init'] == ''
    else:
        assert fresh_path(args['init']) == initial_checkpoint.resolve()
    run, probs = fresh_path(args['out']), fresh_path(args['save_oof'])
    assert read(run / 'args.json') == args
    trained = read(run / 'result.json')
    assert trained['args'] == args
    assert sorted(row['name'] for row in trained['rows']) == sorted(item['name'] for item in valid)
    assert {path.stem for path in probs.glob('*.npy')} == {item['name'] for item in valid}
    logs = [json.loads(line) for line in (run / 'log.jsonl').read_text(encoding='utf8').splitlines() if line]
    assert [row['epoch'] for row in logs] == list(range(1, args['epochs'] + 1))
    microbatches = len(training) * args['spi'] // args['bs']
    updates = math.ceil(microbatches / args['accum_steps'])
    assert all(math.isfinite(row['loss']) and row['microbatches'] == microbatches
               and row['optimizer_updates'] == updates for row in logs)
    checkpoint = run / 'last.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['args'] == args and ckpt['epoch'] == args['epochs']
    assert all(torch.isfinite(value).all() for value in ckpt['model'].values() if value.is_floating_point())
    model = train.build_from_args(args, weights=None)
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    files = [run / name for name in ('args.json', 'result.json', 'log.jsonl', 'last.pt')]
    files += [probs / f"{item['name']}.npy" for item in valid]
    if args['hard_crop']:
        maps = fresh_path(args['hard_maps'])
        assert {path.stem for path in maps.glob('*.png')} == {item['name'] for item in training}
        files += sorted(maps.glob('*.png'))
    manifest = dict(args=args, epochs=args['epochs'], checkpoint=relative(checkpoint),
                    checkpoint_sha256=sha(checkpoint), strict_cpu_reload=True, finite_tensors=True,
                    training_minutes=sum(row['train_sec'] for row in logs) / 60,
                    probabilities={item['name']: relative(probs / f"{item['name']}.npy") for item in valid},
                    output_receipts=[receipt(path) for path in files])
    return model, manifest


def review(identifier):
    import cv2
    import numpy as np
    import torch
    from hybrid30 import install, predict_probs
    train = install()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    plan = read(OUT / 'plan.json')
    verify(plan)
    job = next(job for job in plan['jobs'] if job['id'] == identifier)
    training, valid = dataset(job)
    out = OUT / 'reviews' / identifier
    assert not out.exists(), str(out)
    out.mkdir(parents=True)
    args = job['args']
    expected = {'mit4': ('Unet', 'mit_b2', 60), 'unet3': ('UnetPlusPlus', 'resnet34', 120),
                'context4': ('GlobalContextMiTCNN30', 'mit_b2', 60)}[identifier]
    assert (args['arch'], args['encoder'], args['epochs']) == expected
    stages, rows = {}, {}
    model, manifest = verify_stage(args, job, training, valid)
    label = 'base120_original' if identifier == 'unet3' else 'original'
    stages[label] = manifest
    rows[label] = {item['name']: image_row(item, probability(ROOT / manifest['probabilities'][item['name']], item))
                   for item in valid}
    if identifier == 'unet3':
        del model
        continued = job['continuation']['args']
        assert continued['epochs'] == 6
        model, manifest = verify_stage(continued, job, training, valid, fresh_path(args['out']) / 'last.pt')
        stages['final126_original'] = manifest
        rows['final126_original'] = {
            item['name']: image_row(item, probability(ROOT / manifest['probabilities'][item['name']], item)) for item in valid}
        del model
    else:
        assert 'continuation' not in job
        cap = out / 'cap6_oof'
        cap.mkdir()
        model.cuda().eval()
        paths = {}
        for item in valid:
            name = item['name']
            path = cap / f'{name}.npy'
            if item['gray'].shape[1] == 675:
                with torch.no_grad(), train.autocast(args['amp']):
                    prob = predict_probs(model, item['gray'], args['width'], args['tile'], args['overlap'],
                        tta=True, device='cuda', scale_multiplier=args['scale_multiplier'], extra_scale=.75)
                np.save(path, prob.astype(np.float16))
            else:
                shutil.copyfile(ROOT / manifest['probabilities'][name], path)
            paths[name] = relative(path)
        del model
        torch.cuda.empty_cache()
        stages['cap6'] = dict(checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
            probabilities=paths, output_receipts=[receipt(ROOT / path) for path in paths.values()],
            width675_extra_scale=.75, other_widths='copy original probabilities', tta_views=8)
        rows['cap6'] = {item['name']: image_row(item, probability(ROOT / paths[item['name']], item)) for item in valid}
    verify(plan)
    result = dict(completed=now(), job=identifier, train_images=70, validation_images=20,
        epochs_total=126 if identifier == 'unet3' else 60, plan_sha256=sha(OUT / 'plan.json'),
        stages=stages, scores=scores(rows), training_names=sorted(item['name'] for item in training),
        validation_names=sorted(item['name'] for item in valid),
        shared_alloy_ids=sorted({item['alloy'] for item in training} & {item['alloy'] for item in valid}),
        historical_project_checkpoint_used=False, imagenet_initialization=True,
        same_new_run_continuation=identifier == 'unet3', previously_viewed_validation=True,
        independent_validation=False, test_images_or_labels_used=False,
        checkpoint_selection='fixed last epoch, never best.pt', automatic_submission_change=False)
    save(out / 'per_image_rows.json', rows)
    save(out / 'results.json', result)
    save(out / 'all.done', dict(time=now(), job=identifier, plan_sha256=sha(OUT / 'plan.json'),
        results_sha256=sha(out / 'results.json'), rows_sha256=sha(out / 'per_image_rows.json')))
    print(json.dumps(dict(job=identifier, scores=result['scores']), indent=2), flush=True)


def aggregate():
    import numpy as np
    plan = read(OUT / 'plan.json')
    verify(plan)
    assert not (OUT / 'results.json').exists() and not (OUT / 'all.done').exists()
    jobs = {job['id']: job for job in plan['jobs']}
    _, valid = dataset(jobs['mit4'])
    names = {item['name'] for item in valid}
    results, rows, done_receipts = {}, {}, {}
    for identifier in JOB_IDS:
        dataset(jobs[identifier])
        folder = OUT / 'reviews' / identifier
        done = read(folder / 'all.done')
        assert done['plan_sha256'] == sha(OUT / 'plan.json')
        assert done['results_sha256'] == sha(folder / 'results.json')
        assert done['rows_sha256'] == sha(folder / 'per_image_rows.json')
        result = results[identifier] = read(folder / 'results.json')
        assert result['plan_sha256'] == done['plan_sha256'] and result['validation_names'] == sorted(names)
        stored = read(folder / 'per_image_rows.json')
        for label, stage in result['stages'].items():
            for item in stage['output_receipts']:
                assert sha(fresh_path(item['path'])) == item['sha256'], item['path']
            assert set(stage['probabilities']) == names
            key = identifier + '_' + label
            rows[key] = {}
            for item in valid:
                name = item['name']
                row = image_row(item, probability(fresh_path(stage['probabilities'][name]), item))
                np.testing.assert_allclose(np.asarray(row['tta_pp'], dtype=float),
                    np.asarray(stored[label][name]['tta_pp'], dtype=float), rtol=0, atol=0, equal_nan=True)
                rows[key][name] = row
        done_receipts[identifier] = receipt(folder / 'all.done')
    recipes = {}
    for architecture in ('mit4', 'context4'):
        for inference in ('original', 'cap6'):
            for unet_stage in ('base120_original', 'final126_original'):
                label = f'{architecture}_{inference}_plus_unet3_{unet_stage}'
                recipes[label] = [(architecture, inference), ('unet3', unet_stage)]
                rows[label] = {}
                for item in valid:
                    values = [probability(fresh_path(results[job]['stages'][stage]['probabilities'][item['name']]), item)
                              for job, stage in recipes[label]]
                    rows[label][item['name']] = image_row(item, (values[0] + values[1]) / 2)
    pairs = {}
    for architecture in ('mit4', 'context4'):
        for inference in ('original', 'cap6'):
            prefix = f'{architecture}_{inference}_plus_unet3_'
            final, base = prefix + 'final126_original', prefix + 'base120_original'
            deltas = np.array([np.nanmean(np.asarray(rows[final][name]['tta_pp'], dtype=float)) -
                               np.nanmean(np.asarray(rows[base][name]['tta_pp'], dtype=float)) for name in sorted(names)]) * 100
            pairs[final + '_vs_base120'] = dict(mean_delta_pp=float(deltas.mean()),
                improved=int((deltas > 0).sum()), worsened=int((deltas < 0).sum()), unchanged=int((deltas == 0).sum()))
    result = dict(completed=now(), plan_sha256=sha(OUT / 'plan.json'), train_images=70, validation_images=20,
        scores=scores(rows), recipes={key: dict(models=value, weights=[.5, .5]) for key, value in recipes.items()},
        continuation_comparisons=pairs, job_completion_receipts=done_receipts,
        selected_recipe=None, automatic_model_selection=False, automatic_submission_change=False,
        historical_project_checkpoint_used=False, test_images_or_labels_used=False,
        previously_viewed_validation=True, independent_validation=False,
        metric='Mean across images of per-image class IoU; classes absent from both prediction and GT excluded.',
        sensitivity='both_absent_one_mIoU uses IoU=1 for a class absent in both; official convention unconfirmed.',
        probability_storage='float16, evaluated and combined in float32',
        notes=['Fresh train70 weights only. No earlier OOF probabilities or checkpoints are loaded.',
               'UNet++ continuation starts from this same new train70 run, for 120+6 total epochs.',
               'Validation20 was inspected during earlier architecture selection; this is not an independent test.',
               'Original provided split has shared alloy IDs; image files are disjoint.',
               'All ensemble recipes are fixed 50:50 soft voting; no per-image model choice.'])
    verify(plan)
    save(OUT / 'per_image_rows.json', rows)
    save(OUT / 'results.json', result)
    save(OUT / 'all.done', dict(time=now(), outcome='fresh_train70_valid20_complete',
        plan_sha256=sha(OUT / 'plan.json'), results_sha256=sha(OUT / 'results.json'),
        rows_sha256=sha(OUT / 'per_image_rows.json'), job_completion_receipts=done_receipts,
        automatic_model_selection=False, automatic_submission_change=False))
    print(json.dumps(dict(scores=result['scores'], selected_recipe=None), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--job', choices=JOB_IDS)
    mode.add_argument('--aggregate', action='store_true')
    arguments = parser.parse_args()
    aggregate() if arguments.aggregate else review(arguments.job)
