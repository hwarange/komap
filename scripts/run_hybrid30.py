"""Run the three approved spare-GPU experiments; keep GPU0/Hybrid29 intact."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/review30_hybrid_extensions'
BASE='x4_mit_c1536_bw_r90'
SOURCES=['src/context30.py','src/hybrid30.py','src/train_hybrid30.py','scripts/check_hybrid30.py',
         'scripts/run_hybrid30.py','scripts/review_hybrid30.py','src/hybrid29.py','src/train_hybrid29.py',
         'src/train.py','src/data.py','src/infer.py','scripts/check_scale24_runtime.py',
         'scripts/run_crop18.py','src/review_grains15.py']
CONFIGS={
    'gated_fold0':('GatedMiTCNN30',0,1,'GPU-3fc53ad0-271b-ffd6-c83c-84575e82d19c','x4_gated_mit_cnn_c1536_bw_r90'),
    'context_fold0':('GlobalContextMiTCNN30',0,2,'GPU-e800362c-2a5a-0208-8d32-28107bca6017','x4_context_mit_cnn_c1536_bw_r90'),
    'hybrid_fold1':('HybridMiTCNN29',1,3,'GPU-dce5fed1-b1cd-7a95-9c1b-1f281a29f713','x4_hybrid_mit_cnn_c1536_bw_r90'),
}
ENV=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         NUMEXPR_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
         PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')


def now(): return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify(plan):
    for name,digest in plan['input_hashes'].items(): assert sha(ROOT/name)==digest,name


def idle(job):
    raw=subprocess.check_output(['nvidia-smi','-i',job['gpu_uuid'],
        '--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    index,uuid,name,used,util=map(str.strip,raw.split(','))
    assert int(index)==job['gpu_index'] and uuid==job['gpu_uuid'] and 'L40S' in name and int(used)<1000 and int(util)<10,raw
    return raw


def logs(job):
    p=ROOT/job['args']['out']/'log.jsonl'
    if not p.exists():return []
    data=p.read_text();lines=data.split('\n')[:-1]
    return [json.loads(s) for s in lines if s]


def prepare():
    from data import load_cv
    from hybrid30 import architecture_spec
    assert not OUT.exists()
    old=read(ROOT/'runs/review29_hybrid_pilot/plan.json')
    # The previously validated model/data files remain pinned while GPU0 runs.
    for n,s in old['input_hashes'].items():assert sha(ROOT/n)==s,n
    proof=read(ROOT/'runs/hybrid30_cpu_precheck.json')
    assert proof['context_does_not_read_masks'] and proof['all_dihedral_pixel_maps_passed']
    assert proof['local_crops_labels_and_rng_identical_seeds']==16
    for v in proof['architectures'].values():
        assert v['base_tensors_and_initial_logits_exact'] and v['strict_reload'] and all(x>0 for x in v['final_gradient_l1'].values())
    files={ROOT/n for n in SOURCES}
    files.update([ROOT/'splits/folds5.json',ROOT/'runs/hybrid30_cpu_precheck.json',
        ROOT/'runs/review29_hybrid_pilot/plan.json',ROOT/'runs/review27_finetune_full/results.json',
        ROOT/'runs/review27_finetune_full/per_image_rows.json',ROOT/'runs/review26_final_strengthening/scale_trials/results.json'])
    scale=read(ROOT/'runs/review26_final_strengthening/scale_trials/results.json')
    jobs=[]
    for identifier,(arch,fold,index,uuid,experiment) in CONFIGS.items():
        reference_path=ROOT/f'runs/cv/{BASE}/fold{fold}/args.json'
        reference=read(reference_path)
        assert all(reference[k]==v for k,v in dict(arch='Unet',encoder='mit_b2',epochs=60,fold=fold,seed=0,
            crop=1536,tile=1536,width=2560,scale_multiplier=2.,bs=2,accum_steps=2,spi=8,bw=2.,init='',hires=False,full=False).items())
        a=dict(reference,arch=arch,out=f'runs/cv/{experiment}/fold{fold}',save_oof=f'runs/oof/{experiment}')
        assert {k for k in a if a[k]!=reference[k]}=={'arch','out','save_oof'}
        assert not (ROOT/a['out']).exists()
        training,valid=load_cv('data',fold,'splits/folds5.json')
        assert len(training)==72 and len(valid)==18
        assert {x['alloy'] for x in training}.isdisjoint({x['alloy'] for x in valid})
        cap={r['name']:r for r in scale['output_receipts'] if r['arm']=='cap6' and r['fold']==fold}
        assert set(cap)=={x['name'] for x in valid if x['gray'].shape[1]==675}
        udir='runs/review26_final_strengthening/finetune_plan/oof/interface' if fold==0 else 'runs/review27_finetune_full/oof/interface/fold1'
        for item in training+valid:
            files.update(ROOT/f"data/{item['split']}/{kind}/{item['name']}_{suffix}.png" for kind,suffix in [('images','image'),('masks','mask')])
        for item in valid:
            n=item['name']
            assert not (ROOT/a['save_oof']/f'{n}.npy').exists()
            files.update([ROOT/f'runs/oof/{BASE}/{n}.npy',ROOT/udir/f'{n}.npy'])
        files.update([reference_path,ROOT/f'runs/cv/{BASE}/fold{fold}/result.json',ROOT/f'runs/cv/{BASE}/fold{fold}/last.pt'])
        files.update(ROOT/r['path'] for r in cap.values())
        job=dict(id=identifier,fold=fold,args=a,reference_args=reference,gpu_index=index,gpu_uuid=uuid,
            training_names=sorted(x['name'] for x in training),validation_names=sorted(x['name'] for x in valid),
            interface_oof=udir,cap6_control_records=cap,architecture=architecture_spec(arch),
            entrypoint='src/train_hybrid29.py' if arch=='HybridMiTCNN29' else 'src/train_hybrid30.py')
        idle(job);jobs.append(job)
    # Cached weights pinned by the existing, already-running experiment.
    for row in old['image_net_cache']:assert sha(row['path'])==row['sha256']
    plan=dict(created=now(),jobs=jobs,cpu_validation=proof,image_net_cache=old['image_net_cache'],
        source_hashes={n:sha(ROOT/n) for n in SOURCES},input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        epochs=60,fresh_imagenet=True,seed=0,init_checkpoint=None,test_images_or_labels_used=False,
        inference_recipes=['original','cap6_width675'],ensemble_weights=[.5,.5],
        existing_gpu0_plan_sha256=sha(ROOT/'runs/review29_hybrid_pilot/plan.json'),
        analysis='Gated/context fold0 compared with Hybrid29 and stored MiT on the same18. Hybrid fold1 plus existing fold0 gives36-image comparison.',
        previously_viewed_validation=True,automatic_remaining_folds=[],automatic_submission_change=False,
        maximum_controller_hours=10,comparison_limit='Context adds whole-view inputs and small CNN capacity; not a capacity-matched context-only effect.')
    OUT.mkdir();save('plan.json',plan)
    print(json.dumps(dict(prepared=True,jobs=[j['id'] for j in jobs],input_hashes=len(plan['input_hashes']),plan_sha256=sha(OUT/'plan.json')),indent=2))


def benchmark(identifier):
    from hybrid30 import install
    from check_scale24_runtime import measure
    plan=read(OUT/'plan.json');verify(plan)
    job=next(j for j in plan['jobs'] if j['id']==identifier)
    dest=OUT/f'runtime/{identifier}.json';assert not dest.exists() and not (OUT/'started.json').exists()
    gpu=idle(job);install(job['args']['arch'])
    result=measure(job['args'],warmup=2,updates=4)
    result.update(job=identifier,gpu=gpu,plan_sha256=sha(OUT/'plan.json'),temporary_training_not_reused=True)
    save(f'runtime/{identifier}.json',result)
    print(json.dumps(dict(job=identifier,estimated_total_minutes=result['estimated_total_minutes'],
        peak_train_memory_allocated_gib=result['peak_train_memory_allocated_gib'],all_losses_finite=result['all_losses_finite']),indent=2))


def run():
    from run_crop18 import cli_args
    plan=read(OUT/'plan.json');verify(plan)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not any((OUT/n).exists() for n in ['started.json','all.done','error.json'])
    for job in plan['jobs']:
        idle(job);assert not (ROOT/job['args']['out']).exists()
        runtime=read(OUT/f"runtime/{job['id']}.json")
        assert runtime['all_losses_finite'] and runtime['plan_sha256']==sha(OUT/'plan.json')
    children={};handles=[];status={};started=time.monotonic()
    def stop(*unused): raise RuntimeError('Controller interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def spawn(job,stage):
        identifier=job['id']
        command=[str(ROOT/'.venv/bin/python'),job['entrypoint'],*cli_args(job['args'])] if stage=='training' else [
            str(ROOT/'.venv/bin/python'),'scripts/review_hybrid30.py','--job',identifier]
        stream=(OUT/f'{identifier}_{stage}.log').open('x');handles.append(stream)
        process=subprocess.Popen(command,cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES=job['gpu_uuid']),
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        children[identifier]=process
        status[identifier]=dict(stage=stage,pid=process.pid,gpu_index=job['gpu_index'],started=now(),command=command)
    try:
        for job in plan['jobs']:spawn(job,'training')
        save('started.json',dict(time=now(),controller_pid=os.getpid(),jobs=status,plan_sha256=sha(OUT/'plan.json'),
            runtime_hashes={j['id']:sha(OUT/f"runtime/{j['id']}.json") for j in plan['jobs']}))
        while not all(x['stage'] in ('complete','failed') for x in status.values()):
            assert time.monotonic()-started<plan['maximum_controller_hours']*3600,'Controller execution bound exceeded'
            for job in plan['jobs']:
                identifier=job['id'];st=status[identifier];process=children[identifier]
                if st['stage'] in ('complete','failed'):continue
                rows=logs(job);st.update(completed_epochs=len(rows),latest_epoch=rows[-1] if rows else None)
                assert all(math.isfinite(x['loss']) for x in rows),(identifier,'nonfinite loss')
                if process.poll() is None:continue
                if process.returncode!=0:
                    st.update(stage='failed',exit_code=process.returncode,finished=now());continue
                if st['stage']=='training':
                    assert len(rows)==60;spawn(job,'review')
                else:
                    assert (OUT/f'reviews/{identifier}/all.done').exists()
                    st.update(stage='complete',finished=now(),completed_epochs=60)
            save('state.json',dict(time=now(),controller_pid=os.getpid(),jobs=status,elapsed_minutes=(time.monotonic()-started)/60))
            if not all(x['stage'] in ('complete','failed') for x in status.values()):time.sleep(20)
        assert all(x['stage']=='complete' for x in status.values()),'One or more jobs failed; see per-job logs'
        # GPU0's independent controller is left untouched. Its result is needed
        # only to compare gate/context against Hybrid29 and combine fold0+1.
        while not (ROOT/'runs/review29_hybrid_pilot/all.done').exists():
            assert not (ROOT/'runs/review29_hybrid_pilot/error.json').exists(),'Existing Hybrid29 did not finish successfully'
            assert time.monotonic()-started<plan['maximum_controller_hours']*3600
            save('state.json',dict(time=now(),stage='waiting_existing_hybrid29_review',jobs=status));time.sleep(20)
        verify(plan)
        with (OUT/'aggregate.log').open('x') as stream:
            subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/review_hybrid30.py','--aggregate'],cwd=ROOT,
                env=dict(ENV,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
        assert (OUT/'all.done').exists()
        save('state.json',dict(time=now(),stage='complete_and_stop',jobs=status,automatic_remaining_folds=[]))
    except BaseException as error:
        save('error.json',dict(time=now(),error=f'{type(error).__name__}: {error}',jobs=status));raise
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for stream in handles:stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','benchmark','run'])
    parser.add_argument('--job',choices=list(CONFIGS));opts=parser.parse_args()
    os.chdir(ROOT);sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
    os.environ.update(ENV)
    os.environ['CUDA_VISIBLE_DEVICES']=CONFIGS[opts.job][3] if opts.mode=='benchmark' and opts.job else ''
    if opts.mode=='benchmark':assert opts.job;benchmark(opts.job)
    elif opts.mode=='prepare':prepare()
    else:run()


if __name__=='__main__':main()
