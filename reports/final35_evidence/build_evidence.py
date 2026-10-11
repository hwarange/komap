"""Recompute training70 EDA and official-validation20 report evidence on CPU."""
import os
from pathlib import Path
import sys
import hashlib
import json
from datetime import datetime, timezone, timedelta

ROOT=Path('/home/j-j15a106/komap')
OUT=ROOT/'reports/final35_evidence'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.chdir(ROOT)
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from data import load_split,CLASSES,MASK_RGB
from infer import image_iou,postprocess
from review_submission31 import probability,image_row,scores
cv2.setNumThreads(1)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return Path(path).relative_to(ROOT).as_posix()
def clean(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value
def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as stream:json.dump(clean(value),stream,indent=2,ensure_ascii=False,allow_nan=False)
def summary(values):
    return {k:float(v) for k,v in zip(['min','q25','median','q75','max'],np.percentile(values,[0,25,50,75,100]))}
def complete(folder,plan,pins):
    done=read(folder/'all.done')
    assert done['plan_sha256']==sha(plan)
    for key,name in [('results_sha256','results.json'),('rows_sha256','per_image_rows.json')]:assert done[key]==sha(folder/name)
    for name in ['all.done','results.json','per_image_rows.json']:pins[rel(folder/name)]=sha(folder/name)
    pins[rel(plan)]=sha(plan)
    return read(folder/'results.json'),read(folder/'per_image_rows.json')
def export(fig,name):
    path=OUT/name;assert not path.exists()
    fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    assert not (OUT/'all.done').exists() and not (OUT/'train70_eda.json').exists()
    pins={rel(Path(__file__)):sha(Path(__file__))}
    for name in ['src/data.py','src/infer.py','scripts/review_submission31.py']:pins[name]=sha(ROOT/name)
    training,validation=load_split('data','train'),load_split('data','valid')
    assert len(training)==70 and len(validation)==20
    assert {x['name'] for x in training}.isdisjoint(x['name'] for x in validation)
    for item in training+validation:
        assert item['lab'] is not None and np.isin(item['lab'],range(4)).all()
        for kind,suffix in [('images','image'),('masks','mask')]:
            path=ROOT/'data'/item['split']/kind/(item['name']+'_'+suffix+'.png');pins[rel(path)]=sha(path)
    eda_rows=[];total=np.zeros(4,dtype=np.int64);edge_class=np.zeros(4,dtype=np.int64)
    resolutions={}
    for item in training:
        gray,lab=item['gray'],item['lab'];counts=np.bincount(lab.ravel(),minlength=4)
        total+=counts;edge=cv2.dilate(lab,np.ones((3,3),np.uint8))!=cv2.erode(lab,np.ones((3,3),np.uint8))
        edge_counts=np.asarray([int(((lab==i)&edge).sum()) for i in range(4)]);edge_class+=edge_counts
        p1,p99=np.percentile(gray,[1,99]);shape=f'{gray.shape[1]}x{gray.shape[0]}'
        resolutions[shape]=resolutions.get(shape,0)+1
        eda_rows.append(dict(name=item['name'],alloy=int(item['alloy']),step=int(item['step']),width=gray.shape[1],height=gray.shape[0],
            pixels=int(gray.size),gray_mean=float(gray.mean()),gray_std=float(gray.std()),p1=float(p1),p99=float(p99),
            contrast_span_p99_minus_p1=float(p99-p1),low_contrast_proxy=bool(item['low_contrast']),
            class_pixels={c:int(counts[i]) for i,c in enumerate(CLASSES)},
            class_pixel_percent={c:float(counts[i]/gray.size*100) for i,c in enumerate(CLASSES)},
            absent_classes=[c for i,c in enumerate(CLASSES) if counts[i]==0],
            class_boundary_percent={c:float(edge_counts[i]/counts[i]*100) if counts[i] else None for i,c in enumerate(CLASSES)}))
    train_alloys={x['alloy'] for x in training};valid_alloys={x['alloy'] for x in validation}
    eda=dict(scope='Provided train70 images and masks only',images=70,alloys=len(train_alloys),
        total_pixels=int(total.sum()),resolutions=resolutions,
        pooled_class_pixel_percent={c:float(total[i]/total.sum()*100) for i,c in enumerate(CLASSES)},
        image_mean_class_pixel_percent={c:float(np.mean([r['class_pixel_percent'][c] for r in eda_rows])) for c in CLASSES},
        class_absent_image_count={c:sum(c in r['absent_classes'] for r in eda_rows) for c in CLASSES},
        class_absent_names={c:[r['name'] for r in eda_rows if c in r['absent_classes']] for c in CLASSES},
        gray_mean=summary([r['gray_mean'] for r in eda_rows]),gray_std=summary([r['gray_std'] for r in eda_rows]),
        p1=summary([r['p1'] for r in eda_rows]),p99=summary([r['p99'] for r in eda_rows]),
        contrast_span=summary([r['contrast_span_p99_minus_p1'] for r in eda_rows]),
        low_contrast_proxy_count=sum(r['low_contrast_proxy'] for r in eda_rows),
        low_contrast_proxy_definition='Existing data.py heuristic p1>60; a proxy, not a calibrated contrast label.',
        pooled_class_boundary_percent={c:float(edge_class[i]/total[i]*100) for i,c in enumerate(CLASSES)},
        boundary_definition='Pixels whose3x3 neighbourhood contains another class; includes both sides of a class boundary.',
        normalization='Per image: clip((gray-p1)/max(p99-p1,0.001),0,1), then model input=(value-0.5)/0.25.',
        step_counts={str(s):sum(r['step']==s for r in eda_rows) for s in sorted({r['step'] for r in eda_rows})},
        per_image=eda_rows)
    split=dict(scope='Provided train70 versus validation20; only group metadata comparison',training_images=70,validation_images=20,
        training_alloys=len(train_alloys),validation_alloys=len(valid_alloys),shared_alloy_ids=sorted(train_alloys&valid_alloys),
        shared_alloy_count=len(train_alloys&valid_alloys),shared_image_names=[],
        note='Original split has shared alloy IDs; grouped train70 CV prevents alloy overlap within each fold.')
    save('train70_eda.json',eda);save('original_split_groups.json',split)
    p31=ROOT/'runs/submission31_train70';p32=ROOT/'runs/review32_mit2_train70';p33=ROOT/'runs/review33_mit2_pair'
    result31,rows31=complete(p31,p31/'plan.json',pins)
    result32,rows32=complete(p32/'review',p32/'plan.json',pins)
    result33,rows33=complete(p33,p33/'plan.json',pins)
    paths={'mit2':result32['probabilities']};receipts=list(result32['output_receipts'])
    model_args={'mit2':result32['args']}
    for family,key in [('mit4','original'),('unet3','base120_original'),('context4','original')]:
        result,_=complete(p31/'reviews'/family,p31/'plan.json',pins)
        stage=result['stages'][key];paths[family]=stage['probabilities'];receipts+=stage['output_receipts'];model_args[family]=stage['args']
    for entry in receipts:assert sha(ROOT/entry['path'])==entry['sha256'];pins[entry['path']]=entry['sha256']
    names={x['name'] for x in validation}
    assert all(set(p)==names for p in paths.values())
    recipes={'mit2_plus_unet3':('mit2','unet3'),'mit4_plus_unet3':('mit4','unet3'),'context4_plus_unet3':('context4','unet3')}
    records={key:{} for key in [*paths,*recipes]};selected_predictions={}
    for item in validation:
        name=item['name'];probs={key:probability(ROOT/rows[name],item) for key,rows in paths.items()}
        for key,prob in probs.items():records[key][name]=image_row(item,prob)
        for key,(a,b) in recipes.items():records[key][name]=image_row(item,(probs[a]+probs[b])/2)
        selected_predictions[name]=postprocess((probs['mit2']+probs['unet3'])/2)
        np.testing.assert_allclose(np.asarray(records['mit2_plus_unet3'][name]['tta_pp'],dtype=float),
            np.asarray(rows33['mit2_plus_unet120'][name]['tta_pp'],dtype=float),rtol=0,atol=0,equal_nan=True)
    metrics=scores(records)
    references={'mit2':(result33,'mit2'),'unet3':(result33,'unet120'),'mit4':(result31,'mit4_original'),
        'context4':(result31,'context4_original'),'mit2_plus_unet3':(result33,'mit2_plus_unet120'),
        'mit4_plus_unet3':(result31,'mit4_original_plus_unet3_base120_original'),
        'context4_plus_unet3':(result31,'context4_original_plus_unet3_base120_original')}
    for key,(source,label) in references.items():assert abs(metrics[key]['mIoU']-source['scores'][label]['mIoU'])<1e-12
    ranked=sorted([dict(name=item['name'],mIoU=float(np.nanmean(np.asarray(records['mit2_plus_unet3'][item['name']]['tta_pp'],dtype=float))*100),
        class_IoU={c:records['mit2_plus_unet3'][item['name']]['tta_pp'][i]*100 for i,c in enumerate(CLASSES)}) for item in validation],key=lambda r:(r['mIoU'],r['name']))
    official=dict(scope='Fresh models trained on original train70; evaluation on original validation20',
        scores=metrics,model_args=model_args,selected_recipe='mit2_plus_unet3',weights=[.5,.5],
        selected_per_image_ascending=ranked,all_scores_exactly_reproduced=True,
        metric='Image-wise class IoU mean, then mean across20images; class absent in prediction andGT excluded.',
        limitations=['Validation20 was previously inspected during model development; not an untouched test set.',
                    'MiT/Context60epochs,UNet120epochs; these are recipe comparisons, not matched-epoch architecture ablations.'])
    save('official20_metrics.json',official);save('official20_per_image_rows.json',records)
    p34=ROOT/'runs/review34_train70_group5fold'
    group,_=complete(p34,p34/'plan.json',pins)
    save('train70_groupcv_metrics.json',dict(scope='Internal grouped5fold CV using train70 only:56train/14heldout per fold',
        scores=group['scores'],fold_statistics=group['fold_statistics'],comparisons=group['comparisons'],
        note='Separate evaluation from the officialvalidation20; do not combine scores or describe their difference as improvement.'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=MASK_RGB/255.;display=['Al','Al3Ni','Primary Si','Eutectic Si']
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.3),layout='constrained')
    ax=axes[0,0];v=[eda['pooled_class_pixel_percent'][c] for c in CLASSES];ax.bar(display,v,color=colors,edgecolor='#67717c',linewidth=.5)
    for x,y in enumerate(v):ax.text(x,y+.8,f'{y:.1f}%',ha='center',fontsize=8)
    ax.set(title='A  Class area in training70',ylabel='Pooled pixel share (%)',ylim=(0,max(v)*1.2))
    ax=axes[0,1];keys=list(resolutions);ax.bar(keys,[resolutions[k] for k in keys],color='#537da4')
    for i,k in enumerate(keys):ax.text(i,resolutions[k]+.5,str(resolutions[k]),ha='center')
    ax.set(title='B  Original image sizes',xlabel='Width x height (pixels)',ylabel='Number of images');ax.tick_params(axis='x',labelsize=8)
    ax=axes[1,0];ordered=sorted(eda_rows,key=lambda r:(r['p1'],r['name']))
    for i,r in enumerate(ordered):ax.plot([r['p1'],r['p99']],[i,i],color='#537da4',alpha=.6,linewidth=1)
    ax.set(title='C  Raw luminance range per image',xlabel='Gray value: p1 to p99',ylabel='Images sorted by p1',xlim=(0,255))
    ax=axes[1,1];vals=[eda['pooled_class_boundary_percent'][c] for c in CLASSES];ax.bar(display,vals,color=colors,edgecolor='#67717c',linewidth=.5)
    for x,y in enumerate(vals):ax.text(x,y+1,f'{y:.1f}%',ha='center',fontsize=8)
    ax.set(title='D  Pixels near class boundaries',ylabel='Share of each class (%)',ylim=(0,100))
    export(fig,'train70_eda.png')
    order=['mit2','mit4','unet3','context4','mit4_plus_unet3','context4_plus_unet3','mit2_plus_unet3']
    labels=['MiT2 U-Net','MiT4 U-Net','UNet++3','Context4','MiT4 + UNet++3','Context4 + UNet++3','MiT2 + UNet++3']
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.8),gridspec_kw={'width_ratios':[1.6,1]},layout='constrained')
    values=[metrics[k]['mIoU'] for k in order];axes[0].barh(labels,values,color=['#7395b6']*4+['#a7b9c9']*2+['#168c83'])
    for i,v in enumerate(values):axes[0].text(v+.025,i,f'{v:.3f}',va='center',fontsize=8)
    axes[0].set(xlim=(77,81.35),xlabel='mIoU (%)',title='A  Same original validation20');axes[0].invert_yaxis()
    vals=[metrics['mit2_plus_unet3']['class_IoU'][c] for c in CLASSES];axes[1].bar(display,vals,color=colors,edgecolor='#67717c',linewidth=.5)
    for i,v in enumerate(vals):axes[1].text(i,v+1,f'{v:.2f}',ha='center',fontsize=8)
    axes[1].set(ylim=(0,105),ylabel='Class IoU (%)',title='B  Selected pair');axes[1].tick_params(axis='x',rotation=25)
    export(fig,'official20_scores.png')
    fig,ax=plt.subplots(figsize=(7.6,3.4),layout='constrained')
    for key,label,color in [('mit2_plus_unet3','MiT2 + UNet++3','#168c83'),('mit4_plus_unet3','MiT4 + UNet++3','#537da4'),('context4_plus_unet3','Context4 + UNet++3','#bb7143')]:
        values=[group['fold_statistics'][key]['by_fold'][str(f)]['mIoU'] for f in range(5)]
        ax.plot(range(1,6),values,'o-',color=color,label=f"{label} (OOF {group['scores'][key]['mIoU']:.3f}%)")
    ax.set(xticks=range(1,6),xlabel='Group fold (14 held-out training images each)',ylabel='mIoU (%)',title='Internal train70 grouped CV: separate from validation20')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,loc='upper left')
    export(fig,'train70_groupcv.png')
    cases=[]
    for rank,tag in [(1,'lowest'),(10,'lower_median')]:
        selected=ranked[rank-1];item=next(x for x in validation if x['name']==selected['name']);pred=selected_predictions[item['name']]
        error=pred!=item['lab'];error_rgb=np.full((*error.shape,3),245,dtype=np.uint8);error_rgb[error]=(210,55,50)
        panels=[item['gray'],MASK_RGB[item['lab']],MASK_RGB[pred],error_rgb]
        fig,axes=plt.subplots(1,4,figsize=(12,3.2),layout='constrained')
        for ax,panel,title in zip(axes,panels,['Original','Ground truth','Prediction','Pixel errors (red)']):
            ax.imshow(panel,cmap='gray' if panel.ndim==2 else None,vmin=0 if panel.ndim==2 else None,vmax=255 if panel.ndim==2 else None)
            ax.set_title(title,fontsize=10);ax.axis('off')
        fig.suptitle(f"Validation20 rank {rank}/20: {item['name']} | mIoU {selected['mIoU']:.2f}%",fontsize=11)
        fig.legend(handles=[Patch(facecolor=colors[i],edgecolor='#555',label=c) for i,c in enumerate(CLASSES)],loc='lower center',ncol=4,bbox_to_anchor=(.5,-.04),fontsize=9)
        filename=f'validation_case_{tag}.png';export(fig,filename)
        cases.append(dict(rank_ascending=rank,selection_rule='Ascending per-image mIoU; name breaks ties; rank1 andrank10 fixed before inspecting panels.',
            name=item['name'],mIoU=selected['mIoU'],class_IoU=selected['class_IoU'],width=int(item['gray'].shape[1]),height=int(item['gray'].shape[0]),
            error_pixel_percent=float(error.mean()*100),figure=filename,
            absent_ground_truth_classes=[c for i,c in enumerate(CLASSES) if not (item['lab']==i).any()]))
    save('validation_cases.json',cases)
    source_plan=dict(created=datetime.now(timezone(timedelta(hours=9))).isoformat(),input_hashes=pins,
        train70_eda_only=True,official20_only_for_validation_metrics_and_cases=True,test_images_read=False,
        cpu_only=True,new_training=False,new_model_inference=False,case_selection='Lowest andlower-median validation20 mIoU',
        no_raw_data_download='Only aggregate JSON and report composites are transferred; no standalone source images or masks.')
    save('evidence_plan.json',source_plan)
    for name,digest in pins.items():assert sha(ROOT/name)==digest,name
    outputs=[p for p in OUT.iterdir() if p.is_file() and p.name not in ['all.done']]
    done=dict(completed=datetime.now(timezone(timedelta(hours=9))).isoformat(),cpu_only=True,
        plan_sha256=sha(OUT/'evidence_plan.json'),files=[dict(name=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(outputs)],
        official20_selected_mIoU=metrics['mit2_plus_unet3']['mIoU'],train70_groupcv_mIoU=group['scores']['mit2_plus_unet3']['mIoU'])
    save('all.done',done)
    print(json.dumps(dict(eda={k:v for k,v in eda.items() if k!='per_image'},split=split,
        scores=metrics,cases=cases,files=[p.name for p in outputs]),indent=2),flush=True)

if __name__=='__main__':main()
