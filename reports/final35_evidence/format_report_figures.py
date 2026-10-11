"""Create compact scientific report figures from sealed evidence; CPU only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone, timedelta
ROOT=Path('/home/j-j15a106/komap')
OUT=ROOT/'reports/final35_evidence'
sys.path[:0]=[str(ROOT/'src'), str(ROOT/'scripts')]
os.chdir(ROOT)
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from data import load_split, CLASSES, MASK_RGB
from infer import postprocess
from review_submission31 import probability, image_row
cv2.setNumThreads(1)

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p): return Path(p).relative_to(ROOT).as_posix()
def export(fig, name):
    assert not (OUT/name).exists()
    fig.savefig(OUT/name,dpi=300,facecolor='white')
    plt.close(fig)

def main():
    assert not (OUT/'report_figures_receipt.json').exists()
    done=read(OUT/'all.done'); manifest={e['name']:e for e in done['files']}
    assert sha(OUT/'evidence_plan.json')==done['plan_sha256']
    for name in ['train70_eda.json','validation_cases.json','official20_per_image_rows.json']:
        assert sha(OUT/name)==manifest[name]['sha256']
    oldpins=read(OUT/'evidence_plan.json')['input_hashes']
    pins={rel(OUT/'all.done'):sha(OUT/'all.done'), rel(Path(__file__)):sha(Path(__file__))}
    def check(path):
        key=rel(path)
        if key in oldpins: assert sha(path)==oldpins[key],key
        else: assert path.parent==OUT and sha(path)==manifest[path.name]['sha256']
        pins[key]=sha(path)
    for name in ['src/data.py','src/infer.py','scripts/review_submission31.py']:
        check(ROOT/name)
    for name in ['train70_eda.json','validation_cases.json','official20_per_image_rows.json']:
        check(OUT/name)
    eda=read(OUT/'train70_eda.json')
    colors=MASK_RGB/255.
    labels=['Al','Al3Ni','Primary\nSi','Eutectic\nSi']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(5.6,4.35))
    fig.subplots_adjust(left=.105,right=.99,bottom=.125,top=.94,wspace=.35,hspace=.63)
    ax=axes[0,0];values=[eda['pooled_class_pixel_percent'][c] for c in CLASSES]
    ax.bar(labels,values,color=colors,edgecolor='#67717c',linewidth=.5)
    for i,v in enumerate(values):ax.text(i,v+.8,f'{v:.1f}',ha='center',fontsize=8)
    ax.set(title='A  Class area: train70',ylabel='Pooled pixel share (%)',ylim=(0,70))
    ax=axes[0,1];keys=list(eda['resolutions']);positions=np.arange(len(keys))
    ax.bar(positions,[eda['resolutions'][k] for k in keys],color='#537da4')
    for i,k in enumerate(keys):ax.text(i,eda['resolutions'][k]+.4,str(eda['resolutions'][k]),ha='center',fontsize=8)
    ax.set(xticks=positions,xticklabels=[k.replace('x','\nx') for k in keys],title='B  Original image sizes',ylabel='Images',ylim=(0,29))
    ax=axes[1,0]
    for i,r in enumerate(sorted(eda['per_image'],key=lambda r:(r['p1'],r['name']))):
        ax.plot([r['p1'],r['p99']],[i,i],color='#537da4',alpha=.7,linewidth=.7)
    ax.set(title='C  Raw p1 to p99 per image',xlabel='Gray value (0-255)',ylabel='Image rank by p1',xlim=(0,255),xticks=[0,64,128,192,255])
    ax=axes[1,1];values=[eda['pooled_class_boundary_percent'][c] for c in CLASSES]
    ax.bar(labels,values,color=colors,edgecolor='#67717c',linewidth=.5)
    for i,v in enumerate(values):ax.text(i,v+.9,f'{v:.1f}',ha='center',fontsize=8)
    ax.set(title='D  Near class boundaries*',ylabel='Share of each class (%)',ylim=(0,80))
    fig.text(.105,.018,'* Pixels whose 3x3 neighborhood includes another class.',fontsize=7.5)
    export(fig,'report_eda.png')

    cases=read(OUT/'validation_cases.json')
    sealed_rows=read(OUT/'official20_per_image_rows.json')['mit2_plus_unet3']
    p32=ROOT/'runs/review32_mit2_train70/review/results.json'
    p31=ROOT/'runs/submission31_train70/reviews/unet3/results.json'
    check(p32);check(p31)
    paths2=read(p32)['probabilities']
    paths3=read(p31)['stages']['base120_original']['probabilities']
    wanted={c['name'] for c in cases}
    data={x['name']:x for x in load_split('data','valid') if x['name'] in wanted}
    assert set(data)==wanted
    for item in data.values():
        for directory,suffix in [('images','image'),('masks','mask')]:
            check(ROOT/'data/valid'/directory/(item['name']+'_'+suffix+'.png'))
    fig=plt.figure(figsize=(5.6,3.65))
    grid=fig.add_gridspec(4,4,height_ratios=[.22,1,.22,1],
                         left=.02,right=.985,bottom=.13,top=.90,wspace=.035,hspace=.15)
    for col,title in enumerate(['Original','Ground truth','Prediction','Errors (red)']):
        fig.text(.14+col*.241,.958,title,ha='center',va='center',fontsize=8.5,weight='bold')
    for row,case in enumerate(cases):
        item=data[case['name']];name=item['name']
        check(ROOT/paths2[name]);check(ROOT/paths3[name])
        prob=(probability(ROOT/paths2[name],item)+probability(ROOT/paths3[name],item))/2
        pred=postprocess(prob)
        np.testing.assert_allclose(np.asarray(image_row(item,prob)['tta_pp'],dtype=float),
            np.asarray(sealed_rows[name]['tta_pp'],dtype=float),rtol=0,atol=0,equal_nan=True)
        err=pred!=item['lab'];error=np.full((*err.shape,3),245,dtype=np.uint8);error[err]=(210,55,50)
        titleax=fig.add_subplot(grid[row*2,:]);titleax.axis('off')
        tag='Lowest (rank 1/20)' if row==0 else 'Lower median (rank 10/20)'
        titleax.text(0,.4,f"{tag}: {name} | mIoU {case['mIoU']:.2f}%",va='center',fontsize=8)
        for col,panel in enumerate([item['gray'],MASK_RGB[item['lab']],MASK_RGB[pred],error]):
            ax=fig.add_subplot(grid[row*2+1,col])
            ax.imshow(panel,cmap='gray' if panel.ndim==2 else None,vmin=0 if panel.ndim==2 else None,vmax=255 if panel.ndim==2 else None)
            ax.axis('off')
    fig.legend(handles=[Patch(facecolor=colors[i],edgecolor='#555',label=c) for i,c in enumerate(CLASSES)],
        loc='lower center',bbox_to_anchor=(.5,.01),ncol=4,fontsize=8,columnspacing=1.2,handlelength=1.2,frameon=False)
    export(fig,'report_validation_cases.png')
    for path,digest in pins.items():assert sha(ROOT/path)==digest
    outputs=[dict(name=n,sha256=sha(OUT/n),bytes=(OUT/n).stat().st_size) for n in ['report_eda.png','report_validation_cases.png']]
    receipt=dict(created=datetime.now(timezone(timedelta(hours=9))).isoformat(),cpu_only=True,
        new_training=False,new_model_inference=False,input_hashes=pins,outputs=outputs,
        intended_width_inches=5.6,case_selection='Ascending validation20 image mIoU, ranks1 and10; ties by name',
        scores_identical_to_sealed_evidence=True)
    with (OUT/'report_figures_receipt.json').open('x',encoding='utf8') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
