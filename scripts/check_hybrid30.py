"""Numerical/geometry checks for the proposed gated and whole-view variants."""
import copy
import math
from unittest.mock import patch


def cpu_checks(reference):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from hybrid30 import GATE, GLOBAL, install
    from context30 import (BaseCropDataset, ContextCropDataset, ContextInput, dihedral_matrix,
                           transform_image, aligned_features, CONTEXT_SIDE)
    from infer import predict_probs as ordinary_predict
    from hybrid30 import predict_probs
    torch.set_num_threads(2)
    train = install()
    # All rotations/reflections, including non-square whole images and TTA order.
    gray = np.arange(7*11).reshape(7, 11)
    for first in (False, True):
        for k in range(4):
            for flip in (False, True):
                a = dihedral_matrix(7,11,k,flip,first)
                transformed = transform_image(gray,k,flip,first)
                for y in range(7):
                    for x in range(11):
                        xx,yy,_ = np.rint(a@np.array([x,y,1])).astype(int)
                        assert transformed[yy,xx] == gray[y,x]
    # Sample coordinate fields: catches half-pixel and convolution-stride shifts.
    yy,xx=torch.meshgrid(torch.arange(12)*32.,torch.arange(12)*32.,indexing='ij')
    feature=torch.stack([xx,yy])[None]
    roi_errors=[]
    for k in range(4):
        for flip in (False,True):
            d=dihedral_matrix(128,128,k,flip,True)
            theta=(np.array([[1,0,96],[0,1,96],[0,0,1]])@np.linalg.inv(d))[:2]
            actual=aligned_features(feature,torch.tensor(theta,dtype=torch.float32)[None],(4,4),32)[0].numpy()
            py,px=np.mgrid[:4,:4]*32
            expected=(theta@np.stack([px,py,np.ones_like(px)]).reshape(3,-1)).reshape(2,4,4)
            roi_errors.append(float(np.max(np.abs(actual-expected))))
    assert max(roi_errors)<1e-4
    rng=np.random.default_rng(7)
    items=[dict(gray=rng.integers(0,256,(160,224),dtype=np.uint8),lab=rng.integers(0,4,(160,224),dtype=np.uint8))]
    kw=dict(crop=128,samples_per_image=2,target_width=0,rot='rot90',scale_multiplier=1.)
    ordinary=BaseCropDataset(items,**kw); contextual=ContextCropDataset(items,**kw)
    altered=copy.deepcopy(items); altered[0]['lab']=(altered[0]['lab']+1)%4
    other=ContextCropDataset(altered,**kw)
    for seed in range(16):
        np.random.seed(seed); x,y=ordinary[0]; future=np.random.randint(1<<31)
        np.random.seed(seed); cx,cy=contextual[0]; cfuture=np.random.randint(1<<31)
        assert torch.equal(x,cx.crop) and torch.equal(y,cy) and future==cfuture
        assert cx.whole.shape==(1,CONTEXT_SIDE,CONTEXT_SIDE) and torch.isfinite(cx.whole).all()
        np.random.seed(seed); ox,oy=other[0]
        assert torch.equal(cx.whole,ox.whole) and torch.equal(cx.crop_to_whole,ox.crop_to_whole)
        assert torch.equal(cx.crop,ox.crop) and not torch.equal(cy,oy)
    batch=next(iter(DataLoader(contextual,batch_size=2,num_workers=0)))
    assert isinstance(batch[0],ContextInput) and batch[0].shape==(2,1,128,128)
    assert batch[0].crop_to_whole.shape==(2,2,3)
    proofs={}
    for arch in (GATE,GLOBAL):
        torch.manual_seed(0)
        base=train.build_model('HybridMiTCNN29','mit_b2',weights='imagenet').eval()
        base_rng=torch.get_rng_state().clone()
        torch.manual_seed(0)
        a=dict(reference,arch=arch)
        with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('Factory reseeded CUDA')):
            candidate=train.build_from_args(a,weights='imagenet').eval()
        assert torch.equal(base_rng,torch.get_rng_state())
        own=candidate.state_dict()
        assert all(torch.equal(v,own[n]) for n,v in base.state_dict().items())
        for shape in [(2,1,128,128),(1,1,128,160)]:
            x=torch.randn(shape)
            inputs=ContextInput(x,torch.randn(shape[0],1,CONTEXT_SIDE,CONTEXT_SIDE),
                torch.tensor([[[1.,0,96],[0,1,64]]]).expand(shape[0],-1,-1)) if arch==GLOBAL else x
            with torch.no_grad(): assert torch.equal(base(x),candidate(inputs))
        test_image=rng.integers(0,256,(53,91),dtype=np.uint8)
        for tta,extra in [(False,1.),(True,1.25)]:
            with torch.no_grad():
                p=ordinary_predict(base,test_image,0,64,16,tta,'cpu',extra_scale=extra)
                q=predict_probs(candidate,test_image,0,64,16,tta,'cpu',extra_scale=extra)
            assert np.max(np.abs(p-q))<1e-6 and np.max(np.abs(q.sum(0)-1))<1e-5
        candidate.train(); opt=torch.optim.SGD(candidate.parameters(),lr=.01)
        losses=[]; gradients={}
        groups=['encoder.mit.','encoder.stem.','encoder.stages.','encoder.projections.','decoder.']
        groups += ['encoder.gates.'] if arch==GATE else ['encoder.whole_stem.','encoder.whole_stages.','encoder.context_projections.']
        for step in range(4):
            x=torch.randn(2,1,128,128); y=torch.randint(0,4,(2,128,128))
            inputs=ContextInput(x,torch.randn(2,1,CONTEXT_SIDE,CONTEXT_SIDE),
                torch.tensor([[[1.,0,96],[0,1,64]]]).expand(2,-1,-1)) if arch==GLOBAL else x
            opt.zero_grad(); loss=torch.nn.functional.cross_entropy(candidate(inputs),y);loss.backward()
            assert math.isfinite(loss.item())
            assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
            if step==3:
                for prefix in groups:
                    gradients[prefix]=sum(float(p.grad.abs().sum()) for n,p in candidate.named_parameters() if n.startswith(prefix) and p.grad is not None)
                    assert gradients[prefix]>0,(arch,prefix)
            opt.step();losses.append(loss.item())
        clone=train.build_from_args(a,weights=None);clone.load_state_dict(candidate.state_dict(),strict=True)
        proofs[arch]=dict(parameters=sum(p.numel() for p in candidate.parameters()),
            extra_over_hybrid29=sum(p.numel() for p in candidate.parameters())-sum(p.numel() for p in base.parameters()),
            base_tensors_and_initial_logits_exact=True,cpu_rng_preserved=True,cuda_not_reseeded=True,
            full_inference_zero_branch_equivalence=True,strict_reload=True,synthetic_losses=losses,
            final_gradient_l1=gradients)
        del base,candidate,clone,opt
    return dict(architectures=proofs,all_dihedral_pixel_maps_passed=True,roi_feature_centres_max_error=max(roi_errors),
                local_crops_labels_and_rng_identical_seeds=16,context_does_not_read_masks=True,
                context_batch_collation=True,test_dataset_images_loaded=False)
