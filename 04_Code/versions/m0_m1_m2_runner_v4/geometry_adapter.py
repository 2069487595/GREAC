"""Common external-land initialization plus O anchors; no reference data access."""
from pathlib import Path
import os,sys,hashlib
import numpy as np
from scipy import ndimage as ndi

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SNAP=ROOT/'04_Code/versions/Tao-original/FROZEN-20260921'
sys.path.insert(0,str(ROOT/'03_Data_Processed/tibet_three_lakes_v1/compute_runtime'))
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'04_Code/versions/.runtime_cache/m0_m1_m2_runner_v4')
from numba import njit
sys.path.insert(0,str(SNAP/'project/04_Code/scripts'))
import tibet_tao_kernel_v2 as kernel
import tibet_three_group_run_v3 as tao_v3
tao_run=tao_v3.run

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def initialize(z,R):
    if R[:3].any() or R[-3:].any() or R[:,:3].any() or R[:,-3:].any():
        raise ValueError('R requires 3-cell padding')
    land=np.array(z,dtype=float,copy=True);land[R]=np.nan
    dem,known,slope,info=tao_run.initialize(land,R)
    shore=known&ndi.binary_dilation(R,np.ones((3,3)))
    if not shore.any():raise ValueError('NO_EXTERNAL_LAND')
    theta=float(info['buffer_slope_deg'])
    if not np.isfinite(theta) or theta<=0:raise ValueError('LAND_SLOPE_UNIDENTIFIABLE')
    return dict(dem=dem,known=known,slope=slope,info=info,h0=float(np.median(dem[shore])),
                alpha_rule=1. if theta<=5 else 2.,domain_hash=digest(R))

@njit(cache=True)
def advance(points,pending,water,origin,known,dem,slope,alpha):
    offsets=np.array([[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[-1,1],[1,-1],[1,1]])
    count=0;degenerate=0
    for rr,cc in points:
        if not pending[rr,cc]:continue
        num=0.;den=0.
        for dr,dc in offsets:
            nr=rr+dr;nc=cc+dc
            if not known[nr,nc]:continue
            val,length,deg=kernel.directional(slope,origin,water,rr,cc,nr,nc,dem[nr,nc],alpha,30.)
            if not np.isfinite(val) or length<=0:raise ValueError('INVALID_DIRECTION')
            num+=val/length;den+=1/length;degenerate+=deg
        if den>0:
            dem[rr,cc]=num/den;known[rr,cc]=True;pending[rr,cc]=0;water[rr,cc]=0;count+=1
    return count,degenerate

def reference_advance(points,pending,water,origin,known,dem,slope,alpha):
    """Slow independent Python update used only in acceptance on real grids."""
    count=degenerate=0
    for rr,cc in points:
        if not pending[rr,cc]:continue
        values=[];weights=[]
        for dr,dc in [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr,nc=rr+dr,cc+dc
            if known[nr,nc]:
                value,length,deg=kernel.directional(slope,origin,water,rr,cc,nr,nc,dem[nr,nc],alpha,30.)
                values.append(value/length);weights.append(1/length);degenerate+=deg
        if weights:
            # Match the frozen sequential reduction; Python 3.12+ sum(float)
            # uses compensated summation, which is a different numerical rule.
            numerator=0.;denominator=0.
            for v,w in zip(values,weights):numerator+=v;denominator+=w
            dem[rr,cc]=numerator/denominator;known[rr,cc]=True;pending[rr,cc]=0;water[rr,cc]=0;count+=1
    return count,degenerate

def run(R,visible,z,init,alpha,*,update=advance,component_ids=None):
    """Same slope/geometry for all candidates; isolate scheduler by component.

    External known land is always retained. Anchors remain transparent in the
    ray-water geometry. Components without O use external land alone. Disabling
    execution of a component does not change initialization, origin or another
    component's scheduler. It is an acceptance probe, never domain filtering.
    """
    R=np.asarray(R,bool);visible=np.asarray(visible,bool)
    if digest(R)!=init['domain_hash']:raise ValueError('DOMAIN_CHANGED')
    if (visible&~R).any() or not np.isfinite(z[visible]).all():raise ValueError('INVALID_ANCHORS')
    labels,n=ndi.label(R,np.ones((3,3)))
    output=np.full(R.shape,np.nan);records=[]
    for cid,sl in enumerate(ndi.find_objects(labels),1):
        if sl is None or (component_ids is not None and cid not in component_ids):continue
        r0=max(0,sl[0].start-4);r1=min(R.shape[0],sl[0].stop+4)
        c0=max(0,sl[1].start-4);c1=min(R.shape[1],sl[1].stop+4)
        box=np.s_[r0:r1,c0:c1]
        target=labels[box]==cid;origin=np.ascontiguousarray(R[box],dtype=np.uint8)
        water=origin.copy();anchors=visible[box]&target
        known=init['known'][box].copy();dem=init['dem'][box].copy();slope=np.ascontiguousarray(init['slope'][box])
        dem[anchors]=z[box][anchors];known[anchors]=True
        pending=(target&~anchors).astype(np.uint8);h=init['h0'];trace=[];deg=0
        for it in range(10000):
            if not pending.any():break
            frontier=known&ndi.binary_dilation(pending,np.ones((3,3)))
            if not frontier.any():break
            high=frontier&(dem>h)
            active=high if high.any() else frontier
            if not high.any():h=float(dem[frontier].min())
            points=kernel.frontier(np.argwhere(active),pending,known)
            count,dd=update(points,pending,water,origin,known,dem,slope,float(alpha));deg+=dd
            trace.append([it+1,h,int(count),int(pending.sum())])
            if count==0:break
        part=output[box];part[target&known]=dem[target&known]
        records.append(dict(component=cid,cells=int(target.sum()),anchors=int(anchors.sum()),
            remaining=int(pending.sum()),iterations=len(trace),degenerate=deg,trace=trace,
            initialization='external_land_plus_visible_O' if anchors.any() else 'external_land_only'))
    expected=R if component_ids is None else np.isin(labels,list(component_ids))
    complete=bool(np.isfinite(output[expected]).all())
    if not np.array_equal(output[visible&expected],z[visible&expected]):raise AssertionError('ANCHOR_CHANGED')
    return output,dict(complete=complete,remaining=int((expected&~np.isfinite(output)).sum()),
                       components=records,alpha=float(alpha),input_hashes=dict(domain=digest(R),
                       visible=digest(visible),anchor_values=digest(z[visible]),slope=digest(init['slope']),
                       exterior_dem=digest(init['dem']),exterior_known=digest(init['known'])))
