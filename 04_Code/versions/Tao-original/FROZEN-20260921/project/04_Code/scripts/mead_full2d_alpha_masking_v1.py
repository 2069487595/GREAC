"""Full 2-D iterative recession for each alpha, using outer O only.

Uses released Tao ray geometry and its per-direction height calculation, with
the inactive exponent explicitly enabled in an in-memory AST copy. Mask/seed
handling is adapted to withheld, sparse O; unavailable exterior elevations
are never treated as known land. Not the author's unpublished Mead setup.
"""
from pathlib import Path
import ast,json,hashlib,time,argparse,sys
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject,Resampling

ROOT=Path(__file__).resolve().parents[2]
TAO=ROOT/'10_References/Tao_source_code/scripts'
sys.path.insert(0,str(TAO))
from lake_bathymetry import analysis as source_analysis
from tao_compat import EightNeighbor,draw_extended_line,TotalDistance

BASE=ROOT/'05_Results/codex/by_date/2026-09-10/mead_paired_observation_experiment_v1_waterline'
OUT=ROOT/'05_Results/codex/by_date/2026-09-14/mead_full2d_alpha_masking_v1'
CONFIG={'alphas':[0.,1.,2.,4.5,8.],'withheld_fractions':[.25,.5,.75],
        'gap_m':90.,'resolution_m':90.,'slope_radius_m':600.,'min_slope_samples':3,
        'slope_fallback':'median positive adjacent observed elevation difference / distance',
        'exterior_elevations':'unavailable; only outer O starts recession',
        'geometry_bank_slope_fallback':'nearest slope estimated from outer O',
        'metric':'heldout DEM RMSE on identical test mask; no bathymetry/volume used',
        'subbasin_geometry':'current water mask for current ray; initial masked-water domain for total ray, following supplied core.py',
        'max_rounds':10000}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def alpha_function():
    tree=ast.parse((TAO/'lake_bathymetry/analysis.py').read_text(encoding='utf-8-sig'))
    f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='calculate')
    count=0
    for i,node in enumerate(f.body):
        if isinstance(node,ast.Assign) and ast.unparse(node)=='slope_now = slope_now * (current_len / total_length)':
            f.body[i]=ast.parse('slope_now = slope_now * ((current_len / total_length) ** a)').body[0];count+=1
    assert count==1
    module=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]))
    scope=dict(source_analysis.__dict__);exec(compile(module,'Tao_calculate_alpha_enabled','exec'),scope)
    return scope['calculate'],ast.unparse(f)


def load_inputs():
    meta=json.loads((BASE/'predictions_before_validation.json').read_text())
    with np.load(BASE/'evaluation_masks.npz') as m:r=m['r'];o=m['o'];tr=Affine(*m['transform'][:6])
    with np.load(meta['input_path']) as m:
        z=np.full(r.shape,np.nan,'float32')
        reproject(m['fabdem_m'],z,src_transform=Affine(*m['transform'][:6]),src_crs='EPSG:4326',
                  dst_transform=tr,dst_crs='EPSG:32611',src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.nearest)
    boundary=r&~ndi.binary_erosion(r);d=ndi.distance_transform_edt(r&~boundary,sampling=90)
    return r,o,z,d,tr


def initialize(redacted,train,domain):
    # Never inspect DEM outside train, including withheld or gap pixels.
    assert np.all(~np.isfinite(redacted[~train]))
    coords=np.argwhere(train);vals=redacted[train].astype(float);tree=cKDTree(coords*90.)
    diffs=[]
    for dr,dc in ((1,0),(0,1)):
        aa=train[:train.shape[0]-dr or None,:train.shape[1]-dc or None]
        bb=train[dr:,dc:];ok=aa&bb
        za=redacted[:train.shape[0]-dr or None,:train.shape[1]-dc or None];zb=redacted[dr:,dc:]
        delta=np.abs(za[ok]-zb[ok])/90.;diffs.extend(delta[delta>0].tolist())
    if not diffs:raise ValueError('No observed slope support')
    fallback=float(np.median(diffs));slopes=np.zeros(len(coords));fallback_count=0
    for i,c in enumerate(coords):
        ids=tree.query_ball_point(c*90.,CONFIG['slope_radius_m'])
        xy=(coords[ids]-c)*90.;v=vals[ids]
        design=np.column_stack([np.ones(len(ids)),xy])
        if len(ids)>=CONFIG['min_slope_samples'] and np.linalg.matrix_rank(design)==3:
            co=np.linalg.lstsq(design,v,rcond=None)[0];s=float(np.hypot(co[1],co[2]));slopes[i]=s if s>0 else fallback
        else:slopes[i]=fallback;fallback_count+=1
    # Geometry routines may sample a bank outside R or outside O. Such slopes
    # are explicit nearest-outer-O estimates; no exterior DEM enters the model.
    nearest=ndi.distance_transform_edt(~train,return_distances=False,return_indices=True)
    field=np.zeros(train.shape);field[train]=slopes;field=field[tuple(nearest)]
    return field,{'observed_anchor_count':len(coords),'plane_fit_fallback_count':fallback_count,
                  'fallback_slope_m_per_m':fallback,'slope_p05_p50_p95':np.percentile(slopes,[5,50,95]).tolist()}


def run_recession(domain,train,redacted,slope,alpha,calculate,cache,label):
    started=time.time();water=(domain&~train).astype('uint8');origin=water.copy()
    dem=redacted.astype(float).copy();available=train.copy();initial=int(water.sum())
    H=float(np.max(dem[train]));active=train&ndi.binary_dilation(water,structure=np.ones((3,3)))
    trace=[];degenerate=0
    for iteration in range(1,CONFIG['max_rounds']+1):
        points=np.argwhere((water==1)&ndi.binary_dilation(active,structure=np.ones((3,3))))
        if not len(points):break
        done=0
        for pt in points:
            heights=[];lengths=[]
            for direction,nb in enumerate(EightNeighbor(pt)):
                if not available[nb[0],nb[1]]:continue
                current,vector=draw_extended_line(water,nb,pt)
                key=(int(pt[0]),int(pt[1]),direction)
                if key not in cache:cache[key]=TotalDistance(slope,origin,pt,vector)
                length,p1,p2,low=cache[key]
                step=90.*float(np.linalg.norm(pt-nb))
                # A zero-length half-profile can occur for a discrete minimum.
                if np.linalg.norm(p1-low)==0 or np.linalg.norm(p2-low)==0:
                    value=float(dem[tuple(nb)]);degenerate+=1
                else:
                    value=calculate(slope,pt,float(dem[tuple(nb)]),step,p1,p2,low,0.,current,alpha)
                if not np.isfinite(value):raise ValueError('Nonfinite Tao update')
                heights.append(value);lengths.append(float(length))
            if not heights:continue
            weights=1/np.asarray(lengths);weights/=weights.sum()
            dem[tuple(pt)]=np.dot(heights,weights);water[tuple(pt)]=0;available[tuple(pt)]=True;done+=1
        land=available&ndi.binary_dilation(water,structure=np.ones((3,3)))
        if not water.any():
            trace.append({'round':iteration,'remaining':0,'elapsed_s':time.time()-started});break
        if not land.any():break
        high=land&(dem>H)
        if high.any():active=high
        else:H=float(np.min(dem[land]));active=land
        if iteration%20==0:
            rec={'round':iteration,'remaining':int(water.sum()),'threshold_m':H,'elapsed_s':time.time()-started};trace.append(rec)
            print(label,rec,flush=True)
        if done==0:break
    stats={'alpha':alpha,'iterations':iteration,'initial_unknown':initial,'remaining_unknown':int(water.sum()),
           'complete':not bool(water.any()),'elapsed_s':time.time()-started,'degenerate_half_profile_updates':degenerate,'trace':trace}
    assert np.array_equal(dem[train],redacted[train])
    return dem,stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=float,default=None);parser.add_argument('--alpha',type=float,default=None)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    calculate,patched=alpha_function();(OUT/'calculate_alpha_enabled.py.txt').write_text(patched,encoding='utf-8')
    r,o,z,d,tr=load_inputs()
    CONFIG['source_hashes']={f:sha(TAO/'lake_bathymetry'/f) for f in ('core.py','analysis.py','geometry_utils.py')}
    CONFIG['script_sha256']=sha(Path(__file__))
    (OUT/'protocol.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    # Source parity check: alpha=1 must equal the original fixed first power.
    s=np.full((12,12),.1);pt=np.array([4,5]);p1=np.array([4,1]);p2=np.array([4,9]);low=np.array([4,6])
    a=(s,pt,370.,90.,p1,p2,low,0.,3.)
    assert calculate(*a,1.)==source_analysis.calculate(*a,1.)
    assert calculate(*a,0.)<calculate(*a,2.)
    for fraction in ([args.fold] if args.fold is not None else CONFIG['withheld_fractions']):
        folder=OUT/f'hold_{int(fraction*100)}';folder.mkdir(exist_ok=True)
        cutoff=float(np.quantile(d[o&(d>0)],1-fraction))
        train=o&(d<=cutoff-CONFIG['gap_m']);test=o&(d>=cutoff)
        assert not np.any(train&test)
        redacted=np.where(train,z,np.nan);labels,n=ndi.label(r);supported=np.unique(labels[train]);supported=supported[supported!=0]
        domain=r&np.isin(labels,supported)
        slope,init=initialize(redacted,train,domain)
        frozen={'cutoff_m':cutoff,'train_pixels':int(train.sum()),'test_pixels':int(test.sum()),
                'test_without_seed_component_pixels':int((test&~domain).sum()),'domain_pixels':int(domain.sum()),
                'unsupported_R_pixels':int((r&~domain).sum()),'initialization':init,'only_outer_O_elevations_available':True}
        np.savez_compressed(folder/'frozen_input.npz',redacted_dem=redacted,train=train,test=test,domain=domain,slope=slope,transform=np.array(tr))
        frozen['frozen_input_sha256']=sha(folder/'frozen_input.npz')
        (folder/'input_manifest.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
        cache={}
        for alpha in ([args.alpha] if args.alpha is not None else CONFIG['alphas']):
            name=f'alpha_{alpha:g}';print('START',fraction,alpha,frozen['train_pixels'],frozen['test_pixels'],flush=True)
            pred,stats=run_recession(domain,train,redacted,slope.copy(),alpha,calculate,cache,name)
            path=folder/(name+'.tif')
            with rasterio.open(path,'w',driver='GTiff',height=r.shape[0],width=r.shape[1],count=1,dtype='float32',
                               crs='EPSG:32611',transform=tr,nodata=np.nan,compress='deflate') as s:s.write(np.where(domain,pred,np.nan).astype('float32'),1)
            stats['prediction_sha256']=sha(path)
            # Evaluation only after the entire prediction has been persisted.
            mask=test&np.isfinite(pred)&domain;e=pred[mask]-z[mask]
            stats['test_pixels_requested']=int(test.sum());stats['test_pixels_predicted']=int(mask.sum())
            stats['heldout']={'rmse_m':float(np.sqrt(np.mean(e**2))),'mae_m':float(np.mean(np.abs(e))),
                              'bias_m':float(np.mean(e)),'pearson_r':float(np.corrcoef(pred[mask],z[mask])[0,1])}
            stats['eligible_for_alpha_comparison']=bool(stats['complete'] and mask.sum()==test.sum())
            (folder/(name+'.json')).write_text(json.dumps(stats,indent=2,allow_nan=False),encoding='utf-8')
            print('DONE',fraction,alpha,stats['elapsed_s'],stats['complete'],stats['heldout'],flush=True)


if __name__=='__main__':main()
