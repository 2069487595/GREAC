"""FABDEM source-only feasibility screen on identical TWDB R/W0 masks."""
import csv,json,math,zipfile,shutil,warnings
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from pyproj import Transformer
from pyogrio.raw import read
import shapely
from shapely.ops import transform as geom_transform
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import reproject,Resampling
from twdb_nasadem_intersection_screen import ROOT,TABLE,sample,component
OUT=ROOT/'05_Results/codex/by_date/2026-09-17/twdb_fabdem_intersection_screen'
OLD=ROOT/'05_Results/codex/by_date/2026-09-17/twdb_nasadem_intersection_screen/screening.json'
CACHE=OUT/'tiles'
ARCH=ROOT/'02_Data_Raw/FABDEM_V1-2_all'

def tag(lat,lon):return f'{"N" if lat>=0 else "S"}{abs(lat):02d}{"E" if lon>=0 else "W"}{abs(lon):03d}'

def tilepath(lat,lon):
 name=tag(lat,lon)+'_FABDEM_V1-2.tif';dest=CACHE/name
 if dest.exists():return dest
 for candidate in [ROOT/'02_Data_Raw/FABDEM_tiles_extracted'/name,ROOT/'02_Data_Raw/FABDEM_tiles_extracted'/tag(lat,lon)/name]:
  if candidate.is_file():return candidate
 la=lat//10*10;lo=lon//10*10;archive=ARCH/(tag(la,lo)+'-'+tag(la+10,lo+10)+'_FABDEM_V1-2.zip')
 with zipfile.ZipFile(archive) as z:
  names=[n for n in z.namelist() if Path(n).name==name];assert len(names)==1
  with z.open(names[0]) as s,dest.open('wb') as d:shutil.copyfileobj(s,d,8*1024*1024)
 return dest

def fabdem(lon,lat,tr,crs):
 z=np.full(lon.shape,np.nan,'float32');sources=[]
 for la in np.unique(np.floor(lat).astype(int)):
  for lo in np.unique(np.floor(lon).astype(int)):
   m=(np.floor(lat)==la)&(np.floor(lon)==lo)
   if not m.any():continue
   path=tilepath(la,lo)
   with rasterio.open(path) as s:
    tmp=np.full(z.shape,np.nan,'float32')
    reproject(rasterio.band(s,1),tmp,src_transform=s.transform,src_crs=s.crs,src_nodata=s.nodata,dst_transform=tr,dst_crs=crs,dst_nodata=np.nan,resampling=Resampling.nearest)
    valid=np.isfinite(tmp);z[valid]=tmp[valid]
   sources.append(str(path))
 return z,sources

def main():
 warnings.filterwarnings('ignore',category=DeprecationWarning)
 CACHE.mkdir(parents=True,exist_ok=True)
 protocol={'cohort':'same 114 TWDB v4 entities, 77 publishable identity matches',
 'candidate':'identical R & original NASADEM W0 & 5<=SWO<95, unchanged from NASADEM screen',
 'plateau':'0.5m modal bin in SWO>=95 within R&W0; fallback W0; quantization is diagnostic not water mask',
 'provisional_O':'candidate & finite FABDEM & elevation>plateau+3m & distance from R plateau<=0.5m pixels >60m & valid 5x5 relief>=3m & abs(z-3x3median)<=5m & component>=25pixels',
 'support':'at least100 O pixels and O p95-p05>=3m; exposure not yet certified',
 'adaptive_support':'hide inner25/50/75% ordered by distance from R boundary; gap90m; >=100 train and test pixels; independent confirmation needed',
 'no_truth':'no TWDB elevation,capacity,depth values opened or used for ranking',
 'vertical':'FABDEM EGM2008; do not reuse NASADEM EGM96 H0 for depth validation',
 'status':'input screen only, no alpha selection or model run'}
 (OUT/'protocol_before_screen.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 rows=list(csv.DictReader(TABLE.open(encoding='utf-8-sig')));old={x['entity_id']:x for x in json.loads(OLD.read_text(encoding='utf-8'))}
 ids=sorted({int(r['hylak_id']) for r in rows if r['mapping_publishable']=='True'})
 _,_,geoms,fields=read(ROOT/'02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg',where='Hylak_id IN ('+','.join(map(str,ids))+')',columns=['Hylak_id'])
 lookup={int(i):shapely.from_wkb(g) for i,g in zip(fields[0],geoms)};reports=[]
 for row in rows:
  q={'entity_id':row['entity_id'],'name':row['entity_name'],'hylak_id':row['hylak_id']}
  if row['mapping_publishable']!='True':q['status']='identity_not_publishable';reports.append(q);continue
  try:
   g=lookup[int(row['hylak_id'])];zone=int((g.centroid.x+180)//6)+1;crs=f'EPSG:{32600+zone}'
   fwd=Transformer.from_crs(4326,crs,always_xy=True);inv=Transformer.from_crs(crs,4326,always_xy=True);gp=geom_transform(fwd.transform,g);clip=gp.buffer(3000)
   left,bottom,right,top=clip.bounds;left=math.floor(left/30)*30;top=math.ceil(top/30)*30
   width=math.ceil((right-left)/30)+2;height=math.ceil((top-bottom)/30)+2;tr=from_origin(left,top,30,30)
   rr,cc=np.indices((height,width));lon,lat=inv.transform(left+(cc+.5)*30,top-(rr+.5)*30)
   nz,s,occ,_=sample(lon,lat);anchor=rasterize([(gp,1)],out_shape=nz.shape,transform=tr).astype(bool);extent=rasterize([(clip,1)],out_shape=nz.shape,transform=tr).astype(bool)
   W=component((s==255)&np.isfinite(nz),anchor);R=component((occ>=5)&(occ<=100),anchor)&extent;C=R&W&(occ<95)
   assert int(C.sum())==old[row['entity_id']]['candidate_cells']
   assert np.isclose(R.sum()*.0009,old[row['entity_id']]['R_area_km2'])
   z,sources=fabdem(lon,lat,tr,crs);valid=np.isfinite(z);reference=R&W&(occ>=95)&valid
   if reference.sum()<100:reference=W&valid
   if not reference.any():raise ValueError('no_finite_plateau_reference')
   bins=np.round(z[reference]*2)/2;v,n=np.unique(bins,return_counts=True);plateau=float(v[np.argmax(n)])
   water=R&valid&(abs(z-plateau)<=.5);distance=ndi.distance_transform_edt(~water,sampling=30)
   nearvalid=ndi.minimum_filter(valid.astype('uint8'),5)==1
   safe=np.nan_to_num(z,nan=plateau);relief=ndi.maximum_filter(safe,5)-ndi.minimum_filter(safe,5)
   base=C&valid&(z>plateau+3)&(distance>60)&nearvalid&(relief>=3)&(abs(z-ndi.median_filter(safe,3))<=5)
   labels,_=ndi.label(base);sizes=np.bincount(labels.ravel());keep=sizes>=25;keep[0]=False;O=keep[labels]
   span=float(np.ptp(np.quantile(z[O],[.05,.95]))) if O.any() else 0.;support=bool(O.sum()>=100 and span>=3)
   d=ndi.distance_transform_edt(R,sampling=30);holdouts=[]
   if O.any():
    for fraction in [.25,.5,.75]:
     cut=float(np.quantile(d[O],1-fraction));train=O&(d<=cut-90);test=O&(d>=cut)
     holdouts.append({'hidden_fraction':fraction,'train_cells':int(train.sum()),'test_cells':int(test.sum()),'eligible':bool(train.sum()>=100 and test.sum()>=100)})
   q.update({'crs':crs,'candidate_cells':int(C.sum()),'candidate_area_km2':float(C.sum()*.0009),'FABDEM_coverage':float(valid[C].mean()),'plateau_m':plateau,
     'candidate_plateau_fraction':float(water[C].mean()),'candidate_above_plateau3m_cells':int((C&valid&(z>plateau+3)).sum()),
     'O_cells':int(O.sum()),'O_area_km2':float(O.sum()*.0009),'O_p05_p50_p95_m':np.quantile(z[O],[.05,.5,.95]).tolist() if O.any() else None,
     'O_components':int(keep.sum()),'O_max_inradius_m':float(ndi.distance_transform_edt(O,sampling=30).max()) if O.any() else 0.,
     'terrain_support':support,'holdouts':holdouts,'adaptive_support':support and any(h['eligible'] for h in holdouts),
     'R_touches_3km_clip':old[row['entity_id']]['R_touches_3km_clip'],'source_files':sources,
     'status':'provisional_terrain_candidate' if support else 'insufficient_screened_terrain',
     'unresolved':['source acquisition exposure','Copernicus editing/source masks','survey coverage and vertical datum','terrain continuity and projected-grid duplication']})
   if support:
    np.savez_compressed(OUT/(row['entity_id']+'_inputs.npz'),fabdem=z,R=R,W0=W,candidate=C,O_provisional=O,occurrence=occ,transform=np.array(tr))
  except Exception as exc:q.update(status='screen_error',error=str(exc))
  reports.append(q);print(q['name'],q['status'],q.get('O_cells'),q.get('adaptive_support'),flush=True)
  (OUT/'screening.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'screening.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8');print('DONE',len(reports),flush=True)

if __name__=='__main__':main()
