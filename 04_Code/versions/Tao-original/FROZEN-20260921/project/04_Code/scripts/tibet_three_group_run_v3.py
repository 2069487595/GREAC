"""Repair uninitialized bank-slope reads; retain all v2 data and parameters."""
import tibet_three_group_run_v2 as run
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from pathlib import Path
import argparse
OUT=run.ROOT/'05_Results/codex/by_date/2026-09-16/tibet_three_group_v3_bank_alignment'
original_initialize=run.initialize
original_savejson=run.savejson
def initialize(z,unknown):
    dem,known,slope,info=original_initialize(z,unknown)
    contours=np.concatenate(measure.find_contours(unknown.astype('uint8'),0)).astype(int)
    anchor=np.zeros(unknown.shape,bool);anchor[contours[:,0],contours[:,1]]=True
    distance,nearest=ndi.distance_transform_edt(~anchor,return_indices=True)
    # Rays terminate at first nonwater then sample one further unit step.
    # Raster rounding bounds this footprint by three pixels from a shoreline.
    # Preserve all initialized values, including genuinely zero stretched slopes.
    fill=(~anchor)&(distance<=3.)
    aligned=slope.copy();aligned[fill]=slope[tuple(nearest[:,fill])]
    assert np.array_equal(aligned[anchor],slope[anchor])
    info['bank_alignment']={'rule':'nearest initialized shoreline slope within 3 pixels at previously uninitialized positions; initialized slopes including zeros preserved',
       'filled_positions':int(fill.sum()),'nonzero_filled_positions':int((aligned[fill]>0).sum()),'elevations_modified':False,'new_fitted_parameters':False}
    return dem,known,aligned,info
def savejson(path,obj):
    if path.name=='protocol_before_predictions.json':
        obj['bank_alignment_repair']={'version':3,'wrapper_sha256':run.sha(Path(__file__)),
          'diagnostic_basis':'v2 Mapang axial sample: 1640/2004 rays had both zero slopes, all 1640 with both endpoints outside initialized contours',
          'operation':'nearest initialized shoreline slopes extended at most 3 pixels to satisfy released TotalDistance endpoint sampling; no change to initialized slope values or elevation anchors',
          'not_calibrated_to_soundings':True}
    original_savejson(path,obj)
run.initialize=initialize;run.OUT=OUT;run.savejson=savejson
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lake',choices=run.LAKES,required=True);a=p.parse_args();run.run_lake(a.lake)
