#%%

import importlib
import os.path
import sys

dirs_to_try = [
    ".",
    "/gpfs/slac/staas/fs1/g/neutrino/jwolcott/app",
    "/media/hdd1/jwolcott/app",
    "/dune/app/users/jwolcott/dunesoft",
]

modules_required = {
    # module name -> subdir path
    "mlreco": "lartpc_mlreco3d",
    "larcv": "larcv2/python",
}

for module_name, module_path in modules_required.items():
    software_dir = None
    for d in dirs_to_try:
        d = os.path.join(d, module_path)
        if os.path.isdir(d):
            software_dir = d
            break

    success = False
    if software_dir:
        sys.path.insert(0, software_dir)
        try:
            importlib.import_module(module_name)
            success = True
        except:
            pass

    if not success:
        print("ERROR: couldn't find %s package" % module_name)
    else:
        print("Setup of %s ok from:" % module_name, software_dir)

#%%


import yaml


cfg='''
iotool:
  batch_size: 32
#  batch_size: 1
  shuffle: False
  num_workers: 1
  collate_fn: CollateSparse
  sampler:
    name: RandomSequenceSampler
  dataset:
    name: LArCVDataset
    data_keys:
#      - /gpfs/slac/staas/fs1/g/neutrino/kterao/data/mpvmpr_2020_01_v04/test.root
#      - /gpfs/slac/staas/fs1/g/neutrino/jwolcott/data/nd-lar-reco/supera/nd.fhc.0.supera.root
       - /media/hdd1/jwolcott/data/dune/nd/nd-lar-reco/supera/nd.fhc.0.supera.root
    limit_num_files: 10
    schema:
      input_data:
        - parse_sparse3d_scn
#        - sparse3d_pcluster
        - sparse3d_geant4
      segment_label:
        - parse_sparse3d_scn
        - sparse3d_geant4
        - sparse3d_geant4_semantics
#       particles_label:
#         - parse_particle_points
#         - sparse3d_geant4
#        - sparse3d_pcluster
#        - particle_corrected
model:
  name: uresnet_ppn_chain
  modules:
    ppn:
      num_strides: 6
      filters: 16
      num_classes: 5
      data_dim: 3
      downsample_ghost: False
      use_encoding: False
      ppn_num_conv: 1
      score_threshold: 0.5
      ppn1_size: 24
      ppn2_size: 96
      spatial_size: 768
    uresnet_lonely:
      freeze: False
      num_strides: 6
      filters: 16
      num_classes: 5
      data_dim: 3
      spatial_size: 24576
      ghost: False
      features: 1
  network_input:
    - input_data
  #  - particles_label
  # loss_input:
  #   - segment_label
  #   - particles_label
trainval:
  seed: 123
  learning_rate: 0.001
  unwrapper: unwrap_3d_scn
#  gpus: ''
  gpus: '0'
  weight_prefix: weights/snapshot
  iterations: 10
  report_step: 1
  checkpoint_step: 100
  log_dir: log_inference
#  model_path: '/gpfs/slac/staas/fs1/g/neutrino/jwolcott/data/nd-lar-reco/weights_kazu_sample.ckpt'
  model_path: '/media/hdd1/jwolcott/data/dune/nd/nd-lar-reco/weights_kazu_sample.ckpt'
  train: False
  debug: True
'''

#####

from mlreco.main_funcs import process_config, inference
cfg_dict=yaml.load(cfg,Loader=yaml.Loader)
# pre-process configuration (checks + certain non-specified default settings)
process_config(cfg_dict)


#inference(cfg_dict)


