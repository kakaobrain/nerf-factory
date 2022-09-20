python3 -m run --ginc configs/nerf/360_v2.gin --scene_name garden --ginb run.run_render=True --ginb run.run_eval=False --ginb run.run_train=False
python3 -m run --ginc configs/mipnerf/360_v2.gin --scene_name garden --ginb run.run_render=True --ginb run.run_eval=False --ginb run.run_train=False
python3 -m run --ginc configs/nerfpp/360_v2.gin --scene_name garden --ginb run.run_render=True --ginb run.run_eval=False --ginb run.run_train=False
python3 -m run --ginc configs/mipnerf360/360_v2.gin --scene_name garden --ginb run.run_render=True --ginb run.run_eval=False --ginb run.run_train=False
python3 -m run --ginc configs/plenoxel/360_v2.gin --scene_name garden --ginb run.run_render=True --ginb run.run_eval=False --ginb run.run_train=False
python3 -m run --ginc configs/dvgo/360_v2.gin --scene_name garden --ginb run.run_render=True --ginb run.run_eval=False --ginb run.run_train=False
