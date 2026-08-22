# Flat scene asset

ReplicaCAD "baked lighting" stage `Baked_sc0_staging_00.glb`
(stages_uncompressed variant — the compressed stages use KTX2 textures
trimesh cannot decode), from the Habitat ReplicaCAD dataset
(https://huggingface.co/datasets/ai-habitat/ReplicaCAD_baked_lighting),
an artist recreation of the FRL apartment, licensed CC Attribution 4.0
(commercial use permitted with attribution).

Only used locally for simulation; the raw/processed asset files are
gitignored and re-created by `./run.sh room-assets --glb
assets/room/raw/replicacad_sc0_00_unc.glb --name flat --skip-collision --up y`.
