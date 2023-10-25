sudo singularity build --sandbox singularity/ubuntu_warehouse/ singularity/ubuntu_warehouse_cn.def
sudo singularity run --writable singularity/ubuntu_warehouse
sudo singularity build singularity/ubuntu_warehouse.sif singularity/ubuntu_warehouse