#!/usr/bin/env bash

# Sampling groups, relative to simulations/sampling_configs/.
# Each entry is either a .py file (sampled once) or a directory (every
# .py file inside is sampled).

SAMPLING_GROUP_ALL=(
    sampling_config_test.py
    t20to30_eachcol001024sobol_data001025_bc016384
    t20to30_eachcol002048sobol_data001025_bc016384
    t20to30_eachcol004096sobol_data001025_bc016384
    t20to30_eachcol008192sobol_data001025_bc016384
    t20to30_eachcol016384sobol_data001025_bc016384
    t20to30_eachcol032768sobol_data001025_bc016384
    t20to30_eachcol032768latinhypercube_data001025_bc016384
    t20to30_eachcol032768gridaligned_data001025_bc016384
    t20to30_eachcol032768gridoffset_data001025_bc016384
    t20to30_eachcol032768sobol_data000033_bc016384
    t20to30_eachcol032768sobol_data000065_bc016384
    t20to30_eachcol032768sobol_data000129_bc016384
    t20to30_eachcol032768sobol_data000257_bc016384
    t20to30_eachcol032768sobol_data000513_bc016384
    t20to30_eachcol032768sobol_data001025_bc000016
    t20to30_eachcol032768sobol_data001025_bc000064
    t20to30_eachcol032768sobol_data001025_bc000256
    t20to30_eachcol032768sobol_data001025_bc001024
    t20to30_eachcol032768sobol_data001025_bc004096
)

SAMPLING_GROUP_FIDUCIAL=(
    t20to30_eachcol032768sobol_data001025_bc016384
)

SAMPLING_GROUP_SAMPLING_SCHEME=(
    t20to30_eachcol032768sobol_data001025_bc016384
    t20to30_eachcol032768latinhypercube_data001025_bc016384
    t20to30_eachcol032768gridaligned_data001025_bc016384
    t20to30_eachcol032768gridoffset_data001025_bc016384
)

SAMPLING_GROUP_DATA_SAMPLING=(
    t20to30_eachcol032768sobol_data000033_bc016384
    t20to30_eachcol032768sobol_data000065_bc016384
    t20to30_eachcol032768sobol_data000129_bc016384
    t20to30_eachcol032768sobol_data000257_bc016384
    t20to30_eachcol032768sobol_data000513_bc016384
    t20to30_eachcol032768sobol_data001025_bc016384
)

SAMPLING_GROUP_BC_SAMPLING=(
    t20to30_eachcol032768sobol_data001025_bc000016
    t20to30_eachcol032768sobol_data001025_bc000064
    t20to30_eachcol032768sobol_data001025_bc000256
    t20to30_eachcol032768sobol_data001025_bc001024
    t20to30_eachcol032768sobol_data001025_bc004096
    t20to30_eachcol032768sobol_data001025_bc016384
)

SAMPLING_GROUP_NONE=()

# Simulation -> sampling group
declare -A SIM_CONFIGS=(
    ["1freq_1Ma_500Re_500Rm/sim_config_1freq_1Ma_500Re_500Rm.py"]="SAMPLING_GROUP_ALL"
    ["2freq_1Ma_500Re_500Rm/sim_config_2freq_1Ma_500Re_500Rm.py"]="SAMPLING_GROUP_ALL"
    ["3freq_1Ma_500Re_500Rm/sim_config_3freq_1Ma_500Re_500Rm.py"]="SAMPLING_GROUP_ALL"

    ["1freq_1Ma_500Re_500Rm_nononlin/sim_config_1freq_1Ma_500Re_500Rm_nononlin.py"]="SAMPLING_GROUP_NONE"
    ["2freq_1Ma_500Re_500Rm_nononlin/sim_config_2freq_1Ma_500Re_500Rm_nononlin.py"]="SAMPLING_GROUP_NONE"
    ["3freq_1Ma_500Re_500Rm_nononlin/sim_config_3freq_1Ma_500Re_500Rm_nononlin.py"]="SAMPLING_GROUP_NONE"
)
