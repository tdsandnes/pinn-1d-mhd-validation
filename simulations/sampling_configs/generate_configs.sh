#!/usr/bin/env bash
# Regenerate every sampling config from template.py.

set -euo pipefail

SC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SC_DIR"

NUM_REPEATS=3

# Fiducial values for each swept quantity
FID_T_MIN=20
FID_T_MAX=30
FID_N_PHYS=32768
FID_N_DATA=1025
FID_N_BC=16384


# Wipe subdirectories
find . -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +


# Overwrite one top-level constant in a generated file
set_const() {
    sed -i "s|^$2 = .*|$2 = $3|" "$1"
}

# Write NUM_REPEATS repeats of one basename
generate_repeats() {
    local basename=$1; shift
    local subdir="$SC_DIR/$basename"
    [[ -d "$subdir" ]] && return
    mkdir -p "$subdir"
    for r in $(seq 1 $NUM_REPEATS); do
        local rr=$(printf "%02d" $r)
        local seed=$((16#$(printf "%s_%s" "$basename" "$rr" | md5sum | cut -c1-8)))
        local out="$subdir/sampling_config_${basename}_repeat${rr}.py"
        cp template.py "$out"
        set_const "$out" BASENAME "\"$basename\""
        set_const "$out" REPEAT "$r"
        set_const "$out" SEED "$seed"
        for kv in "$@"; do
            set_const "$out" "${kv%%=*}" "${kv#*=}"
        done
    done
}


sobol() {
    # t_min t_max N_phys N_data N_bc
    local base=$(printf "t%02dto%02d_eachcol%06dsobol_data%06d_bc%06d" \
                 "$1" "$2" "$3" "$4" "$5")
    generate_repeats "$base" T_MIN=$1.0 T_MAX=$2.0 N_DATA=$4 N_PHYS=$3 N_BC=$5 \
        N_X_PHYS=1 N_T_PHYS=1 TRAIN_METHOD='"sobol"' VAL_METHOD='"sobol"'
}

latin_hypercube() {
    # t_min t_max N_phys N_data N_bc
    local base=$(printf "t%02dto%02d_eachcol%06dlatinhypercube_data%06d_bc%06d" \
                 "$1" "$2" "$3" "$4" "$5")
    generate_repeats "$base" T_MIN=$1.0 T_MAX=$2.0 N_DATA=$4 N_PHYS=$3 N_BC=$5 \
        N_X_PHYS=1 N_T_PHYS=1 TRAIN_METHOD='"latin_hypercube"' VAL_METHOD='"latin_hypercube"'
}

grid_aligned() {
    # t_min t_max N_x N_t N_data N_bc
    local N=$(($3 * $4))
    local base=$(printf "t%02dto%02d_eachcol%06dgridaligned_data%06d_bc%06d" \
                 "$1" "$2" "$N" "$5" "$6")
    generate_repeats "$base" T_MIN=$1.0 T_MAX=$2.0 N_DATA=$5 N_PHYS=$N N_BC=$6 \
        N_X_PHYS=$3 N_T_PHYS=$4 TRAIN_METHOD='"grid"' VAL_METHOD='"sobol"'
}

grid_offset() {
    # t_min t_max N_x N_t N_data N_bc
    local N=$(($3 * $4))
    local base=$(printf "t%02dto%02d_eachcol%06dgridoffset_data%06d_bc%06d" \
                 "$1" "$2" "$N" "$5" "$6")
    generate_repeats "$base" T_MIN=$1.0 T_MAX=$2.0 N_DATA=$5 N_PHYS=$N N_BC=$6 \
        N_X_PHYS=$3 N_T_PHYS=$4 TRAIN_METHOD='"grid"' VAL_METHOD='"sobol"' \
        ZP_OFFSET_X=-0.25 ZP_OFFSET_T=-0.25 ZM_OFFSET_X=0.25 ZM_OFFSET_T=0.25
}


# col_sampling: vary N_phys, hold N_data and N_bc at fiducial
for N in 1024 2048 4096 8192 16384 32768; do
    sobol $FID_T_MIN $FID_T_MAX $N $FID_N_DATA $FID_N_BC
done

# data_sampling: vary N_data, hold N_phys and N_bc at fiducial
for N in 33 65 129 257 513 1025; do
    sobol $FID_T_MIN $FID_T_MAX $FID_N_PHYS $N $FID_N_BC
done

# bc_sampling: vary N_bc, hold N_phys and N_data at fiducial
for N in 16 64 256 1024 4096 16384; do
    sobol $FID_T_MIN $FID_T_MAX $FID_N_PHYS $FID_N_DATA $N
done

# sampling scheme comparison: all at fiducial values
latin_hypercube $FID_T_MIN $FID_T_MAX $FID_N_PHYS $FID_N_DATA $FID_N_BC
grid_aligned    $FID_T_MIN $FID_T_MAX 128 256 $FID_N_DATA $FID_N_BC
grid_offset     $FID_T_MIN $FID_T_MAX 128 256 $FID_N_DATA $FID_N_BC


echo "Done: $(find . -mindepth 1 -maxdepth 1 -type d | wc -l) configs, \
$(find . -mindepth 2 -type f -name '*.py' | wc -l) files."
