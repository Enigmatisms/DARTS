scene="cbox-trans"
scatter_param=(0.2 0.4)
time_widths=(0.02 0.1)
start_time=16.1
image_num=30
atime=87
start_num=0
elliptical=(0 1)
da_guiding=(0 1)
names=('origin' 'da-only' 'tsample' 'darts')

# photon points
photon_cnt=(1000000 600000)
vphoton_cnt=(20000 40000)
v_radii=(0.3 0.2 0.1 0.1)
echo "Progress:" &> progress.txt

for((scat_id=0;scat_id<2;scat_id++)); do
    sigma_s=${scatter_param[scat_id]}
    p_cnt=${photon_cnt[scat_id]}
    vp_cnt=${vphoton_cnt[scat_id]}
    for twidth in ${time_widths[@]}; do
        echo "" >> progress.txt
        echo "--------------- Scene = $scene-$sigma_s-$twidth (`date`) --------------" >> progress.txt
        echo "" >> progress.txt
        for((i=$start_num;i<${image_num};i++)); do
            for use_ell in ${elliptical[@]}; do
            for use_da in ${da_guiding[@]}; do
            val=$(($use_ell*2+$use_da))
            name=${names[val]}
            v_radius=${v_radii[val]}
            target_folder=./results/ablation/$scene-$sigma_s-$twidth/$name/
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            else 
                if [ -f ${target_folder}result_${i}.pfm ]; then
                    echo "$scene-$sigma_s-$twidth/$name: $i Exists" >> progress.txt
                    continue
                fi
            fi
            valid=1

            output_name="test.pfm"
            output_folder=./exp_scenes/$scene/
            if [ ! -d $output_folder ]; then
                mkdir -p $output_folder
            fi
            pfm_file="${output_folder}${output_name}"
            renamed_file="${target_folder}result_$i.pfm"

            scene_json=./exp_scenes/$scene/transient.json
            python3 ./modifier.py -f $scene_json --sigma_s $sigma_s --tw $twidth \
                    --at $atime --start_time $start_time --p_cnt $p_cnt --vp_cnt $vp_cnt \
                    --elliptical $use_ell --da_distance $use_da --vg_radius $v_radius
            ./build/release/tungsten $scene_json -t 104 --seed $i

            mv $pfm_file $renamed_file
            # echo "$scene-$sigma_s-$twidth/$name: $use_ell $use_da (`date`)" >> progress.txt
            echo "$scene-$sigma_s-$twidth/$name: $i (`date`)" >> progress.txt
            done
            done
        done
        if [ ! $valid -eq 1 ]; then
            continue
        fi
    done
done