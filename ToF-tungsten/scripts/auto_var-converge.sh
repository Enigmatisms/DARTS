scene="dragon"
strategy="points"
settings=("short" "long")
start_times=(4.1 7.1)
start_num=0
# dragon points
photon_cnt=(5000 10000 100000 1000000 7000000 7000000)
vphoton_cnt=(1000 2000 10000 40000 200000 200000)
volgather_r=(0.4 0.35 0.3 0.3 0.3 0.3)
# dragon beams
photon_cnt=(3000 5000 10000 50000 100000 100000)
vphoton_cnt=(500 1000 2000 10000 20000 20000)
volgather_r=(0.5 0.45 0.4 0.4 0.4 0.4)
allowed_time=(1 2 13.5 135 1350 13500)      # short time
end_nums=(200 40 30 30 30 20)                 # short time
echo "Progress:" &> progress.txt

for((id=0;id<1;id++)); do
    setting=${settings[id]}
    start_time=${start_times[id]}
    echo "" >> progress.txt
    echo "--------------- Scene = $scene-$strategy-$setting (`date`) --------------" >> progress.txt
    echo "" >> progress.txt
    for((k=0;k<6;k++)); do
        p_cnt=${photon_cnt[k]}
        vp_cnt=${vphoton_cnt[k]}
        vg_r=${volgather_r[k]}
        image_num=${end_nums[k]}
        atime=${allowed_time[k]}
        valid=0
        for((i=$start_num;i<${image_num};i++)); do
            # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
            target_folder=./results/converge/$scene-$strategy-$setting/$atime/
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            else 
                if [ -f ${target_folder}result_${i}.pfm ]; then
                    echo "$strategy-$setting-$atime: $i Exists" >> progress.txt
                    continue
                fi
            fi
            valid=1

            output_name="TungstenRender.pfm"
            output_folder=./exp_scenes/$scene/$setting-$strategy/
            if [ ! -d $output_folder ]; then
                mkdir -p $output_folder
            fi
            pfm_file="${output_folder}${output_name}"
            renamed_file="${target_folder}result_$i.pfm"

            scene_json=./exp_scenes/$scene/scene-$strategy-$setting.json
            python3 ./modifier.py -f $scene_json --sigma_s 1.0 --tw 0.05 --at $atime --start_time $start_time --p_cnt $p_cnt --vp_cnt $vp_cnt --vg_radius $vg_r
            ./build/release/tungsten $scene_json -t 104 --seed $i

            mv $pfm_file $renamed_file
            echo "$strategy-$setting-$atime: $i (`date`)" >> progress.txt
        done
        if [ ! $valid -eq 1 ]; then
            continue
        fi
    done
done
