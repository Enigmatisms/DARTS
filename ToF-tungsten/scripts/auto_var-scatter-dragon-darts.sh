scene="dragon"
strategy="points-darts"
settings=("short" "long")
start_times=(4.1 7.1)
all_sigma_s=(0.25 0.5 0.75 1.0 1.25 1.5)
start_num=0
image_num=40

# photon points
photon_cnt=(1000000 90000 800000 700000 600000 500000)
vphoton_cnt=(20000 30000 40000 50000 60000 70000)

volgather_r=(0.225 0.2 0.175 0.15 0.15 0.15)
echo "Progress:" &> progress.txt

for((id=0;id<2;id++)); do
    setting=${settings[id]}
    start_time=${start_times[id]}
    echo "" >> progress.txt
    echo "--------------- Scene = $scene-$strategy-$setting (`date`) --------------" >> progress.txt
    echo "" >> progress.txt
    if [ $id -eq 0 ]; then
        allowed_time=(163 209 243 275 300 330)      # short time
    else
        allowed_time=(265 347 425 495 565 635)      # long time
    fi
    for((k=0;k<6;k++)); do
        sigma_s=${all_sigma_s[k]}
        p_cnt=${photon_cnt[k]}
        vp_cnt=${vphoton_cnt[k]}
        vg_r=${volgather_r[k]}
        atime=${allowed_time[k]}
        valid=0
        for((i=$start_num;i<${image_num};i++)); do
            # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
            target_folder=./results/scatter/$scene-$strategy-$setting/$sigma_s/
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            else 
                if [ -f ${target_folder}result_${i}.pfm ]; then
                    echo "$strategy-$setting-$sigma_s: $i Exists" >> progress.txt
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
            python3 ./modifier.py -f $scene_json --sigma_s $sigma_s --tw 0.05 --at $atime --start_time $start_time --p_cnt $p_cnt --vp_cnt $vp_cnt --vg_radius $vg_r
            ./build/release/tungsten $scene_json -t 104 --seed $i

            mv $pfm_file $renamed_file
            echo "$strategy-$setting-$sigma_s: $i (`date`)" >> progress.txt
        done
        if [ ! $valid -eq 1 ]; then
            continue
        fi
    done
done
