scene="staircase"
strategy="beams"
settings=("short" "long")
start_times=(12.5 15.25)
all_time_widths=(0.05 0.125 0.25 0.5 1.0 2.0)
allowed_time=(340 345 351 365 390 440)
start_num=0
image_num=40
echo "Progress:" &> progress.txt

for((id=0;id<2;id++)); do
    setting=${settings[id]}
    start_time=${start_times[id]}
    echo "" >> progress.txt
    echo "--------------- Scene = $scene-$strategy-$setting (`date`) --------------" >> progress.txt
    echo "" >> progress.txt
    for((k=0;k<6;k++)); do 
        time_width=${all_time_widths[k]}
        atime=${allowed_time[k]}
        valid=0
        for((i=$start_num;i<${image_num};i++)); do
            # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
            target_folder=./results/gate-width/$scene-$strategy-$setting/$time_width/
            if [ -f ${target_folder}result_${i}.pfm ]; then
                echo "$strategy-$setting-$time_width: $i Exists" >> progress.txt
                continue
            fi
            valid=1
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            fi

            output_name="TungstenRender.pfm"
            output_folder=./exp_scenes/$scene/$setting-$strategy/
            if [ ! -d $output_folder ]; then
                mkdir -p $output_folder
            fi
            pfm_file="${output_folder}${output_name}"
            renamed_file="${target_folder}result_$i.pfm"

            scene_json=./exp_scenes/$scene/scene-$strategy-$setting.json
            python3 ./modifier.py -f $scene_json --sigma_s 0.5 --tw $time_width --at $atime --start_time $start_time
            ./build/release/tungsten $scene_json -t 104 --seed $i

            mv $pfm_file $renamed_file
            echo "$strategy-$setting-$time_width: $i (`date`)" >> progress.txt
        done
        if [ ! $valid -eq 1 ]; then
            continue
        fi
    done
done
