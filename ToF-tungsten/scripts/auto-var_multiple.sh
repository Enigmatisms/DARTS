image_num=30
time_points=("short" "long")
scenes=("staircase")
type="beams"
echo "Progress:" &> progress.txt
echo "" >> progress.txt

for scene in ${scenes[@]}; do
    echo "--------------- scene: $scene  (`date`) --------------" >> progress.txt
    echo "" >> progress.txt
    for((i=0;i<${image_num};i++)); do
        # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
        for time_point in ${time_points[@]}; do
            output_folder=./exp_scenes/$scene/$time_point-$type/
            pfm_file="${output_folder}TungstenRender.pfm"
            renamed_file="${output_folder}result_$i.pfm"
            if [ ! -d $output_folder ]; then
                mkdir -p $output_folder
            fi
            if [ -f $pfm_file ]; then
                mv $pfm_file $renamed_file
                echo "$time_point: $i  Exists" >> progress.txt
                continue;
            fi
            if [ -f $renamed_file ]; then
                echo "$time_point: $i  Exists" >> progress.txt
                continue;
            fi
        
            pbrt_file=./exp_scenes/$scene/scene-$type-darts-$time_point.json
            ./build/release/tungsten $pbrt_file -t 104 --seed $i
            mv $pfm_file "${output_folder}result_$i.pfm"
            echo "$scene-$time_point: $i  (`date`)" >> progress.txt
        done
    done
done
