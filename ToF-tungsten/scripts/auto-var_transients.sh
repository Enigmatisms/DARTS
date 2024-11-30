# Shell script for running path reuse DARTS-PP transient rendering

image_nums=(20 20 20)
scene="staircase"
methods=("" "-dartspp-reuse-hr" "-originpp-hr")
echo "Progress:" &> progress.txt
echo "" >> progress.txt

echo "--------------- Transient rendering (`date`) --------------" >> progress.txt
echo "" >> progress.txt
for((num_id=0;num_id<3;num_id++)); do
    method=${methods[num_id]}
    image_num=${image_nums[num_id]}
    for((i=0;i<${image_num};i++)); do
        # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
        output_folder=./exp_scenes/$scene/transients/
        if [ ! -d $output_folder ]; then
            mkdir -p $output_folder
        fi

        mv_folder=./results/transients/trs$method-$i/
        if [ -d $mv_folder ]; then
            echo "$scene-$method: $i  Exists" >> progress.txt
            continue;
        fi

        pbrt_file=./exp_scenes/$scene/transient$method.json
        ./build/release/tungsten $pbrt_file -t 104 --seed $i
        if [ ! -d $mv_folder ]; then
            mv $output_folder $mv_folder
        fi
        echo "$scene-$method: $i  (`date`)" >> progress.txt
    done
done