# Shell script for running no path reuse DARTS-PP transient rendering

total_image_num=20
frame_num=40
scene="staircase"
# type=""
type=""
echo "Progress:" &> progress.txt
echo "" >> progress.txt
frame_width=0.25
frame_time_second=90

add_floats() {
    echo "$1 + $2" | bc
}

for((img_num=1;img_num<${total_image_num};img_num++)); do

echo "--------------- scene$type: $scene - [$img_num]  (`date`) --------------" >> progress.txt
echo "" >> progress.txt

mv_folder=./results/transients/$scene/trs-darts-pp-$img_num/
if [ -d $mv_folder ]; then
    echo "$scene: $i  Exists" >> progress.txt
    continue;
fi
start_time=8.5

for((i=0;i<${frame_num};i++)); do
    output_folder=./exp_scenes/$scene/transients/
    pfm_file="${output_folder}result.pfm"
    renamed_file="${output_folder}result_$i.pfm"
    if [ ! -d $output_folder ]; then
        mkdir -p $output_folder
    fi
    if [ -f $pfm_file ]; then
        mv $pfm_file $renamed_file
        echo "$start_time: $i  Exists" >> progress.txt
        start_time=$(add_floats $start_time $frame_width)
        continue;
    fi
    if [ -f $renamed_file ]; then
        echo "$start_time: $i  Exists" >> progress.txt
        start_time=$(add_floats $start_time $frame_width)
        continue;
    fi

    scene_json=./exp_scenes/$scene/transient${type}.json
    python3 ./modifier.py -f $scene_json --sigma_s 0.5 --tw $frame_width --at $frame_time_second --start_time $start_time
    ./build/release/tungsten $scene_json -t 104 --seed $img_num
    mv $pfm_file "${output_folder}result_$i.pfm"
    echo "$scene-$start_time: $i  (`date`)" >> progress.txt
    start_time=$(add_floats $start_time $frame_width)
done

if [ ! -d $mv_folder ]; then
    mv $output_folder $mv_folder
fi

echo "Trial finished (`date`)" >> progress.txt

done
