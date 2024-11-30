image_num=30
methods=('darts')
frameworks=('pt')
output_name="kitchen.pfm"

echo "Progress:" &> progress.txt
echo "" >> progress.txt

echo "--------------- (`date`) --------------" >> progress.txt
echo "" >> progress.txt
for((i=16;i<${image_num};i++)); do
    for method in ${methods[@]}; do
    for framework in ${frameworks[@]}; do
    # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
    output_folder=./results/teaser/$framework-$method/
    if [ ! -d $output_folder ]; then
        mkdir -p $output_folder
    fi
    pfm_file="./exp_scenes/kitchen/${output_name}"
    renamed_file="${output_folder}result_$i.pfm"
    if [ -f $renamed_file ]; then
        echo "teaser: $i  Exists" >> progress.txt
        continue;
    fi

    pbrt_file=./exp_scenes/kitchen/$framework-$method.json
    ./build/release/tungsten $pbrt_file -t 104 --seed $i
    mv $pfm_file "${output_folder}result_$i.pfm"
    echo "Teaser: $i  (`date`)" >> progress.txt

    done
    done
done