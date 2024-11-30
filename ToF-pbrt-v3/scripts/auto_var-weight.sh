scenes=("staircase")
strategies=("no-sample" "sample")
image_num=8
echo "Progress:" &> progress.txt

for scene in ${scenes[@]}; do
    echo "" >> progress.txt
    echo "--------------- Scene = $scene (`date`) --------------" >> progress.txt
    echo "" >> progress.txt

    for((i=0;i<${image_num};i++)); do
        # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
        for strategy in ${strategies[@]}; do
            target_folder=./results/weight-$scene-$strategy-$i/
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            fi
            pbrt_file=./experiments/darts/$scene/$scene-$strategy.pbrt
            ./build/pbrt $pbrt_file --nthreads=104

            mv ./test_result/*.exr ${target_folder}
            echo "$strategy: $i (`date`)" >> progress.txt
        done
    done
done
