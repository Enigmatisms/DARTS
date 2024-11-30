scenes=("staircase" "dragon")
strategies=("darts")
settings=("-short" "-long")
image_num=120
echo "Progress:" &> progress.txt

for((i=0;i<${image_num};i++)); do
    for scene in ${scenes[@]}; do
        for setting in ${settings[@]}; do
            echo "" >> progress.txt
            echo "--------------- Scene = $scene$setting (`date`) --------------" >> progress.txt
            echo "" >> progress.txt

            # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
            for strategy in ${strategies[@]}; do
                target_folder=./results/$scene-$strategy$setting/
                if [ ! -d $target_folder ]; then
                    mkdir -p $target_folder
                fi
                output_file=${target_folder}result_${i}.exr
                if [ -f $output_file ]; then
                    echo "$strategy$setting: $i exists" >> progress.txt
                    continue
                fi
                ./build/pbrt ./experiments/$strategy/$scene/$scene-gated$setting.pbrt --nthreads=104

                mv ./${scene}_0000.exr ${output_file} 
                echo "$strategy$setting: $i  (`date`)" >> progress.txt
            done
        done
    done
done
