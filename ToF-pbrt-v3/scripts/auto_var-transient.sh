scenes=("dragon")
strategies=("origin")
image_num=20
echo "Progress:" &> progress.txt

for scene in ${scenes[@]}; do
    echo "" >> progress.txt
    echo "--------------- Scene = $scene (`date`) --------------" >> progress.txt
    echo "" >> progress.txt

    for((i=0;i<${image_num};i++)); do
        # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
        for strategy in ${strategies[@]}; do
            target_folder=./results/transient-$scene-$strategy-$i/
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            fi
            ./build/pbrt ./experiments/$strategy/$scene/$scene-transient.pbrt --nthreads=104

            mv ./test_result/*.exr ${target_folder}
            echo "$strategy: $i (`date`)" >> progress.txt
        done
    done
    if [ -f time.log ]; then
        for strategy in ${strategies[@]}; do
            target_folder=./results/transient-$scene-$strategy-$i/
            if [ -f ${target_folder}time.log ]; then
                cat time.log >> ${target_folder}time.log
            else
                cp time.log ${target_folder}time.log
            fi
        done
        rm time.log
    fi
done
