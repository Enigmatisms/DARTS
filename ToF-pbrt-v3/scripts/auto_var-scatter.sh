scenes=("staircase")
strategies=("darts")
settings=("-short" "-long")
image_num=40
# echo "Progress:" &> progress.txt

for scene in ${scenes[@]}; do
    for setting in ${settings[@]}; do
        echo "" >> progress.txt
        echo "--------------- Scene = $scene$setting (`date`) --------------" >> progress.txt
        echo "" >> progress.txt
        if [ $scene = "staircase" ]; then
            all_sigma_s=("0.125 0.25 0.375 0.5 0.625 0.75")
        else 
            all_sigma_s=("0.25 0.5 0.75 1.0 1.25 1.5")              # 1.0 is not added
        fi
        for sigma_s in  ${all_sigma_s[@]}; do 
            valid=0
            for((i=0;i<${image_num};i++)); do
                # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
                for strategy in ${strategies[@]}; do
                    target_folder=./results/$scene-$strategy$setting/$sigma_s/
                    if [ ! -d $target_folder ]; then
                        mkdir -p $target_folder
                    fi
                    if [ -f ${target_folder}result_${i}.exr ]; then
                        continue
                    fi
                    valid=1

                    pbrt_file=./experiments/$strategy/$scene/$scene-gated$setting.pbrt
                    python3 ./modifier.py $pbrt_file $sigma_s
                    ./build/pbrt $pbrt_file --nthreads=104

                    mv ./${scene}_0000.exr ${target_folder}result_${i}.exr
                    echo "$strategy$setting-$sigma_s: $i (`date`)" >> progress.txt
                done
            done
            if [ ! $valid -eq 1 ]; then
                continue
            fi
            if [ -f time.log ]; then
                for strategy in ${strategies[@]}; do
                    target_folder=./results/$scene-$strategy$setting/$sigma_s/
                    if [ -f ${target_folder}time.log ]; then
                        cat time.log >> ${target_folder}time.log
                    else
                        cp time.log ${target_folder}time.log
                    fi
                done
                rm time.log
            fi

        done
    done
done
