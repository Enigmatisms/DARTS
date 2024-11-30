scene="staircase"
strategies=("darts" "origin")
settings=("-short" "-long")
all_time_widths=(0.05 0.125 0.25 0.5 1.0 2.0)
image_num=40
echo "Progress:" &> progress.txt

for setting in ${settings[@]}; do
    for strategy in ${strategies[@]}; do
    echo "" >> progress.txt
    echo "--------------- Scene = $scene-$strategy$setting (`date`) --------------" >> progress.txt
    echo "" >> progress.txt
    for time_width in ${all_time_widths[@]}; do 
        valid=0
        for((i=0;i<${image_num};i++)); do
            # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
            target_folder=./results/$scene-$strategy$setting/$time_width/
            if [ -f ${target_folder}result_${i}.exr ]; then
                echo "$strategy$setting-$time_width: $i Exists" >> progress.txt
                continue
            fi
            valid=1
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            fi
            pbrt_file=./experiments/$strategy/$scene/$scene-gated$setting.pbrt
            python3 ./modifier.py $pbrt_file $time_width
            ./build/pbrt $pbrt_file --nthreads=104

            mv ./${scene}_0000.exr ${target_folder}result_${i}.exr
            echo "$strategy$setting-$time_width: $i (`date`)" >> progress.txt
        done
        if [ ! $valid -eq 1 ]; then
            continue
        fi
        if [ -f time.log ]; then
            target_folder=./results/$scene-$strategy$setting/$time_width/
            if [ -f ${target_folder}time.log ]; then
                cat time.log >> ${target_folder}time.log
            else
                cp time.log ${target_folder}time.log
            fi
            rm time.log
        fi
    done
    done
done
