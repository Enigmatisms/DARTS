scenes=("staircase" "dragon")
strategy="darts"
image_num=60
echo "Progress:" &> progress.txt

for scene in ${scenes[@]}; do
    echo "" >> progress.txt
    echo "--------------- Scene = $scene-$strategy --------------" >> progress.txt
    echo "" >> progress.txt
    valid=0
    target_folder=./results/bdpt-$scene-$strategy/
    for((i=0;i<${image_num};i++)); do
        # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
        if [ -f ${target_folder}result_${i}.exr ]; then
            continue
        fi
        valid=1
        if [ ! -d $target_folder ]; then
            mkdir -p $target_folder
        fi
        pbrt_file=./experiments/$strategy/$scene/$scene-gated-bdpt.pbrt
        python3 ./modifier.py $pbrt_file $time_width
        ./build/pbrt $pbrt_file --nthreads=104

        mv ./${scene}_0000.exr ${target_folder}result_${i}.exr
        echo "$strategy$setting-$time_width: $i" >> progress.txt
    done
    if [ ! $valid -eq 1 ]; then
        continue
    fi
    if [ -f time.log ]; then
        if [ -f ${target_folder}time.log ]; then
            cat time.log >> ${target_folder}time.log
        else
            cp time.log ${target_folder}time.log
        fi
        rm time.log
    fi
done
