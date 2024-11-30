scene="cbox"
strategy="darts-full"
image_num=40
echo "Progress:" &> progress.txt

for((i=0;i<${image_num};i++)); do
    echo "" >> progress.txt
    echo "--------------- Scene = $scene --------------" >> progress.txt
    echo "" >> progress.txt

    # I opt for an interlaced running pattern: to get rid of the effect from cache coherence
    target_folder=./results/$scene-$strategy-path/
    if [ ! -d $target_folder ]; then
        mkdir -p $target_folder
    fi
    ./build/pbrt ./experiments/$scene/$scene-equi-prop.pbrt --nthreads=104

    mv ./${scene}_0000.exr ${target_folder}result_${i}.exr
    echo "$strategy-$strategy: $i" >> progress.txt
done
if [ -f time.log ]; then
    target_folder=./results/$scene-$strategy-path/
    if [ -f ${target_folder}time.log ]; then
        cat time.log >> ${target_folder}time.log
    else
        cp time.log ${target_folder}time.log
    fi
    rm time.log
fi
