scene="dragon"
strategy="darts"
setting="-short"
image_num=(160 40 30 30 30 20)
sample_num=(1 10 100 1000 10000 100000)
echo "Progress:" &> progress.txt

pbrt_file=./experiments/$strategy/$scene/$scene-gated$setting.pbrt
echo $pbrt_file
for((k=0;k<6;k++)); do
    img_num=${image_num[$k]};
    smp_num=${sample_num[$k]};
    target_folder=./results/converge-$scene-$strategy$setting/$smp_num/
    python3 ./modifier.py $pbrt_file $smp_num
    echo "" >> progress.txt
    echo "--------------- Scene = $scene$setting sample num: $smp_num (`date`) --------------" >> progress.txt
    echo "" >> progress.txt
    valid=0
    for((i=1;i<${img_num};i++)); do
        if [ ! -d $target_folder ]; then
            mkdir -p $target_folder
        fi
        if [ -f ${target_folder}result_${i}.exr ]; then
            continue
        fi
        valid=1
        ./build/pbrt $pbrt_file --nthreads=104

        mv ./${scene}_0000.exr ${target_folder}result_${i}.exr
        echo "$strategy$setting-$smp_num: $i (`date`)" >> progress.txt
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
