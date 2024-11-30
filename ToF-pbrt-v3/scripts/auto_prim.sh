strategies=("eps" "ts")
image_num=40
echo "Progress:" &> progress.txt

for strategy in ${strategies[@]}; do
    echo "" >> progress.txt
    echo "--------------- strategy = $strategy --------------" >> progress.txt
    echo "" >> progress.txt
    target_folder=./results/dir-$strategy-18.1-20000/
    if [ ! -d $target_folder ]; then
        mkdir -p $target_folder
    fi
    for((i=0;i<${image_num};i++)); do
        ./build/pbrt ./curves/$strategy/vpt-18.1-0.05.pbrt --nthreads=104
        mv ./cornell-vpt_0000.exr ${target_folder}cornell-vpt_${i}.exr
        echo "$strategy/vpt-18.1-0.05: $i" >> progress.txt
    done
    if [ -f ${target_folder}time.log ]; then
        cat time.log >> ${target_folder}time.log
        rm time.log
    else
        mv time.log ${target_folder}time.log
    fi
done
