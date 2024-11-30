strategies=("origin")
sigma_ts=(0.01 0.02 0.1 0.2 0.5 1.0)
image_num=40
echo "Progress:" &> progress.txt

for sigma_t in ${sigma_ts[@]}; do
    echo "" >> progress.txt
    echo "--------------- Sigma_t = $sigma_t --------------" >> progress.txt
    echo "" >> progress.txt
    for strategy in ${strategies[@]}; do
        target_folder=./results/curve-$strategy-$sigma_t-16.1-16384/
        if [ ! -d $target_folder ]; then
            mkdir -p $target_folder
        fi
        echo "--------------- ($strategy) --------------" >> progress.txt
        for((i=0;i<${image_num};i++)); do
            ./build/pbrt ./curves/$strategy/vpt-16.1-$sigma_t.pbrt --nthreads=104
            mv ./cornell-vpt_0000.exr ${target_folder}cornell-vpt_${i}.exr
            echo "$strategy/vpt-16.1-$sigma_t: $i" >> progress.txt
        done
        if [ -f ${target_folder}time.log ]; then
            cat time.log >> ${target_folder}time.log
            rm time.log
        else
            mv time.log ${target_folder}time.log
        fi
    done
done
