scene="cbox-pt"
scatter_param=(0.4 0.2)
time_widths=(0.02 0.1)
image_num=40
start_num=0
elliptical=(0 1)
da_guiding=(0 1)
names=('origin' 'da-only' 'tsample' 'darts')

# photon points
echo "Progress:" &> progress.txt

for((scat_id=0;scat_id<2;scat_id++)); do
    sigma_s=${scatter_param[scat_id]}
    for twidth in ${time_widths[@]}; do
        echo "" >> progress.txt
        echo "--------------- Scene = $scene-$sigma_s-$twidth (`date`) --------------" >> progress.txt
        echo "" >> progress.txt
        for use_ell in ${elliptical[@]}; do
        for use_da in ${da_guiding[@]}; do
        val=$(($use_ell*2+$use_da))
        name=${names[val]}
        v_radius=${v_radii[val]}
        target_folder=./results/ablation/$scene-$sigma_s-$twidth/$name/
        for((i=$start_num;i<${image_num};i++)); do
            if [ ! -d $target_folder ]; then
                mkdir -p $target_folder
            else 
                if [ -f ${target_folder}result_${i}.exr ]; then
                    echo "$scene-$sigma_s-$twidth/$name: $i Exists" >> progress.txt
                    continue
                fi
            fi
            valid=1

            output_name="cbox_0000.exr"
            output_folder=./
            if [ ! -d $output_folder ]; then
                mkdir -p $output_folder
            fi
            pfm_file="${output_folder}${output_name}"
            renamed_file="${target_folder}result_$i.exr"

            scene_json=./experiments/darts/cbox-vpt-${name}.pbrt
            python3 ./modifier.py $scene_json $sigma_s $twidth
            ./build/pbrt $scene_json --nthreads 104

            mv $pfm_file $renamed_file
            # echo "$scene-$sigma_s-$twidth/$name: $use_ell $use_da (`date`)" >> progress.txt
            echo "$scene-$sigma_s-$twidth/$name: $i (`date`)" >> progress.txt
        done
        if [ -f time.log ]; then
            if [ -f ${target_folder}time.log ]; then
                cat time.log >> ${target_folder}time.log
            else
                cp time.log ${target_folder}time.log
            fi
            rm time.log
        fi

        done
        done
        if [ ! $valid -eq 1 ]; then
            continue
        fi
    done
done