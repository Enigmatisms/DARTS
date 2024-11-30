input_folder=$1
output_folder=$2
obj_path=${output_folder}objs/

python3 ./ply2obj.py --input_folder ${input_folder} --output_folder ${obj_path}
for file in `ls ${obj_path}*.obj`; do
    no_ext_name=`python3 ./get_no_ext.py ${file}`
    ../build/release/obj2json $file ${output_folder}${no_ext_name}.json
    echo "$no_ext_name is processed." 
    # sleep 0.1
    mv ${output_folder}Mesh1.wo3 ${output_folder}${no_ext_name}.wo3
    rm ${output_folder}*.json
done