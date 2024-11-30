for file in `ls ./configs/vids/*.conf`; do
    echo $file
    python3 ./make_video.py --config $file
done