for file in $(find /home/crc00098/ReACG/result-n100/2025-06-13/ -name "run.yaml")
do
    cp $file "${file}.old3"
    sed -i /dictitems/d $file
    sed -i /state/d $file
    sed -i /param/d $file
    sed -i /^batch_size/d $file
    sed -i 's/^[[:space:]]*//' $file
    sed -i '/^-g/d' $file
done