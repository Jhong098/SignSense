# Use to move files generated recorder.py to appropriate locations (run from top level)
# Usage: number_of_vids outfile_prefix vid_location npy_location
for n in $(seq 0 $1); do
    mv vid$n.mp4 $3/$2${n}.mp4
    mv vid$n.npy $4/$2${n}.npy 
done
