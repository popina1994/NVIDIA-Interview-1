# X = 600
for ((yGrid=1;yGrid <=150;yGrid++))
do
	echo "Choosing block x size $i yGridSize: $yGrid"
	nvcc -D X_BLOCK=1024 -D Y_GRID="$yGrid" main.cu -o main && ./main  2>&1 | tee log.txt
done
