#!/bin/bash -l
#
# Runs the SedovBlast_3D and EAGLE_12 examples using naive, serial and vectorised particle interactions. Then compares the output between versions using check_ngbs.py

./autogen.sh

echo
echo "# Running SedovBlast_3D and EAGLE_12 with naive interactions and neighbour logging, 16 thread"
echo

cd ../

# Naive interactions run
./configure --disable-mpi --enable-debug-interactions --disable-vec --enable-naive-interactions
make clean; make -j 6

cd examples/SedovBlast_3D/

./getGlass.sh
python makeIC.py

../swift -s -t 16 -n 5 sedov.yml -P SPH:h_tolerance:10

mv sedov_0000.hdf5 sedov_naive.hdf5

cd ../EAGLE_12/

# Link to ICs
ln -s /gpfs/data/Swift/web-storage/ICs/EAGLE_ICs_12.hdf5 EAGLE_ICs_12.hdf5

../swift -s -t 16 -n 5 eagle_12.yml -P SPH:h_tolerance:10  

mv eagle_0000.hdf5 eagle_12_naive.hdf5

cd ../../

echo
echo "# Running SedovBlast_3D and EAGLE_12 with serial interactions and neighbour logging, 16 thread"
echo

# Serial interactions run
./configure --disable-mpi --enable-debug-interactions --disable-vec
make clean; make -j 6

cd examples/SedovBlast_3D/

../swift -s -t 16 -n 5 sedov.yml -P SPH:h_tolerance:10

mv sedov_0000.hdf5 sedov_serial.hdf5

cd ../EAGLE_12/

../swift -s -t 16 -n 5 eagle_12.yml -P SPH:h_tolerance:10  

mv eagle_0000.hdf5 eagle_12_serial.hdf5

cd ../../

echo
echo "# Running SedovBlast_3D and EAGLE_12 with vectorised interactions and neighbour logging, 16 thread"
echo

# Vectorised interactions run
./configure --disable-mpi --enable-debug-interactions
make clean; make -j 6

cd examples/SedovBlast_3D/

../swift -s -t 16 -n 5 sedov.yml -P SPH:h_tolerance:10

mv sedov_0000.hdf5 sedov_vec.hdf5

# Compare outputs
python ../check_ngbs.py sedov_naive.hdf5 sedov_serial.hdf5 
python ../check_ngbs.py sedov_naive.hdf5 sedov_vec.hdf5 
python ../check_ngbs.py sedov_serial.hdf5 sedov_vec.hdf5 

cd ../EAGLE_12/

../swift -s -t 16 -n 5 eagle_12.yml -P SPH:h_tolerance:10  

mv eagle_0000.hdf5 eagle_12_vec.hdf5

# Compare outputs
python ../check_ngbs.py eagle_12_naive.hdf5 eagle_12_serial.hdf5 
python ../check_ngbs.py eagle_12_naive.hdf5 eagle_12_vec.hdf5 
python ../check_ngbs.py eagle_12_serial.hdf5 eagle_12_vec.hdf5 
