# Mandelbrot Set Computation

## Usage

- The project contains three folders:
  - csc4005-assignment-3-mpi-pth-omp:
    
    - Contains MPI, pthread, OpenMP version
    - These projects can be compiled as the following
      ```bash
      cd /path/to/project
      mkdir build && cd build
      cmake .. -DCMAKE_BUILD_TYPE=Release
      cmake --build . -j4
      ```
    - The project will generate four executable files
      - csc4005_imgui_serial:
        
        The given serial version. It receives no arguments.
      - csc4005_imgui_mpi:
        
        The MPI version. It receives one integer that indicates number of bodies.
        - The way to run:
        use sbatch, example: 
        example.sh
        ```bash
        #!/bin/bash
        #SBATCH --account=csc4005
        #SBATCH --time=1
        #SBATCH --partition=debug
        #SBATCH --qos=normal
        #SBATCH --ntasks=1
        #SBATCH --nodes=1

        echo "mainmode: " && /bin/hostname
        xvfb-run -a mpirun -np 1 ./path/to/csc4005_imgui_mpi 200
        ```
        then type in terminal:
        ```bash
        sbatch example.sh
        ```

      - csc4005_imgui_pthread:

        The pthread version. It receives two integers that indicate number of thread and number of bodies respectively.

        - The way to run:
        use sbatch, example: 
        example.sh
        ```bash
        #!/bin/bash
        #SBATCH --account=csc4005
        #SBATCH --time=1
        #SBATCH --partition=debug
        #SBATCH --qos=normal
        #SBATCH --ntasks=1
        #SBATCH --nodes=1

        echo "mainmode: " && /bin/hostname
        xvfb-run -a mpirun -np 1 ./path/to/csc4005_imgui_pthread 2 200
        ```
        then type in terminal:
        ```bash
        sbatch example.sh
        ```

      - csc4005_imgui_omp

        The OpenMP version. It receives two integers that indicate number of bodies and number of threads respectively.

        - The way to run:
        use sbatch, example: 
        example.sh
        ```bash
        #!/bin/bash
        #SBATCH --account=csc4005
        #SBATCH --time=1
        #SBATCH --partition=debug
        #SBATCH --qos=normal
        #SBATCH --ntasks=1
        #SBATCH --nodes=1

        echo "mainmode: " && /bin/hostname
        xvfb-run -a mpirun -np 1 ./path/to/csc4005_imgui_omp 200 2
        ```
        then type in terminal:
        ```bash
        sbatch example.sh
        ```

  
  - csc4005-assignment-3-cuda:

    - Contains cuda version

    - This project can be compiled as the following:
      ```bash
      cd /path/to/project
      mkdir build && cd build
      source scl_source enable devtoolset-10
      CC=gcc CXX=g++ cmake ..
      make -j4
      ```
    
    - The project will generate one executable file
      - csc4005_imgui_cuda: 

        The cuda version. It receives two integers that indicate number of bodies and number of blocks or threads respectively.
        - The way to run:
        use sbatch, example: 
        example.sh
        ```bash
        #!/bin/bash
        #SBATCH --account=csc4005
        #SBATCH --time=1
        #SBATCH --partition=debug
        #SBATCH --qos=normal
        #SBATCH --ntasks=1
        #SBATCH --nodes=1

        echo "mainmode: " && /bin/hostname
        xvfb-run -a mpirun -np 1 ./path/to/csc4005_imgui_cuda 200 10
        ```
        then type in terminal:
        ```bash
        sbatch example.sh
        ```

  - csc4005-assignment-3-hybrid
    
    - Contains the MPI+OpenMP version
    - This projects can be compiled as the following:
      ```bash
      cd /path/to/project
      mkdir build && cd build
      cmake .. -DCMAKE_BUILD_TYPE=Release
      cmake --build . -j4
      ```
    - The project will generate one executable file
      - csc4005_imgui_hybrid: It receives two integers that indicate number of bodies, number of total threads respectively.
        - The way to run:
        use sbatch, example: 
        example.sh
        ```bash
        #!/bin/bash
        #SBATCH --account=csc4005
        #SBATCH --time=1
        #SBATCH --partition=debug
        #SBATCH --qos=normal
        #SBATCH --ntasks=1
        #SBATCH --nodes=1

        echo "mainmode: " && /bin/hostname
        xvfb-run -a mpirun -np 1 ./path/to/csc4005_imgui_hybrid 200 33
        ```
        then type in terminal:
        ```bash
        sbatch example.sh
        ```

      
