# MoveUrBody (`MUrB`), a n-body code

This is a n-body code that simulates the Newtonian gravity equations.

## How to compile and run the code

This project uses `cmake` in order to generate any type of projects (Makefile, 
Visual Studio projects, Eclipse projects, etc.).

On Linux/Debian-like systems:

```bash
sudo apt-get install cmake
```

### Get the Git Submodules

`MUrB` depends on some other Git repositories (or submodules). It is highly 
recommended to get those submodules before trying to do anything else. Here is 
the command to get all the required submodules:

```bash
git submodule update --init --recursive
```

### Other Dependencies

`MUrB` comes with a *cool* real time display engine. To enjoy it, you have to 
install some dependencies: `OpenGL >= 3.0`, `GLEW >= 1.11.0`, `GLM >= 0.9.5.4` 
and `GLFW >= 3.0.4` libraries are required. If one on these libraries is missing 
on your system, then `MUrB` will be compiled in console mode only.

On Ubuntu 20.04 you can install the required libraries as follow:

```bash
sudo apt install libglfw3-dev libglew-dev libglm-dev
```

### Compilation with Makefile

Open a terminal and type (from the `MUrB` root folder):

```bash
mkdir build
cd build
cmake .. -G"Unix Makefiles" -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -g" -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native"
make -j4
```

## Run the code

Run 1000 bodies (`-n`) during 1000 iterations (`-i`) and enable the verbose mode 
(`-v`):

```bash
./bin/murb -n 1000 -i 1000 -v
```

Expected output:

```
n-body simulation configuration:
--------------------------------
  -> bodies scheme     (-s    ): galaxy
  -> implementation    (--im  ): cpu+naive
  -> nb. of bodies     (-n    ): 1000
  -> nb. of iterations (-i    ): 1000
  -> verbose mode      (-v    ): enable
  -> precision                 : fp32
  -> mem. allocated            : 0.0724792 MB
  -> geometry shader   (--ngs ): enable
  -> time step         (--dt  ): 3600.000000 sec
  -> softening factor  (--soft): 2e+08
Compiling shader: ../src/common/ogl/shaders/vertex330_color_v2.glsl
Compiling shader: ../src/common/ogl/shaders/geometry330_color_v2.glsl
Compiling shader: ../src/common/ogl/shaders/fragment330_color_v2.glsl
Linking shader program... SUCCESS !

Simulation started...
Iteration n°1000 ( 729.9 FPS), physic time:   41d   16h    0m 0.000s
Simulation ended.

Entire simulation took 1370.0 ms (729.9 FPS)
```

### Command Line Parameters

Here is the help (`-h`) of `MUrB`:
```
Usage: ./bin/murb -i nIterations -n nBodies [--dt timeStep] [--gf] [--help] [--im ImplTag] [--ngs] [--nv] [--nvc] [--soft softeningFactor] [--wg workGroup] [--wh winHeight] [--ww winWidth] [-h] [-s Bodies scheme] [-v]

  -i      the number of iterations to compute.
  -n      the number of generated bodies.
  --dt    select a fixed time step in second (default is 3600.000000 sec).
  --gf    display the number of GFlop/s.
  --help  display this help.
  --im    code implementation tag:
           - "cpu+naive"
           ----
  --ngs   disable geometry shader for visu (slower but it should work with old GPUs).
  --nv    no visualization (disable visu).
  --nvc   visualization without colors.
  --soft  softening factor.
  --wh    the height of the window in pixel (default is 768).
  --ww    the width of the window in pixel (default is 1024).
  -h      display this help.
  -s      bodies scheme (initial conditions can be "galaxy" or "random").
  -v      enable verbose mode.
```

