#include "core/CUDABodies.hpp"

template <typename T>
CUDABodies<T>::CUDABodies(const unsigned long n, const std::string &scheme, const unsigned long randInit) 
    : Bodies<T>(n, scheme, randInit)    
{
    this->allocateBuffersOnDevice();
    this->memcpyBuffersOnDevice();
}

template <typename T>
void CUDABodies<T>::allocateBuffersOnDevice() {
    cudaMalloc(&this->devDataSoA.m, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devDataSoA.r, (this->n + this->padding)*sizeof(T));

    cudaMalloc(&this->devDataSoA.qx, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devDataSoA.qy, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devDataSoA.qz, (this->n + this->padding)*sizeof(T));

    cudaMalloc(&this->devDataSoA.vx, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devDataSoA.vy, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devDataSoA.vz, (this->n + this->padding)*sizeof(T));

    // Internal buffers for Leapfrog
    cudaMalloc(&this->devIntermVelocities.x, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devIntermVelocities.y, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devIntermVelocities.z, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devNextPositions.x, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devNextPositions.y, (this->n + this->padding)*sizeof(T));
    cudaMalloc(&this->devNextPositions.z, (this->n + this->padding)*sizeof(T));
}

template <typename T>
void CUDABodies<T>::memcpyBuffersOnDevice() {
    cudaMemcpy(this->devDataSoA.m, this->dataSoA.m.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devDataSoA.r, this->dataSoA.r.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(this->devDataSoA.qx, this->dataSoA.qx.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devDataSoA.qy, this->dataSoA.qy.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devDataSoA.qz, this->dataSoA.qz.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(this->devDataSoA.vx, this->dataSoA.vx.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devDataSoA.vy, this->dataSoA.vy.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devDataSoA.vz, this->dataSoA.vz.data(), (this->n + this->padding)*sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(this->devNextPositions.x, this->devDataSoA.qx, (this->n + this->padding)*sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->devNextPositions.y, this->devDataSoA.qy, (this->n + this->padding)*sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->devNextPositions.z, this->devDataSoA.qz, (this->n + this->padding)*sizeof(T), cudaMemcpyDeviceToDevice);
}
// ======================================================================================
// =============================== GETTERS and SETTERS ==================================
// ======================================================================================
template <typename T>
const devDataSoA_t<T>& CUDABodies<T>::getDevDataSoA() const {
    return this->devDataSoA;
}

template <typename T>
void CUDABodies<T>::invalidateDataSoA() {
    this->dataOnCPU = false;
}

template <typename T>
const dataSoA_t<T>& CUDABodies<T>::getDataSoA() const {
    if ( this->dataOnCPU == false ) {
        // Copies the data from device to host, then returns it 

        // Mass and radius do not change

        // Positions
        cudaMemcpy(Bodies<T>::dataSoA.qx.data(), this->devDataSoA.qx, (this->n + this->padding)*sizeof(T), 
                cudaMemcpyDeviceToHost);

        cudaMemcpy(Bodies<T>::dataSoA.qy.data(), this->devDataSoA.qy, (this->n + this->padding)*sizeof(T), 
                cudaMemcpyDeviceToHost);

        cudaMemcpy(Bodies<T>::dataSoA.qz.data(), this->devDataSoA.qz, (this->n + this->padding)*sizeof(T), 
                cudaMemcpyDeviceToHost);

        // Velocities
        cudaMemcpy(Bodies<T>::dataSoA.vx.data(), this->devDataSoA.vx, (this->n + this->padding)*sizeof(T), 
                cudaMemcpyDeviceToHost);

        cudaMemcpy(Bodies<T>::dataSoA.vy.data(), this->devDataSoA.vy, (this->n + this->padding)*sizeof(T), 
                cudaMemcpyDeviceToHost);
        cudaMemcpy(Bodies<T>::dataSoA.vz.data(), this->devDataSoA.vz, (this->n + this->padding)*sizeof(T), 
                cudaMemcpyDeviceToHost);

        this->dataOnCPU = true;
    }

    return this->dataSoA;
}

template <typename T>
const std::vector<dataAoS_t<T>>& CUDABodies<T>::getDataAoS() const {
    this->getDataSoA();
    printf("\n\ngetDataAoS NOT IMPLEMENTED!!\n\n");
    return this->dataAoS;
}

template <typename T>
const devAccSoA_t<T>& CUDABodies<T>::getDevPositionsBuffer() const {
    return this->devNextPositions;
}

// template <typename T>
// void CUDABodies<T>::swapPositionsBuffers() {
//     T* tmp;
//     tmp = this->devPositionsBuffer.x;
//     this->devPositionsBuffer.x = this->devDataSoA.qx;
//     this->devDataSoA.qx = tmp;
//     tmp = this->devPositionsBuffer.y;
//     this->devPositionsBuffer.y = this->devDataSoA.qy;
//     this->devDataSoA.qy = tmp;
//     tmp = this->devPositionsBuffer.z;
//     this->devPositionsBuffer.z = this->devDataSoA.qz;
//     this->devDataSoA.qz = tmp;
// }

// ======================================================================================
// ========================== UPDATE POSITIONS & VELOCITIES =============================
// ======================================================================================

template <typename T>
__global__ void devUpdatePositionsAndVelocities(const devAccSoA_t<T> devAccelerations, T dt, devDataSoA_t<T> devDataSoA, unsigned long n) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (iBody >= n) return;

    T aixDt = devAccelerations.x[iBody] * dt;
    T aiyDt = devAccelerations.y[iBody] * dt;
    T aizSt = devAccelerations.z[iBody] * dt;

    T vix = devDataSoA.vx[iBody];
    T viy = devDataSoA.vy[iBody];
    T viz = devDataSoA.vz[iBody];

    T qixNew = devDataSoA.qx[iBody] + (vix + aixDt * 0.5) * dt;
    T qiyNew = devDataSoA.qy[iBody] + (viy + aiyDt * 0.5) * dt;
    T qizNew = devDataSoA.qz[iBody] + (viz + aizSt * 0.5) * dt;

    T vixNew = vix + aixDt;
    T viyNew = viy + aiyDt;
    T vizNew = viz + aizSt;

    devDataSoA.qx[iBody] = qixNew;
    devDataSoA.qy[iBody] = qiyNew;
    devDataSoA.qz[iBody] = qizNew;
    devDataSoA.vx[iBody] = vixNew;
    devDataSoA.vy[iBody] = viyNew;
    devDataSoA.vz[iBody] = vizNew;
}

template <typename T>
void CUDABodies<T>::updatePositionsAndVelocitiesOnDevice(const devAccSoA_t<T> &devAccelerations, T &dt) 
{
    this->invalidateDataSoA();
    int threads = 1024;
    int blocks = (this->n + threads - 1) / threads;  
    if ( blocks == 1 ) {
        threads = this->n;
    }
    devUpdatePositionsAndVelocities<T><<<blocks, threads>>>(devAccelerations, dt, this->devDataSoA, this->n);
    cudaDeviceSynchronize();
}


/* 
    LEAPFROG SCHEME:
    Original formulation:
        v_{n+1/2} = v_n + a_n * DeltaT / 2
        x_{n+1} = x_n + v_{n+1/2} * DeltaT
        v_{n+1} = v_{n+1/2} + a_{n+1} * DeltaT / 2

    Looking at consecutive iterations I have:
                    | v_{2+1/2} =  v_{5/2} = v_2 + a_2 * DeltaT / 2
            iter 2  | x_{2+1} = x_3 = x_2 + v_{2+1/2} * DeltaT
                    |         = x_2 + v_{5/2}
                    | v_{2+1} = v_3 = v_{2+1/2} + a_{2+1} * DeltaT / 2
                    |         = v_{5/2} + a_3 * Delta T / 2

                    | v_{3+1/2} = v_{7/2} = v_3 + a_3 * DeltaT / 2 
            iter 3  | x_{3+1} = x_4 = x_3 + v_{3+1/2} * DeltaT
                    |         = x_3 + v_{7/2}
                    | v_{3+1} = v_4 = v_{3+1/2} + a_{3+1} * DeltaT / 2
                    |         = v_{7/2} + a_4 * DeltaT / 2
    
    I can reorder them such that:
        v_{3} = v_{5/2} + a_3 * DeltaT / 2
        v_{7/2} = v_{3} + a_3 * DeltaT / 2
                = v_{5/2} + a_3 * DeltaT
        x_4 = x_3 + v_{7/2} * DeltaT
        
    ### Iterations
        - First iteration: v_0 is already computed, so we do
            v_{1/2} = v_{0} + a_0 * DeltaT / 2
            x_1 = x_0 + v_{1/2} * DeltaT
        - Middle iterations (with n=1...last-1)
            v_{n} = v_{n-1/2} + a_n * DeltaT / 2
            v_{n+1/2} = v_{n} + a_n * DeltaT / 2
            x_{n+1} = x_n + v_{n+1/2} * DeltaT
        - Last iteration: 
            v_{last} ~= v_{last-1/2}
                theoretically, I should have one calculation of the acceleration more than
                the number of considered iterations, but I will not do so and, for 
                semplicity, I will do v_{last} = v_{last-1/2}
            
*/


template <typename T>
__global__ void devLeapfrogFirst(const devAccSoA_t<T> devAccelerations, 
                                  devAccSoA_t<T> devIntermVelocities, devAccSoA_t<T> devNextPositions, 
                                 devDataSoA_t<T> devDataSoA, 
                                 unsigned long n, T dt) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (iBody >= n) return;

    // === Doing iteration 0
    // Result:
    // devDataSoA not modified -> contains v_0 and x_0
    // x_1 and v_{1/2} stored in temporary buffers

    // v_{1/2} = v_{0} + a_0 * DeltaT / 2
    T vixHalf = devDataSoA.vx[iBody] + devAccelerations.x[iBody] * dt / 2;
    T viyHalf = devDataSoA.vy[iBody] + devAccelerations.y[iBody] * dt / 2;
    T vizHalf = devDataSoA.vz[iBody] + devAccelerations.z[iBody] * dt / 2;
    devIntermVelocities.x[iBody] = vixHalf;
    devIntermVelocities.y[iBody] = viyHalf;
    devIntermVelocities.z[iBody] = vizHalf;

    // x_1 = x_0 + v_{1/2} * DeltaT
    // Not storing x_1 in devDataSoA to avoid having mismatching velocities
    // and positions
    devNextPositions.x[iBody] = devDataSoA.qx[iBody] + vixHalf * dt;
    devNextPositions.y[iBody] = devDataSoA.qy[iBody] + viyHalf * dt;
    devNextPositions.z[iBody] = devDataSoA.qz[iBody] + vizHalf * dt;

}

template <typename T>
__global__ void devLeapfrogMiddle(const devAccSoA_t<T> devAccelerations, 
                                  devAccSoA_t<T> devIntermVelocities, devAccSoA_t<T> devNextPositions, 
                                 devDataSoA_t<T> devDataSoA, 
                                 unsigned long n, T dt) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (iBody >= n) return;
    // === Doing iteration n
    //    1. v_{n} = v_{n-1/2} + a_n * DeltaT / 2
    //    2. v_{n+1/2} = v_{n} + a_n * DeltaT / 2
    //    3. x_{n+1} = x_n + v_{n+1/2} * DeltaT
    //
    // Results:
    // devDataSoA contains x_n and v_n
    // x_{n+1} and v_{n+1/2} stored in temporary buffers

    T qixN = devNextPositions.x[iBody];
    T qiyN = devNextPositions.y[iBody];
    T qizN = devNextPositions.z[iBody];
    devDataSoA.qx[iBody] = qixN;   // devDataSoA contains x_n
    devDataSoA.qy[iBody] = qiyN;  
    devDataSoA.qz[iBody] = qizN;

    T aixN_dt_half = devAccelerations.x[iBody] * dt / 2; // a_n * dt / 2
    T aiyN_dt_half = devAccelerations.y[iBody] * dt / 2;
    T aizN_dt_half = devAccelerations.z[iBody] * dt / 2;

    // === 1. v_{n} = v_{n-1/2} + a_n * DeltaT / 2
    T vixN = devIntermVelocities.x[iBody] + aixN_dt_half;
    T viyN = devIntermVelocities.y[iBody] + aiyN_dt_half;
    T vizN = devIntermVelocities.z[iBody] + aizN_dt_half;
    devDataSoA.vx[iBody] = vixN;    // devDataSoA contains v_n
    devDataSoA.vy[iBody] = viyN;
    devDataSoA.vz[iBody] = vizN;

    // === 2. v_{n+1/2} = v_{n} + a_n * DeltaT / 2
    T vixNPlusHalf = vixN + aixN_dt_half;
    T viyNPlusHalf = viyN + aiyN_dt_half;
    T vizNPlusHalf = vizN + aizN_dt_half;
    devIntermVelocities.x[iBody] = vixNPlusHalf;   // saving v_{n+1/2} to temporary buffer
    devIntermVelocities.y[iBody] = viyNPlusHalf;
    devIntermVelocities.z[iBody] = vizNPlusHalf;

    // === 3. x_{n+1} = x_n + v_{n+1/2} * DeltaT
    T qixNPlus1 = qixN + vixNPlusHalf * dt;
    T qiyNPlus1 = qiyN + viyNPlusHalf * dt;
    T qizNPlus1 = qizN + vizNPlusHalf * dt;
    devNextPositions.x[iBody] = qixNPlus1; // saving x_{n+1} to temporary buffer
    devNextPositions.y[iBody] = qiyNPlus1;
    devNextPositions.z[iBody] = qizNPlus1;

}


template <typename T>
__global__ void devLeapfrogLast(const devAccSoA_t<T> devAccelerations, 
                                devAccSoA_t<T> devIntermVelocities, devAccSoA_t<T> devNextPositions, 
                                 devDataSoA_t<T> devDataSoA, 
                                 unsigned long n, T dt) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (iBody >= n) return;

    // === Doing last iteration
    //   1. v_{last} ~= v_{last-1/2}
    //
    // Results:
    // devDataSoA contains x_n and v_n

    //  1. v_{last} ~= v_{last-1/2}
    devDataSoA.vx[iBody] = devIntermVelocities.x[iBody]; // Setting v_n
    devDataSoA.vy[iBody] = devIntermVelocities.y[iBody];
    devDataSoA.vz[iBody] = devIntermVelocities.z[iBody];

    devDataSoA.qx[iBody] = devNextPositions.x[iBody];   // Setting x_n
    devDataSoA.qy[iBody] = devNextPositions.y[iBody];
    devDataSoA.qz[iBody] = devNextPositions.z[iBody];
}

template <typename T>
void CUDABodies<T>::updatePositionsAndVelocitiesLeapfrogOnDevice(const devAccSoA_t<T> &devAccelerations, T &dt, int iteration, int totalIterations) 
{
    this->invalidateDataSoA();
    int threads = 1024;
    int blocks = (this->n + threads - 1) / threads;  
    if ( blocks == 1 ) {
        threads = this->n;
    }
    if ( iteration == 0 ) {
        devLeapfrogFirst<T><<<blocks, threads>>>(devAccelerations, 
                                                 this->devIntermVelocities, this->devNextPositions,
                                                 this->devDataSoA, this->n, dt);
    } else if ( iteration < totalIterations - 1 ) {
        devLeapfrogMiddle<T><<<blocks, threads>>>(devAccelerations, 
                                                 this->devIntermVelocities, this->devNextPositions,
                                                 this->devDataSoA, this->n, dt);
    } else {   // Last iteration!
        devLeapfrogLast<T><<<blocks, threads>>>(devAccelerations, 
                                                 this->devIntermVelocities, this->devNextPositions,
                                                 this->devDataSoA, this->n, dt);
    }

    // Maybe removable?
    cudaDeviceSynchronize();
}

template <typename T>
void CUDABodies<T>::updatePositionsAndVelocities(const accSoA_t<T> &accelerations, T &dt) 
{
    devAccSoA_t<T> devAccelerations;
    cudaMalloc(&devAccelerations.x, (Bodies<T>::n + this->padding) * sizeof(T));
    cudaMalloc(&devAccelerations.y, (Bodies<T>::n + this->padding) * sizeof(T));
    cudaMalloc(&devAccelerations.z, (Bodies<T>::n + this->padding) * sizeof(T));
    cudaMemcpy(devAccelerations.x, accelerations.ax.data(), Bodies<T>::n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(devAccelerations.y, accelerations.ay.data(), Bodies<T>::n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(devAccelerations.z, accelerations.az.data(), Bodies<T>::n * sizeof(T), cudaMemcpyHostToDevice);

    this->updatePositionsAndVelocitiesOnDevice(devAccelerations, dt);

    cudaFree(devAccelerations.x);
    cudaFree(devAccelerations.y);
    cudaFree(devAccelerations.z);
}

template <typename T>
void CUDABodies<T>::updatePositionsAndVelocities(const std::vector<accAoS_t<T>> &accelerations, T &dt)
{
    printf("\n\n updatePositionsAndVelocities NOT IMPLEMENTED");
}

template <typename T>
CUDABodies<T>::~CUDABodies() {
    cudaFree(devDataSoA.m);
    cudaFree(devDataSoA.r);

    cudaFree(devDataSoA.qx);
    cudaFree(devDataSoA.qy);
    cudaFree(devDataSoA.qz);

    cudaFree(devDataSoA.vx);
    cudaFree(devDataSoA.vy);
    cudaFree(devDataSoA.vz);

    // Internal buffers for Leapfrog
    cudaFree(devIntermVelocities.x);
    cudaFree(devIntermVelocities.y);
    cudaFree(devIntermVelocities.z);
    cudaFree(devNextPositions.x);
    cudaFree(devNextPositions.y);
    cudaFree(devNextPositions.z);
}

// ==================================================================================== explicit template instantiation
template class CUDABodies<float>;
template class CUDABodies<double>;
// ==================================================================================== explicit template instantiation