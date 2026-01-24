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


// ======================================================================================
// ========================== UPDATE POSITIONS & VELOCITIES =============================
// ======================================================================================
template <typename T>
__global__ void devUpdatePositionsAndVelocities(const devAccSoA_t<T> devAccelerations, T dt, devDataSoA_t<T> devDataSoA, unsigned long n) {
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (iBody >= n) return;

    T aixDt = devAccelerations.ax[iBody] * dt;
    T aiyDt = devAccelerations.ay[iBody] * dt;
    T aizSt = devAccelerations.az[iBody] * dt;

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

template <typename T>
void CUDABodies<T>::updatePositionsAndVelocities(const accSoA_t<T> &accelerations, T &dt) 
{
    devAccSoA_t<T> devAccelerations;
    cudaMalloc(&devAccelerations.ax, (Bodies<T>::n + this->padding) * sizeof(T));
    cudaMalloc(&devAccelerations.ay, (Bodies<T>::n + this->padding) * sizeof(T));
    cudaMalloc(&devAccelerations.az, (Bodies<T>::n + this->padding) * sizeof(T));
    cudaMemcpy(devAccelerations.ax, accelerations.ax.data(), Bodies<T>::n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(devAccelerations.ay, accelerations.ay.data(), Bodies<T>::n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(devAccelerations.az, accelerations.az.data(), Bodies<T>::n * sizeof(T), cudaMemcpyHostToDevice);

    this->updatePositionsAndVelocitiesOnDevice(devAccelerations, dt);

    cudaFree(devAccelerations.ax);
    cudaFree(devAccelerations.ay);
    cudaFree(devAccelerations.az);
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
}

// ==================================================================================== explicit template instantiation
template class CUDABodies<float>;
template class CUDABodies<double>;
// ==================================================================================== explicit template instantiation