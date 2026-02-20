#include "core/BodiesAllocator.hpp"

// --- PARTE CPU (LASCIALA ATTIVA) ---

template <typename T>
BodiesAllocator<T>::BodiesAllocator(const unsigned long n, const std::string &scheme, 
                                 const unsigned long randInit) 
    : n{n}, scheme{scheme}, randInit{randInit}
{

}

template <typename T>
std::unique_ptr<Bodies<T>> BodiesAllocator<T>::allocate_unique() const
{
    return std::make_unique<Bodies<T>>(n, scheme, randInit);
}

template <typename T>
std::shared_ptr<Bodies<T>> BodiesAllocator<T>::allocate_shared() const
{
    return std::make_shared<Bodies<T>>(n, scheme, randInit);
}

// --- PARTE CUDA (DISABILITATA PER MAC) ---
// Commentiamo tutto questo blocco per evitare errori di Linker

/* template <typename T>
CUDABodiesAllocator<T>::CUDABodiesAllocator(const unsigned long n, const std::string &scheme, 
                                 const unsigned long randInit) 
    : n{n}, scheme{scheme}, randInit{randInit}
{

}

template <typename T>
std::unique_ptr<Bodies<T>> CUDABodiesAllocator<T>::allocate_unique() const
{
    return std::make_unique<CUDABodies<T>>(n, scheme, randInit);
}

template <typename T>
std::shared_ptr<Bodies<T>> CUDABodiesAllocator<T>::allocate_shared() const
{
    return std::make_shared<CUDABodies<T>>(n, scheme, randInit);
}
*/

// --- ISTANZIAZIONI ---

template class BodiesAllocator<float>;
template class BodiesAllocator<double>;

// Commentiamo anche le istanziazioni CUDA
// template class CUDABodiesAllocator<float>;
// template class CUDABodiesAllocator<double>;