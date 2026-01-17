#ifndef BODIES_ALLOCATOR_HPP_
#define BODIES_ALLOCATOR_HPP_

#include "core/Bodies.hpp"
#include "core/CUDABodies.hpp"
#include <memory>

template <typename T>
class BodiesAllocatorInterface {
    public:
        virtual std::unique_ptr<Bodies<T>> allocate_unique() const = 0;
        virtual std::shared_ptr<Bodies<T>> allocate_shared() const = 0;
};

template <typename T>
class BodiesAllocator : public BodiesAllocatorInterface<T> {
    public:
        BodiesAllocator(const unsigned long n, 
                        const std::string &scheme = "galaxy", 
                        const unsigned long randInit = 0);
        virtual std::unique_ptr<Bodies<T>> allocate_unique() const;
        virtual std::shared_ptr<Bodies<T>> allocate_shared() const;
        virtual ~BodiesAllocator() = default;
    private:
        const unsigned long n;
        const std::string &scheme;
        const unsigned long randInit;
};

template <typename T>
class CUDABodiesAllocator : public BodiesAllocatorInterface<T> {
    public:
        CUDABodiesAllocator(const unsigned long n, 
                        const std::string &scheme = "galaxy", 
                        const unsigned long randInit = 0);
        virtual std::unique_ptr<Bodies<T>> allocate_unique() const;
        virtual std::shared_ptr<Bodies<T>> allocate_shared() const;
        virtual ~CUDABodiesAllocator() = default;
    private:
        const unsigned long n;
        const std::string &scheme;
        const unsigned long randInit;
};

#endif