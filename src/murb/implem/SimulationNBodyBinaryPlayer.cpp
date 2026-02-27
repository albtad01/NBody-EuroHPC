#include "SimulationNBodyBinaryPlayer.hpp"
#include <iostream>

template <typename T>
SimulationNBodyBinaryPlayer<T>::SimulationNBodyBinaryPlayer(const BodiesAllocatorInterface<T>& allocator, const T soft, std::string filename)
    : SimulationNBodyInterface<T>(allocator, soft), fileName(filename)
{
    inFile.open(fileName, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << fileName << std::endl;
        exit(-1);
    }

    // Leggiamo l'header per "saltarlo"
    unsigned long fileNBodies, fileNIte;
    inFile.read(reinterpret_cast<char*>(&fileNBodies), sizeof(unsigned long));
    inFile.read(reinterpret_cast<char*>(&fileNIte), sizeof(unsigned long));

    if (fileNBodies != (unsigned long)this->getBodies()->getN()) {
        std::cerr << "Attenzione: Il numero di corpi nel file (" << fileNBodies 
                  << ") non coincide con quello richiesto (" << this->getBodies()->getN() << ")!" << std::endl;
    }
}

template <typename T>
void SimulationNBodyBinaryPlayer<T>::computeOneIteration() {
    int n = this->getBodies()->getN();
    // Otteniamo il riferimento SoA
    auto& data = this->getBodies()->getDataSoA();

    // FIX: Usiamo const_cast per rimuovere il qualificatore 'const' dal puntatore.
    // Questo è necessario perché inFile.read deve poter modificare la memoria,
    // e noi sappiamo che nel Player è corretto sovrascrivere le posizioni.
    inFile.read(reinterpret_cast<char*>(const_cast<T*>(data.qx.data())), n * sizeof(T));
    inFile.read(reinterpret_cast<char*>(const_cast<T*>(data.qy.data())), n * sizeof(T));
    inFile.read(reinterpret_cast<char*>(const_cast<T*>(data.qz.data())), n * sizeof(T));

    if (inFile.eof()) {
        std::cout << "\n[PLAYER] Fine del file raggiunta. Ricomincio..." << std::endl;
        inFile.clear();
        inFile.seekg(2 * sizeof(unsigned long), std::ios::beg); // Torna all'inizio dopo l'header
    }
}

template <typename T>
SimulationNBodyBinaryPlayer<T>::~SimulationNBodyBinaryPlayer() {
    if (inFile.is_open()) inFile.close();
}

template class SimulationNBodyBinaryPlayer<float>;
template class SimulationNBodyBinaryPlayer<double>;