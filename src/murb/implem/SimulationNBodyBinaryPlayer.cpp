#include "SimulationNBodyBinaryPlayer.hpp"

#include <stdexcept>
#include <type_traits>

namespace {

template <typename T>
std::vector<T> convertValues(const std::vector<float>& source) {
    if constexpr (std::is_same_v<T, float>) {
        return source;
    } else {
        return std::vector<T>(source.begin(), source.end());
    }
}

} // namespace

template <typename T>
SimulationNBodyBinaryPlayer<T>::SimulationNBodyBinaryPlayer(
    const BodiesAllocatorInterface<T>& allocator, const T soft, const std::string& filename)
    : SimulationNBodyInterface<T>(allocator, soft), fileName(filename), reader(filename)
{
    if (reader.getMetadata().bodyCount != this->getBodies()->getN())
        throw std::invalid_argument("trajectory body count does not match player allocation");
    this->getBodies()->setRadii(convertValues<T>(reader.getMetadata().radii));
}

template <typename T>
void SimulationNBodyBinaryPlayer<T>::computeOneIteration() {
    readNextFrame();
}

template <typename T>
bool SimulationNBodyBinaryPlayer<T>::readNextFrame() {
    murb::TrajectoryFrame frame;
    if (!reader.readNextFrame(frame)) return false;
    this->getBodies()->setPositionsAndVelocities(
        convertValues<T>(frame.qx), convertValues<T>(frame.qy), convertValues<T>(frame.qz),
        convertValues<T>(frame.vx), convertValues<T>(frame.vy), convertValues<T>(frame.vz));
    return true;
}

template <typename T>
void SimulationNBodyBinaryPlayer<T>::restart() {
    reader.rewind();
}

template <typename T>
const murb::TrajectoryMetadata& SimulationNBodyBinaryPlayer<T>::getMetadata() const {
    return reader.getMetadata();
}

template class SimulationNBodyBinaryPlayer<float>;
template class SimulationNBodyBinaryPlayer<double>;
