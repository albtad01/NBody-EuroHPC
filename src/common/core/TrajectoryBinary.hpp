#ifndef TRAJECTORY_BINARY_HPP_
#define TRAJECTORY_BINARY_HPP_

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "core/Bodies.hpp"

namespace murb {

inline constexpr std::uint32_t TrajectoryFormatVersion = 1;
inline constexpr std::uint32_t TrajectoryEndianMarker = 0x01020304u;
inline constexpr std::uint32_t TrajectoryScalarFp32 = 1;

struct TrajectoryMetadata {
    std::uint32_t version = 0;
    std::uint32_t scalarType = 0;
    std::uint64_t bodyCount = 0;
    std::uint64_t frameCount = 0;
    std::uint64_t recordingStride = 0;
    double timestep = 0;
    std::string backend;
    std::string sourceCommit;
    std::vector<float> radii;
};

struct TrajectoryFrame {
    std::uint64_t iteration = 0;
    std::vector<float> qx;
    std::vector<float> qy;
    std::vector<float> qz;
    std::vector<float> vx;
    std::vector<float> vy;
    std::vector<float> vz;
};

class TrajectoryWriter {
  public:
    TrajectoryWriter(const std::string& path,
                     std::uint64_t bodyCount,
                     std::uint64_t recordingStride,
                     double timestep,
                     const std::string& backend,
                     const std::string& sourceCommit,
                     const std::vector<float>& radii);
    ~TrajectoryWriter();

    TrajectoryWriter(const TrajectoryWriter&) = delete;
    TrajectoryWriter& operator=(const TrajectoryWriter&) = delete;

    void writeFrame(std::uint64_t iteration, const dataSoA_t<float>& data);
    void finalize();
    std::uint64_t getFrameCount() const;

  private:
    std::ofstream out;
    std::uint64_t bodyCount;
    std::uint64_t recordingStride;
    std::uint64_t frameCount = 0;
    bool finalized = false;
};

class TrajectoryReader {
  public:
    explicit TrajectoryReader(const std::string& path);

    TrajectoryReader(const TrajectoryReader&) = delete;
    TrajectoryReader& operator=(const TrajectoryReader&) = delete;
    TrajectoryReader(TrajectoryReader&&) = default;
    TrajectoryReader& operator=(TrajectoryReader&&) = default;

    const TrajectoryMetadata& getMetadata() const;
    bool readNextFrame(TrajectoryFrame& frame);

  private:
    std::ifstream in;
    TrajectoryMetadata metadata;
    std::uint64_t framesRead = 0;
};

} // namespace murb

#endif
