#include "core/TrajectoryBinary.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace murb {
namespace {

constexpr std::array<char, 8> Magic = {'M', 'U', 'R', 'B', 'T', 'R', 'J', '\0'};
constexpr std::uint32_t HasRadii = 1u;
constexpr std::uint64_t FixedHeaderBytes = 72;
constexpr std::streamoff FrameCountOffset = 40;
constexpr std::uint32_t MaxTextBytes = 4096;

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "The trajectory format requires IEEE-754 binary32 floats");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "The trajectory format requires IEEE-754 binary64 doubles");

void failIf(bool condition, const std::string& message) {
    if (condition) throw std::runtime_error("trajectory: " + message);
}

template <typename UInt>
void writeLittle(std::ostream& stream, UInt value) {
    static_assert(std::is_unsigned_v<UInt>);
    std::array<char, sizeof(UInt)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<char>((value >> (index * 8)) & UInt{0xff});
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    failIf(!stream, "failed while writing file");
}

template <typename UInt>
UInt readLittle(std::istream& stream, const char* field) {
    static_assert(std::is_unsigned_v<UInt>);
    std::array<unsigned char, sizeof(UInt)> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    failIf(stream.gcount() != static_cast<std::streamsize>(bytes.size()),
           std::string("truncated ") + field);
    UInt value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= static_cast<UInt>(bytes[index]) << (index * 8);
    return value;
}

void writeF32(std::ostream& stream, float value) {
    writeLittle(stream, std::bit_cast<std::uint32_t>(value));
}

float readF32(std::istream& stream, const char* field) {
    return std::bit_cast<float>(readLittle<std::uint32_t>(stream, field));
}

void writeF64(std::ostream& stream, double value) {
    writeLittle(stream, std::bit_cast<std::uint64_t>(value));
}

double readF64(std::istream& stream, const char* field) {
    return std::bit_cast<double>(readLittle<std::uint64_t>(stream, field));
}

void writeFloatArray(std::ostream& stream, const float* values, std::uint64_t count) {
    if constexpr (std::endian::native == std::endian::little) {
        stream.write(reinterpret_cast<const char*>(values),
                     static_cast<std::streamsize>(count * sizeof(float)));
        failIf(!stream, "failed while writing float array");
    } else {
        for (std::uint64_t index = 0; index < count; ++index) writeF32(stream, values[index]);
    }
}

void readFloatArray(std::istream& stream, std::vector<float>& values, std::uint64_t count,
                    const char* field) {
    values.resize(static_cast<std::size_t>(count));
    if constexpr (std::endian::native == std::endian::little) {
        const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
        stream.read(reinterpret_cast<char*>(values.data()), bytes);
        failIf(stream.gcount() != bytes, std::string("truncated frame field ") + field);
    } else {
        for (std::uint64_t index = 0; index < count; ++index)
            values[static_cast<std::size_t>(index)] = readF32(stream, field);
    }
}

std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right, const char* description) {
    failIf(right > std::numeric_limits<std::uint64_t>::max() - left,
           std::string(description) + " size overflows");
    return left + right;
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right, const char* description) {
    failIf(left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left,
           std::string(description) + " size overflows");
    return left * right;
}

void validateTextLength(const std::string& value, const char* name) {
    failIf(value.size() > MaxTextBytes, std::string(name) + " is too long");
}

void requireBodyArrays(const dataSoA_t<float>& data, std::uint64_t count) {
    const auto expected = static_cast<std::size_t>(count);
    failIf(data.qx.size() < expected || data.qy.size() < expected || data.qz.size() < expected ||
           data.vx.size() < expected || data.vy.size() < expected || data.vz.size() < expected,
           "body snapshot is smaller than the declared body count");
}

} // namespace

TrajectoryWriter::TrajectoryWriter(const std::string& path,
                                   std::uint64_t bodyCount,
                                   std::uint64_t recordingStride,
                                   double timestep,
                                   const std::string& backend,
                                   const std::string& sourceCommit,
                                   const std::vector<float>& radii)
    : out(path, std::ios::binary | std::ios::trunc), bodyCount(bodyCount),
      recordingStride(recordingStride) {
    failIf(path.empty(), "output path is empty");
    failIf(!out.is_open(), "cannot create " + path);
    failIf(bodyCount == 0, "body count must be positive");
    failIf(recordingStride == 0, "recording stride must be positive");
    failIf(!std::isfinite(timestep) || timestep <= 0, "timestep must be finite and positive");
    failIf(radii.size() != static_cast<std::size_t>(bodyCount),
           "radius count does not match body count");
    validateTextLength(backend, "backend name");
    validateTextLength(sourceCommit, "source commit");

    const auto radiusBytes = checkedMultiply(bodyCount, sizeof(float), "radius section");
    auto headerBytes = checkedAdd(FixedHeaderBytes, backend.size(), "header");
    headerBytes = checkedAdd(headerBytes, sourceCommit.size(), "header");
    headerBytes = checkedAdd(headerBytes, radiusBytes, "header");

    out.write(Magic.data(), static_cast<std::streamsize>(Magic.size()));
    writeLittle(out, TrajectoryFormatVersion);
    writeLittle(out, TrajectoryEndianMarker);
    writeLittle(out, TrajectoryScalarFp32);
    writeLittle(out, HasRadii);
    writeLittle(out, headerBytes);
    writeLittle(out, bodyCount);
    writeLittle(out, std::uint64_t{0});
    writeLittle(out, recordingStride);
    writeF64(out, timestep);
    writeLittle(out, static_cast<std::uint32_t>(backend.size()));
    writeLittle(out, static_cast<std::uint32_t>(sourceCommit.size()));
    out.write(backend.data(), static_cast<std::streamsize>(backend.size()));
    out.write(sourceCommit.data(), static_cast<std::streamsize>(sourceCommit.size()));
    failIf(!out, "failed while writing metadata");
    writeFloatArray(out, radii.data(), bodyCount);
}

TrajectoryWriter::~TrajectoryWriter() {
    if (!finalized) {
        try { finalize(); } catch (...) {}
    }
}

void TrajectoryWriter::writeFrame(std::uint64_t iteration, const dataSoA_t<float>& data) {
    failIf(finalized, "cannot write a finalized file");
    const auto expectedIteration = checkedMultiply(frameCount + 1, recordingStride, "iteration");
    failIf(iteration != expectedIteration, "frame iteration does not match recording stride");
    requireBodyArrays(data, bodyCount);
    writeLittle(out, iteration);
    writeFloatArray(out, data.qx.data(), bodyCount);
    writeFloatArray(out, data.qy.data(), bodyCount);
    writeFloatArray(out, data.qz.data(), bodyCount);
    writeFloatArray(out, data.vx.data(), bodyCount);
    writeFloatArray(out, data.vy.data(), bodyCount);
    writeFloatArray(out, data.vz.data(), bodyCount);
    ++frameCount;
}

void TrajectoryWriter::finalize() {
    if (finalized) return;
    failIf(!out, "cannot finalize after an earlier write failure");
    out.seekp(FrameCountOffset, std::ios::beg);
    failIf(!out, "cannot seek to frame count");
    writeLittle(out, frameCount);
    out.flush();
    failIf(!out, "cannot finalize file");
    out.close();
    finalized = true;
}

std::uint64_t TrajectoryWriter::getFrameCount() const { return frameCount; }

TrajectoryReader::TrajectoryReader(const std::string& path) : in(path, std::ios::binary) {
    failIf(!in.is_open(), "cannot open " + path);
    std::error_code sizeError;
    const auto actualFileBytes = std::filesystem::file_size(path, sizeError);
    failIf(static_cast<bool>(sizeError), "cannot determine file size");

    std::array<char, Magic.size()> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    failIf(in.gcount() != static_cast<std::streamsize>(magic.size()), "truncated header");
    failIf(magic != Magic, "wrong magic identifier");

    metadata.version = readLittle<std::uint32_t>(in, "version");
    failIf(metadata.version != TrajectoryFormatVersion, "unsupported format version");
    failIf(readLittle<std::uint32_t>(in, "endianness marker") != TrajectoryEndianMarker,
           "unsupported endianness policy");
    metadata.scalarType = readLittle<std::uint32_t>(in, "scalar type");
    failIf(metadata.scalarType != TrajectoryScalarFp32, "unsupported scalar type");
    const auto flags = readLittle<std::uint32_t>(in, "flags");
    failIf(flags != HasRadii, "unsupported static-body flags");
    const auto headerBytes = readLittle<std::uint64_t>(in, "header size");
    metadata.bodyCount = readLittle<std::uint64_t>(in, "body count");
    metadata.frameCount = readLittle<std::uint64_t>(in, "frame count");
    metadata.recordingStride = readLittle<std::uint64_t>(in, "recording stride");
    metadata.timestep = readF64(in, "timestep");
    const auto backendBytes = readLittle<std::uint32_t>(in, "backend length");
    const auto commitBytes = readLittle<std::uint32_t>(in, "commit length");

    failIf(metadata.bodyCount == 0, "body count must be positive");
    failIf(metadata.bodyCount > static_cast<std::uint64_t>(std::numeric_limits<int>::max() - 1024),
           "body count exceeds supported visualization dimensions");
    failIf(metadata.bodyCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
           "body count is too large for this system");
    failIf(metadata.recordingStride == 0, "recording stride must be positive");
    failIf(!std::isfinite(metadata.timestep) || metadata.timestep <= 0,
           "timestep must be finite and positive");
    failIf(backendBytes > MaxTextBytes || commitBytes > MaxTextBytes, "metadata text is too long");

    const auto radiusBytes = checkedMultiply(metadata.bodyCount, sizeof(float), "radius section");
    auto expectedHeaderBytes = checkedAdd(FixedHeaderBytes, backendBytes, "header");
    expectedHeaderBytes = checkedAdd(expectedHeaderBytes, commitBytes, "header");
    expectedHeaderBytes = checkedAdd(expectedHeaderBytes, radiusBytes, "header");
    failIf(headerBytes != expectedHeaderBytes, "inconsistent header size");
    failIf(actualFileBytes < headerBytes, "truncated static-body section");

    metadata.backend.resize(backendBytes);
    metadata.sourceCommit.resize(commitBytes);
    in.read(metadata.backend.data(), static_cast<std::streamsize>(backendBytes));
    failIf(in.gcount() != static_cast<std::streamsize>(backendBytes), "truncated backend name");
    in.read(metadata.sourceCommit.data(), static_cast<std::streamsize>(commitBytes));
    failIf(in.gcount() != static_cast<std::streamsize>(commitBytes), "truncated source commit");
    readFloatArray(in, metadata.radii, metadata.bodyCount, "radii");
    frameDataOffset = in.tellg();
    failIf(frameDataOffset == std::streampos{-1}, "cannot locate first frame");

    const auto valuesPerFrame = checkedMultiply(metadata.bodyCount, std::uint64_t{6}, "frame");
    auto frameBytes = checkedMultiply(valuesPerFrame, sizeof(float), "frame");
    frameBytes = checkedAdd(frameBytes, sizeof(std::uint64_t), "frame");
    const auto payloadBytes = checkedMultiply(metadata.frameCount, frameBytes, "payload");
    const auto expectedFileBytes = checkedAdd(headerBytes, payloadBytes, "file");
    failIf(actualFileBytes != expectedFileBytes, "file size does not match header and frame count");
}

const TrajectoryMetadata& TrajectoryReader::getMetadata() const { return metadata; }

bool TrajectoryReader::readNextFrame(TrajectoryFrame& frame) {
    if (framesRead == metadata.frameCount) return false;

    TrajectoryFrame next;
    next.iteration = readLittle<std::uint64_t>(in, "frame iteration");
    const auto expectedIteration =
        checkedMultiply(framesRead + 1, metadata.recordingStride, "iteration");
    failIf(next.iteration != expectedIteration, "frame iteration does not match recording stride");
    readFloatArray(in, next.qx, metadata.bodyCount, "qx");
    readFloatArray(in, next.qy, metadata.bodyCount, "qy");
    readFloatArray(in, next.qz, metadata.bodyCount, "qz");
    readFloatArray(in, next.vx, metadata.bodyCount, "vx");
    readFloatArray(in, next.vy, metadata.bodyCount, "vy");
    readFloatArray(in, next.vz, metadata.bodyCount, "vz");
    frame = std::move(next);
    ++framesRead;
    return true;
}

void TrajectoryReader::rewind() {
    in.clear();
    in.seekg(frameDataOffset);
    failIf(!in, "cannot rewind to first frame");
    framesRead = 0;
}

} // namespace murb
