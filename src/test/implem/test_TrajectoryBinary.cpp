#include <catch.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/TrajectoryBinary.hpp"

namespace {

class TemporaryTrajectory {
  public:
    explicit TemporaryTrajectory(const std::string& label) {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("murb-" + label + "-" + std::to_string(nonce) + ".murbtraj");
    }

    ~TemporaryTrajectory() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }

    std::filesystem::path path;
};

dataSoA_t<float> sampleBodies() {
    dataSoA_t<float> data;
    data.qx = {1.0f, 2.0f, 3.0f};
    data.qy = {4.0f, 5.0f, 6.0f};
    data.qz = {7.0f, 8.0f, 9.0f};
    data.vx = {-1.0f, -2.0f, -3.0f};
    data.vy = {-4.0f, -5.0f, -6.0f};
    data.vz = {-7.0f, -8.0f, -9.0f};
    data.m = {10.0f, 11.0f, 12.0f};
    data.r = {0.25f, 0.5f, 0.75f};
    return data;
}

void writeSample(const std::filesystem::path& path, std::uint64_t frames = 2) {
    auto data = sampleBodies();
    murb::TrajectoryWriter writer(path.string(), 3, 5, 3600.0,
                                  "gpu+tile+full", "0123456789abcdef", data.r);
    for (std::uint64_t frame = 0; frame < frames; ++frame) {
        data.qx[0] += static_cast<float>(frame);
        data.vz[2] -= static_cast<float>(frame);
        writer.writeFrame((frame + 1) * 5, data);
    }
    writer.finalize();
}

void overwriteU32Little(const std::filesystem::path& path, std::streamoff offset,
                        std::uint32_t value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    REQUIRE(file.is_open());
    char bytes[4];
    for (int index = 0; index < 4; ++index)
        bytes[index] = static_cast<char>((value >> (index * 8)) & 0xffu);
    file.seekp(offset);
    file.write(bytes, sizeof(bytes));
    REQUIRE(static_cast<bool>(file));
}

} // namespace

TEST_CASE("trajectory round trip preserves metadata and complete frames", "[trajectory]") {
    TemporaryTrajectory file("roundtrip");
    writeSample(file.path);

    murb::TrajectoryReader reader(file.path.string());
    const auto& metadata = reader.getMetadata();
    REQUIRE(metadata.version == murb::TrajectoryFormatVersion);
    REQUIRE(metadata.scalarType == murb::TrajectoryScalarFp32);
    REQUIRE(metadata.bodyCount == 3);
    REQUIRE(metadata.frameCount == 2);
    REQUIRE(metadata.recordingStride == 5);
    REQUIRE(metadata.timestep == 3600.0);
    REQUIRE(metadata.backend == "gpu+tile+full");
    REQUIRE(metadata.sourceCommit == "0123456789abcdef");
    REQUIRE((metadata.radii == std::vector<float>{0.25f, 0.5f, 0.75f}));

    murb::TrajectoryFrame first;
    REQUIRE(reader.readNextFrame(first));
    REQUIRE(first.iteration == 5);
    REQUIRE((first.qx == std::vector<float>{1.0f, 2.0f, 3.0f}));
    REQUIRE((first.qy == std::vector<float>{4.0f, 5.0f, 6.0f}));
    REQUIRE((first.qz == std::vector<float>{7.0f, 8.0f, 9.0f}));
    REQUIRE((first.vx == std::vector<float>{-1.0f, -2.0f, -3.0f}));
    REQUIRE((first.vy == std::vector<float>{-4.0f, -5.0f, -6.0f}));
    REQUIRE((first.vz == std::vector<float>{-7.0f, -8.0f, -9.0f}));

    murb::TrajectoryFrame second;
    REQUIRE(reader.readNextFrame(second));
    REQUIRE(second.iteration == 10);
    REQUIRE((second.qx == std::vector<float>{2.0f, 2.0f, 3.0f}));
    REQUIRE((second.vz == std::vector<float>{-7.0f, -8.0f, -10.0f}));
    REQUIRE_FALSE(reader.readNextFrame(second));
}

TEST_CASE("trajectory reader rejects wrong magic", "[trajectory]") {
    TemporaryTrajectory file("magic");
    writeSample(file.path);
    std::fstream bytes(file.path, std::ios::binary | std::ios::in | std::ios::out);
    REQUIRE(bytes.is_open());
    bytes.put('X');
    bytes.close();
    REQUIRE_THROWS_WITH(murb::TrajectoryReader(file.path.string()),
                        Catch::Matchers::Contains("wrong magic"));
}

TEST_CASE("trajectory reader rejects unsupported version", "[trajectory]") {
    TemporaryTrajectory file("version");
    writeSample(file.path);
    overwriteU32Little(file.path, 8, murb::TrajectoryFormatVersion + 1);
    REQUIRE_THROWS_WITH(murb::TrajectoryReader(file.path.string()),
                        Catch::Matchers::Contains("unsupported format version"));
}

TEST_CASE("trajectory reader rejects unsupported scalar type", "[trajectory]") {
    TemporaryTrajectory file("scalar");
    writeSample(file.path);
    overwriteU32Little(file.path, 16, 99);
    REQUIRE_THROWS_WITH(murb::TrajectoryReader(file.path.string()),
                        Catch::Matchers::Contains("unsupported scalar type"));
}

TEST_CASE("trajectory reader enforces the endianness policy", "[trajectory]") {
    TemporaryTrajectory file("endian");
    writeSample(file.path);
    overwriteU32Little(file.path, 12, 0x04030201u);
    REQUIRE_THROWS_WITH(murb::TrajectoryReader(file.path.string()),
                        Catch::Matchers::Contains("unsupported endianness policy"));
}

TEST_CASE("trajectory reader rejects a truncated header", "[trajectory]") {
    TemporaryTrajectory file("short-header");
    {
        std::ofstream bytes(file.path, std::ios::binary);
        bytes.write("MURBTRJ", 7);
    }
    REQUIRE_THROWS_WITH(murb::TrajectoryReader(file.path.string()),
                        Catch::Matchers::Contains("truncated header"));
}

TEST_CASE("trajectory reader rejects a truncated frame", "[trajectory]") {
    TemporaryTrajectory file("short-frame");
    writeSample(file.path, 1);
    const auto size = std::filesystem::file_size(file.path);
    std::filesystem::resize_file(file.path, size - 1);
    REQUIRE_THROWS_WITH(murb::TrajectoryReader(file.path.string()),
                        Catch::Matchers::Contains("file size does not match"));
}

TEST_CASE("trajectory header validates its declared size", "[trajectory]") {
    TemporaryTrajectory file("header-size");
    writeSample(file.path);
    overwriteU32Little(file.path, 24, 0);
    REQUIRE_THROWS_WITH(murb::TrajectoryReader(file.path.string()),
                        Catch::Matchers::Contains("inconsistent header size"));
}
