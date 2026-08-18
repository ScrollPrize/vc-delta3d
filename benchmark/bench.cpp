// Benchmark VC-Delta3D on raw chunk files fetched by run.py.
//
// Usage: bench <level-dir> [reps]
//
// <level-dir> must contain *.raw chunk files plus a shape.txt holding
// "z y x elem_size" for every chunk in the directory. Reports the
// compression ratio and best-of-<reps> encode/decode throughput.
#include <vc_delta3d/codec.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static double now()
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: bench <level-dir> [reps]\n");
        return 1;
    }
    const fs::path dir = argv[1];
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;

    int z = 0, y = 0, x = 0, elem = 0;
    {
        std::ifstream meta(dir / "shape.txt");
        if (!(meta >> z >> y >> x >> elem) || z <= 0 || y <= 0 || x <= 0 ||
            (elem != 1 && elem != 2)) {
            std::fprintf(stderr, "bench: missing or invalid %s\n",
                         (dir / "shape.txt").c_str());
            return 1;
        }
    }
    const std::size_t N = std::size_t(z) * y * x * elem;

    std::vector<std::vector<std::byte>> chunks;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".raw")
            continue;
        std::ifstream f(e.path(), std::ios::binary);
        std::vector<std::byte> buf(N);
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(N));
        if (f.gcount() == static_cast<std::streamsize>(N))
            chunks.push_back(std::move(buf));
        else
            std::fprintf(stderr, "bench: skipping short file %s\n",
                         e.path().c_str());
    }
    if (chunks.empty()) {
        std::fprintf(stderr, "bench: no usable .raw chunks in %s\n",
                     dir.c_str());
        return 1;
    }

    // Warmup pass that also verifies every chunk round-trips exactly.
    std::size_t totalComp = 0;
    std::vector<std::vector<std::byte>> encoded;
    for (const auto& c : chunks) {
        auto enc = vc_delta3d::compress({c.data(), c.size()}, {z, y, x},
                                        static_cast<std::size_t>(elem), 1);
        std::vector<std::byte> dec(N);
        if (!vc_delta3d::decompressInto({enc.data(), enc.size()},
                                        {dec.data(), dec.size()}) ||
            std::memcmp(dec.data(), c.data(), N) != 0) {
            std::fprintf(stderr, "bench: roundtrip FAILED in %s\n",
                         dir.c_str());
            return 1;
        }
        totalComp += enc.size();
        encoded.push_back(std::move(enc));
    }

    const double rawBytes = double(chunks.size()) * double(N);

    double bestEnc = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now();
        for (const auto& c : chunks) {
            auto enc = vc_delta3d::compress({c.data(), c.size()}, {z, y, x},
                                            static_cast<std::size_t>(elem), 1);
            asm volatile("" ::"r"(enc.data()) : "memory");
        }
        bestEnc = std::min(bestEnc, now() - t0);
    }

    std::vector<std::byte> out(N);
    double bestDec = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now();
        for (const auto& e : encoded) {
            vc_delta3d::decompressInto({e.data(), e.size()}, {out.data(), N});
            asm volatile("" ::"r"(out.data()) : "memory");
        }
        bestDec = std::min(bestDec, now() - t0);
    }

    std::printf("%-24s chunks=%-3zu shape=%dx%dx%d elem=%d ratio=%.3f "
                "enc=%.1f MB/s dec=%.1f MB/s\n",
                dir.filename().c_str(), chunks.size(), z, y, x, elem,
                rawBytes / double(totalComp), rawBytes / bestEnc / 1e6,
                rawBytes / bestDec / 1e6);
    return 0;
}
