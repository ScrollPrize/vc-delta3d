#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <vc_delta3d/codec.hpp>
#include <vc_delta3d/indexed.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// Every entry point releases the GIL while the codec runs so that callers
// (zarr, dask, thread pools) can encode and decode chunks in parallel. Python
// objects are only touched with the GIL held: input pointers are taken before
// the release and result bytes objects are created before or after it.

namespace {

vc_delta3d::WireMagic parseMagic(nb::bytes magic)
{
    if (magic.size() != vc_delta3d::kWireMagic.size())
        throw std::invalid_argument("wire magic must contain exactly 4 bytes");
    vc_delta3d::WireMagic parsed;
    std::memcpy(parsed.data(), magic.c_str(), parsed.size());
    return parsed;
}

std::span<const std::byte> bytesSpan(const nb::bytes& b)
{
    return {reinterpret_cast<const std::byte*>(b.c_str()), b.size()};
}

vc_delta3d::Predictor parsePredictor(const std::string& name)
{
    if (name == "delta")
        return vc_delta3d::Predictor::Delta;
    if (name == "fast") return vc_delta3d::Predictor::Fast;
    if (name == "medium") return vc_delta3d::Predictor::Medium;
    if (name == "slow") return vc_delta3d::Predictor::Slow;
    if (name == "slower") return vc_delta3d::Predictor::Slower;
    throw std::invalid_argument("unknown predictor/preset");
}

nb::bytes toBytes(const std::vector<std::byte>& v)
{
    return nb::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

nb::bytes compressSpan(std::span<const std::byte> input,
                       std::array<int, 3> shapeZYX,
                       std::size_t elemSize,
                       int quant,
                       const std::string& predictor)
{
    const auto pred = parsePredictor(predictor);
    std::vector<std::byte> out;
    {
        nb::gil_scoped_release release;
        out = vc_delta3d::compress(input, shapeZYX, elemSize, quant, pred);
    }
    return toBytes(out);
}

template <typename T>
nb::bytes compressArray(
    nb::ndarray<nb::numpy, const T, nb::ndim<3>, nb::c_contig> array,
    int quant,
    const std::string& predictor)
{
    const auto z = static_cast<int>(array.shape(0));
    const auto y = static_cast<int>(array.shape(1));
    const auto x = static_cast<int>(array.shape(2));
    const auto nbytes = static_cast<std::size_t>(array.size()) * sizeof(T);
    return compressSpan(
        {reinterpret_cast<const std::byte*>(array.data()), nbytes},
        {z, y, x}, sizeof(T), quant, predictor);
}

// Decodes straight into a freshly allocated bytes object, so the result is
// produced without the GIL and without an intermediate copy.
nb::bytes decompressToBytes(nb::bytes payload,
                            std::size_t expectedSize,
                            const vc_delta3d::WireMagic& magic,
                            const char* error)
{
    PyObject* raw = PyBytes_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(expectedSize));
    if (!raw)
        throw nb::python_error();
    auto out = nb::steal<nb::bytes>(raw);
    const auto input = bytesSpan(payload);
    const std::span<std::byte> output{
        reinterpret_cast<std::byte*>(PyBytes_AS_STRING(raw)), expectedSize};
    bool ok;
    {
        nb::gil_scoped_release release;
        ok = vc_delta3d::decompressIntoWithMagic(input, output, magic);
    }
    if (!ok)
        throw std::runtime_error(error);
    return out;
}

void decompressToArray(nb::bytes payload,
                       nb::ndarray<nb::numpy, std::uint8_t, nb::c_contig> output,
                       const vc_delta3d::WireMagic& magic,
                       const char* error)
{
    const auto input = bytesSpan(payload);
    const std::span<std::byte> out{
        reinterpret_cast<std::byte*>(output.data()), output.size()};
    bool ok;
    {
        nb::gil_scoped_release release;
        ok = vc_delta3d::decompressIntoWithMagic(input, out, magic);
    }
    if (!ok)
        throw std::runtime_error(error);
}

} // namespace

NB_MODULE(_codec, m)
{
    m.doc() = "VC-Delta3D compression bindings";

    nb::enum_<vc_delta3d::LeafCodec>(m,"LeafCodec")
        .value("Choice",vc_delta3d::LeafCodec::Choice)
        .value("Rectangular",vc_delta3d::LeafCodec::Rectangular);
    nb::class_<vc_delta3d::IndexedLeaf>(m,"IndexedLeaf")
        .def_ro("channel",&vc_delta3d::IndexedLeaf::channel)
        .def_ro("origin",&vc_delta3d::IndexedLeaf::originZYX)
        .def_ro("shape",&vc_delta3d::IndexedLeaf::shapeZYX)
        .def_ro("codec",&vc_delta3d::IndexedLeaf::codec)
        .def_ro("offset",&vc_delta3d::IndexedLeaf::offset)
        .def_ro("length",&vc_delta3d::IndexedLeaf::length);
    nb::class_<vc_delta3d::ChunkIndex>(m,"ChunkIndex")
        .def_ro("shape",&vc_delta3d::ChunkIndex::shapeZYX)
        .def_ro("channels",&vc_delta3d::ChunkIndex::channels)
        .def_ro("header_bytes",&vc_delta3d::ChunkIndex::headerBytes)
        .def_ro("encoded_bytes",&vc_delta3d::ChunkIndex::encodedBytes)
        .def_ro("decoded_bytes",&vc_delta3d::ChunkIndex::decodedBytes)
        .def_prop_ro("channels_last",[](const vc_delta3d::ChunkIndex&i){return i.layout==vc_delta3d::ChannelLayout::Last;})
        .def_ro("leaves",&vc_delta3d::ChunkIndex::leaves);
    m.def("indexed_header_size",[](nb::bytes b){return vc_delta3d::indexedHeaderSize(bytesSpan(b));},"prefix"_a);
    m.def("read_index",[](nb::bytes b,std::uint64_t size){
        auto src=bytesSpan(b);vc_delta3d::ChunkIndex index;
        {nb::gil_scoped_release release;index=vc_delta3d::readIndex(src,size);}return index;
    },"header"_a,"object_size"_a);
    m.def("decode_indexed_leaf",[](nb::bytes b,const vc_delta3d::IndexedLeaf&leaf){
        std::vector<std::byte> out(size_t(leaf.shapeZYX[0])*leaf.shapeZYX[1]*leaf.shapeZYX[2]);bool ok;
        auto src=bytesSpan(b);
        {nb::gil_scoped_release release;ok=vc_delta3d::decodeIndexedLeaf(src,leaf,out);}
        if(!ok)throw std::runtime_error("invalid indexed leaf");return toBytes(out);
    },"payload"_a,"leaf"_a);
    m.def("compress_indexed",[](nb::bytes raw,int z,int y,int x,int channels,bool last,const std::string&p,
        const std::vector<std::array<std::uint32_t,3>>&plan){
        vc_delta3d::IndexedOptions options;options.preset=parsePredictor(p);
        for(auto n:plan){if(n[0]>4||n[2]>1)throw std::invalid_argument("invalid partition node");
            options.partition.push_back({vc_delta3d::SplitAxis(n[0]),n[1],bool(n[2])});}
        std::vector<std::byte> out;auto src=bytesSpan(raw);
        {nb::gil_scoped_release release;out=vc_delta3d::compressIndexed(src,{z,y,x},channels,
            last?vc_delta3d::ChannelLayout::Last:vc_delta3d::ChannelLayout::First,options);}
        return toBytes(out);
    },"raw"_a,"z"_a,"y"_a,"x"_a,"channels"_a=1,"channels_last"_a=false,
      "predictor"_a="fast","partition"_a=std::vector<std::array<std::uint32_t,3>>{});

    m.def(
        "compress",
        [](nb::bytes raw, int z, int y, int x, int elem_size, int quant,
           const std::string& predictor) {
            return compressSpan(bytesSpan(raw), {z, y, x},
                                static_cast<std::size_t>(elem_size), quant,
                                predictor);
        },
        "raw"_a, "z"_a, "y"_a, "x"_a, "elem_size"_a, "quant"_a = 1,
        "predictor"_a = "fast");

    m.def(
        "compress_channels",
        [](nb::bytes raw, int z, int y, int x, int channels, bool channels_last,
           int elem_size, int quant, const std::string& predictor) {
            const auto input = bytesSpan(raw);
            const auto pred = parsePredictor(predictor);
            std::vector<std::byte> out;
            {
                nb::gil_scoped_release release;
                out = vc_delta3d::compressChannels(
                    input, {z, y, x}, channels,
                    channels_last ? vc_delta3d::ChannelLayout::Last
                                  : vc_delta3d::ChannelLayout::First,
                    static_cast<std::size_t>(elem_size), quant, pred);
            }
            return toBytes(out);
        },
        "raw"_a, "z"_a, "y"_a, "x"_a, "channels"_a, "channels_last"_a,
        "elem_size"_a, "quant"_a = 1, "predictor"_a = "fast");
    m.def(
        "channel_count",
        [](nb::bytes payload) { return vc_delta3d::channelCount(bytesSpan(payload)); },
        "payload"_a);
    m.def("decoded_size", [](nb::bytes payload) { return vc_delta3d::decodedSize(bytesSpan(payload)); }, "payload"_a);
    m.def("compress_array", &compressArray<std::uint8_t>,
          "array"_a, "quant"_a = 1, "predictor"_a = "fast");
    m.def("compress_array", &compressArray<std::uint16_t>,
          "array"_a, "quant"_a = 1, "predictor"_a = "fast");

    m.def(
        "decompress",
        [](nb::bytes payload, std::size_t expected_size) {
            return decompressToBytes(
                payload, expected_size, vc_delta3d::kWireMagic,
                "not a valid VC-Delta3D payload of the expected size");
        },
        "payload"_a, "expected_size"_a);

    m.def(
        "decompress_into",
        [](nb::bytes payload,
           nb::ndarray<nb::numpy, std::uint8_t, nb::c_contig> output) {
            decompressToArray(
                payload, output, vc_delta3d::kWireMagic,
                "not a valid VC-Delta3D payload for the output size");
        },
        "payload"_a, "output"_a);

    m.def(
        "decompress_with_magic",
        [](nb::bytes payload, std::size_t expected_size, nb::bytes magic) {
            return decompressToBytes(
                payload, expected_size, parseMagic(magic),
                "not a valid payload of the expected size and wire magic");
        },
        "payload"_a, "expected_size"_a, "magic"_a);

    m.def(
        "decompress_into_with_magic",
        [](nb::bytes payload,
           nb::ndarray<nb::numpy, std::uint8_t, nb::c_contig> output,
           nb::bytes magic) {
            decompressToArray(
                payload, output, parseMagic(magic),
                "not a valid payload for the output size and wire magic");
        },
        "payload"_a, "output"_a, "magic"_a);
}
