use std::path::Path;

use bevy::{
    prelude::Image,
    render::{render_asset::RenderAssetUsages, render_resource::Extent3d},
};
use ktx2::SupercompressionScheme;
use ktx2_writer::{Header, KTX2Writer, WriterLevel};
use rgb9e5::float3_to_rgb9e5;

pub mod ktx2_writer;
pub mod rgb9e5;

pub fn to_vec_f16_from_byte_slice(vecs: &[u8]) -> &[half::f16] {
    unsafe { std::slice::from_raw_parts(vecs.as_ptr() as *const _, vecs.len() / 2) }
}

pub fn u32_to_bytes(vecs: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(vecs.as_ptr() as *const _, vecs.len() * 4) }
}

pub fn write_ktx2(image: &Image, output_path: &Path) {
    if image.is_compressed() {
        panic!("Only uncompressed images supported");
    }

    let mut mips = Vec::new();
    for mip_level in 0..image.texture_descriptor.mip_level_count {
        let mut rgb9e5 = Vec::new();
        for face in 0..6 {
            let mip_data = extract_mip_level(image, mip_level, face);
            let f16data = to_vec_f16_from_byte_slice(&mip_data.data);

            for v in f16data.chunks(4) {
                rgb9e5.push(float3_to_rgb9e5(&[
                    v[0].to_f32(),
                    v[1].to_f32(),
                    v[2].to_f32(),
                ]));
            }
        }

        let rgb9e5_bytes = u32_to_bytes(&rgb9e5).to_vec();
        mips.push(WriterLevel {
            uncompressed_length: rgb9e5_bytes.len(),
            bytes: zstd::bulk::compress(&rgb9e5_bytes, 0).unwrap(),
        });
    }

    // Create DFD for RGB9E5 format
    let dfd_bytes = create_rgb9e5_dfd();

    // https://github.khronos.org/KTX-Specification/
    let writer = KTX2Writer {
        header: Header {
            format: Some(ktx2::Format::E5B9G9R9_UFLOAT_PACK32),
            type_size: 4,
            pixel_width: image.texture_descriptor.size.width,
            pixel_height: image.texture_descriptor.size.height,
            pixel_depth: 0, // Must be 0 for cube maps according to KTX2 spec
            layer_count: 0, // Must be 0 for non-array cube maps according to KTX2 spec
            face_count: 6,
            supercompression_scheme: Some(SupercompressionScheme::Zstandard),
        },
        dfd_bytes: &dfd_bytes,
        levels_descending: mips,
    };

    writer
        .write(&mut std::fs::File::create(output_path).unwrap())
        .unwrap();
}

/// Builds a KTX 2.0 Data-Format Descriptor for `VK_FORMAT_E5B9G9R9_UFLOAT_PACK32`.
///
/// The descriptor follows the sample layout shown in the specification and uses
/// one BASIC descriptor block (vendor 0, type 0, version 2).  Six samples are
/// written so that the validator sees the expected RGB mantissas and their
/// shared exponent.
///
/// Every texel occupies a single 32-bit word, therefore `bytesPlane0` is `4`.
/// The function returns the descriptor as a little-endian byte vector ready to
/// be written to the file.
fn create_rgb9e5_dfd() -> Vec<u8> {
    // Helper to push a 32-bit little-endian word
    fn push(word: u32, out: &mut Vec<u8>) {
        out.extend_from_slice(&word.to_le_bytes());
    }

    let mut dfd: Vec<u8> = Vec::with_capacity(132);
    dfd.extend_from_slice(&0u32.to_le_bytes()); // will be overwritten later

    // Data-format-descriptor header (2 × u32)
    // word0: descriptorType (lower 15 b) | vendorId (upper 17 b) – both 0 → 0
    push(0, &mut dfd);

    // The BASIC block length in bytes = 24 (header) + 16 × numSamples.
    const NUM_SAMPLES: usize = 6;
    const BASIC_BLOCK_BYTE_LENGTH: u32 = 24 + 16 * NUM_SAMPLES as u32;
    const VERSION_NUMBER: u32 = 2;
    // word1: versionNumber (low 16 b) | descriptorBlockSize (high 16 b)
    let word1 = (BASIC_BLOCK_BYTE_LENGTH << 16) | VERSION_NUMBER;
    push(word1, &mut dfd);

    // word2: colourModel | colourPrimaries | transferFunction | flags
    const COLOR_MODEL_RGBSDA: u32 = 1; // KHR_DF_MODEL_RGBSDA
    const COLOR_PRIMARIES_BT709: u32 = 1; // Recommended default
    const TRANSFER_LINEAR: u32 = 1; // KHR_DF_TRANSFER_LINEAR
    const FLAGS_STRAIGHT_ALPHA: u32 = 0; // no premultiplied alpha
    let word2 = COLOR_MODEL_RGBSDA
        | (COLOR_PRIMARIES_BT709 << 8)
        | (TRANSFER_LINEAR << 16)
        | (FLAGS_STRAIGHT_ALPHA << 24);
    push(word2, &mut dfd);

    // word3: texelBlockDimensions – for a 1×1×1 block we store each dimension − 1 = 0
    push(0, &mut dfd);

    // word4 & word5: bytesPlane0-3 / bytesPlane4-7 (8 × u8)
    // For a packed 32-bit texel bytesPlane0 = 4, the rest 0.
    push(4, &mut dfd); // bytesPlane0 = 4, others 0
    push(0, &mut dfd); // bytesPlane4-7 = 0

    fn push_sample(
        out: &mut Vec<u8>,
        bit_offset: u32,
        bit_length_bits: u32,
        channel_type: u32,
        qualifiers: u32,
        lower: u32,
        upper: u32,
    ) {
        let first_word =
            bit_offset | ((bit_length_bits - 1) << 16) | (channel_type << 24) | (qualifiers << 28);
        push(first_word, out);
        push(0, out); // samplePosition – not used → 0
        push(lower, out);
        push(upper, out);
    }

    // Qualifier bits (see ChannelTypeQualifiers in ktx2 crate)
    const QUAL_NONE: u32 = 0;
    const QUAL_EXPONENT: u32 = 1 << 1; // EXPONENT flag

    // Channel-type codes (KDF §A.3): 0=R,1=G,2=B,3=A/SharedExponent
    const CH_R: u32 = 0;
    const CH_G: u32 = 1;
    const CH_B: u32 = 2;
    const CH_EXPONENT: u32 = 3; // shared exponent stored in alpha slot

    // For each colour channel we write: mantissa sample followed by its exponent sample.

    // RED mantissa & exponent
    push_sample(&mut dfd, 0, 9, CH_R, QUAL_NONE, 0, 8448); // R mantissa (bits 0-8)
    push_sample(&mut dfd, 27, 5, CH_R, QUAL_EXPONENT, 15, 31); // R exponent (bits 27-31)

    // GREEN mantissa & exponent
    push_sample(&mut dfd, 9, 9, CH_G, QUAL_NONE, 0, 8448); // G mantissa (bits 9-17)
    push_sample(&mut dfd, 27, 5, CH_G, QUAL_EXPONENT, 15, 31); // G exponent (shared bits)

    // BLUE mantissa & exponent
    push_sample(&mut dfd, 18, 9, CH_B, QUAL_NONE, 0, 8448); // B mantissa (bits 18-26)
    push_sample(&mut dfd, 27, 5, CH_B, QUAL_EXPONENT, 15, 31); // B exponent (shared bits)

    // Patch totalSize ------------------------------------------------------------------
    let total_size = dfd.len() as u32;
    dfd[0..4].copy_from_slice(&total_size.to_le_bytes());

    dfd
}

/// Extract a specific individual mip level as a new image.
pub fn extract_mip_level(image: &Image, mip_level: u32, face: u32) -> Image {
    let descriptor = &image.texture_descriptor;

    if descriptor.mip_level_count < mip_level {
        panic!(
            "Mip level {mip_level} requested, but only {} are avaliable.",
            descriptor.mip_level_count
        );
    }

    let block_size = descriptor.format.block_copy_size(None).unwrap() as usize;

    let mut byte_offset = 0usize;

    for _ in 0..face {
        let mut width = descriptor.size.width as usize;
        let mut height = descriptor.size.height as usize;
        for _ in 0..descriptor.mip_level_count {
            byte_offset += width * block_size * height;
            width /= 2;
            height /= 2;
        }
    }

    let mut width = descriptor.size.width as usize;
    let mut height = descriptor.size.height as usize;

    for _ in 0..mip_level {
        byte_offset += width * block_size * height;
        width /= 2;
        height /= 2;
    }

    let mut new_descriptor = descriptor.clone();

    new_descriptor.mip_level_count = 1;
    new_descriptor.size = Extent3d {
        width: width as u32,
        height: height as u32,
        depth_or_array_layers: 1,
    };

    Image {
        data: image.data[byte_offset..byte_offset + (width * block_size * height)].to_vec(),
        texture_descriptor: new_descriptor,
        sampler: image.sampler.clone(),
        texture_view_descriptor: image.texture_view_descriptor.clone(),
        asset_usage: RenderAssetUsages::default(),
    }
}
