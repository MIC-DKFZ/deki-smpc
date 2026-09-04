# FHE baseline encryption and decryption pipelines

This document describes the message-processing pipelines used for the CKKS,
BFV, and BGV baselines in *Efficient Privacy-Preserving Medical Cross-Silo
Federated Learning*. These pipelines document the experimental baselines; they
are not part of the deki-smpc protocol or client API.

## CKKS encryption pipeline

### Inputs

- State dictionary `S` containing numeric arrays or tensors
- Ring dimension `N`
- CKKS scale `Delta`
- Public key `pk`
- Ciphertext serializer `Ser`
- Compressor `Comp`

### Outputs

- Compressed ciphertext bytes `B`
- Metadata needed to reconstruct the state dictionary

### Steps

1. **Flatten the state dictionary.** Traverse `S` in a fixed order, record each
   tensor's shape and data type in the metadata, and concatenate the values into
   one list `L` of complex numbers.
2. **Chunk by ring dimension.** Split `L` into chunks `C_1, ..., C_m` of length
   at most `N`. Pad the final chunk with zeros when necessary and record the
   padding length.
3. **Encode as plaintext polynomials.** Apply CKKS encoding with scale `Delta`
   to every chunk `C_i`, producing a plaintext polynomial `p_i` in `R_Q`.
4. **Encrypt the plaintexts.** Compute `ct_i = Enc_pk(p_i)` for every plaintext
   polynomial.
5. **Serialize the ciphertexts.** Convert every `ct_i` to bytes with `Ser` and
   pack the serialized ciphertexts into a single container `B'`.
6. **Compress the container.** Compute `B = Comp(B')`.

The metadata stores `N`, `Delta`, the number of chunks, the padding length, the
state-dictionary traversal order, each tensor's shape and data type, codec
versions, CKKS context parameters, and key identifiers.

## CKKS decryption pipeline

### Inputs

- Compressed ciphertext bytes `B`
- The recorded metadata
- Secret key `sk`
- Matching decompressor and ciphertext deserializer

### Output

- Reconstructed state dictionary `S`

### Steps

1. **Decompress the container.** Apply the matching decompressor to recover
   `B'`.
2. **Unpack and deserialize.** Extract the serialized ciphertexts and
   deserialize each one to recover `ct_1, ..., ct_m`.
3. **Decrypt the ciphertexts.** Compute `p_i = Dec_sk(ct_i)` for every
   ciphertext.
4. **Decode the plaintexts.** Apply CKKS decoding with the recorded context and
   scale to recover the chunks `C_1, ..., C_m`.
5. **Concatenate and trim.** Concatenate the decoded chunks in their recorded
   order and remove the recorded zero padding from the final chunk.
6. **Reconstruct the state dictionary.** Split the resulting list using the
   recorded traversal order, shapes, and data types, then restore every tensor
   in `S`.

## BFV and BGV differences

BFV and BGV use the same flattening, chunking, encryption, serialization, and
compression pipeline, with an additional float-to-integer conversion after
flattening:

1. Choose a scale `s` and map each real value `x` to
   `x' = round(s * x) mod t`, where `t` is the plaintext modulus.
2. Record `s` and `t` in the metadata, then encode, encrypt, serialize, and
   compress as above.
3. During decryption, after decoding, concatenation, and padding removal,
   convert the recovered integers back to floating-point values with
   $x \approx x'/s$ before reconstructing the state dictionary.
