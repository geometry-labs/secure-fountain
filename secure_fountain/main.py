""" Secure Fountain Architecture

Description
-----------

This library employs a secure fountain architecture from [1] and using the degree distributions from [2] to reduce
local storage requirements for blockchain nodes and maintain some transaction verification capability. A good way of
using this repo is for nodes to encode all block epochs that have been unused for transaction verification for a
sufficiently long period of time, only to re-bootstrap that epoch later if necessary.

References:
    [1] Kadhe, Swanand, Jichan Chung, and Kannan Ramchandran. "Sef: A secure fountain architecture for slashing storage
        costs in blockchains." arXiv preprint arXiv:1906.12140 (2019). https://arxiv.org/abs/1906.12140
    [2] Hyytiä, Esa, Tuomas Tirronen, and Jorma Virtamo. "Optimizing the degree distribution of LT codes with an
        importance sampling approach." RESIM 2006, 6th International Workshop on Rare Event Simulation. 2006.
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.8992&rep=rep1&type=pdf

OVERALL IDEA:
Full nodes become droplet nodes by partitioning their blockchain into 128-block epochs, then encoding each epoch into a
list ("bucket") of 40 encoded blocks ("droplets"). The size of a droplet is only a few bits more than a block. Droplet
nodes save their header chain and their droplets, but otherwise can erase any un-encoded confirmed blocks they like.
Droplet nodes serve their header chains and buckets to bootstrapping nodes. New nodes can become droplet nodes either by
first bootstrapping the full blockchain as a full node and then encoding the blockchain at the end, or instead by
bootstrapping the blockchain epoch-by-epoch, encoding epochs as they go. Many droplets contain whole blocks, or can be
XORd together to obtain whole blocks, so droplet nodes are still capable of a lot of transaction verification.

TERMINOLOGY:
    BlockHeader: (usual meaning)
    Block: (usual meaning)
    Epoch: A sequence of at most 128 adjacent blocks.
    BlockChain: The entirety of the blockchain.
    Droplet: An encoded block.
    Bucket: A list of encoded blocks.
    BucketChain: A list of buckets.
    Full Node: (usual meaning)
    Droplet Node: A node that serves encoded blocks instead of blocks to their peers.
    Bucket Node: A new node that collects encoded blocks from droplet nodes to bootstrap.

STORAGE SAVINGS
Storage savings are (i) scalable, (ii) controlled by the DROPLETS_TO_SAVE_PER_EPOCH variable, and (iii) in trade-off
with the number of neighbors N we must query for droplets before decoding.
    # >>> relative_storage_savings = 1.0 - DROPLETS_TO_SAVE_PER_EPOCH / 128
    # >>> number_of_neighbors_to_query = 2 * ceil(160 / DROPLETS_TO_SAVE_PER_EPOCH)

Examples:
    - If number_of_neighbors_to_query <= 128, then DROPLETS_TO_SAVE_PER_EPOCH >= 2.5, but we need an integer. To
      maximize relative_storage_savings, we minimize DROPLETS_TO_SAVE_PER_EPOCH with DROPLETS_TO_SAVE_PER_EPOCH = 3.
      This gives us relative_storage_savings = 1.0 - 3/128 = 0.9765625, or ~97.6% savings.
    - If relative_storage_savings >= 0.9, then DROPLETS_TO_SAVE_PER_EPOCH <= 12.8, but we need an integer. To minimize
      number_of_neighbors_to_query, we maximize DROPLETS_TO_SAVE_PER_EPOCH, so we set DROPLETS_TO_SAVE_PER_EPOCH = 12.
      This gives us number_of_neighbors_to_query = 28.
    - If we use the Bitcoin model of contacting number_of_neighbors_to_query = 8, then DROPLETS_TO_SAVE_PER_EPOCH = 40.
      This gives us relative_storage_savings = 0.6875, or ~68.8% savings. This is the default value of the repo.

DROPLET NODE TRANSACTION VERIFICATION:
Droplet nodes by default carry 40 droplets and some of these correspond to only one block. We can verify
transactions from these singleton droplets without problem. More than that, if two droplets have index sets with a
singleton symmetric difference, those two droplets can be XORed to validate transactions out of the block touched by
both.

More than that, if some N droplets have index sets such that some combination of symmetric differences of these sets are
singletons, the droplet node can also validate transactions out of that singleton block by simply computing some XORs.

Even if a transaction is relayed to you for a block that you cannot verify out of in these ways, you can query a full
node neighbor just for the block you are missing, or a droplet node neighbor for a new droplet.

Lastly, if none of the above work, you can attempt to re-bootstrap the block epoch entirely as in the previous section.

ENCODING ENTIRE BLOCKCHAIN (FULL NODE -> DROPLET NODE):
Full nodes turn themselves into droplet nodes with the following.
    # >>> encoded_block_chain, remaining_blocks = encode(blocks)
Save encoded_block_chain and remaining_blocks. The full node is now a droplet node, you can erase any confirmed blocks from the original blockchain you like. The encoded_block_chain consists of the original header chain and, for each epoch of 128 blocks, a list of Droplets we call a bucket.
    # >>> assert isinstance(encoded_block_chain.hdr_chain, list)
    # >>> assert all(isinstance(next_hdr, BlockHeader) for next_hdr in encoded_block_chain.hdr_chain)
    # >>> assert isinstance(encoded_block_chain.buckets, list)
    # >>> assert all(isinstance(next_bucket, list) for next_bucket in encoded_block_chain.buckets)
    # >>> assert all(len(next_bucket)==DROPLETS_TO_SAVE_PER_EPOCH for next_bucket in encoded_block_chain.buckets)
    # >>> assert all(all(isinstance(next_droplet, Droplet) for next_droplet in next_bucket) for next_bucket in encoded_block_chain.buckets)
Droplet nodes should respond to peer queries for the j-th block epoch by responding with their j-th bucket.
    # >>> response = encoded_block_chain.buckets[j]
Droplet nodes should also share their block headers with peers when queried.

ENCODING A SINGLE EPOCH:
New nodes can bootstrap into being a full node the classic way, and become a droplet node after that at any time. On the
other hand, bootstrapping nodes that do not wish to become a full node en route to becoming a droplet node can encode
epochs as they go. Use encode_epoch to encode an epoch of blocks to a bucket.
    # >>> hdr_chain: List[BlockHeader] = [b.hdr for b in epoch] # be sure to save the header chain!
    # >>> encoded_blocks: List[Droplet] = encode_epoch(epoch)
Now encoded_blocks is a list of Droplet objects.
    # >>> assert isinstance(encoded_blocks, list)
    # >>> assert len(encoded_blocks) == DROPLETS_TO_SAVE_PER_EPOCH
    # >>> assert all(isinstance(next_droplet, Droplet) for next_droplet in encoded_blocks)
Droplet nodes should respond to peer queries for this epoch by responding with encoded_blocks. Droplet nodes should also
share their block headers with peers when queried.

DECODING ENTIRE BLOCKCHAIN (DROPLET NODE -> FULL NODE):
Obtain the longest honest valid header chain from your neighbors as usual. Download an encoded_block_chain from each
neighbor, say encoded_block_chains[0], ..., encoded_block_chains[N-1]. In our default case, we contact 8 neighbors in
total. The encoded blockchains can be summed to merge their droplets, bucket by bucket.
    # >>> total_encoded_block_chain: EncodedBlockChain = sum(encoded_block_chains)
Then decode the total to obtain a decoded block_chain or bool.
    # >>> decoded_block_chain: List[Block] = decode(total_encoded_block_chain)
If decode returns the bool False, then decoding failed; download more droplets and try again. Otherwise, if decode  does not
return the bool False, then the result is just a list of blocks.
    # >>> assert isinstance(decoded_block_chain, list)
    # >>> assert all(isinstance(next_block, Block) for next_block in decoded_block_chain)
    # >>> assert len(decoded_block_chain) == BLOCKS_PER_EPOCH * len(encoded_block_chain.buckets)
Make sure you verify that the header chain matches.
    # >>> assert [b.hdr for b in decoded_block_chain] == encoded_block_chain.hdr_chain
Of course, make sure you verify all blockchain data before attempting transaction verification.

DECODING A SINGLE EPOCH:
Obrtain the longest honest valid header chain longest_hdr_chain from your neighbors as usual. Contact full node peers
and download their blocks, downloaded_blocks, from the desired epoch (say the j-th epoch), if possible. If you can
obtain the whole epoch this way, then do so. Otherwise, create a bucket of singleton droplets for the desired epoch
from any blocks you've already downloaded.
    # >>> singleton_bucket = [Droplet(indices=idx, val=block) for idx, block in zip(indices, downloaded_blocks)]
Then, contact droplet peers and download their bucket for the desired epoch and merge buckets before attempting to
decode.
    # >>> merged_bucket = singleton_bucket + sum(downloaded_buckets)
Use the headers for this epoch to decode.
    # >>> decoded_epoch = decode_epoch(longest_hdr_chain[j * BLOCKS_PER_EPOCH: j * BLOCKS_PER_EPOCH], merged_bucket)
If bucket_to_epoch does not fail by producing the bool False, you have obtained the whole epoch this way.

Hosting
-------
The repository for this project can be found at [https://www.github.com/geometry-labs/secure_fountain](https://www.github.com/geometry-labs/secure_fountain).

License
-------
Released under the MIT License, see LICENSE file for details.

Copyright
---------
Copyright (c) 2022 Geometry Labs, Inc.
Funded by The QRL Foundation.

Contributors
------------
Brandon Goodell (lead author), Mitchell "Isthmus" Krawiec-Thayer

"""
from copy import deepcopy
from random import random, sample
from typing import List, Tuple, Union

# SEF Constants (only change these if you understand the implications)
BLOCKS_PER_EPOCH: int = 128
DROPLETS_TO_SAVE_PER_EPOCH: int = 40
DROPLETS_TO_DECODE_EPOCH: int = 160
PMF: List[float] = [
    0.18,
    0.33,
    0.26,
    0.14,
    0.05,
    0.01,
    1 - 0.18 - 0.33 - 0.26 - 0.14 - 0.05 - 0.01,
]
CMF: List[float] = [sum(PMF[:i]) for i in range(1, 1 + len(PMF))]

# Block chain-specific constants.
TXN_LEN_IN_BYTES: int = 128

# Type aliases for readability
HashDigest = List[int]
MerkleRoot = List[int]
MetaData = List[int]
Payload = List[int]
BlockIndex = int
BlockIndices = List[BlockIndex]
Degree = int
Degrees = List[int]
SampleSize = int
Proportion = float
Proportions = List[float]
EpochCount = int


def payload_to_merkle_root(x: bytes) -> MerkleRoot:
    """Dummy function for computing merkle roots. Mock for testing."""
    ...


class BlockHeader(object):
    """BlockHeader class. Interpret bitstrings (i.e. hash outputs) as lists of binary expansions of 32-bit signed integers.

    Attributes
    ----------
        hash_parent_hdr: List[int]
            The hash output of the parent BlockHeader written as a bitstring then encoded as a list of 32-bit signed ints.
        root: List[int]
            A Merkle root written as a bitstring then encoded as a list of 32-bit signed ints.
        metadata: List[int]
    """

    hash_parent_hdr: HashDigest
    root: MerkleRoot
    metadata: MetaData

    def __init__(self, **data):
        self.hash_parent_hdr = data["hash_parent_hdr"]
        self.root = data["root"]
        self.metadata = data["metadata"]

    def __xor__(self, other):
        """XOR two BlockHeaders by XORing the integers in each attribute coordinate-wise."""
        result = deepcopy(self)
        result.hash_parent_hdr = [
            i ^ j for i, j in zip(self.hash_parent_hdr, other.hash_parent_hdr)
        ]
        result.root = [i ^ j for i, j in zip(self.root, other.root)]
        result.metadata = [i ^ j for i, j in zip(self.metadata, other.metadata)]
        return result

    def __eq__(self, other) -> bool:
        """Test two BlockHeaders for equality by comparing all attributes."""
        return (
            isinstance(self, type(other))
            and self.hash_parent_hdr == other.hash_parent_hdr
            and self.root == other.root
            and self.metadata == other.metadata
        )


HeaderChain = List[BlockHeader]


class Block(object):
    """Block class.

    Attributes
    ----------
        hdr: BlockHeader
        payload: List[int]
            Block payload written as a bitstring then encoded as a list of 32-bit signed ints.
    """

    hdr: BlockHeader
    payload: Payload

    def __init__(self, **data):
        self.hdr = BlockHeader(
            hash_parent_hdr=data["hdr"].hash_parent_hdr,
            root=data["hdr"].root,
            metadata=data["hdr"].metadata,
        )
        self.payload = data["payload"]

    def __xor__(self, other):
        """XOR two Blocks by XORing the BlockHeaders and payloads."""
        result = deepcopy(self)
        result.hdr ^= other.hdr
        result.payload = [i ^ j for i, j in zip(self.payload, other.payload)]
        return result

    def __eq__(self, other) -> bool:
        """Test equality of Blocks by comparing headers and payloads."""
        return (
            isinstance(self, type(other))
            and self.hdr == other.hdr
            and self.payload == other.payload
        )


# Type aliases for readability
BlockChain: type = List[Block]
Epoch: type = List[Block]  # sub-sequence of BlockChain
BlockList: type = List[Block]  # non-sequential
SampledBlocks: type = Tuple[BlockIndices, BlockList]


def block_verify(expected_hdr: BlockHeader, block: Block) -> bool:
    """Verify that the input block BlockHeader matches the input expected BlockHeader and that the input block payload Merkle root matches the input BlockHeader Merkle root."""
    a: bool = expected_hdr == block.hdr
    payload_as_list_of_bytes = [
        i.to_bytes(length=TXN_LEN_IN_BYTES, byteorder="big") for i in block.payload
    ]
    i: int = 0
    payload_as_bytes: bytes = payload_as_list_of_bytes[i]
    while i + 1 < len(payload_as_list_of_bytes):
        i += 1
        payload_as_bytes += payload_as_list_of_bytes[i]
    b: bool = payload_to_merkle_root(payload_as_bytes) == block.hdr.root
    return a and b


def epoch_verify(expected_hdr_chain: HeaderChain, epoch: Epoch) -> bool:
    """Verify an epoch stored as a list of blocks."""
    return all(
        block_verify(expected_hdr=next_hdr, block=next_block)
        for next_hdr, next_block in zip(expected_hdr_chain, epoch)
    )


class Droplet(object):
    """Droplet class (XOR'd blocks).

    Attributes
    ----------
        indices: List[int]
            List of integer indexes indicating the blocks touched by this Droplet.
        val: List[int]
            The payloads of touched blocks XORd together.

    (Potential upgrade: Modify so that indices is a bitstring characteristic vector to save space and making XORing two droplets simpler. Example: use 11100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 instead of [0, 1, 2, 127]).
    """

    indices: BlockIndices
    val: Block

    def __init__(self, **data):
        self.indices = sorted(data["indices"])
        self.val = data["val"]

    def __xor__(self, other):
        """XOR two Droplets by XORing their val Blocks and XORing the index characteristic vectors (equiv symmetric difference of sets of indices)."""
        result = deepcopy(self)
        result.indices = [x for x in self.indices if x not in other.indices]
        result.indices += [y for y in other.indices if y not in self.indices]
        result.indices = sorted(result.indices)
        result.val ^= other.val
        return result

    def __eq__(self, other) -> bool:
        """Test equality of droplets by comparing attributes."""
        return (
            isinstance(self, type(other))
            and self.indices == other.indices
            and self.val == other.val
        )

    def __len__(self) -> int:
        """Define the length of a droplet as the number of touched blocks. Convenience."""
        return len(self.indices)


# Type aliases for readability
Bucket: type = List[Droplet]
PeeledSingleton: type = Tuple[
    Union[None, Block], Union[None, BlockIndex], Bucket
]  # Contains a single block, the index, and the bucket from whence it came
BucketChain: type = List[Bucket]
EncodedEpoch: type = Tuple[
    HeaderChain, Bucket
]  # Encode an epoch into a header chain and a bucket of droplets.
RemainingBlocks: type = List[
    Block
]  # Trailing (unconfirmed) blocks before we can fill up a whole new epoch.


def sample_degree_w_replacement(sample_size: SampleSize) -> Degrees:
    """Sample from the degree distribution in "Optimizing the Degree Distribution of LT Codes with an Importance Sampling Approach" from https://www.netlab.tkk.fi/~esa/pub/files/hyytia-resim-2006.pdf

    Note: if block epochs are not length 128, this distribution is very likely to perform inadequately.

    :param sample_size: int
    :return: Tuple of sampled degrees
    :rtype: List[int]
    """
    result: List[int] = []
    while len(result) < sample_size:
        u: Proportions = sorted([random() for _ in range(sample_size)])
        for i, next_cmf in enumerate(CMF):
            successes, u = (
                len([v for v in u if v < next_cmf]),
                [v for v in u if v >= next_cmf],
            )
            result += [2**i] * successes
            if not len(u):  # (if `u` is empty, break)
                break
    return result[:sample_size]


def degree_to_indices_and_blocks(degree: Degree, epoch: Epoch) -> SampledBlocks:
    """Input degree and block epoch, output a sampled list of indices and the corresponding list of blocks.

    :param degree: int
    :param epoch: List[Block]
    :return: (block_indices, blocks)
    :rtype: List[list[int], list[Block]]
    """
    if len(epoch) != BLOCKS_PER_EPOCH:
        raise ValueError(
            f"Input epoch must have {BLOCKS_PER_EPOCH} blocks, but had {len(epoch)}."
        )
    population: BlockIndices = list(range(BLOCKS_PER_EPOCH))
    sampled_indices: BlockIndices = sorted(
        sample(population=population, k=degree)
    )  # Not necessary to sort.
    sampled_blocks: List[Block] = [epoch[i] for i in sampled_indices]
    return sampled_indices, sampled_blocks


def indices_and_blocks_to_droplet_val(sampled_blocks: SampledBlocks) -> Block:
    """Input some sampled blocks, output the payload of the corresponding droplet.

    :param sampled_blocks: SampledBlocks
    :return: droplet_payload
    :rtype: Block
    """
    blocklist: List[Block] = sampled_blocks[1]
    droplet_payload = blocklist[0]

    for val in blocklist[1:]:
        droplet_payload ^= val

    return droplet_payload


def indices_and_blocks_to_droplet(sampled_blocks: SampledBlocks) -> Droplet:
    """Input some sampled blocks, output the corresponding droplet. Different from indices_and_blocks_to_droplet_val, which only outputs the payload.

    :param sampled_blocks: SampledBlocks
    :return: droplet
    :rtype: Droplet
    """
    val = indices_and_blocks_to_droplet_val(sampled_blocks=sampled_blocks)
    indices: List[int] = sampled_blocks[0]
    droplet = Droplet(indices=indices, val=val)
    return droplet


def encode_epoch(sample_size: SampleSize, epoch: Epoch) -> Bucket:
    """Input a sample size and an epoch, output a bucket of sample_size droplets.

    :param sample_size: SampleSize
    :param epoch: Epoch
    :return: bucket
    :rtype: Bucket
    """
    degrees: Degrees = sample_degree_w_replacement(sample_size=sample_size)
    if not isinstance(degrees, list) or not all(isinstance(i, int) for i in degrees):
        raise ValueError("Integer?")

    bucket: List[Droplet] = []
    for degree in degrees:
        sampled_blocks: SampledBlocks = degree_to_indices_and_blocks(
            degree=degree, epoch=epoch
        )
        bucket += [indices_and_blocks_to_droplet(sampled_blocks=sampled_blocks)]
    return bucket


def find_singleton(bucket: Bucket) -> Union[bool, Droplet]:
    """Input a bucket of droplets, output a boolean indicating no singletons were found, or a droplet corresponding to the first singleton found.

    :param bucket: Bucket
    :return: droplet
    :rtype: Union[bool, Droplet]
    """
    droplet: Union[bool, Droplet] = False
    for next_droplet in bucket:
        if len(next_droplet) == 1:
            droplet = next_droplet
            break
    return droplet


def peel_singleton(
    expected_hdr: BlockHeader, singleton: Droplet, bucket: Bucket
) -> PeeledSingleton:
    """Input an expected block header, a singleton, and a bucket. Output the corresponding PeeledSingleton, which
    is a tuple with the singleton, its index, and the bucket with all the droplets XORd against the singleton.

    :param expected_hdr: BlockHeader
    :param singleton: Droplet
    :param bucket: Bucket
    :return: peeled_singleton
    :rtype: PeeledSingleton
    """
    resulting_bucket: Bucket = [d for d in bucket if d != singleton and len(d)]
    if not block_verify(expected_hdr=expected_hdr, block=singleton.val):
        return None, None, resulting_bucket
    for i in range(len(resulting_bucket)):
        if singleton.indices[0] in resulting_bucket[i].indices:
            resulting_bucket[i] ^= singleton
    resulting_bucket = [d for d in resulting_bucket if len(d) > 0]
    peeled_singleton: PeeledSingleton = (
        singleton.val,
        singleton.indices[0],
        resulting_bucket,
    )
    return peeled_singleton


def find_and_peel_singleton(
    exp_hdr_chain: HeaderChain, bucket: Bucket
) -> Union[bool, PeeledSingleton]:
    """Calls find_singleton and peel_singleton.

    :param exp_hdr_chain: HeaderChain
    :param bucket: Bucket
    :return: peeled_singleton
    :rtype: PeeledSingleton
    """
    next_singleton: Union[bool, Droplet] = find_singleton(bucket=bucket)
    if not next_singleton:
        return False
    next_singleton_index: int = next_singleton.indices[0]
    peeled_singleton: PeeledSingleton = peel_singleton(
        expected_hdr=exp_hdr_chain[next_singleton_index],
        singleton=next_singleton,
        bucket=bucket,
    )
    return peeled_singleton


class EncodedBlockChain(object):
    """EncodedBlockChain (aka bucket chain). Note that this object does not concern itself with the trailing/unconfirmed blocks that do not fit into an epoch.

    Attributes
    ----------
        hdr_chain: HeaderChain
            Chain of block headers.
        buckets: BucketChain
            Buckets corresponding to each epoch.
    """

    hdr_chain: HeaderChain
    buckets: BucketChain

    def __init__(self, **data):
        self.hdr_chain = data["hdr_chain"]
        self.buckets = data["buckets"]

    def __add__(self, other):
        """Adding two EncodedBlockChains concatenates the buckets in each EncodedBlockChain together."""
        if not isinstance(self, type(other)):
            raise ValueError("")
        elif self.hdr_chain != other.hdr_chain:
            raise ValueError("Header chains must match.")
        elif len(self.buckets) != len(other.buckets) or len(
            self.buckets
        ) * BLOCKS_PER_EPOCH != len(self.hdr_chain):
            raise ValueError(
                "Both encoded blockchains must have the same number of encoded epochs, which must match header chain length."
            )
        result = deepcopy(self)
        for i, (a, b) in enumerate(zip(self.buckets, other.buckets)):
            result.buckets[i] = a + b
        return result

    def __radd__(self, other):
        """Allows us to use sum() since __add__() defined above."""
        if other == 0 or other is None:
            return self
        return self + other


def encode(block_chain: BlockChain) -> Tuple[EncodedBlockChain, Epoch]:
    """Inputs a block_chain, outputs a tuple with the EncodedBlockChain and a partial epoch of the trailing blocks (unconfirmed, usually).

    :param block_chain: BlockChain
    :return: encoded_block_chain_and_trailing_epoch
    :rtype: Tuple[EncodedBlockChain, Epoch]
    """
    hdr_chain: HeaderChain = [b.hdr for b in block_chain]
    num_epochs: int = len(block_chain) // BLOCKS_PER_EPOCH
    epochs = [
        block_chain[i * BLOCKS_PER_EPOCH : (i + 1) * BLOCKS_PER_EPOCH]
        for i in range(num_epochs)
    ]
    remaining_blocks: List[Block] = block_chain[num_epochs * BLOCKS_PER_EPOCH :]
    encoded_block_chain: EncodedBlockChain = EncodedBlockChain(
        hdr_chain=hdr_chain,
        buckets=[
            encode_epoch(sample_size=DROPLETS_TO_SAVE_PER_EPOCH, epoch=epoch)
            for epoch in epochs
        ],
    )
    encoded_block_chain_and_trailing_epoch: Tuple[EncodedBlockChain, Epoch] = (
        encoded_block_chain,
        remaining_blocks,
    )
    return encoded_block_chain_and_trailing_epoch


def decode_epoch(exp_hdr_chain: HeaderChain, bucket: Bucket) -> Union[bool, Epoch]:
    """Inputs an expected header chain and a bucket, outputs a boolean (indicating failure) or a decoded epoch.

    :param exp_hdr_chain: HeaderChain
    :param bucket: Bucket
    :return: decoded_epoch
    :rtype: Union[bool, Epoch]
    """
    srt_bucket: List[Droplet] = sorted(bucket, key=lambda x: len(x))
    decoded_epoch: List[Union[None, Block]] = [None] * BLOCKS_PER_EPOCH
    while len(srt_bucket) > 0 and any(i is None for i in decoded_epoch):
        peel_function_output: Union[bool, list] = find_and_peel_singleton(
            exp_hdr_chain=exp_hdr_chain, bucket=srt_bucket
        )

        # If epoch does not have singleton, pass up the False
        if isinstance(peel_function_output, bool):
            if not peel_function_output:
                return False  # Did not have sufficient data to decode
            else:
                raise Exception(
                    "Unexpected True from find_and_peel_singleton. You should never see this message."
                )

        # If decoding was successful, unpack the output proceed as usual
        else:
            block, block_index, srt_bucket = peel_function_output
        if block is None or block_index is None:
            return False
        decoded_epoch[block_index] = block

    if any(i is None for i in decoded_epoch):
        return False

    return decoded_epoch


def decode(encoded_block_chain: EncodedBlockChain) -> Union[bool, BlockChain]:
    """Inputs an encoded blockchain and outputs a boolean (indicating decoding failure) or a blockchain.

    :param encoded_block_chain: EncodedBlockChain
    :return: decoded_block_chain
    :rtype: Union[bool, BlockChain]
    """
    decoded_block_chain: List[Block] = []
    for i, next_bucket in enumerate(encoded_block_chain.buckets):
        hdrs: HeaderChain = encoded_block_chain.hdr_chain[
            i * BLOCKS_PER_EPOCH : (i + 1) * BLOCKS_PER_EPOCH
        ]
        try:
            epoch = decode_epoch(
                exp_hdr_chain=hdrs, bucket=encoded_block_chain.buckets[i]
            )
        except ValueError as _:
            return False
        if isinstance(epoch, bool):
            return epoch
        decoded_block_chain += epoch
    return decoded_block_chain
