# Testing suite requires pytest and pytest-mock (see requirements_for_tests.txt)
import sys
from random import randint, seed
from typing import List, Union

import pytest

from secure_fountain import (
    BlockHeader,
    Block,
    block_verify,
    epoch_verify,
    Droplet,
    sample_degree_w_replacement,
    degree_to_indices_and_blocks,
    indices_and_blocks_to_droplet_val,
    indices_and_blocks_to_droplet,
    encode_epoch,
    find_singleton,
    peel_singleton,
    find_and_peel_singleton,
    BLOCKS_PER_EPOCH,
    DROPLETS_TO_SAVE_PER_EPOCH,
    EncodedBlockChain,
    encode,
    decode_epoch,
    decode,
)

TEST_SAMPLE_SIZE: int = 2 ** 3


# Check interpreter version


def test_python_version_3_8_or_above():
    """Checks that the interpreter understands the built-in types used in the secure fountain codebase"""
    try:
        True and (_ := False)
    except TypeError:
        raise Exception(
            f"\n\nPrototype requires python 3.8+, but this interpreter is {sys.version}"
        )


@pytest.mark.skip(
    reason="This release has been modified to be compatible with 3.8+ onward"
)
def test_python_version_3_10_or_above():
    """Checks that the interpreter understands the built-in types used in the secure fountain codebase"""
    try:
        _: list[str] = [
            "attempt type hint with list[] syntax",
            "instead of List[]",
        ]  # noqa: is test
    except TypeError:
        raise Exception(
            f"\n\nPrototype requires python 3.10 style typing, but this interpreter is {sys.version}"
        )


MAX_ENTRY_LEN: int = 2 ** 12
HASH_PARENT_HDR_LEN: int = 10
ROOT_LEN: int = 10
METADATA_LEN: int = 10


def test_block_header_det():
    """Deterministically tests the block headers."""
    for _ in range(TEST_SAMPLE_SIZE):
        x_simulated_hash_parent_hdr: List[int] = [
            2 ** i % MAX_ENTRY_LEN for i in range(HASH_PARENT_HDR_LEN)
        ]
        x_simulated_merkle_root: List[int] = [
            3 ** i % MAX_ENTRY_LEN for i in range(ROOT_LEN)
        ]
        x_simulated_metadata: List[int] = [
            5 ** i % MAX_ENTRY_LEN for i in range(METADATA_LEN)
        ]

        x: BlockHeader = BlockHeader(
            hash_parent_hdr=x_simulated_hash_parent_hdr,
            root=x_simulated_merkle_root,
            metadata=x_simulated_metadata,
        )

        y_simulated_hash_parent_hdr = [
            7 ** i % MAX_ENTRY_LEN for i in range(HASH_PARENT_HDR_LEN)
        ]
        y_simulated_merkle_root = [11 ** i % MAX_ENTRY_LEN for i in range(ROOT_LEN)]
        y_simulated_metadata = [13 ** i % MAX_ENTRY_LEN for i in range(METADATA_LEN)]

        y: BlockHeader = BlockHeader(
            hash_parent_hdr=y_simulated_hash_parent_hdr,
            root=y_simulated_merkle_root,
            metadata=y_simulated_metadata,
        )

        expected_xord_hash_parent_hdr = [
            (2 ** i % MAX_ENTRY_LEN) ^ (7 ** i % MAX_ENTRY_LEN)
            for i in range(HASH_PARENT_HDR_LEN)
        ]
        expected_xord_merkle_root = [
            (3 ** i % MAX_ENTRY_LEN) ^ (11 ** i % MAX_ENTRY_LEN)
            for i in range(ROOT_LEN)
        ]
        expected_xord_metadata = [
            (5 ** i % MAX_ENTRY_LEN) ^ (13 ** i % MAX_ENTRY_LEN)
            for i in range(METADATA_LEN)
        ]
        expected_x_xor_y: BlockHeader = BlockHeader(
            hash_parent_hdr=expected_xord_hash_parent_hdr,
            root=expected_xord_merkle_root,
            metadata=expected_xord_metadata,
        )

        observed_x_xor_y: BlockHeader = x ^ y

        assert all(
            isinstance(i, BlockHeader) for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert all(
            i.hash_parent_hdr == expected_xord_hash_parent_hdr
            for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert all(
            i.root == expected_xord_merkle_root
            for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert all(
            i.metadata == expected_xord_metadata
            for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert expected_x_xor_y == observed_x_xor_y

        observed_x: BlockHeader = observed_x_xor_y ^ y

        assert isinstance(observed_x, BlockHeader)
        assert observed_x.hash_parent_hdr == x.hash_parent_hdr
        assert observed_x.root == x.root
        assert observed_x.metadata == x.metadata
        assert x == observed_x

        observed_y: BlockHeader = observed_x_xor_y ^ x

        assert isinstance(observed_y, BlockHeader)
        assert observed_y.hash_parent_hdr == y.hash_parent_hdr
        assert observed_y.root == y.root
        assert observed_y.metadata == y.metadata
        assert y == observed_y


def test_block_header_ran():
    """Randomly tests the block headers."""
    for _ in range(TEST_SAMPLE_SIZE):
        x_simulated_hash_parent_hdr: List[int] = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(HASH_PARENT_HDR_LEN)
        ]
        x_simulated_merkle_root: List[int] = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(ROOT_LEN)
        ]
        x_simulated_metadata: List[int] = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(METADATA_LEN)
        ]

        x: BlockHeader = BlockHeader(
            hash_parent_hdr=x_simulated_hash_parent_hdr,
            root=x_simulated_merkle_root,
            metadata=x_simulated_metadata,
        )

        y_simulated_hash_parent_hdr = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(HASH_PARENT_HDR_LEN)
        ]
        y_simulated_merkle_root = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(ROOT_LEN)
        ]
        y_simulated_metadata = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(METADATA_LEN)
        ]

        y: BlockHeader = BlockHeader(
            hash_parent_hdr=y_simulated_hash_parent_hdr,
            root=y_simulated_merkle_root,
            metadata=y_simulated_metadata,
        )

        expected_xord_hash_parent_hdr = [
            a ^ b
            for a, b in zip(x_simulated_hash_parent_hdr, y_simulated_hash_parent_hdr)
        ]
        expected_xord_merkle_root = [
            a ^ b for a, b in zip(x_simulated_merkle_root, y_simulated_merkle_root)
        ]
        expected_xord_metadata = [
            a ^ b for a, b in zip(x_simulated_metadata, y_simulated_metadata)
        ]
        expected_x_xor_y: BlockHeader = BlockHeader(
            hash_parent_hdr=expected_xord_hash_parent_hdr,
            root=expected_xord_merkle_root,
            metadata=expected_xord_metadata,
        )

        observed_x_xor_y: BlockHeader = x ^ y

        assert all(
            isinstance(i, BlockHeader) for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert all(
            i.hash_parent_hdr == expected_xord_hash_parent_hdr
            for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert all(
            i.root == expected_xord_merkle_root
            for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert all(
            i.metadata == expected_xord_metadata
            for i in [expected_x_xor_y, observed_x_xor_y]
        )
        assert expected_x_xor_y == observed_x_xor_y

        observed_x: BlockHeader = observed_x_xor_y ^ y

        assert isinstance(observed_x, BlockHeader)
        assert observed_x.hash_parent_hdr == x.hash_parent_hdr
        assert observed_x.root == x.root
        assert observed_x.metadata == x.metadata
        assert x == observed_x

        observed_y: BlockHeader = observed_x_xor_y ^ x

        assert isinstance(observed_y, BlockHeader)
        assert observed_y.hash_parent_hdr == y.hash_parent_hdr
        assert observed_y.root == y.root
        assert observed_y.metadata == y.metadata
        assert y == observed_y


SAMPLE_SIZE: int = 2 ** 3
TEST_PRIME: int = 2 ** 31 - 2 ** 17 + 1
BLOCK_HEADER_CASES: list = [
    [
        [2 ** i % TEST_PRIME],  # 0
        [3 ** i % TEST_PRIME],  # 1
        [5 ** i % TEST_PRIME],  # 2
        BlockHeader(
            hash_parent_hdr=[2 ** i % TEST_PRIME],
            root=[3 ** i % TEST_PRIME],
            metadata=[5 ** i % TEST_PRIME],
        ),  # 3
        [7 ** i % TEST_PRIME],  # 4
        [11 ** i % TEST_PRIME],  # 5
        [13 ** i % TEST_PRIME],  # 6
        BlockHeader(
            hash_parent_hdr=[7 ** i % TEST_PRIME],
            root=[11 ** i % TEST_PRIME],
            metadata=[13 ** i % TEST_PRIME],
        ),  # 7
        [(2 ** i % TEST_PRIME) ^ (7 ** i % TEST_PRIME)],  # 8
        [(3 ** i % TEST_PRIME) ^ (11 ** i % TEST_PRIME)],  # 9
        [(5 ** i % TEST_PRIME) ^ (13 ** i % TEST_PRIME)],  # 10
        BlockHeader(
            hash_parent_hdr=[(2 ** i % TEST_PRIME) ^ (7 ** i % TEST_PRIME)],
            root=[(3 ** i % TEST_PRIME) ^ (11 ** i % TEST_PRIME)],
            metadata=[(5 ** i % TEST_PRIME) ^ (13 ** i % TEST_PRIME)],
        ),
        BlockHeader(
            hash_parent_hdr=[2 ** i % TEST_PRIME],
            root=[3 ** i % TEST_PRIME],
            metadata=[5 ** i % TEST_PRIME],
        )
        ^ BlockHeader(
            hash_parent_hdr=[7 ** i % TEST_PRIME],
            root=[11 ** i % TEST_PRIME],
            metadata=[13 ** i % TEST_PRIME],
        ),
        [17 ** i % TEST_PRIME],  # 4
        [19 ** i % TEST_PRIME],  # 5
        [23 ** i % TEST_PRIME],  # 6
        BlockHeader(
            hash_parent_hdr=[17 ** i % TEST_PRIME],
            root=[19 ** i % TEST_PRIME],
            metadata=[23 ** i % TEST_PRIME],
        ),  # 7
    ]
    for i in range(SAMPLE_SIZE)
]


@pytest.mark.parametrize(
    "a_hash_parent_hdr,a_root,a_metadata,a,b_hash_parent_hdr,b_root,b_metadata,b,c_hash_parent_hdr,c_root,c_metadata,c,d,e_hash_parent_hdr,e_root,e_metadata,e",
    BLOCK_HEADER_CASES,
)
def test_block_header_par_det(
        a_hash_parent_hdr,
        a_root,
        a_metadata,
        b_hash_parent_hdr,
        b_root,
        b_metadata,
        c_hash_parent_hdr,
        c_root,
        c_metadata,
        a: BlockHeader,
        b: BlockHeader,
        c: BlockHeader,
        d: BlockHeader,
        e_hash_parent_hdr,
        e_root,
        e_metadata,
        e,
):
    """Deterministically tests block headers with parameterization"""
    assert all(isinstance(x, BlockHeader) for x in [a, b, c, d])
    assert a.hash_parent_hdr == a_hash_parent_hdr
    assert a.root == a_root
    assert a.metadata == a_metadata

    assert b.hash_parent_hdr == b_hash_parent_hdr
    assert b.root == b_root
    assert b.metadata == b_metadata

    assert c.hash_parent_hdr == c_hash_parent_hdr
    assert c.root == c_root
    assert c.metadata == c_metadata

    assert d.hash_parent_hdr == c_hash_parent_hdr
    assert d.root == c_root
    assert d.metadata == c_metadata

    # Test xor and equality
    assert d == c == a ^ b != e


BLOCK_CASES = [
    case
    + [
        [29 ** i % TEST_PRIME],
        [31 ** i % TEST_PRIME],
        [37 ** i % TEST_PRIME],
        Block(hdr=case[3], payload=[29 ** i % TEST_PRIME]),
        Block(hdr=case[7], payload=[31 ** i % TEST_PRIME]),
        Block(
            hdr=case[3] ^ case[7],
            payload=[(29 ** i % TEST_PRIME) ^ (31 ** i % TEST_PRIME)],
        ),
        Block(hdr=case[3], payload=[29 ** i % TEST_PRIME])
        ^ Block(hdr=case[7], payload=[31 ** i % TEST_PRIME]),
        Block(hdr=case[-1], payload=[37 ** i % TEST_PRIME]),
    ]
    for i, case in enumerate(BLOCK_HEADER_CASES)
]


@pytest.mark.parametrize(
    "a_hash_parent_hdr,a_root,a_metadata,a,b_hash_parent_hdr,b_root,b_metadata,b,c_hash_parent_hdr,c_root,c_metadata,c,d,e_hash_parent_hdr,e_root,e_metadata,e,a_payload,b_payload,e_payload,a_block,b_block,c_block,d_block,e_block",
    BLOCK_CASES,
)
def test_block_par_det(
        a_hash_parent_hdr,
        a_root,
        a_metadata,
        b_hash_parent_hdr,
        b_root,
        b_metadata,
        c_hash_parent_hdr,
        c_root,
        c_metadata,
        a,
        b,
        c,
        d,
        e_hash_parent_hdr,
        e_root,
        e_metadata,
        e,
        a_payload,
        b_payload,
        e_payload,
        a_block,
        b_block,
        c_block,
        d_block,
        e_block,
):
    """Deterministically test blocks with parameterization."""
    assert all(
        isinstance(x, Block) for x in [a_block, b_block, c_block, d_block, e_block]
    )
    assert a_block ^ b_block == c_block == d_block != e_block


PAYLOAD_LEN: int = 10


def test_block_det():
    """Deterministically test blocks without parameterization."""
    for _ in range(TEST_SAMPLE_SIZE):
        x_simulated_hash_parent_hdr: List[int] = [
            2 ** i % MAX_ENTRY_LEN for i in range(HASH_PARENT_HDR_LEN)
        ]
        x_simulated_merkle_root: List[int] = [
            3 ** i % MAX_ENTRY_LEN for i in range(ROOT_LEN)
        ]
        x_simulated_metadata: List[int] = [
            5 ** i % MAX_ENTRY_LEN for i in range(METADATA_LEN)
        ]
        x_simulated_block_header: BlockHeader = BlockHeader(
            hash_parent_hdr=x_simulated_hash_parent_hdr,
            root=x_simulated_merkle_root,
            metadata=x_simulated_metadata,
        )
        x_simulated_payload = [7 ** i % MAX_ENTRY_LEN for i in range(PAYLOAD_LEN)]

        x: Block = Block(hdr=x_simulated_block_header, payload=x_simulated_payload)

        y_simulated_hash_parent_hdr = [
            11 ** i % MAX_ENTRY_LEN for i in range(HASH_PARENT_HDR_LEN)
        ]
        y_simulated_merkle_root = [13 ** i % MAX_ENTRY_LEN for i in range(ROOT_LEN)]
        y_simulated_metadata = [17 ** i % MAX_ENTRY_LEN for i in range(METADATA_LEN)]
        y_simulated_block_header: BlockHeader = BlockHeader(
            hash_parent_hdr=y_simulated_hash_parent_hdr,
            root=y_simulated_merkle_root,
            metadata=y_simulated_metadata,
        )
        y_simulated_payload = [19 ** i % MAX_ENTRY_LEN for i in range(PAYLOAD_LEN)]

        y: Block = Block(hdr=y_simulated_block_header, payload=y_simulated_payload)

        expected_xord_block_header: BlockHeader = (
                x_simulated_block_header ^ y_simulated_block_header
        )
        expected_xord_payload: List[int] = [
            a ^ b for a, b in zip(x_simulated_payload, y_simulated_payload)
        ]

        expected_xord_block: Block = Block(
            hdr=expected_xord_block_header, payload=expected_xord_payload
        )
        observed_xord_block: Block = x ^ y

        assert all(
            isinstance(i, Block) for i in [expected_xord_block, observed_xord_block]
        )
        assert all(
            i.hdr == expected_xord_block_header
            for i in [expected_xord_block, observed_xord_block]
        )
        assert all(
            i.payload == expected_xord_payload
            for i in [expected_xord_block, observed_xord_block]
        )
        assert expected_xord_block == observed_xord_block

        observed_x: Block = observed_xord_block ^ y

        assert isinstance(observed_x, Block)
        assert observed_x.hdr == x.hdr
        assert observed_x.payload == x.payload
        assert x == observed_x

        observed_y: Block = observed_xord_block ^ x

        assert isinstance(observed_y, Block)
        assert observed_y.hdr == y.hdr
        assert observed_y.payload == y.payload
        assert y == observed_y

        z_simulated_hash_parent_hdr: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.hdr.hash_parent_hdr
        ]
        z_simulated_merkle_root: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.hdr.root
        ]
        z_simulated_metadata: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.hdr.metadata
        ]
        z_simulated_hdr: BlockHeader = BlockHeader(
            hash_parent_hdr=z_simulated_hash_parent_hdr,
            root=z_simulated_merkle_root,
            metadata=z_simulated_metadata,
        )
        z_simulated_payload: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.payload
        ]
        z: Block = Block(hdr=z_simulated_hdr, payload=z_simulated_payload)

        assert z != x and z != y


def test_block_ran():
    """Randomly test blocks."""
    for _ in range(TEST_SAMPLE_SIZE):
        x_simulated_hash_parent_hdr: List[int] = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(HASH_PARENT_HDR_LEN)
        ]
        x_simulated_merkle_root: List[int] = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(ROOT_LEN)
        ]
        x_simulated_metadata: List[int] = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(METADATA_LEN)
        ]
        x_simulated_block_header: BlockHeader = BlockHeader(
            hash_parent_hdr=x_simulated_hash_parent_hdr,
            root=x_simulated_merkle_root,
            metadata=x_simulated_metadata,
        )
        x_simulated_payload = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(PAYLOAD_LEN)
        ]

        x: Block = Block(hdr=x_simulated_block_header, payload=x_simulated_payload)

        y_simulated_hash_parent_hdr = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(HASH_PARENT_HDR_LEN)
        ]
        y_simulated_merkle_root = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(ROOT_LEN)
        ]
        y_simulated_metadata = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(METADATA_LEN)
        ]
        y_simulated_block_header: BlockHeader = BlockHeader(
            hash_parent_hdr=y_simulated_hash_parent_hdr,
            root=y_simulated_merkle_root,
            metadata=y_simulated_metadata,
        )
        y_simulated_payload = [
            randint(0, MAX_ENTRY_LEN - 1) for i in range(PAYLOAD_LEN)
        ]

        y: Block = Block(hdr=y_simulated_block_header, payload=y_simulated_payload)

        expected_xord_block_header: BlockHeader = (
                x_simulated_block_header ^ y_simulated_block_header
        )
        expected_xord_payload: List[int] = [
            a ^ b for a, b in zip(x_simulated_payload, y_simulated_payload)
        ]
        expected_xord_block: Block = Block(
            hdr=expected_xord_block_header, payload=expected_xord_payload
        )
        observed_xord_block: Block = x ^ y

        assert all(
            isinstance(i, Block) for i in [expected_xord_block, observed_xord_block]
        )
        assert all(
            i.hdr == expected_xord_block_header
            for i in [expected_xord_block, observed_xord_block]
        )
        assert all(
            i.payload == expected_xord_payload
            for i in [expected_xord_block, observed_xord_block]
        )
        assert expected_xord_block == observed_xord_block

        observed_x: Block = observed_xord_block ^ y

        assert isinstance(observed_x, Block)
        assert observed_x.hdr == x.hdr
        assert observed_x.payload == x.payload
        assert x == observed_x

        observed_y: Block = observed_xord_block ^ x

        assert isinstance(observed_y, Block)
        assert observed_y.hdr == y.hdr
        assert observed_y.payload == y.payload
        assert y == observed_y

        z_simulated_hash_parent_hdr: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.hdr.hash_parent_hdr
        ]
        z_simulated_merkle_root: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.hdr.root
        ]
        z_simulated_metadata: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.hdr.metadata
        ]
        z_simulated_hdr: BlockHeader = BlockHeader(
            hash_parent_hdr=z_simulated_hash_parent_hdr,
            root=z_simulated_merkle_root,
            metadata=z_simulated_metadata,
        )
        z_simulated_payload: List[int] = [
            i - 1 % MAX_ENTRY_LEN for i in observed_xord_block.payload
        ]
        z: Block = Block(hdr=z_simulated_hdr, payload=z_simulated_payload)

        assert z != x and z != y


@pytest.mark.parametrize(
    "a_hash_parent_hdr,a_root,a_metadata,a,b_hash_parent_hdr,b_root,b_metadata,b,c_hash_parent_hdr,c_root,c_metadata,c,d,e_hash_parent_hdr,e_root,e_metadata,e,a_payload,b_payload,e_payload,a_block,b_block,c_block,d_block,e_block",
    BLOCK_CASES,
)
def test_block_verify(
        mocker,
        a_hash_parent_hdr,
        a_root,
        a_metadata,
        b_hash_parent_hdr,
        b_root,
        b_metadata,
        c_hash_parent_hdr,
        c_root,
        c_metadata,
        a,
        b,
        c,
        d,
        e_hash_parent_hdr,
        e_root,
        e_metadata,
        e,
        a_payload,
        b_payload,
        e_payload,
        a_block,
        b_block,
        c_block,
        d_block,
        e_block,
):
    """Deterministically tests block_verify with parameterization."""
    mocker.patch(
        "secure_fountain.main.payload_to_merkle_root",
        side_effect=[
            a_root,
            b_root,
            c_root,
            c_root,
            e_root,
            c_root,
            c_root,
            c_root,
            c_root,
            c_root,
        ],
    )
    assert block_verify(expected_hdr=a, block=a_block)
    assert block_verify(expected_hdr=b, block=b_block)
    assert block_verify(expected_hdr=c, block=c_block)
    assert block_verify(expected_hdr=d, block=d_block)
    assert block_verify(expected_hdr=e, block=e_block)
    assert block_verify(expected_hdr=d, block=c_block)
    assert block_verify(expected_hdr=c, block=d_block)
    assert block_verify(expected_hdr=a ^ b, block=c_block)
    assert block_verify(expected_hdr=a ^ b, block=d_block)
    assert not block_verify(expected_hdr=e, block=c_block)
    assert c == d == a ^ b != e
    assert c_block == d_block == a_block ^ b_block != e_block


EPOCH_CASES_ROOT_CHAIN = [
    [[2 ** i * 2 ** j % TEST_PRIME] for i in range(SAMPLE_SIZE)]
    for j in range(SAMPLE_SIZE)
]
EPOCH_CASES_PAYLOAD_CHAIN = [
    [[2 ** i * 3 ** j % TEST_PRIME] for i in range(SAMPLE_SIZE)]
    for j in range(SAMPLE_SIZE)
]
EPOCH_CASES_HEADER_CHAIN = [
    [
        BlockHeader(
            hash_parent_hdr=[(2 ** i * 5 ** j) % TEST_PRIME],
            root=EPOCH_CASES_ROOT_CHAIN[i][j],
            metadata=[(2 ** i * 5 ** j) % TEST_PRIME],
        )
        for i in range(SAMPLE_SIZE)
    ]
    for j in range(SAMPLE_SIZE)
]
EPOCH_CASES_BLOCK_CHAIN = [
    [Block(hdr=h, payload=p) for h, p in zip(hdrs, payloads)]
    for hdrs, payloads in zip(EPOCH_CASES_HEADER_CHAIN, EPOCH_CASES_PAYLOAD_CHAIN)
]
EPOCH_CASES = [
    [root_chain, payload_chain, hdr_chain, block_chain]
    for root_chain, payload_chain, hdr_chain, block_chain in zip(
        EPOCH_CASES_ROOT_CHAIN,
        EPOCH_CASES_PAYLOAD_CHAIN,
        EPOCH_CASES_HEADER_CHAIN,
        EPOCH_CASES_BLOCK_CHAIN,
    )
]


@pytest.mark.parametrize("root_chain,payload_chain,hdr_chain,block_chain", EPOCH_CASES)
def test_epoch_verify_by_patching_merkle_root_computation(
        mocker, root_chain, payload_chain, hdr_chain, block_chain
):
    """Weak deterministic test for epoch_verify with parameterization by patching merkle root computation."""
    mocker.patch("secure_fountain.main.payload_to_merkle_root", side_effect=root_chain)
    assert epoch_verify(expected_hdr_chain=hdr_chain, epoch=block_chain)


@pytest.mark.parametrize("root_chain,payload_chain,hdr_chain,block_chain", EPOCH_CASES)
def test_epoch_verify_by_patching_block_verify(
        mocker, root_chain, payload_chain, hdr_chain, block_chain
):
    """Weak deterministic test for epoch_verify with parameterization by patching block_verify."""
    mocker.patch("secure_fountain.main.block_verify", return_value=True)
    assert epoch_verify(expected_hdr_chain=hdr_chain, epoch=block_chain)

    mocker.patch("secure_fountain.main.block_verify", return_value=False)
    assert not epoch_verify(expected_hdr_chain=hdr_chain, epoch=block_chain)


DROPLETS_CASES = [
    i
    + [  # noqa: correct format
        Droplet(indices=[0], val=i[-1][0]),
        Droplet(indices=[0, 1], val=i[-1][0] ^ i[-1][1]),  # noqa: is not list
        Droplet(indices=[0], val=i[-1][0]),
        Droplet(indices=[1], val=i[-1][1]),
    ]
    for i in EPOCH_CASES
]


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,droplet_one,droplet_two,droplet_three,droplet_four",
    DROPLETS_CASES,
)
def test_droplet_par(
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        droplet_one,
        droplet_two,
        droplet_three,
        droplet_four,
):
    """Deterministic test for droplets using parameterization."""
    assert droplet_one.indices == [0]
    assert droplet_one.val == block_chain[0]
    assert len(droplet_one) == 1
    assert droplet_two.indices == [0, 1]
    assert droplet_two.val == block_chain[0] ^ block_chain[1]
    assert len(droplet_two) == 2
    assert droplet_one != droplet_two
    assert droplet_one == droplet_three
    assert droplet_two ^ droplet_one == droplet_four
    assert droplet_two ^ droplet_four == droplet_one == droplet_three


SAMPLE_DEGREE_W_REPLACEMENT_CASE: list = [
    [
        [
            0.01,
            0.175,
            0.185,
            0.505,
            0.515,
            0.765,
            0.775,
            0.905,
            0.915,
            0.955,
            0.965,
            0.969,
            0.971,
            0.99999,
        ],
        [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64],
    ]
]


@pytest.mark.parametrize("side_effect,exp_out", SAMPLE_DEGREE_W_REPLACEMENT_CASE)
def test_sample_degree_w_replacement(mocker, side_effect, exp_out):
    """Deterministic test of sample_degree_w_replacement with parameterization."""
    sample_size = len(side_effect)
    mocker.patch("secure_fountain.main.random", side_effect=side_effect)
    assert sample_degree_w_replacement(sample_size=sample_size) == exp_out


DEGREE_TO_INDICES_AND_BLOCKS_CASES_SINGLETONS = [
    i + [1, [0], [i[3][0]]] for i in EPOCH_CASES
]
DEGREE_TO_INDICES_AND_BLOCKS_CASES_DOUBLETS = [
    i + [2, [0, 1], [i[3][0], i[3][1]]] for i in EPOCH_CASES
]
DEGREE_TO_INDICES_AND_BLOCKS_CASES = (
        DEGREE_TO_INDICES_AND_BLOCKS_CASES_SINGLETONS
        + DEGREE_TO_INDICES_AND_BLOCKS_CASES_DOUBLETS
)


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks",
    DEGREE_TO_INDICES_AND_BLOCKS_CASES,
)
def test_degree_to_indices_and_blocks_by_patching_sample(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
):
    """Deterministically test degree_to_indices_and_blocks by patching sample."""
    mocker.patch("secure_fountain.main.sample", return_value=exp_indices)
    assert exp_indices, exp_blocks == degree_to_indices_and_blocks(
        degree=degree, epoch=block_chain
    )


INDICES_AND_BLOCKS_TO_DROPLET_VAL = [
    i + [i[-1][0] if len(i[-1]) == 1 else i[-1][0] ^ i[-1][1]]  # noqa: is not list
    for i in DEGREE_TO_INDICES_AND_BLOCKS_CASES
]


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val",
    INDICES_AND_BLOCKS_TO_DROPLET_VAL,
)
def test_indices_and_blocks_to_droplet_val(
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
):
    """Deterministically test indices_and_blocks_to_droplet_val with parameterization."""
    assert (
            indices_and_blocks_to_droplet_val(sampled_blocks=(exp_indices, exp_blocks))
            == exp_droplet_val
    )


INDICES_AND_BLOCKS_TO_DROPLET_CASES = [
    i + [Droplet(indices=i[-3], val=i[-1])]
    for i in INDICES_AND_BLOCKS_TO_DROPLET_VAL  # noqa: is proper format
]


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet",
    INDICES_AND_BLOCKS_TO_DROPLET_CASES,
)
def test_indices_and_blocks_to_droplet(
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
):
    """Deterministically test indices_and_blocks_to_droplet with parameterization."""
    assert (
            indices_and_blocks_to_droplet(sampled_blocks=(exp_indices, exp_blocks))
            == exp_droplet
    )


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet",
    INDICES_AND_BLOCKS_TO_DROPLET_CASES[:8],
)
def test_indices_and_blocks_to_droplet_by_patching_indices_and_blocks_to_droplet_val(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
):
    """Weak deterministic test for indices_and_blocks_to_droplet."""
    mocker.patch(
        "secure_fountain.main.indices_and_blocks_to_droplet_val",
        return_value=exp_blocks[0],
    )
    obs_droplet: Droplet = indices_and_blocks_to_droplet(
        sampled_blocks=(exp_indices, exp_blocks)
    )
    assert isinstance(obs_droplet, Droplet)
    assert isinstance(exp_droplet, Droplet)
    assert obs_droplet.indices == exp_droplet.indices
    assert obs_droplet.val.hdr.hash_parent_hdr == exp_droplet.val.hdr.hash_parent_hdr
    assert obs_droplet.val.hdr.root == exp_droplet.val.hdr.root
    assert obs_droplet.val.hdr.metadata == exp_droplet.val.hdr.metadata
    assert obs_droplet.val.hdr == exp_droplet.val.hdr
    assert obs_droplet.val.payload == exp_droplet.val.payload
    assert obs_droplet.val == exp_droplet.val
    assert obs_droplet == exp_droplet


EPOCH_TO_BUCKET_CASES = [i + [[i[-1]]] for i in INDICES_AND_BLOCKS_TO_DROPLET_CASES]


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet,exp_bucket",
    EPOCH_TO_BUCKET_CASES,
)
def test_epoch_to_bucket_by_patching_sample_degree_w_replacement_and_degree_to_indices_and_blocks(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
        exp_bucket,
):
    """Weak deterministic test of epoch_to_bucket by patching sample_degree_w_replacement and degree_to_indices_and_blocks."""
    mocker.patch(
        "secure_fountain.main.sample_degree_w_replacement",
        return_value=[len(exp_indices)],
    )
    mocker.patch(
        "secure_fountain.main.degree_to_indices_and_blocks",
        return_value=(exp_indices, exp_blocks),
    )
    obs_bucket: List[Droplet] = encode_epoch(sample_size=1, epoch=block_chain)
    assert isinstance(obs_bucket, list)
    assert all(isinstance(i, Droplet) for i in obs_bucket)
    assert all(isinstance(i, Droplet) for i in exp_bucket)
    assert len(obs_bucket) == len(exp_bucket)
    for obs_droplet, exp_droplet in zip(obs_bucket, exp_bucket):
        assert obs_droplet.indices == exp_droplet.indices
        assert obs_droplet.val == exp_droplet.val
        assert obs_droplet == exp_droplet


EPOCH_TO_BUCKET_HOLLOW_CASES = [[1, None, [1, 2, 32]] for i in range(SAMPLE_SIZE)]


@pytest.mark.parametrize(
    "sample_size,epoch,sampled_degrees", EPOCH_TO_BUCKET_HOLLOW_CASES
)
def test_epoch_to_bucket_by_patching_all(mocker, sample_size, epoch, sampled_degrees):
    """Very weak deterministic test of epoch_to_bucket by patching everything."""
    mocker.patch(
        "secure_fountain.main.sample_degree_w_replacement", return_value=sampled_degrees
    )
    mocker.patch("secure_fountain.main.degree_to_indices_and_blocks", return_value=None)
    mocker.patch(
        "secure_fountain.main.indices_and_blocks_to_droplet", return_value=None
    )
    x = encode_epoch(sample_size=sample_size, epoch=epoch)
    assert isinstance(x, list)
    assert len(x) == 3
    assert x == [None, None, None]


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet,exp_bucket",
    EPOCH_TO_BUCKET_CASES[:8],
)
def test_find_singleton_successes(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
        exp_bucket,
):
    """Deterministically test find_singleton_successes with parameterization."""
    x = find_singleton(bucket=exp_bucket)
    assert not isinstance(x, bool)
    assert isinstance(x, Droplet)
    assert len(x) == 1
    assert all(i in x.indices for i in exp_indices)


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet,exp_bucket",
    EPOCH_TO_BUCKET_CASES[8:],
)
def test_find_singleton_in_bucket_failures(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
        exp_bucket,
):
    """Deterministically test find_singleton_in_bucket failures with parameterization."""
    x = find_singleton(bucket=exp_bucket)
    assert not x


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet,exp_bucket",
    EPOCH_TO_BUCKET_CASES[:8],
)
def test_peel_singleton(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
        exp_bucket,
):
    """Deterministically test peel_singleton with parameterization."""
    mocker.patch(
        "secure_fountain.main.payload_to_merkle_root",
        return_value=exp_droplet.val.hdr.root,
    )
    x = peel_singleton(
        expected_hdr=hdr_chain[exp_indices[0]], singleton=exp_droplet, bucket=exp_bucket
    )
    assert not isinstance(x, bool)
    assert isinstance(x, tuple)
    assert len(x) == 3
    assert x[0] is not None
    assert isinstance(x[0], Block)
    assert x[1] is not None
    assert isinstance(x[1], int)
    assert 0 <= x[1]
    assert isinstance(x[2], list)
    assert len(x[2]) == 0  # in these special cases.


PEEL_SINGLETON_BAD_BLOCKS = [
    (
        BlockHeader(hash_parent_hdr=[0], root=[0], metadata=[0]),
        Droplet(indices=[1], val=None),
        [],
    )
    for i in range(SAMPLE_SIZE)
]


@pytest.mark.parametrize("expected_hdr,singleton,bucket", PEEL_SINGLETON_BAD_BLOCKS)
def test_peel_singleton_bad_blocks(mocker, expected_hdr, singleton, bucket):
    """Deterministically test peel_singleton with bad blocks using parameterization."""
    mocker.patch("secure_fountain.main.block_verify", return_value=False)
    assert peel_singleton(
        expected_hdr=expected_hdr, singleton=singleton, bucket=bucket
    ) == (None, None, [])


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet,exp_bucket",
    EPOCH_TO_BUCKET_CASES[:8],
)
def test_find_and_peel_singleton_by_patching_merkle(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
        exp_bucket,
):
    """Deterministically test find_and_peel_singleton with parameterization by patching merkle root."""
    mocker.patch(
        "secure_fountain.main.payload_to_merkle_root",
        return_value=exp_droplet.val.hdr.root,
    )
    x = find_and_peel_singleton(exp_hdr_chain=hdr_chain, bucket=exp_bucket)
    assert not isinstance(x, bool)
    assert isinstance(x, tuple)
    assert x[0] is not None
    assert isinstance(x[0], Block)
    assert x[1] is not None
    assert isinstance(x[1], int)
    assert 0 <= x[1]
    assert isinstance(x[2], list)
    assert len(x[2]) == 0  # in these special cases.


@pytest.mark.parametrize(
    "root_chain,payload_chain,hdr_chain,block_chain,degree,exp_indices,exp_blocks,exp_droplet_val,exp_droplet,exp_bucket",
    EPOCH_TO_BUCKET_CASES[:8],
)
def test_find_and_peel_singleton_by_patching_all(
        mocker,
        root_chain,
        payload_chain,
        hdr_chain,
        block_chain,
        degree,
        exp_indices,
        exp_blocks,
        exp_droplet_val,
        exp_droplet,
        exp_bucket,
):
    """Weak deterministic test of find_and_peel_singleton with parameterization and by patching find_singleton, block_verify, and peel_singleton."""
    mocker.patch("secure_fountain.main.find_singleton", return_value=exp_droplet)
    mocker.patch("secure_fountain.main.block_verify", return_value=True)
    mocker.patch(
        "secure_fountain.main.peel_singleton",
        return_value=[exp_blocks[0], exp_indices[0], []],
    )
    x = find_and_peel_singleton(exp_hdr_chain=hdr_chain, bucket=exp_bucket)
    assert not isinstance(x, bool)
    assert isinstance(x, list)
    assert x[0] is not None
    assert isinstance(x[0], Block)
    assert x[1] is not None
    assert isinstance(x[1], int)
    assert 0 <= x[1]
    assert isinstance(x[2], list)
    assert len(x[2]) == 0  # in these special cases.


NUM_EPOCHS: int = 3
NUM_NEIGHBORS: int = 4


def test_encoded_block_chain():
    """Deterministically test encoded_block_chain"""
    for _ in range(TEST_SAMPLE_SIZE):
        block_chain = [
            Block(
                hdr=BlockHeader(
                    hash_parent_hdr=[2 ** i % MAX_ENTRY_LEN],
                    root=[3 ** i % MAX_ENTRY_LEN],
                    metadata=[5 ** i % MAX_ENTRY_LEN],
                ),
                payload=[7 ** i % MAX_ENTRY_LEN],
            )
            for i in range(NUM_EPOCHS * BLOCKS_PER_EPOCH)
        ]
        num_epochs = len(block_chain) // BLOCKS_PER_EPOCH
        partitioned_block_chain = [
            block_chain[i * BLOCKS_PER_EPOCH: (i + 1) * BLOCKS_PER_EPOCH]
            for i in range(num_epochs)
        ]
        hdr_chain = [b.hdr for b in block_chain]
        buckets = []
        for i, next_epoch in enumerate(partitioned_block_chain):
            next_bucket = [
                Droplet(
                    indices=[
                        (11 ** i * (13 ** j % BLOCKS_PER_EPOCH)) % BLOCKS_PER_EPOCH
                    ],
                    val=next_epoch[
                        (11 ** i * (13 ** j % BLOCKS_PER_EPOCH)) % BLOCKS_PER_EPOCH
                        ],
                )
                for j in range(DROPLETS_TO_SAVE_PER_EPOCH)
            ]
            buckets += [next_bucket]
        x = EncodedBlockChain(hdr_chain=hdr_chain, buckets=buckets)
        assert isinstance(x, EncodedBlockChain)
        assert x.hdr_chain == hdr_chain
        assert x.buckets == buckets


def test_encode():
    """Deterministically test encode"""
    for _ in range(TEST_SAMPLE_SIZE):
        block_chain = [
            Block(
                hdr=BlockHeader(
                    hash_parent_hdr=[2 ** i % MAX_ENTRY_LEN],
                    root=[3 ** i % MAX_ENTRY_LEN],
                    metadata=[5 ** i % MAX_ENTRY_LEN],
                ),
                payload=[7 ** i % MAX_ENTRY_LEN],
            )
            for i in range(NUM_EPOCHS * BLOCKS_PER_EPOCH)
        ]
        hdr_chain = [b.hdr for b in block_chain]
        encoded_block_chain: EncodedBlockChain
        remaining_blocks: List[Block]
        encoded_block_chain, remaining_blocks = encode(block_chain=block_chain)
        assert not isinstance(encoded_block_chain, bool)
        assert isinstance(encoded_block_chain, EncodedBlockChain)
        assert encoded_block_chain.hdr_chain == hdr_chain
        assert isinstance(encoded_block_chain.buckets, list)
        assert all(
            isinstance(i, Droplet)
            for bucket in encoded_block_chain.buckets
            for i in bucket
        )
        assert all(
            len(bucket) == DROPLETS_TO_SAVE_PER_EPOCH
            for bucket in encoded_block_chain.buckets
        )
        assert isinstance(remaining_blocks, list)
        assert len(remaining_blocks) == 0


def test_bucket_to_epoch(mocker):
    """Deterministically test bucket_to_epoch."""
    for _ in range(TEST_SAMPLE_SIZE):
        block_chain = [
            Block(
                hdr=BlockHeader(
                    hash_parent_hdr=[2 ** i % MAX_ENTRY_LEN],
                    root=[3 ** i % MAX_ENTRY_LEN],
                    metadata=[5 ** i % MAX_ENTRY_LEN],
                ),
                payload=[7 ** i % MAX_ENTRY_LEN],
            )
            for i in range(NUM_EPOCHS * BLOCKS_PER_EPOCH)
        ]
        hdr_chain = [b.hdr for b in block_chain]
        encoded_block_chain: EncodedBlockChain
        remaining_blocks: List[Block]
        try:
            encoded_block_chain, _ = encode(block_chain=block_chain)
        except ValueError as ve:
            assert False

        ct = 0
        while ct < NUM_NEIGHBORS:
            try:
                next_encoded_block_chain, _ = encode(block_chain=block_chain)
                try:
                    encoded_block_chain += next_encoded_block_chain
                except ValueError as ve:
                    assert False
            except ValueError as ve:
                assert False
            ct += 1

        observed_hdr_chain: List[BlockHeader] = encoded_block_chain.hdr_chain
        partitioned_hdr_chain: List[List[BlockHeader]] = [
            observed_hdr_chain[i * BLOCKS_PER_EPOCH: (i + 1) * BLOCKS_PER_EPOCH]
            for i in range(NUM_EPOCHS)
        ]
        buckets: List[List[Droplet]] = encoded_block_chain.buckets
        for i, (obs_hdr_chain, bucket) in enumerate(
                zip(partitioned_hdr_chain, buckets)
        ):
            mocker.patch("secure_fountain.main.block_verify", return_value=False)
            decoded_epoch: Union[bool, List[Block]] = decode_epoch(
                exp_hdr_chain=obs_hdr_chain, bucket=bucket
            )
            assert isinstance(decoded_epoch, bool)
            assert not decoded_epoch

            mocker.patch("secure_fountain.main.block_verify", return_value=True)
            decoded_epoch: Union[bool, List[Block]] = decode_epoch(
                exp_hdr_chain=obs_hdr_chain, bucket=bucket
            )
            if isinstance(decoded_epoch, bool):
                assert decoded_epoch is False
            else:
                assert isinstance(decoded_epoch, list)
                assert all(isinstance(i, Block) for i in decoded_epoch)
                assert len(decoded_epoch) == BLOCKS_PER_EPOCH


def test_decode(mocker):
    """Deterministically test decode."""
    expected_not_decodable: List[int] = [4, 12]  # out of the first 15
    observed_not_decodable: List[int] = []
    for x in range(15):
        seed(x)
        print(x)
        block_chain = [
            Block(
                hdr=BlockHeader(
                    hash_parent_hdr=[2 ** i % MAX_ENTRY_LEN],
                    root=[3 ** i % MAX_ENTRY_LEN],
                    metadata=[5 ** i % MAX_ENTRY_LEN],
                ),
                payload=[7 ** i % MAX_ENTRY_LEN],
            )
            for i in range(NUM_EPOCHS * BLOCKS_PER_EPOCH)
        ]
        hdr_chain = [b.hdr for b in block_chain]
        encoded_block_chain: EncodedBlockChain
        remaining_blocks: List[Block]
        try:
            encoded_block_chain, _ = encode(block_chain=block_chain)
        except ValueError as ve:
            assert False

        ct = 0
        while ct < NUM_NEIGHBORS:
            try:
                next_encoded_block_chain, _ = encode(block_chain=block_chain)
                try:
                    encoded_block_chain += next_encoded_block_chain
                except ValueError as ve:
                    assert False
            except ValueError as ve:
                assert False
            ct += 1
        mocker.patch("secure_fountain.main.block_verify", return_value=True)
        decoded_block_chain: Union[bool, List[Block]] = decode(
            encoded_block_chain=encoded_block_chain
        )
        if isinstance(decoded_block_chain, bool):
            observed_not_decodable.append(x)
        else:
            assert not isinstance(decoded_block_chain, bool)
            assert isinstance(decoded_block_chain, list)
            assert all(isinstance(i, Block) for i in decoded_block_chain)
            assert decoded_block_chain == block_chain
    assert observed_not_decodable == expected_not_decodable


def test_bucket_to_epoch_verify_non_decodable(mocker):
    """Deterministically test bucket_to_epoch with several buckets."""
    expected_not_decodable: List[int] = [4, 12]  # out of the first 15
    observed_not_decodable: List[int] = []
    for randomness_seed in range(15):
        seed(randomness_seed)
        block_chain = [
            Block(
                hdr=BlockHeader(
                    hash_parent_hdr=[2 ** i % MAX_ENTRY_LEN],
                    root=[3 ** i % MAX_ENTRY_LEN],
                    metadata=[5 ** i % MAX_ENTRY_LEN],
                ),
                payload=[7 ** i % MAX_ENTRY_LEN],
            )
            for i in range(NUM_EPOCHS * BLOCKS_PER_EPOCH)
        ]
        hdr_chain = [b.hdr for b in block_chain]
        encoded_block_chain: EncodedBlockChain
        remaining_blocks: List[Block]
        try:
            encoded_block_chain, _ = encode(block_chain=block_chain)
        except ValueError as ve:
            assert False

        ct = 0
        while ct < NUM_NEIGHBORS:
            try:
                next_encoded_block_chain, _ = encode(block_chain=block_chain)
                try:
                    encoded_block_chain += next_encoded_block_chain
                except ValueError as ve:
                    assert False
            except ValueError as ve:
                assert False
            ct += 1

        observed_hdr_chain: List[BlockHeader] = encoded_block_chain.hdr_chain
        partitioned_hdr_chain: List[List[BlockHeader]] = [
            observed_hdr_chain[i * BLOCKS_PER_EPOCH: (i + 1) * BLOCKS_PER_EPOCH]
            for i in range(NUM_EPOCHS)
        ]
        buckets: List[List[Droplet]] = encoded_block_chain.buckets
        for i, (obs_hdr_chain, bucket) in enumerate(
                zip(partitioned_hdr_chain, buckets)
        ):
            mocker.patch("secure_fountain.main.block_verify", return_value=False)
            decoded_epoch: Union[bool, List[Block]] = decode_epoch(
                exp_hdr_chain=obs_hdr_chain, bucket=bucket
            )
            assert isinstance(decoded_epoch, bool)
            assert not decoded_epoch

            mocker.patch("secure_fountain.main.block_verify", return_value=True)
            decoded_epoch: Union[bool, List[Block]] = decode_epoch(
                exp_hdr_chain=obs_hdr_chain, bucket=bucket
            )
            if isinstance(decoded_epoch, bool):
                assert decoded_epoch is False
                observed_not_decodable.append(randomness_seed)
            else:
                assert isinstance(decoded_epoch, list)
                assert all(isinstance(i, Block) for i in decoded_epoch)
                assert len(decoded_epoch) == BLOCKS_PER_EPOCH

    assert observed_not_decodable == expected_not_decodable
