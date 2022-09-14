# Secue fountain architecture

Prototype implementation of the encoding and decoding framework for the secure fountain architecture (for python 3.8+)


## Introduction
The secure fountain architecture offers a quantum-resistant off-chain approach for reducing node storage requirements (with a tradeoff of increased bandwidth), without compromising blockchain verifiability.

This prototype library is blockchain-agnostic. Since the secure fountain architecture is simply an opt-in decentralized peer-to-peer storage layer,  it can be run alongside the existing message distribution and storage system (with no changes to the underlying consensus protocol).

(Note that quantum resistance relies using a quantum-secure hash function, of which there are many)

## References
Our description of the secure fountain architecture can be found here:
+ [Techniques for efficient post-quantum finance (Part 4: reducing storage requirements)](https://www.theqrl.org/blog/techniques-for-efficient-post-quantum-finance-part-4-reducing-storage-requirements/)

This implementation is inspired by:
+ Kadhe, Swanand, Jichan Chung, and Kannan Ramchandran. "SeF: A secure fountain architecture for slashing storage costs in blockchains." arXiv preprint arXiv:1906.12140 (2019).
+ Luby, Michael. "LT codes." The 43rd Annual IEEE Symposium on Foundations of Computer Science, 2002. Proceedings. IEEE Computer Society, 2002.

## Contributors

Brandon Goodell (lead author), Mitchell "Isthmus" Krawiec-Thayer

Built by [Geometry Labs](https://www.geometrylabs.io) with funding from [The QRL Foundation](https://qrl.foundation/).

## Tests

To install requirements for tests:

```pip install -r test_requirements.txt```

Then run:

```tests/test_main.py```

## License

This library is released as free and open-source software under the MIT License, see LICENSE file for details.
