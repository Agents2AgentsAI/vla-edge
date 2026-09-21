# Third-party notices

## GELLO

Portions of `examples/bimanual-yam/gello_min/` are adapted from GELLO as
distributed in the [YAM repository](https://github.com/williamtsai726/YAM/tree/main/gello_software).

MIT License

Copyright (c) 2023 Philipp Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ABC-VLA, Gemma, and OpenPI

The native ABC-VLA sampler follows [amazon-far/abc](https://github.com/amazon-far/abc)
(Apache-2.0). The serving bundle retains the selected upstream `abc_minimal`
source and its notices outside this repository. Gemma model/tokenizer assets
retain their upstream terms. Model weights and compiled engines are not
included in this source distribution.

`src/vla_edge/protocol/abc_msgpack.py` adapts the numeric NumPy wire format
used by ABC and [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)
(Apache-2.0), with explicit rejection of unsupported dtypes. That codec derives
from [msgpack-numpy](https://github.com/lebedov/msgpack-numpy), whose BSD notice
follows:

<!---
-*- mode:markdown -*-
vi:ft=markdown
-->
### msgpack-numpy license

Copyright (c) 2013-2022, Lev E. Givon.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.
* Neither the name of Lev E. Givon nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Pi0.5 and OpenPI

The Pi0.5 host preprocessing follows [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi),
licensed under Apache-2.0. The runtime implements the public Pi0.5 observation,
tokenization and normalization contracts without bundling the training stack.
The BimanualYAM model originates from [robocurve/pi0.5-yam](https://huggingface.co/robocurve/pi0.5-yam).
Model weights, tokenizer and derived engine plans retain their upstream terms,
including the Gemma terms. They are not included in this source repository.
