nvidia_copyright = """
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

convolution_template = nvidia_copyright + """
problem:
  shape:
    name: "CNN-Layer"
    dimensions: [ C, M, R, S, N, P, Q ]
    coefficients:
      - name: Wstride
        default: 1
      - name: Hstride
        default: 1
      - name: Wdilation
        default: 1
      - name: Hdilation
        default: 1
    data-spaces:
      - name: Weights
        projection:
          - [ [C] ]
          - [ [M] ]
          - [ [R] ]
          - [ [S] ]
      - name: Inputs
        projection:
          - [ [N] ]
          - [ [C] ]
          - [ [R, Wdilation], [P, Wstride] ] # SOP form: R*Wdilation + P*Wstride
          - [ [S, Hdilation], [Q, Hstride] ] # SOP form: S*Hdilation + Q*Hstride
      - name: Outputs
        projection:
          - [ [N] ]
          - [ [M] ]
          - [ [Q] ]
          - [ [P] ]
        read-write: True
  instance:
    C: {c}
    M: {m}
    N: {n}
    P: {p}
    Q: {q}
    R: {r}
    S: {s}
    Wdilation: {wd}
    Wstride: {ws}
    Hdilation: {hd}
    Hstride: {hs}"""

depthwise_convolution_template = nvidia_copyright + """
problem:
  shape:
    name: "CNN-Layer"
    dimensions: [ C, R, S, N, P, Q ]
    coefficients:
      - name: Wstride
        default: 1
      - name: Hstride
        default: 1
      - name: Wdilation
        default: 1
      - name: Hdilation
        default: 1
    data-spaces:
      - name: Weights
        projection:
          - [ [C] ]
          - [ [R] ]
          - [ [S] ]
      - name: Inputs
        projection:
          - [ [N] ]
          - [ [C] ]
          - [ [R, Wdilation], [P, Wstride] ] # SOP form: R*Wdilation + P*Wstride
          - [ [S, Hdilation], [Q, Hstride] ] # SOP form: S*Hdilation + Q*Hstride
      - name: Outputs
        projection:
          - [ [N] ]
          - [ [C] ]
          - [ [Q] ]
          - [ [P] ]
        read-write: True
  instance:
    C: {c}
    N: {n}
    P: {p}
    Q: {q}
    R: {r}
    S: {s}
    Wdilation: {wd}
    Wstride: {ws}
    Hdilation: {wd}
    Hstride: {hs}"""