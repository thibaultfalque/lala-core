// Copyright 2021 Pierre Talbot, Frédéric Pinel, Cem Guvel

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DUAL_HPP
#define DUAL_HPP

#include <cmath>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>

template <typename T> 
struct Dual {
  T element;

  CUDA Dual<T> (T element): element(element) {}

  CUDA static Dual<T> bot () {
    return Dual<T>(T::top());
  }

  CUDA static Dual<T> top () {
    return Dual<T>(T::bot());
  }

  CUDA void join (const Dual<T>& other) {
    element.meet(other.element);
  }

  CUDA void meet (const Dual<T>& other) {
    element.join(other.element);
  }
};

#endif
